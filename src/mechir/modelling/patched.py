"""Activation patching shared by every ranker family.

The patching loops used to be copied into each ranker -- ``Dot``, ``Cat``,
``MonoT5`` and ``GenerativeRanker`` each carried a near-identical
``get_act_patch_block_every`` / ``get_act_patch_attn_head_out_all_pos`` /
``get_act_patch_attn_head_by_pos``. Only two things actually varied between
the copies, and both are now strategy seams on :class:`PatchedMixin`:

``_patch_score``
    How a patched forward pass becomes one score per batch item. One
    override per *ranker family*.
``_hook_names``
    How a ``(component, layer)`` pair becomes a hook name. One strategy per
    *architecture*; see :mod:`mechir.modelling.hook_names`.

So a new ranker family costs one scoring method and a new patching method
costs one driver, rather than one method per (family, method) pair.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, Iterator, List, Tuple

import torch
from jaxtyping import Float
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint

from mechir.modelling.hook_names import BLOCK_HOOK_NAMES, HookNameStrategy
from mechir.util import linear_rank_function

#: Residual-stream components swept by ``get_act_patch_block_every``, in the
#: order of the leading axis of its result.
BLOCK_COMPONENTS: Tuple[str, ...] = ("resid_pre", "attn_out", "mlp_out")

#: Attention components swept by ``get_act_patch_attn_head_by_pos``, in the
#: order of the leading axis of its result.
HEAD_COMPONENTS: Tuple[str, ...] = ("z", "pattern")

#: ``(index, patched_output)`` pairs, one per corrupted forward pass.
PatchStream = Iterator[Tuple[Tuple[int, ...], torch.Tensor]]


class PatchedMixin(ABC):
    """Activation patching for a hooked ranker.

    Subclasses supply a ``forward`` that returns a score (or, for
    :class:`~mechir.Dot`, document embeddings plus a ``_patch_score``
    override) and inherit every patching method.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = None

    @property
    def _patch_funcs(self) -> Dict[str, Callable]:
        """Patching methods by the ``patch_type`` name callers pass."""
        return {
            "block_all": self.get_act_patch_block_every,
            "head_all": self.get_act_patch_attn_head_out_all_pos,
            "head_by_pos": self.get_act_patch_attn_head_by_pos,
        }

    # -- the two strategy seams ---------------------------------------------

    @property
    def _hook_names(self) -> HookNameStrategy:
        """Hook naming for this ranker's architecture.

        Defaults to the single-stack ``blocks.{layer}.…`` scheme. Encoder-
        decoder rankers override this; see
        :mod:`mechir.modelling.hook_names`.
        """
        return BLOCK_HOOK_NAMES

    def _patch_score(self, patched_output: torch.Tensor, **context) -> torch.Tensor:
        """Reduce one patched forward pass to a score per batch item.

        The default is the identity, which suits every ranker whose
        ``forward`` already returns a score. ``Dot`` returns document
        embeddings instead and overrides this to contract them against the
        query embeddings handed through ``context``.

        Parameters
        ----------
        patched_output: torch.Tensor
            Whatever ``forward`` returned under the patching hooks.
        **context
            Surplus keyword arguments from the patching call, carrying
            whatever the ranker family needs to score with.
        """
        return patched_output

    # -- the patching drivers ----------------------------------------------

    def get_act_patch_block_every(
        self,
        corrupted_tokens: Dict[str, torch.Tensor],
        clean_cache: ActivationCache,
        patching_metric: Callable = linear_rank_function,
        scores: torch.Tensor = None,
        scores_p: torch.Tensor = None,
        **kwargs,
    ) -> Float[torch.Tensor, "3 layer pos"]:
        """Patch every position of every layer in the residual stream.

        Returns ``(3, n_layers, seq)``, one plane per component of
        :data:`BLOCK_COMPONENTS`, each entry the batch mean of
        ``patching_metric``.
        """
        seq_len = corrupted_tokens["input_ids"].shape[1]
        return self._reduce_patches(
            self._new_results(len(BLOCK_COMPONENTS), self._n_layers, seq_len),
            self._get_act_patch_block_every(corrupted_tokens, clean_cache),
            patching_metric,
            scores,
            scores_p,
            kwargs,
        )

    def get_act_patch_attn_head_out_all_pos(
        self,
        corrupted_tokens: Dict[str, torch.Tensor],
        clean_cache: ActivationCache,
        patching_metric: Callable = linear_rank_function,
        scores: torch.Tensor = None,
        scores_p: torch.Tensor = None,
        **kwargs,
    ) -> Float[torch.Tensor, "layer head"]:
        """Patch each attention head's output at every position.

        Returns ``(n_layers, n_heads)``, each entry the batch mean of
        ``patching_metric``.
        """
        return self._reduce_patches(
            self._new_results(self._n_layers, self._model.cfg.n_heads),
            self._get_act_patch_attn_head_out_all_pos(corrupted_tokens, clean_cache),
            patching_metric,
            scores,
            scores_p,
            kwargs,
        )

    def get_act_patch_attn_head_by_pos(
        self,
        corrupted_tokens: Dict[str, torch.Tensor],
        clean_cache: ActivationCache,
        layer_head_list: List[Tuple[int, int]] = (),
        patching_metric: Callable = linear_rank_function,
        scores: torch.Tensor = None,
        scores_p: torch.Tensor = None,
        **kwargs,
    ) -> Float[torch.Tensor, "2 head pos"]:
        """Patch selected heads one position at a time.

        Returns ``(2, len(layer_head_list), seq)``, one plane per component
        of :data:`HEAD_COMPONENTS`, each entry the batch mean of
        ``patching_metric``.
        """
        seq_len = corrupted_tokens["input_ids"].shape[1]
        return self._reduce_patches(
            self._new_results(len(HEAD_COMPONENTS), len(layer_head_list), seq_len),
            self._get_act_patch_attn_head_by_pos(
                corrupted_tokens, clean_cache, layer_head_list
            ),
            patching_metric,
            scores,
            scores_p,
            kwargs,
        )

    # -- scoring, once for every driver and every family -------------------

    def _reduce_patches(
        self,
        results: torch.Tensor,
        stream: PatchStream,
        patching_metric: Callable,
        scores: torch.Tensor,
        scores_p: torch.Tensor,
        context: Dict[str, object],
    ) -> torch.Tensor:
        """Score each patched run in ``stream`` and write it into ``results``.

        This is the single place where a patched model output becomes a
        number, for every ranker family and every patching method.
        """
        for index, patched_output in stream:
            score = self._patch_score(patched_output, **context)
            results[index] = patching_metric(score, scores, scores_p).mean()
        return results

    # -- enumeration: what gets patched ------------------------------------

    def _get_act_patch_block_every(
        self,
        corrupted_tokens: Dict[str, torch.Tensor],
        clean_cache: ActivationCache,
        **kwargs,
    ) -> PatchStream:
        """Yield one patched run per (component, layer, position)."""
        self._model.reset_hooks()
        batch = self._to_device(corrupted_tokens)
        seq_len = batch["input_ids"].shape[1]

        for component_idx, component in enumerate(BLOCK_COMPONENTS):
            for layer in range(self._n_layers):
                for position in range(seq_len):
                    hook_fn = partial(
                        self._patch_residual_component,
                        pos=position,
                        clean_cache=clean_cache,
                    )
                    yield (
                        (component_idx, layer, position),
                        self._run_patched(batch, component, layer, hook_fn),
                    )

    def _get_act_patch_attn_head_out_all_pos(
        self,
        corrupted_tokens: Dict[str, torch.Tensor],
        clean_cache: ActivationCache,
        **kwargs,
    ) -> PatchStream:
        """Yield one patched run per (layer, head)."""
        self._model.reset_hooks()
        batch = self._to_device(corrupted_tokens)

        for layer in range(self._n_layers):
            for head in range(self._model.cfg.n_heads):
                hook_fn = partial(
                    self._patch_head_vector, head_index=head, clean_cache=clean_cache
                )
                yield (
                    (layer, head),
                    self._run_patched(batch, "z", layer, hook_fn),
                )

    def _get_act_patch_attn_head_by_pos(
        self,
        corrupted_tokens: Dict[str, torch.Tensor],
        clean_cache: ActivationCache,
        layer_head_list: List[Tuple[int, int]] = (),
        **kwargs,
    ) -> PatchStream:
        """Yield one patched run per (component, selected head, position)."""
        self._model.reset_hooks()
        batch = self._to_device(corrupted_tokens)
        seq_len = batch["input_ids"].shape[1]

        for component_idx, component in enumerate(HEAD_COMPONENTS):
            patch_fn = (
                self._patch_head_vector_by_pos_pattern
                if component == "pattern"
                else self._patch_head_vector_by_pos
            )
            for head_idx, (layer, head) in enumerate(layer_head_list):
                for position in range(seq_len):
                    hook_fn = partial(
                        patch_fn,
                        pos=position,
                        head_index=head,
                        clean_cache=clean_cache,
                    )
                    yield (
                        (component_idx, head_idx, position),
                        self._run_patched(batch, component, layer, hook_fn),
                    )

    # -- running one corrupted forward pass --------------------------------

    def _run_patched(
        self,
        batch: Dict[str, torch.Tensor],
        component: str,
        layer: int,
        hook_fn: Callable,
    ) -> torch.Tensor:
        """Run ``forward`` on ``batch`` with ``hook_fn`` on one hook point."""
        return self.run_with_hooks(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **self._extra_forward_kwargs(batch),
            fwd_hooks=[(self._hook_names(component, layer), hook_fn)],
        )

    def _extra_forward_kwargs(self, batch: dict) -> dict:
        """Forward kwargs beyond input_ids/attention_mask, when the batch has them.

        ``token_type_ids`` carries the query/document split for
        cross-encoders; dropping it during patching scores a different input
        than ``score`` did. Encoder-decoder batches have no token types, so
        this stays empty there.
        """
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is None:
            return {}
        return {"token_type_ids": token_type_ids}

    @property
    def _n_layers(self) -> int:
        """Number of layers in the stack being patched."""
        return self._model.cfg.n_layers

    def _new_results(self, *shape: int) -> torch.Tensor:
        """A zeroed float32 result tensor on the ranker's device."""
        return torch.zeros(*shape, device=self._device, dtype=torch.float32)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Copy of ``batch`` with its tensors on the ranker's device.

        Index and mask tensors have to share a device with the activations
        they are used against. The caller's batch is left untouched, so
        patching the same batch twice is reproducible.
        """
        return {
            key: value.to(self._device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    # -- hook functions ----------------------------------------------------

    def _patch_residual_component(
        self,
        corrupted_component: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
        pos: int,
        clean_cache: ActivationCache,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Patch one sequence position of the residual stream."""
        corrupted_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
        return corrupted_component

    def _patch_head_vector(
        self,
        corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint,
        head_index: int,
        clean_cache: ActivationCache,
        **kwargs,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        """Patch one head's output at every sequence position."""
        corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][
            :, :, head_index
        ]
        return corrupted_head_vector

    def _patch_head_vector_by_pos_pattern(
        self,
        corrupted_activation: Float[torch.Tensor, "batch pos head_index pos_q pos_k"],
        hook: HookPoint,
        pos: int,
        head_index: int,
        clean_cache: ActivationCache,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        """Patch one head's attention pattern at one query position."""
        corrupted_activation[:, head_index, pos, :] = clean_cache[hook.name][
            :, head_index, pos, :
        ]
        return corrupted_activation

    def _patch_head_vector_by_pos(
        self,
        corrupted_activation: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint,
        pos: int,
        head_index: int,
        clean_cache: ActivationCache,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        """Patch one head's output at one sequence position."""
        corrupted_activation[:, pos, head_index] = clean_cache[hook.name][
            :, pos, head_index
        ]
        return corrupted_activation

    # -- supplied by the hooked ranker -------------------------------------

    @abstractmethod
    def forward(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the "
            "_model_forward method"
        )

    @abstractmethod
    def run_with_cache(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the "
            "_model_run_with_cache method"
        )

    @abstractmethod
    def run_with_hooks(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the "
            "_model_run_with_hooks method"
        )


__all__ = ["BLOCK_COMPONENTS", "HEAD_COMPONENTS", "PatchStream", "PatchedMixin"]
