import torch
from typing import Any, Dict, Tuple, Callable
from jaxtyping import Float
from transformer_lens import ActivationCache
import transformer_lens.utils as utils
from .hooked.loading_from_pretrained import get_official_model_name
from abc import ABC, abstractmethod
from functools import partial
from transformer_lens.hook_points import HookPoint


class PatchedMixin(ABC):
    def __init__(self) -> None:
        super().__init__()

    def _patch_residual_component(
        self,
        corrupted_component: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
        pos: int,
        clean_cache: ActivationCache,
    ):
        """
        Patches a given sequence position in the residual stream, using the value
        from the clean cache.
        """
        corrupted_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
        return corrupted_component

    def _patch_head_vector(
        self,
        corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
        hook,  #: HookPoint,
        head_index: int,
        clean_cache: ActivationCache,
        **kwargs,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        """
        Patches the output of a given head (before it's added to the residual stream) at
        every sequence position, using the value from the clean cache.
        """

        corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][
            :, :, head_index
        ]
        return corrupted_head_vector

    def _patch_head_vector_by_pos_pattern(
        self,
        corrupted_activation: Float[torch.Tensor, "batch pos head_index pos_q pos_k"],
        hook,  #: HookPoint,
        pos,
        head_index: int,
        clean_cache: ActivationCache,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

        corrupted_activation[:, head_index, pos, :] = clean_cache[hook.name][
            :, head_index, pos, :
        ]
        return corrupted_activation

    def _patch_head_vector_by_pos(
        self,
        corrupted_activation: Float[torch.Tensor, "batch pos head_index d_head"],
        hook,  #: HookPoint,
        pos,
        head_index: int,
        clean_cache: ActivationCache,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

        corrupted_activation[:, pos, head_index] = clean_cache[hook.name][
            :, pos, head_index
        ]
        return corrupted_activation

    def _get_act_patch_block_every(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        **kwargs,
    ) -> Float[torch.Tensor, "3 layer pos"]:
        """
        Returns an array of results of patching each position at each layer in the residual
        stream, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's logit output.
        """

        self._model.reset_hooks()
        _, seq_len = corrupted_tokens["input_ids"].size()
        # send tokens to device if not already there
        corrupted_tokens["input_ids"] = corrupted_tokens["input_ids"].to(self._device)
        corrupted_tokens["attention_mask"] = corrupted_tokens["attention_mask"].to(
            self._device
        )

        for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
            for layer in range(self._model.cfg.n_layers):
                for position in range(seq_len):
                    hook_fn = partial(
                        self._patch_residual_component,
                        pos=position,
                        clean_cache=clean_cache,
                    )
                    patched_outputs = self._model_run_with_hooks(
                        corrupted_tokens["input_ids"],
                        one_zero_attention_mask=corrupted_tokens["attention_mask"],
                        fwd_hooks=[(utils.get_act_name(component, layer), hook_fn)],
                    )
                    yield (component_idx, layer, position), patched_outputs

    def _get_act_patch_attn_head_out_all_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        **kwargs,
    ) -> Float[torch.Tensor, "layer head"]:
        """
        Returns an array of results of patching at all positions for each head in each
        layer, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's embedding output.
        """

        self._model.reset_hooks()
        for layer in range(self._model.cfg.n_layers):
            for head in range(self._model.cfg.n_heads):
                hook_fn = partial(
                    self._patch_head_vector, head_index=head, clean_cache=clean_cache
                )
                patched_outputs = self._model_run_with_hooks(
                    corrupted_tokens["input_ids"],
                    one_zero_attention_mask=corrupted_tokens["attention_mask"],
                    fwd_hooks=[(utils.get_act_name("z", layer), hook_fn)],
                )
                yield (layer, head), patched_outputs

    def _get_act_patch_attn_head_by_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        layer_head_list,
        **kwargs,
    ) -> Float[torch.Tensor, "layer pos head"]:
        self._model.reset_hooks()
        _, seq_len = corrupted_tokens["input_ids"].size()

        for component_idx, component in enumerate(["z", "pattern"]):
            for i, layer_head in enumerate(layer_head_list):
                layer = layer_head[0]
                head = layer_head[1]
                for position in range(seq_len):
                    patch_fn = (
                        self._patch_head_vector_by_pos_pattern
                        if component == "pattern"
                        else self._patch_head_vector_by_pos
                    )
                    hook_fn = partial(
                        patch_fn, pos=position, head_index=head, clean_cache=clean_cache
                    )
                    patched_outputs = self._model_run_with_hooks(
                        corrupted_tokens["input_ids"],
                        one_zero_attention_mask=corrupted_tokens["attention_mask"],
                        fwd_hooks=[(utils.get_act_name(component, layer), hook_fn)],
                    )
                    yield (component_idx, i, position), patched_outputs

    @abstractmethod
    def _model_forward(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the _model_forward method"
        )

    @abstractmethod
    def _model_run_with_cache(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the _model_run_with_cache method"
        )

    @abstractmethod
    def _model_run_with_hooks(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the _model_run_with_hooks method"
        )

    @abstractmethod
    def get_act_patch_attn_head_out_all_pos(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the get_act_patch_attn_head_out_all_pos method"
        )

    @abstractmethod
    def get_act_patch_attn_head_by_pos(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the get_act_patch_attn_head_by_pos method"
        )

    @abstractmethod
    def get_act_patch_block_every(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the get_act_patch_block_every method"
        )


class PatchedModel(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        model_func: Any,
        hook_obj: Any,
        return_cache: bool = False,
    ) -> None:
        torch.set_grad_enabled(False)
        self._device = utils.get_device()
        self.model_name_or_path = get_official_model_name(model_name_or_path)
        self.__hf_model = model_func(self.model_name_or_path).eval().to(self._device)

        self._model = hook_obj.from_pretrained(
            self.model_name_or_path, device=self._device, hf_model=self.__hf_model
        )

        self._model_forward = None
        self._model_run_with_cache = None
        self._model_run_with_hooks = None

        self._patch_funcs = {
            "block_all": self._get_act_patch_block_every,
            "head_all": self._get_act_patch_attn_head_out_all_pos,
            "head_by_pos": self._get_act_patch_attn_head_by_pos,
        }

        self._return_cache = return_cache

    @abstractmethod
    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        raise NotImplementedError(
            "Instantiate a subclass of PatchedModel and implement the _forward method"
        )

    @abstractmethod
    def _forward_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        raise NotImplementedError(
            "Instantiate a subclass of PatchedModel and implement the _forward method"
        )

    def _patch_residual_component(
        self,
        corrupted_component: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
        pos: int,
        clean_cache: ActivationCache,
    ):
        """
        Patches a given sequence position in the residual stream, using the value
        from the clean cache.
        """
        corrupted_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
        return corrupted_component

    def _patch_head_vector(
        self,
        corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
        hook,  #: HookPoint,
        head_index: int,
        clean_cache: ActivationCache,
        **kwargs,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        """
        Patches the output of a given head (before it's added to the residual stream) at
        every sequence position, using the value from the clean cache.
        """

        corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][
            :, :, head_index
        ]
        return corrupted_head_vector

    def _patch_head_vector_by_pos_pattern(
        self,
        corrupted_activation: Float[torch.Tensor, "batch pos head_index pos_q pos_k"],
        hook,  #: HookPoint,
        pos,
        head_index: int,
        clean_cache: ActivationCache,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

        corrupted_activation[:, head_index, pos, :] = clean_cache[hook.name][
            :, head_index, pos, :
        ]
        return corrupted_activation

    def _patch_head_vector_by_pos(
        self,
        corrupted_activation: Float[torch.Tensor, "batch pos head_index d_head"],
        hook,  #: HookPoint,
        pos,
        head_index: int,
        clean_cache: ActivationCache,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

        corrupted_activation[:, pos, head_index] = clean_cache[hook.name][
            :, pos, head_index
        ]
        return corrupted_activation

    @abstractmethod
    def _get_act_patch_attn_head_out_all_pos(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedModel and implement the _get_act_patch_attn_head_out_all_pos method"
        )

    @abstractmethod
    def _get_act_patch_attn_head_by_pos(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedModel and implement the _get_act_patch_attn_head_by_pos method"
        )

    @abstractmethod
    def _get_act_patch_block_every(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedModel and implement the _get_act_patch_block_every method"
        )

    @abstractmethod
    def score(self, queries: dict, documents: dict, cache=False):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedModel and implement the score method"
        )

    @abstractmethod
    def __call__(
        self,
        queries: dict,
        documents: dict,
        queries_p: dict,
        documents_p: dict,
        patching_metric: Callable,
        patch_type: str = "block_all",
        layer_head_list: list = [],
    ):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedModel and implement the __call__ method"
        )


from .cat import Cat
from .dot import Dot
from .t5 import MonoT5
