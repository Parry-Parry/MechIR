"""Backbone dispatch and pooling.

Maps a checkpoint's HuggingFace architecture onto the hooked class that can
host it. Encoder checkpoints go to mechir's own ported encoders; decoder
checkpoints go to TransformerLens's HookedTransformer, which already covers
the Qwen/Llama/Mistral families that modern IR models are built on.
"""

import logging
from typing import Callable, Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformer_lens import HookedTransformer

from mechir.modelling.architectures import HookedDistilBert, HookedEncoder

logger = logging.getLogger(__name__)

# Architectures TransformerLens hosts as decoder-only transformers. Matched as
# substrings, so the fine-tuned IR variants (Qwen3-Embedding, Qwen3-Reranker,
# gte-Qwen2, RankLLaMA) are covered automatically -- they leave the
# architecture field untouched.
DECODER_ARCHITECTURES = (
    "Qwen3ForCausalLM",
    "Qwen2ForCausalLM",
    "QWenLMHeadModel",
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Phi3ForCausalLM",
    "GPTNeoXForCausalLM",
    "GPT2LMHeadModel",
)


def _is_decoder(architecture: str) -> bool:
    return any(a.lower() in architecture.lower() for a in DECODER_ARCHITECTURES)


def hf_architecture(model_name_or_path: str) -> str:
    """The checkpoint's declared architecture, or "" when it declares none."""
    config = AutoConfig.from_pretrained(model_name_or_path)
    architectures = getattr(config, "architectures", None) or []
    return architectures[0] if architectures else ""


def get_backbone(model_name_or_path: str) -> Tuple[Callable, Callable, bool]:
    """Resolve a checkpoint to the classes that can load it.

    Returns
    -------
    (hooked_cls, hf_cls, is_decoder)
        ``hooked_cls`` hosts the hooked model, ``hf_cls`` loads the source
        weights, and ``is_decoder`` selects the forward convention --
        decoders have no ``return_type="embeddings"`` and need the final
        norm applied by hand.
    """
    architecture = hf_architecture(model_name_or_path)

    if _is_decoder(architecture):
        return HookedTransformer, AutoModelForCausalLM, True
    if "distilbert" in architecture.lower():
        return HookedDistilBert, AutoModel, False
    if not architecture:
        logger.warning(
            "%s declares no architecture; assuming a BERT-style encoder.",
            model_name_or_path,
        )
    return HookedEncoder, AutoModel, False


def load_backbone(model_name_or_path: str, official_name: str, device, **kwargs):
    """Build the hooked model and return it with its source model.

    Decoder backbones load with weight processing disabled so activations
    match the source model, which is what diagnostic work needs.
    """
    hooked_cls, hf_cls, is_decoder = get_backbone(model_name_or_path)
    hf_model = hf_cls.from_pretrained(model_name_or_path).eval().to(device)

    if is_decoder:
        hooked = hooked_cls.from_pretrained(
            official_name,
            hf_model=hf_model,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            **kwargs,
        )
    else:
        hooked = hooked_cls.from_pretrained(
            official_name, hf_model=hf_model, device=device, **kwargs
        )

    return hooked, hf_model, is_decoder


def _pool_cls(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return x[:, 0, :]


def _pool_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean over non-padding positions.

    Averaging over padding, as this previously did, makes the score depend
    on how far the batch happened to be padded: the same document scores
    differently at ``max_length=180`` and ``max_length=220``. Masking is
    also what the encoder bi-encoders this wraps -- TAS-B and friends --
    actually do, so the unmasked version did not reproduce the source model.

    The mask is optional only so that unpadded single sequences still work;
    omitting it when the batch *is* padded reintroduces the bug.
    """
    if mask is None:
        return x.mean(dim=1)

    mask = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def _pool_last(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Final non-padding token, as used by decoder bi-encoders.

    The mask is required for correctness on padded batches: with right
    padding the last real token sits at a different index in every row.
    """
    if mask is None:
        return x[:, -1, :]
    last = mask.sum(dim=1) - 1
    return x[torch.arange(x.shape[0], device=x.device), last]


POOLING: Dict[str, Callable] = {
    "cls": _pool_cls,
    "mean": _pool_mean,
    "last": _pool_last,
}

__all__ = [
    "POOLING",
    "DECODER_ARCHITECTURES",
    "get_backbone",
    "hf_architecture",
    "load_backbone",
]
