from functools import partial
from typing import Callable, Dict, Tuple
import logging
import os
import torch 
from tqdm import tqdm
from jaxtyping import Float
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformer_lens import ActivationCache
import transformer_lens.utils as utils
from . import PatchedModel
from .hooked.HookedDistilBert import HookedDistilBert
from .hooked.HookedEncoder import HookedEncoder
from .hooked.loading_from_pretrained import get_official_model_name
from ..util import batched_dot_product, linear_rank_function

logger = logging.getLogger(__name__)

POOLING = {
    'cls' : lambda x : x[:,0,:],
    'mean' : lambda x : x.mean(dim=1),
}

def get_hooked(architecture):
    huggingface_token = os.environ.get("HF_TOKEN", None)
    hf_config = AutoConfig.from_pretrained(
        get_official_model_name(architecture),
        token=huggingface_token
    )
    architecture = hf_config.architectures[0]
    if "distilbert" in architecture.lower(): return HookedDistilBert
    return HookedEncoder

class Dot(PatchedModel):
    def __init__(self, 
                 model_name_or_path : str,
                 pooling_type : str = 'cls',
                 tokenizer = None,
                 special_token: str = "X",
                 ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        self.special_token = special_token
        
        super().__init__(model_name_or_path, AutoModel.from_pretrained, get_hooked(model_name_or_path))

        self._model_forward = partial(self._model, return_type="embeddings")
        self._model_run_with_cache = partial(self._model.run_with_cache, return_type="embeddings")
        self._model_run_with_hooks = partial(self._model.run_with_hooks, return_type="embeddings")

        self._pooling_type = pooling_type
        self._pooling = POOLING[pooling_type]

    def _forward(self, 
                input_ids : torch.Tensor,
                attention_mask : torch.Tensor,
                ) -> torch.Tensor:

        return self._pooling(self._model_forward(input_ids, one_zero_attention_mask=attention_mask))

    def _forward_cache(self, 
                input_ids : torch.Tensor,
                attention_mask : torch.Tensor,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        reps, cached = self._model_run_with_cache(input_ids, one_zero_attention_mask=attention_mask)
        return self._pooling(reps), cached

    def _get_act_patch_block_every(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"], 
        clean_cache: ActivationCache, 
        reps_q : Float[torch.Tensor, "batch pos model"],
        patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], float],
        scores : Float[torch.Tensor, "batch pos"],
        scores_p : Float[torch.Tensor, "batch pos"],
        **kwargs
    ) -> Float[torch.Tensor, "3 layer pos"]:
        '''
        Returns an array of results of patching each position at each layer in the residual
        stream, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's logit output.
        '''

        self._model.reset_hooks()
        _, seq_len = corrupted_tokens["input_ids"].size()
        results = torch.zeros(3, self._model.cfg.n_layers, seq_len, device=self._device, dtype=torch.float32)

        # send tokens to device if not already there
        corrupted_tokens["input_ids"] = corrupted_tokens["input_ids"].to(self._device)
        corrupted_tokens["attention_mask"] = corrupted_tokens["attention_mask"].to(self._device)

        for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
            for layer in range(self._model.cfg.n_layers):
                for position in range(seq_len):
                    hook_fn = partial(self._patch_residual_component, pos=position, clean_cache=clean_cache)
                    patched_outputs =  self._model_run_with_hooks(
                        corrupted_tokens["input_ids"],
                        one_zero_attention_mask=corrupted_tokens["attention_mask"],
                        fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
                    )
                    patched_outputs = batched_dot_product(reps_q, self._pooling(patched_outputs))
                    if patched_outputs.size(0) != 1: patched_outputs = patched_outputs.mean(dim=0)
                    results[component_idx, layer, position] = patching_metric(patched_outputs, scores, scores_p)

        return results

    
    def _get_act_patch_attn_head_out_all_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"], 
        clean_cache: ActivationCache, 
        reps_q : Float[torch.Tensor, "batch pos model"],
        patching_metric: Callable,
        scores : Float[torch.Tensor, "batch pos"],
        scores_p : Float[torch.Tensor, "batch pos"],
        **kwargs
    ) -> Float[torch.Tensor, "layer head"]:
        '''
        Returns an array of results of patching at all positions for each head in each
        layer, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's embedding output.
        '''

        self._model.reset_hooks()
        results = torch.zeros(self._model.cfg.n_layers, self._model.cfg.n_heads, device=self._device, dtype=torch.float32)

        for layer in range(self._model.cfg.n_layers):
            for head in range(self._model.cfg.n_heads):
                hook_fn = partial(self._patch_head_vector, head_index=head, clean_cache=clean_cache)
                patched_outputs =  self._model_run_with_hooks(
                        corrupted_tokens["input_ids"],
                        one_zero_attention_mask=corrupted_tokens["attention_mask"],
                        fwd_hooks = [(utils.get_act_name("z", layer), hook_fn)],
                    )
                patched_outputs = batched_dot_product(reps_q, self._pooling(patched_outputs))
                if patched_outputs.size(0) != 1: patched_outputs = patched_outputs.mean(dim=0)
                batch_results = patching_metric(patched_outputs, scores, scores_p)
                results[layer, head] = batch_results if batch_results.size(0) == 1 else batch_results.mean(dim=0).mean() # hacky fix, should change later

        return results


    def _get_act_patch_attn_head_by_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"], 
        reps_q : Float[torch.Tensor, "batch pos model"],
        clean_cache: ActivationCache, 
        layer_head_list,
        patching_metric: Callable,
        scores : Float[torch.Tensor, "batch pos"],
        scores_p : Float[torch.Tensor, "batch pos"],
        **kwargs
    ) -> Float[torch.Tensor, "layer pos head"]:
        
        self._model.reset_hooks()
        _, seq_len = corrupted_tokens["input_ids"].size()
        results = torch.zeros(2, len(layer_head_list), seq_len, device=self._device, dtype=torch.float32)

        for component_idx, component in enumerate(["z", "pattern"]):
            for i, layer_head in enumerate(layer_head_list):
                layer = layer_head[0]
                head = layer_head[1]
                for position in range(seq_len):
                    patch_fn = self._patch_head_vector_by_pos_pattern if component == "pattern" else self._patch_head_vector_by_pos
                    hook_fn = partial(patch_fn, pos=position, head_index=head, clean_cache=clean_cache)
                    patched_outputs = self._model_run_with_hooks(
                        corrupted_tokens["input_ids"],
                        one_zero_attention_mask=corrupted_tokens["attention_mask"],
                        fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
                    )
                    patched_outputs = batched_dot_product(reps_q, self._pooling(patched_outputs))
                    if patched_outputs.size(0) != 1: patched_outputs = patched_outputs.mean(dim=0)
                    results[component_idx, i, position] = patching_metric(patched_outputs, scores, scores_p)

        return results
    

    def score(self,
            queries : dict,
            documents : dict,
            reps_q=None,
            cache=False
    ):
        if reps_q is None: reps_q = self._forward(queries['input_ids'], queries['attention_mask'])
        if cache: 
            reps_d, cache_d = self._forward_cache(documents['input_ids'], documents['attention_mask'])
            return batched_dot_product(reps_q, reps_d), reps_q, reps_d, cache_d
        reps_d = self._forward(documents['input_ids'], documents['attention_mask'])
        return batched_dot_product(reps_q, reps_d), reps_q, reps_d
    
    def __call__(
            self, 
            queries : dict, 
            documents : dict,
            documents_p : dict,
            patch_type : str = 'block_all',
            layer_head_list : list = [],
            patching_metric: Callable = linear_rank_function,
    ):  
        assert patch_type in self._patch_funcs, f"Patch type {patch_type} not recognized. Choose from {self._patch_funcs.keys()}"
        scores, reps_q, _ = self.score(queries, documents)
        scores_p, _, _, cache_d = self.score(queries, documents_p, cache=True, reps_q=reps_q)

        patching_kwargs = {
            'corrupted_tokens' : documents,
            'clean_cache' : cache_d,
            'reps_q' : reps_q,
            'patching_metric' : patching_metric,
            'layer_head_list' : layer_head_list,
            'scores' : scores,
            'scores_p' : scores_p,
        }

        patched_output = self._patch_funcs[patch_type](**patching_kwargs)
        if self._return_cache: return patched_output, cache_d
        return patched_output #PatchingOutput(output, scores, scores_p)