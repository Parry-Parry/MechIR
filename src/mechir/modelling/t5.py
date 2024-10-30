from functools import partial
from typing import Callable, Dict, Tuple
import logging
import torch 
from jaxtyping import Float
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformer_lens import HookedEncoderDecoder, ActivationCache
import transformer_lens.utils as utils
from . import PatchedModel
from ..util import linear_rank_function

logger = logging.getLogger(__name__)

'''
monoT5
'''

class MonoT5(PatchedModel):
    def __init__(self, 
                 model_name_or_path : str,
                 pos_token : str = "true",
                 neg_token : str = "false",
                 special_token: str = "X",
                 softmax_output: bool = False
                 ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path) 
        
        self.special_token = special_token
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.special_token]})
        self.special_token_id = self.tokenizer.convert_tokens_to_ids(self.special_token)

        self.pos_token = self.tokenizer.encode(pos_token, return_tensors="pt")[0]
        self.neg_token = self.tokenizer.encode(neg_token, return_tensors="pt")[0]

        super().__init__(model_name_or_path, AutoModelForSeq2SeqLM.from_pretrained, HookedEncoderDecoder, tokenizer_len=len(self.tokenizer))
        self._model_forward = partial(self._model, return_type="logits")
        self._model_run_with_cache = partial(self._model.run_with_cache, return_type="logits")
        self._model_run_with_hooks = partial(self._model.run_with_hooks, return_type="logits")

        self.softmax_output = softmax_output


    def _forward(self, 
                input_ids : torch.Tensor,
                attention_mask : torch.Tensor,
                ) -> torch.Tensor:

        return self._model_forward(input_ids, one_zero_attention_mask=attention_mask)

    def _forward_cache(self, 
                input_ids : torch.Tensor,
                attention_mask : torch.Tensor,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        logits, cached = self._model_run_with_cache(input_ids, one_zero_attention_mask=attention_mask)
        return logits, cached


    def _get_act_patch_block_every(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"], 
        clean_cache: ActivationCache, 
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
                    results[component_idx, layer, position] = patching_metric(patched_outputs, scores, scores_p)

        return results

    
    def _get_act_patch_attn_head_out_all_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"], 
        clean_cache: ActivationCache, 
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
                results[layer, head] = patching_metric(patched_outputs, scores, scores_p)
                
        return results


    def _get_act_patch_attn_head_by_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"], 
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
                    
                    results[component_idx, i, position] = patching_metric(patched_outputs, scores, scores_p)

        return results
    

    def score(self,
            sequences : dict,
            cache=False
    ):
        if cache: 
            logits, cache = self._forward_cache(sequences['input_ids'], sequences['attention_mask'])
            scores = logits[:, 0, (self.pos_token, self.neg_token)].softmax(dim=-1)[:, 0] if self.softmax_output else logits[:, 0, (self.pos_token, self.neg_token)][:, 0]
            return scores, logits, cache
       
        logits = self._forward(sequences['input_ids'], sequences['attention_mask'])
        scores = logits[:, 0, (self.pos_token, self.neg_token)].softmax(dim=-1)[:, 0] if self.softmax_output else logits[:, 0, (self.pos_token, self.neg_token)][:, 0]
        return scores, logits
    

    def __call__(
            self, 
            sequences : dict,
            sequences_p : dict,
            patch_type : str = 'block_all',
            layer_head_list : list = [],
            patching_metric: Callable = linear_rank_function,
    ):  
        assert patch_type in self._patch_funcs, f"Patch type {patch_type} not recognized. Choose from {self._patch_funcs.keys()}"
        scores, _ = self.score(sequences)
        scores_p, _, cache = self.score(sequences_p, cache=True)

        patching_kwargs = {
            'corrupted_tokens' : sequences,
            'clean_cache' : cache,
            'patching_metric' : patching_metric,
            'layer_head_list' : layer_head_list,
            'scores' : scores,
            'scores_p' : scores_p,
        }
        patched_output = self._patch_funcs[patch_type](**patching_kwargs)
        return patched_output