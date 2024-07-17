import torch 
from functools import partial
from typing import Any, Dict, Tuple, Callable
from jaxtyping import Float
from transformer_lens import ActivationCache
import transformer_lens.utils as utils
from abc import ABC, abstractmethod

class PatchedModel(object, ABC):
    def __init__(self,
                 model_name_or_path : str,
                 model_func : Any,
                 hook_obj : Any,
                 ) -> None:
        torch.set_grad_enabled(False)
        self._device = utils.get_device()

        self.__hf_model = model_func(model_name_or_path).eval().to(self._device)
        self._model = hook_obj.from_pretrained(model_name_or_path, device=self._device, hf_model=self.__hf_model)

        self._model_forward = None
        self._model_run_with_cache = None
        self._model_run_with_hooks = None

        self._patch_funcs = {
            'block_all' : self._get_act_patch_block_every,
            'head_all' : self._get_act_patch_attn_head_out_all_pos,
            'head_by_pos' : self._get_act_patch_attn_head_by_pos,
        }


    @abstractmethod
    def _forward(self, 
                input_ids : torch.Tensor,
                attention_mask : torch.Tensor,
                ) -> torch.Tensor:

        raise NotImplementedError("Instantiate a subclass of PatchedModel and implement the _forward method")


    @abstractmethod
    def _forward_cache(self, 
                input_ids : torch.Tensor,
                attention_mask : torch.Tensor,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        raise NotImplementedError("Instantiate a subclass of PatchedModel and implement the _forward method")


    def _patch_head_vector(
        corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
        hook, #: HookPoint, 
        head_index: int, 
        clean_cache: ActivationCache,
        **kwargs
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        '''
        Patches the output of a given head (before it's added to the residual stream) at
        every sequence position, using the value from the clean cache.
        '''
    
        corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
        return corrupted_head_vector

    
    def _patch_head_vector_by_pos_pattern(
        corrupted_activation: Float[torch.Tensor, "batch pos head_index pos_q pos_k"],
        hook, #: HookPoint, 
        pos,
        head_index: int, 
        clean_cache: ActivationCache
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

        corrupted_activation[:,head_index,pos,:] = clean_cache[hook.name][:,head_index,pos,:]
        return corrupted_activation
    

    def _patch_head_vector_by_pos(
        corrupted_activation: Float[torch.Tensor, "batch pos head_index d_head"],
        hook, #: HookPoint, 
        pos,
        head_index: int, 
        clean_cache: ActivationCache
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

        corrupted_activation[:, pos, head_index] = clean_cache[hook.name][:, pos, head_index]
        return corrupted_activation
    
    @abstractmethod
    def _get_act_patch_attn_head_out_all_pos(*args, **kwargs):
        raise NotImplementedError("Instantiate a subclass of PatchedModel and implement the _get_act_patch_attn_head_out_all_pos method")
    

    @abstractmethod
    def _get_act_patch_attn_head_by_pos(*args, **kwargs):
        raise NotImplementedError("Instantiate a subclass of PatchedModel and implement the _get_act_patch_attn_head_by_pos method")
    

    @abstractmethod
    def _get_act_patch_block_every(*args, **kwargs):
        raise NotImplementedError("Instantiate a subclass of PatchedModel and implement the _get_act_patch_block_every method")
    
    @abstractmethod
    def score(self,
            queries : dict,
            documents : dict,
            cache=False
    ):
        raise NotImplementedError("Instantiate a subclass of PatchedModel and implement the score method")

    @abstractmethod
    def __call__(
            self, 
            queries : dict, 
            documents : dict,
            queries_p : dict,
            documents_p : dict,
            patching_metric: Callable,
            patch_type : str = 'block_all',
            layer_head_list : list = [],
    ):  
        raise NotImplementedError("Instantiate a subclass of PatchedModel and implement the __call__ method")