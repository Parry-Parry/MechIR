# Description: This file contains the state dictionary for the models in the hooked library.
import torch
from .loading import register_architecture

def GPTNeoForCausalLM_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_heads,
        "n_heads": hf_config.num_heads,
        "d_mlp": hf_config.hidden_size * 4,
        "n_layers": hf_config.num_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.layer_norm_epsilon,
        "d_vocab": hf_config.vocab_size,
        "attn_types": hf_config.attention_layers,
        "act_fn": hf_config.activation_function,
        "use_attn_scale": False,
        "use_local_attn": True,
        "window_size": hf_config.window_size,
        "scale_attn_by_inverse_layer_idx": False,
        "normalization_type": "LN",
        }

register_architecture("GPTNeoForCausalLM", GPTNeoForCausalLM_state_dict)

def GPT2LMHeadModel_state_dict(hf_config):
    return {
        "d_model": hf_config.n_embd,
        "d_head": hf_config.n_embd // hf_config.n_head,
        "n_heads": hf_config.n_head,
        "d_mlp": hf_config.n_embd * 4,
        "n_layers": hf_config.n_layer,
        "n_ctx": hf_config.n_ctx,
        "eps": hf_config.layer_norm_epsilon,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.activation_function,
        "use_attn_scale": True,
        "use_local_attn": False,
        "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
        "normalization_type": "LN",
        }

register_architecture("GPT2LMHeadModel", GPT2LMHeadModel_state_dict)

def OPTForCausalLM_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.ffn_dim,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": 1e-5,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.activation_function,
        "use_attn_scale": True,
        "use_local_attn": False,
        "scale_attn_by_inverse_layer_idx": False,
        "normalization_type": "LN",
        }

register_architecture("OPTForCausalLM", OPTForCausalLM_state_dict)

def GPTJForCausalLM_state_dict(hf_config):
    return {
        "d_model": hf_config.n_embd,
        "d_head": hf_config.n_embd // hf_config.n_head,
        "n_heads": hf_config.n_head,
        "d_mlp": 4 * hf_config.n_embd,
        "n_layers": hf_config.n_layer,
        "n_ctx": hf_config.n_positions,
        "eps": 1e-5,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.activation_function,
        "use_attn_scale": True,
        "use_local_attn": False,
        "scale_attn_by_inverse_layer_idx": False,
        "parallel_attn_mlp": True,
        "positional_embedding_type": "rotary",
        "rotary_dim": hf_config.rotary_dim,
        "rotary_adjacent_pairs": True,
        "normalization_type": "LN",
        }

register_architecture("GPTJForCausalLM", GPTJForCausalLM_state_dict)

def GPT2LMHeadModel_state_dict(hf_config):
    state = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "normalization_type": "LN",
        }
    rotary_pct = hf_config.rotary_pct
    state["rotary_dim"] = round(rotary_pct * state["d_head"])
    return state

register_architecture("GPT2LMHeadModel", GPT2LMHeadModel_state_dict)

def GPTNeoForCausalLM_state_dict(hf_config):
    state = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_heads,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": False,
            "use_local_attn": True,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "normalization_type": "LN",
        }
    rotary_pct = hf_config.rotary_pct
    state["rotary_dim"] = round(rotary_pct * state["d_head"])
    return state

register_architecture("GPTNeoForCausalLM", GPTNeoForCausalLM_state_dict)

def BertModel_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.layer_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": "gelu",
        "attention_dir": "bidirectional",
    }

register_architecture("BertModel", BertModel_state_dict)
register_architecture("BertForMaskedLM", BertModel_state_dict)

def BertForSequenceClassification_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.layer_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": "gelu",
        "attention_dir": "bidirectional",
        "num_labels": hf_config.num_labels,
    }

register_architecture("BertForSequenceClassification", BertForSequenceClassification_state_dict)

def ElectraForPreTraining_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.layer_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": "gelu",
        "attention_dir": "bidirectional",
    }

register_architecture("ElectraForPreTraining", ElectraForPreTraining_state_dict)

def ElectraForSequenceClassification_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.layer_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": "gelu",
        "attention_dir": "bidirectional",
        "num_labels": hf_config.num_labels,
    }

register_architecture("ElectraForSequenceClassification", ElectraForSequenceClassification_state_dict)

def DistilBert_state_dict(hf_config):
    return {
        "d_model": hf_config.dim,
        "d_head": hf_config.dim // hf_config.n_heads,
        "n_heads": hf_config.n_heads,
        "d_mlp": hf_config.dim * 4,
        "n_layers": hf_config.n_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.layer_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": "gelu",
        "attention_dir": "bidirectional",
    }

register_architecture("DistilBert", DistilBert_state_dict)

def DistilBertForSequenceClassification_state_dict(hf_config):
    return {
        "d_model": hf_config.dim,
        "d_head": hf_config.dim // hf_config.n_heads,
        "n_heads": hf_config.n_heads,
        "d_mlp": hf_config.dim * 4,
        "n_layers": hf_config.n_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.layer_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": "gelu",
        "attention_dir": "bidirectional",
        "num_labels": hf_config.num_labels,
    }

register_architecture("DistilBertForSequenceClassification", DistilBertForSequenceClassification_state_dict)

def MistralForCausalLM_state_dict(hf_config):
    return {
        "d_model": 4096,
        "d_head": 4096 // 32,
        "n_heads": 32,
        "d_mlp": 14336,
        "n_layers": 32,
        "n_ctx": 2048,  # Capped due to memory issues
        "d_vocab": 32000,
        "act_fn": "silu",
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "window_size": 4096,
        "attn_types": ["local"] * 32,
        "eps": 1e-05,
        "n_key_value_heads": 8,
        "gated_mlp": True,
        "use_local_attn": True,
        "rotary_dim": 4096 // 32,
    }

register_architecture("MistralForCausalLM", MistralForCausalLM_state_dict)

def MixtralForCausalLM_state_dict(hf_config):
    return {
        "dtype": torch.bfloat16,
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,  # Capped due to memory issues
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.hidden_act,
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "rotary_base": hf_config.rope_theta,
        "window_size": hf_config.sliding_window,  # This is None, as no sliding window was used
        "attn_types": ["global"] * 32,
        "eps": hf_config.rms_norm_eps,
        "n_key_value_heads": hf_config.num_key_value_heads,
        "gated_mlp": True,
        "use_local_attn": False,
        "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
        "num_experts": hf_config.num_local_experts,
        "experts_per_token": hf_config.num_experts_per_tok,
    }

register_architecture("MixtralForCausalLM", MixtralForCausalLM_state_dict)

def BloomForCausalLM_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.n_head,
        "n_heads": hf_config.n_head,
        "d_mlp": hf_config.hidden_size * 4,
        "n_layers": hf_config.n_layer,
        "n_ctx": 2048,  # Capped due to HF Tokenizer Constraints
        "d_vocab": hf_config.vocab_size,
        "act_fn": "gelu_fast",
        "eps": hf_config.layer_norm_epsilon,
        "normalization_type": "LN",
        "post_embedding_ln": True,
        "positional_embedding_type": "alibi",
    }

register_architecture("BloomForCausalLM", BloomForCausalLM_state_dict)

def GPT2LMHeadCustomModel_state_dict(hf_config):
    return {
        "d_model": hf_config.n_embd,
        "d_head": hf_config.n_embd // hf_config.n_head,
        "n_heads": hf_config.n_head,
        "d_mlp": hf_config.n_embd * 4,
        "n_layers": hf_config.n_layer,
        "n_ctx": hf_config.n_positions,
        "eps": hf_config.layer_norm_epsilon,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.activation_function,
        "use_attn_scale": True,
        "use_local_attn": False,
        "trust_remote_code": False,  # Only santacoder needs trust_remote_code
        "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
        "normalization_type": "LN",
    }

register_architecture("GPT2LMHeadCustomModel", GPT2LMHeadCustomModel_state_dict)

def LlamaForCausalLM_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.rms_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.hidden_act,
        "n_key_value_heads": (
            hf_config.num_key_value_heads
            if hf_config.num_key_value_heads != hf_config.num_attention_heads
            else None
        ),
        # This is done because the current implementation of GQA will use Grouped-Query Attention if
        # n_key_value_heads is not None, but hf_config.num_key_value_heads is sometimes specified as
        # the same as hf_config.num_attention_heads, in which case GQA should not be used.
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "rotary_adjacent_pairs": False,
        "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
        "final_rms": True,
        "gated_mlp": True,
    }

register_architecture("LlamaForCausalLM", LlamaForCausalLM_state_dict)

def QWenLMHeadModel_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size // 2,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
        "eps": hf_config.layer_norm_epsilon,
        "d_vocab": hf_config.vocab_size,
        "act_fn": "silu",
        "use_attn_scale": hf_config.scale_attn_weights,
        "initializer_range": hf_config.initializer_range,
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "rotary_dim": hf_config.kv_channels,
        "rotary_adjacent_pairs": False,
        "tokenizer_prepends_bos": True,
        "trust_remote_code": True,
        "final_rms": True,
        "gated_mlp": True,
    }

register_architecture("QWenLMHeadModel", QWenLMHeadModel_state_dict)

def Qwen2ForCausalLM_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
        "eps": hf_config.rms_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.hidden_act,
        "use_attn_scale": True,
        "initializer_range": hf_config.initializer_range,
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "rotary_base": hf_config.rope_theta,
        "rotary_adjacent_pairs": False,
        "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
        "tokenizer_prepends_bos": True,
        "final_rms": True,
        "gated_mlp": True,
    }

register_architecture("Qwen2ForCausalLM", Qwen2ForCausalLM_state_dict)

def PhiForCausalLM_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.layer_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.hidden_act,
        "initializer_range": hf_config.initializer_range,
        "normalization_type": "LN",
        "positional_embedding_type": "rotary",
        "trust_remote_code": True,
        "rotary_base": hf_config.rope_theta,
        "use_attn_scale": True,
        "parallel_attn_mlp": True,
    }

register_architecture("PhiForCausalLM", PhiForCausalLM_state_dict)

def Phi3ForCausalLM_state_dict(hf_config):
    return {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": hf_config.max_position_embeddings,
        "eps": hf_config.rms_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.hidden_act,
        "initializer_range": hf_config.initializer_range,
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "trust_remote_code": True,
        "rotary_base": hf_config.rope_theta,
        "use_attn_scale": True,
        "gated_mlp": True,
        "parallel_attn_mlp": False,
        "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
    }

register_architecture("Phi3ForCausalLM", Phi3ForCausalLM_state_dict)

def T5forConditionalGeneration_state_dict(hf_config):
    return {
        "d_model": hf_config.d_model,
        "d_head": hf_config.d_kv,
        "n_heads": hf_config.num_heads,
        "d_mlp": hf_config.d_ff,
        "d_vocab": hf_config.vocab_size,
        "n_layers": hf_config.num_layers,
        "n_ctx": hf_config.max_length,
        "eps": hf_config.layer_norm_epsilon,
        "act_fn": hf_config.feed_forward_proj,
        "positional_embedding_type": "relative_positional_bias",
        "relative_attention_max_distance": hf_config.relative_attention_max_distance,
        "relative_attention_num_buckets": hf_config.relative_attention_num_buckets,
        "decoder_start_token_id": hf_config.decoder_start_token_id,
        "attention_dir": "bidirectional",
        "use_attn_scale": False,
        "tie_word_embeddings": hf_config.tie_word_embeddings,
    }

register_architecture("T5forConditionalGeneration", T5forConditionalGeneration_state_dict)