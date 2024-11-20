import einops
from functools import partial
from .loading_from_pretrained import register_with_transformer_lens
from .HookedTransformerConfig import HookedTransformerConfig


def convert_distilbert_weights(
    distilbert, cfg: HookedTransformerConfig, sequence_classification=False, raw=False
):
    embeddings = distilbert.embeddings
    state_dict = {
        "embed.embed.W_E": embeddings.word_embeddings.weight,
        "embed.pos_embed.W_pos": embeddings.position_embeddings.weight,
        # currently does not support sinusoidal embeddings
        "embed.ln.w": embeddings.LayerNorm.weight,
        "embed.ln.b": embeddings.LayerNorm.bias,
    }

    for l in range(cfg.n_layers):
        block = distilbert.transformer.layer[l]
        state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            block.attention.q_lin.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            block.attention.q_lin.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            block.attention.k_lin.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            block.attention.k_lin.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            block.attention.v_lin.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            block.attention.v_lin.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            block.attention.out_lin.weight,
            "m (i h) -> i h m",
            i=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_O"] = block.attention.out_lin.bias
        state_dict[f"blocks.{l}.ln1.w"] = block.sa_layer_norm.weight
        state_dict[f"blocks.{l}.ln1.b"] = block.sa_layer_norm.bias
        state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(
            block.ffn.lin1.weight, "mlp model -> model mlp"
        )
        state_dict[f"blocks.{l}.mlp.b_in"] = block.ffn.lin1.bias
        state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(
            block.ffn.lin2.weight, "model mlp -> mlp model"
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = block.ffn.lin2.bias
        state_dict[f"blocks.{l}.ln2.w"] = block.output_layer_norm.weight
        state_dict[f"blocks.{l}.ln2.b"] = block.output_layer_norm.bias

        # no MLM and unembed layers

        if not raw:
            if sequence_classification:
                classification_head = distilbert.pre_classifier
                state_dict["classifier.W"] = classification_head.weight
                state_dict["classifier.b"] = classification_head.bias

    return state_dict


register_with_transformer_lens(
    convert_distilbert_weights,
    ["DistilBert", "DistilBertModel"],
    function_type="conversion",
)
register_with_transformer_lens(
    partial(convert_distilbert_weights, sequence_classification=True),
    "DistilBertForSequenceClassification",
    function_type="conversion",
)


def convert_bert_based_weights(
    bert,
    cfg: HookedTransformerConfig,
    sequence_classification=False,
    raw=False,
    model_name: str = "bert",
):
    embeddings = getattr(bert, model_name).embeddings if not raw else bert.embeddings
    state_dict = {
        "embed.embed.W_E": embeddings.word_embeddings.weight,
        "embed.pos_embed.W_pos": embeddings.position_embeddings.weight,
        "embed.token_type_embed.W_token_type": embeddings.token_type_embeddings.weight,
        "embed.ln.w": embeddings.LayerNorm.weight,
        "embed.ln.b": embeddings.LayerNorm.bias,
    }

    for l in range(cfg.n_layers):
        block = (
            getattr(bert, model_name).encoder.layer[l]
            if not raw
            else bert.encoder.layer[l]
        )
        state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            block.attention.self.query.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            block.attention.self.query.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            block.attention.self.key.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            block.attention.self.key.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            block.attention.self.value.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            block.attention.self.value.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            block.attention.output.dense.weight,
            "m (i h) -> i h m",
            i=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_O"] = block.attention.output.dense.bias
        state_dict[f"blocks.{l}.ln1.w"] = block.attention.output.LayerNorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = block.attention.output.LayerNorm.bias
        state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(
            block.intermediate.dense.weight, "mlp model -> model mlp"
        )
        state_dict[f"blocks.{l}.mlp.b_in"] = block.intermediate.dense.bias
        state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(
            block.output.dense.weight, "model mlp -> mlp model"
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = block.output.dense.bias
        state_dict[f"blocks.{l}.ln2.w"] = block.output.LayerNorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = block.output.LayerNorm.bias
    if not raw:
        if sequence_classification:
            classification_head = bert.classifier
            if "electra" in model_name:
                state_dict["classifier.dense.W"] = classification_head.dense.weight
                state_dict["classifier.dense.b"] = classification_head.dense.bias
                state_dict["classifier.out_proj.W"] = einops.rearrange(
                    classification_head.out_proj.weight, "labels model -> model labels"
                )
                state_dict["classifier.out_proj.b"] = classification_head.out_proj.bias
            else:
                state_dict["classifier.W"] = classification_head.weight
                state_dict["classifier.b"] = classification_head.bias
        else:
            if not "electra" in model_name:
                mlm_head = bert.cls.predictions
                state_dict["mlm_head.W"] = mlm_head.transform.dense.weight
                state_dict["mlm_head.b"] = mlm_head.transform.dense.bias
                state_dict["mlm_head.ln.w"] = mlm_head.transform.LayerNorm.weight
                state_dict["mlm_head.ln.b"] = mlm_head.transform.LayerNorm.bias
                # "unembed.W_U": mlm_head.decoder.weight.T,
                state_dict["unembed.b_U"] = mlm_head.bias
    # Note: BERT uses tied embeddings
    state_dict["unembed.W_U"] = embeddings.word_embeddings.weight.T

    return state_dict


def convert_bert_weights(
    bert, cfg: HookedTransformerConfig, sequence_classification=False, raw=False
):
    print(dir(bert))
    embeddings = bert.bert.embeddings if not raw else bert.embeddings
    state_dict = {
        "embed.embed.W_E": embeddings.word_embeddings.weight,
        "embed.pos_embed.W_pos": embeddings.position_embeddings.weight,
        "embed.token_type_embed.W_token_type": embeddings.token_type_embeddings.weight,
        "embed.ln.w": embeddings.LayerNorm.weight,
        "embed.ln.b": embeddings.LayerNorm.bias,
    }

    for l in range(cfg.n_layers):
        block = bert.bert.encoder.layer[l] if not raw else bert.encoder.layer[l]
        state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            block.attention.self.query.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            block.attention.self.query.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            block.attention.self.key.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            block.attention.self.key.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            block.attention.self.value.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            block.attention.self.value.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            block.attention.output.dense.weight,
            "m (i h) -> i h m",
            i=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_O"] = block.attention.output.dense.bias
        state_dict[f"blocks.{l}.ln1.w"] = block.attention.output.LayerNorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = block.attention.output.LayerNorm.bias
        state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(
            block.intermediate.dense.weight, "mlp model -> model mlp"
        )
        state_dict[f"blocks.{l}.mlp.b_in"] = block.intermediate.dense.bias
        state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(
            block.output.dense.weight, "model mlp -> mlp model"
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = block.output.dense.bias
        state_dict[f"blocks.{l}.ln2.w"] = block.output.LayerNorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = block.output.LayerNorm.bias
    if not raw:
        if sequence_classification:
            classification_head = bert.classifier
            state_dict["classifier.W"] = classification_head.weight
            state_dict["classifier.b"] = classification_head.bias
        else:
            mlm_head = bert.cls.predictions
            state_dict["mlm_head.W"] = mlm_head.transform.dense.weight
            state_dict["mlm_head.b"] = mlm_head.transform.dense.bias
            state_dict["mlm_head.ln.w"] = mlm_head.transform.LayerNorm.weight
            state_dict["mlm_head.ln.b"] = mlm_head.transform.LayerNorm.bias
            # "unembed.W_U": mlm_head.decoder.weight.T,
            state_dict["unembed.b_U"] = mlm_head.bias
    # Note: BERT uses tied embeddings
    state_dict["unembed.W_U"] = embeddings.word_embeddings.weight.T

    return state_dict


register_with_transformer_lens(
    partial(convert_bert_weights, raw=True),
    ["BertModel", "BertForMaskedLM"],
    function_type="conversion",
)
register_with_transformer_lens(
    partial(convert_bert_weights, sequence_classification=True),
    "BertForSequenceClassification",
    function_type="conversion",
)
register_with_transformer_lens(
    partial(convert_bert_based_weights, model_name="roberta", raw=True),
    ["RobertaModel", "RobertaForMaskedLM"],
    function_type="conversion",
)
register_with_transformer_lens(
    partial(
        convert_bert_based_weights, sequence_classification=True, model_name="roberta"
    ),
    "RobertaForSequenceClassification",
    function_type="conversion",
)
register_with_transformer_lens(
    partial(convert_bert_based_weights, model_name="electra", raw=True),
    ["ElectraModel"],
    function_type="conversion",
)
register_with_transformer_lens(
    partial(
        convert_bert_based_weights, sequence_classification=True, model_name="electra"
    ),
    "ElectraForSequenceClassification",
    function_type="conversion",
)
