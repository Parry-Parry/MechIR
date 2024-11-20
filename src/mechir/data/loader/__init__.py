import torch


def pad(a: list, b: list, tok: str):
    assert type(a) == type(b) == list, "Both a and b must be lists"

    padded = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            padded.append(a[i])
            i += 1
            j += 1
        else:
            padded.append(tok)
            j += 1

    while j < len(b):
        padded.append(tok)
        j += 1

    return padded


class BaseCollator(object):
    tokenizer = None
    transformation_func: callable = None
    special_mask: bool = False
    q_max_length: int = 30
    d_max_length: int = 300
    special_token: int = "a"
    perturb_type: str = "append"
    pre_perturbed: bool = False

    def __init__(
        self,
        tokenizer,
        transformation_func=None,
        special_mask=False,
        q_max_length=30,
        d_max_length=200,
        special_token="a",
        perturb_type="append",
        pre_perturbed=False,
    ) -> None:
        assert (
            transformation_func is not None or pre_perturbed
        ), "Either a transformation function or pre-perturbed data must be provided."
        self.tokenizer = tokenizer
        # self.tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
        # self.special_token_id = self.tokenizer.convert_tokens_to_ids(special_token)

        self.transformation_func = transformation_func
        self.special_mask = special_mask
        self.perturb_type = perturb_type
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length
        self.special_token = special_token
        self.special_token_id = self.tokenizer.convert_tokens_to_ids(self.special_token)
        self.perturb_type = perturb_type
        self.pre_perturbed = pre_perturbed

    def get_data(self, batch):
        if self.pre_perturbed:
            queries, docs, perturbed = zip(*batch)
        else:
            queries, docs = zip(*batch)
            perturbed = [
                self.transformation_func(doc, query=query) for query, doc in batch
            ]

        batch_padded_docs, batch_padded_perturbed_docs = [], []

        for doc_a, doc_b in zip(docs, perturbed):
            padded_a, padded_b = self.pad_by_perturb_type(doc_a, doc_b)
            batch_padded_docs.append(padded_a)
            batch_padded_perturbed_docs.append(padded_b)

        return queries, batch_padded_docs, batch_padded_perturbed_docs

    def pad(self, a: str, b: str):
        # turn both sequences into list of tokenized elements
        a = self.tokenizer.tokenize(a)
        b = self.tokenizer.tokenize(b)

        return self.tokenizer.decode(
            self.tokenizer.tokens_to_ids(pad(a, b, self.special_token))
        )

    def pad_by_perturb_type(self, doc_a: str, doc_b: str):
        accepted_perturb_types = ["append", "prepend", "replace", "inject"]
        assert (
            self.perturb_type in accepted_perturb_types
        ), f"Perturbation type must be one of the following: {accepted_perturb_types}"

        doc_a = self.tokenizer.tokenize(doc_a)
        doc_b = self.tokenizer.tokenize(doc_b)

        if self.perturb_type == "append":
            assert len(doc_a) < len(
                doc_b
            ), "Perturbed document should be longer than original for append perturbation."
            doc_a = doc_a + [self.special_token] * (len(doc_b) - len(doc_a))
        elif self.perturb_type == "prepend":
            assert len(doc_a) < len(
                doc_b
            ), "Perturbed document should be longer than original for prepend perturbation."
            doc_a = [self.special_token] * (len(doc_b) - len(doc_a)) + doc_a
        elif self.perturb_type == "replace":
            if len(doc_a) == len(doc_b):
                pass  # no padding needed
            else:
                padded_a, padded_b = [], []
                idx_a, idx_b = 0, 0
                while idx_a < len(doc_a) and idx_b < len(doc_b):
                    if doc_a[idx_a] == doc_b[idx_b]:
                        padded_a.append(doc_a[idx_a])
                        padded_b.append(doc_b[idx_b])
                        idx_a += 1
                        idx_b += 1
                    else:
                        padded_a.append(doc_a[idx_a])
                        padded_b.append(doc_b[idx_b])
                        idx_a += 1
                        idx_b += 1

                        if len(doc_a) < len(doc_b):
                            # Replaced term is shorter in length than the term it was replaced with
                            while idx_b < len(doc_b) and (
                                idx_a >= len(doc_a) or doc_b[idx_b] != doc_a[idx_a]
                            ):
                                padded_a.append(self.special_token)
                                padded_b.append(doc_b[idx_b])
                                idx_b += 1
                        if len(doc_a) > len(doc_b):
                            # Replaced term is longer than the term it was replaced with
                            while idx_a < len(doc_a) and (
                                idx_b >= len(doc_b) or doc_b[idx_b] != doc_a[idx_a]
                            ):
                                padded_a.append(doc_a[idx_a])
                                padded_b.append(self.special_token)
                                idx_a += 1

                doc_a, doc_b = padded_a, padded_b

        elif self.perturb_type == "inject":
            pass

        assert len(doc_a) == len(
            doc_b
        ), "Failed to pad input pairs, mismatch in document lengths post-padding."
        return self.tokenizer.convert_tokens_to_string(
            doc_a
        ), self.tokenizer.convert_tokens_to_string(doc_b)


def pad_tokenized(
    a_batch: torch.Tensor,
    b_batch: torch.Tensor,
    pad_tok: int,
):

    a_batch_input_ids, b_batch_input_ids = a_batch["input_ids"], b_batch["input_ids"]
    a_batch_attn_mask, b_batch_attn_mask = (
        a_batch["attention_mask"],
        b_batch["attention_mask"],
    )

    a_batch_final, b_batch_final = [], []
    a_batch_attn_final, b_batch_attn_final = [], []

    for a_tokens, b_tokens, a_mask, b_mask in zip(
        a_batch_input_ids, b_batch_input_ids, a_batch_attn_mask, b_batch_attn_mask
    ):
        a_padded_tokens, b_padded_tokens = [], []
        a_padded_attn_mask, b_padded_attn_mask = [], []

        if len(a_tokens) == len(b_tokens):
            # No padding needed
            a_padded_tokens.append(a_tokens)
            b_padded_tokens.append(b_tokens)
            a_padded_attn_mask.append(a_mask)
            b_padded_attn_mask.append(b_mask)
        else:
            # Determine where to pad
            idx_a, idx_b = 0, 0
            while idx_a < len(a_tokens) and idx_b < len(b_tokens):
                if a_tokens[idx_a] == b_tokens[idx_b]:
                    a_padded_tokens.append(a_tokens[idx_a])
                    b_padded_tokens.append(b_tokens[idx_b])
                    a_padded_attn_mask.append(a_mask[idx_a])
                    b_padded_attn_mask.append(b_mask[idx_b])
                    idx_a += 1
                    idx_b += 1
                elif len(a_tokens) < len(b_tokens):
                    # Accounts for the following perturbations: append, prepend, insert
                    # Also for replacement where the replaced term is equal to or shorter in length than the term is was replaced with
                    a_padded_tokens.append(torch.tensor([pad_tok], dtype=torch.int32))
                    b_padded_tokens.append(b_tokens[idx_b])
                    a_padded_attn_mask.append(a_mask[idx_a])
                    b_padded_attn_mask.append(b_mask[idx_b])
                    idx_b += 1
                elif len(a_tokens) > len(b_tokens):
                    # Account for replacement perturbation where the replaced term is longer than the term is was replaced with
                    a_padded_tokens.append(a_tokens[idx_a])
                    b_padded_tokens.append(torch.tensor([pad_tok], dtype=torch.int32))
                    a_padded_attn_mask.append(a_mask[idx_a])
                    b_padded_attn_mask.append(b_mask[idx_b])
                    idx_a += 1

        a_batch_final.append(torch.tensor(a_padded_tokens))
        b_batch_final.append(torch.tensor(b_padded_tokens))
        a_batch_attn_final.append(torch.tensor(a_padded_attn_mask))
        b_batch_attn_final.append(torch.tensor(b_padded_attn_mask))

    finalized_tokenized_a_batch = {
        "input_ids": torch.stack(a_batch_final),
        "attention_mask": torch.stack(a_batch_attn_final),
    }
    finalized_tokenized_b_batch = {
        "input_ids": torch.stack(b_batch_final),
        "attention_mask": torch.stack(b_batch_attn_final),
    }

    return finalized_tokenized_a_batch, finalized_tokenized_b_batch


from .cat import __all__ as cat_all
from .cat import *
from .dot import __all__ as dot_all
from .dot import *
from .t5 import __all__ as t5_all
from .t5 import *

__all__ = cat_all + dot_all + t5_all + ["pad", "pad_tokenized", "BaseCollator"]
