from . import BaseCollator


class DotDataCollator(BaseCollator):
    def __init__(
        self,
        tokenizer,
        transformation_func: callable = None,
        special_mask=False,
        q_max_length=30,
        d_max_length=300,
        special_token="a",
        perturb_type="append",
        pre_perturbed=False,
    ) -> None:
        super().__init__(
            tokenizer,
            transformation_func,
            special_mask,
            q_max_length,
            d_max_length,
            special_token,
            perturb_type,
            pre_perturbed,
        )

    def __call__(self, batch) -> dict:
        queries, batch_padded_docs, batch_padded_perturbed_docs = self.get_data(batch)
        tokenized_queries = self.tokenizer(
            queries,
            padding="max_length",
            truncation=False,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )
        tokenized_docs = self.tokenizer(
            batch_padded_docs,
            padding="max_length",
            truncation=False,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )

        tokenized_perturbed_docs = self.tokenizer(
            batch_padded_perturbed_docs,
            padding="max_length",
            truncation=False,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )

        return {
            "queries": dict(tokenized_queries),
            "documents": dict(tokenized_docs),
            "perturbed_documents": dict(tokenized_perturbed_docs),
        }


__all__ = ["DotDataCollator"]
