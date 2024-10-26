from . import BaseCollator, pad

class DotDataCollator(BaseCollator):
    def __init__(self, 
                 tokenizer, 
                 transformation_func : callable,
                 special_mask=False,
                 q_max_length=30,
                 d_max_length=200,
                 special_token="X",
                 perturb_type="append",
                 ) -> None:
        super().__init__(tokenizer, transformation_func, special_mask, perturb_type, q_max_length, d_max_length, special_token)

    def __call__(self, batch) -> dict:
        batch_perturbed_docs = [self.transformation_func(doc, query=query) for query, doc in batch]
        batch_og_docs = [doc for _, doc in batch]
        # batch_docs = [self.pad_by_perturb_type(doc_a, doc_b) for doc_a, doc_b in zip(batch_og_docs, batch_perturbed_docs)]
        batch_padded_docs, batch_padded_perturbed_docs = [], []
        for doc_a, doc_b in zip(batch_og_docs, batch_perturbed_docs):
            padded_a, padded_b = self.pad_by_perturb_type(doc_a, doc_b)
            batch_padded_docs.append(padded_a)
            batch_padded_perturbed_docs.append(padded_b)

        tokenized_queries = self.tokenizer(
            [query for query, _ in batch],
            padding='max_length',
            truncation=False,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )
        tokenized_docs = self.tokenizer(
            batch_padded_docs,
            padding='max_length',
            truncation=False,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask
        )

        tokenized_perturbed_docs = self.tokenizer(
            batch_padded_perturbed_docs,
            padding='max_length',
            truncation=False,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask
        )
 
        return {
            "queries": dict(tokenized_queries),
            "documents": dict(tokenized_docs),
            "perturbed_documents": dict(tokenized_perturbed_docs),
        }

__all__ = ["DotDataCollator"]