from . import BaseCollator, pad_tokenized

class DotDataCollator(BaseCollator):
    def __init__(self, 
                 tokenizer, 
                 transformation_func : callable,
                 special_mask=False,
                 q_max_length=30,
                 d_max_length=200,
                 special_token="X",
                 ) -> None:
        super().__init__(tokenizer, transformation_func, q_max_length, d_max_length, special_token)
        self.special_mask = special_mask

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_queries_cat = []
        batch_docs = []
        # for (q, dx) in batch:
        #     batch_queries.append(q)
        #     batch_queries_cat.extend([q]*len(dx))
        #     batch_docs.extend(dx)

        # batch_perturbed_docs = [self.transformation_func(dx, query=q) for q, dx in zip(batch_queries_cat, batch_docs)]
        # batch_docs = [pad(a, b, self.special_token) for a, b in zip(batch_docs, batch_perturbed_docs)]
        batch_perturbed_docs = [self.transformation_func(doc, query=query) for query, doc in batch]
        batch_docs = [doc for _, doc in batch]

        tokenized_queries = self.tokenizer(
            # batch_queries,
            [query for query, _ in batch],
            padding=True,
            truncation=False,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )
        tokenized_docs = self.tokenizer(
            batch_docs,
            padding=True,
            truncation=False,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask
        )

        tokenized_perturbed_docs = self.tokenizer(
            batch_perturbed_docs,
            padding=True,
            truncation=False,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask
        )

        # Adjust padding for documents
        padded_tokenized_docs, padded_tokenized_perturbed_docs = pad_tokenized(
            tokenized_docs,
            tokenized_perturbed_docs,
            self.special_token_id,
        )
 
        return {
            "queries": dict(tokenized_queries),
            "documents": dict(padded_tokenized_docs),
            "perturbed_documents": dict(padded_tokenized_perturbed_docs),
        }

__all__ = ["DotDataCollator"]