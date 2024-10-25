from . import BaseCollator, pad_tokenized

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
        super().__init__(tokenizer, transformation_func, q_max_length, d_max_length, special_token)
        self.special_mask = special_mask
        self.perturb_type = perturb_type

    def pad_by_perturb_type(self, doc_a : str, doc_b : str):
        accepted_perturb_types = ["append", "prepend", "replace", "inject"]
        assert self.perturb_type in accepted_perturb_types, f"Perturbation type must be one of the following: {accepted_perturb_types}"

        doc_a = self.tokenizer.tokenize(doc_a)
        doc_b = self.tokenizer.tokenize(doc_b) 

        if self.perturb_type == "append":
            assert len(doc_a) < len(doc_b), "Perturbed document should be longer than original for append perturbation."
            doc_a = doc_a + [self.special_token] * (len(doc_b) - len(doc_a))
        elif self.perturb_type == "prepend":
            assert len(doc_a) < len(doc_b), "Perturbed document should be longer than original for prepend perturbation."
            doc_a = [self.special_token] * (len(doc_b) - len(doc_a)) + doc_a
        elif self.perturb_type == "replace":
            pass
        elif self.perturb_type == "inject":
            pass

        assert len(doc_a) == len(doc_b), "Failed to pad input pairs, mismatch in document lengths post-padding."
        return self.tokenizer.convert_tokens_to_string(doc_a)


    def __call__(self, batch) -> dict:
        batch_perturbed_docs = [self.transformation_func(doc, query=query) for query, doc in batch]
        batch_og_docs = [doc for _, doc in batch]
        batch_docs = [self.pad_by_perturb_type(doc_a, doc_b) for doc_a, doc_b in zip(batch_og_docs, batch_perturbed_docs)]

        tokenized_queries = self.tokenizer(
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
 
        return {
            "queries": dict(tokenized_queries),
            "documents": dict(tokenized_docs),
            "perturbed_documents": dict(tokenized_perturbed_docs),
        }

__all__ = ["DotDataCollator"]