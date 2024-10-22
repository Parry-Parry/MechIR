from . import BaseCollator

class MonoT5DataCollator(BaseCollator):
    def __init__(self, 
                 tokenizer,
                 transformation_func : callable,
                 q_max_length=30,
                 d_max_length=200,
                 special_token="X",
                 ) -> None:
        super().__init__(tokenizer, transformation_func, q_max_length, d_max_length, special_token)

    def prompt(query : str, document : str):
        return f"query: {query} document: {document} relevant:"

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for (q, dx, *args) in batch:
            batch_queries.extend([q]*len(dx))
            batch_docs.extend(dx)
            if len(args) == 0:
                continue
            batch_scores.extend(args[0])
        
        batch_perturbed_docs = [self.transformation_func(dx, query=q) for q, dx in zip(batch_queries, batch_docs)]
        batch_docs = [self.pad(a, b, self.special_token) for a, b in zip(batch_docs, batch_perturbed_docs)]

        tokenized_sequences = self.tokenizer(
            [self.prompt(q, dx) for q, dx in zip(batch_queries, batch_docs)],
            padding=True,
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        tokenized_perturbed_sequences = self.tokenizer(
            [self.prompt(q, dx) for q, dx in zip(batch_queries, batch_perturbed_docs)],
            padding=True,
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        
        return {
            "sequences": dict(tokenized_sequences),
            "perturbed_sequences": dict(tokenized_perturbed_sequences),
        }

__all__ = ["MonoT5DataCollator"]