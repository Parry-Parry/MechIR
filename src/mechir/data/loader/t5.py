from . import pad

class MonoT5DataCollator:
    def __init__(self, 
                 tokenizer,
                 transformation_func : callable,
                 q_max_length=30,
                 d_max_length=200,
                 special_token="X",
                 ) -> None:
        self.tokenizer = tokenizer
        self.transformation_func = transformation_func
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length
        self.special_token = special_token
        self.special_token_id = self.tokenizer.convert_tokens_to_ids(self.special_token)

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
        batch_docs = [pad(a, b, self.special_token) for a, b in zip(batch_docs, batch_perturbed_docs)]

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