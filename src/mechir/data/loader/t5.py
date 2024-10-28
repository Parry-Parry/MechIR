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
        batch_queries, batch_docs, batch_perturbed_docs = self.get_data(batch)

        tokenized_sequences = self.tokenizer(
            [self.prompt(q, dx) for q, dx in zip(batch_queries, batch_docs)],
            padding='max_length',
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        tokenized_perturbed_sequences = self.tokenizer(
            [self.prompt(q, dx) for q, dx in zip(batch_queries, batch_perturbed_docs)],
            padding='max_length',
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        
        return {
            "sequences": dict(tokenized_sequences),
            "perturbed_sequences": dict(tokenized_perturbed_sequences),
        }

__all__ = ["MonoT5DataCollator"]