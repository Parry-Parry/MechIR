class MonoT5DataCollator:
    def __init__(self, 
                 tokenizer,
                 q_max_length=30,
                 d_max_length=200,
                 ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

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

        tokenized_sequences = self.tokenizer(
            [self.prompt(q, dx) for q, dx in zip(batch_queries, batch_docs)],
            padding=True,
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        return {
            "sequences": dict(tokenized_sequences),
        }