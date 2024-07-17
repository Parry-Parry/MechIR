class DotDataCollator:
    def __init__(self, 
                 tokenizer, 
                 transformation_func : callable,
                 special_mask=False,
                 q_max_length=30,
                 d_max_length=200,
                 ) -> None:
        self.tokenizer = tokenizer
        self.transformation_func = transformation_func
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length
        self.special_mask = special_mask

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_queries_cat = []
        batch_docs = []
        for (q, dx) in batch:
            batch_queries.append(q)
            batch_queries_cat.extend([q]*len(dx))
            batch_docs.extend(dx)

        batch_docs = [self.transformation_func(dx, query=q) for q, dx in zip(batch_queries_cat, batch_docs)]

        tokenized_queries = self.tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )
        tokenized_docs = self.tokenizer(
            batch_docs,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask
        )
 
        return {
            "queries": dict(tokenized_queries),
            "documents": dict(tokenized_docs),
        }