class CatDataCollator:
    def __init__(self, 
                 tokenizer,
                 q_max_length=30,
                 d_max_length=200,
                 ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

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
            batch_queries,
            batch_docs,
            padding=True,
            truncation='only_second',
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        return {
            "sequences": dict(tokenized_sequences),
        }
    
def _make_pos_pairs(texts) -> list:
    output = []
    pos = texts[0]
    for i in range(1, len(texts)):
        output.append([pos, texts[i]])
    return output
    
class PairDataCollator:
    def __init__(self, tokenizer, max_length=512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for (q, dx, *args) in batch:
            batch_queries.append(q)
            batch_document_pairs = _make_pos_pairs(dx)
            batch_docs.append(batch_document_pairs)
            if len(args) == 0:
                continue
            batch_score_pairs = _make_pos_pairs(args[0])
            batch_scores.extend(batch_score_pairs)
            
        # tokenize each pair with each query
        texts = []
        for query, pairs in zip(batch_queries, batch_docs):
            for pair in pairs:
                texts.append(f"[CLS] {query} [SEP] {pair[0]} [SEP] {pair[1]}")
        tokenized_sequences = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
                
        return {
            "sequences": dict(tokenized_sequences),
        }