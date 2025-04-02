from functools import partial
from mechir import Dot, Cat
import pandas as pd
import pyterrier as pt


class DummyTransformer(pt.Transformer):
    def __init__(self, transform_func):
        super().__init__()
        self.transform_func = transform_func

    def transform(self, inps: pd.DataFrame):
        outs = inps.copy()
        outs['score'] = self.transform_func(inps)
        return outs


CROSS_ENCODER_CHECKPOINT = "crystina-z/monoELECTRA_LCE_nneg31"
BI_ENCODER_CHECKPOINT = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"


def score_cat(model, df):
    queries = df.query.to_list()
    docs = df.text.to_list()

    tokenizer = model.tokenizer

    sequences = tokenizer(queries, docs, return_tensors="pt", padding=True, truncation=True)

    scores, _ = model.score(dict(sequences))
    return scores.cpu().numpy().tolist()


def score_dot(model, df):
    queries = df.query.to_list()
    docs = df.text.to_list()

    tokenizer = model.tokenizer

    queries = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
    docs = tokenizer(docs, return_tensors="pt", padding=True, truncation=True)

    scores, _, _, _ = model.score(dict(queries), dict(docs))
    return scores.cpu().numpy().tolist()


cat_score = partial(score_cat, Cat.from_pretrained(CROSS_ENCODER_CHECKPOINT))
CatTransformer = DummyTransformer(cat_score)

dot_score = partial(score_dot, Dot.from_pretrained(BI_ENCODER_CHECKPOINT))
DotTransformer = DummyTransformer(dot_score)
