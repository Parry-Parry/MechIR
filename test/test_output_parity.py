from mechir import Dot, Cat
import pandas as pd
try:
    from pyterrier_dr import ElectraScorer, HgfBiEncoder
except ImportError:
    return

test_dataframe = pd.DataFrame([
    {
        'qid': "1",
        "query": "What is the capital of France?",
        "docno": "100",
        "text": "Paris is the capital of France."
    },
    {
        "qid": "1",
        "query" : "What is the capital of France?",
        "docno" : "101",
        "text" : "The capital of China is Beijing."
    },
    {
        "qid" : "2",
        "query" : "What is the capital of China?",
        "docno" : "100",
        "text" : "Paris is the capital of France."
    },
    {
        "qid" : "2",
        "query" : "What is the capital of China?",
        "docno" : "101",
        "text" : "The capital of China is Beijing."
    },
])

CROSS_ENCODER_CHECKPOINT = ""
BI_ENCODER_CHECKPOINT = ""

def score_cat(model, df):
    


def test_electra_equivelance():
    hgf_cat = ElectraScorer()
    mechir_cat = Cat(CROSS_ENCODER_CHECKPOINT)

    hgf_scores = hgf_cat.transform(test_dataframe)


def test_bi_equivelance():
    hgf_dot = HgfBiEncoder(BI_ENCODER_CHECKPOINT)
    mechir_dot = Dot(BI_ENCODER_CHECKPOINT)

    hgf_scores = hgf_dot.transform(test_dataframe)