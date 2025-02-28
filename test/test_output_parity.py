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


def test_electra_equivelance():
    hgf_cat = ElectraScorer()
    mechir_cat = Cat(CROSS_ENCODER_CHECKPOINT, softmax_output=True)

    hgf_scores = hgf_cat.transform(test_dataframe).score.to_list()
    mechir_scores = score_cat(mechir_cat, test_dataframe)
    query_id_doc_id_pairs = zip(test_dataframe.qid.to_list(), test_dataframe.docno.to_list())

    # check they are close
    for hgf, mechir, pair in zip(hgf_scores, mechir_scores, query_id_doc_id_pairs):
        assert abs(hgf - mechir) < 0.01, f"Pair {pair} is not close, {hgf} != {mechir}"


def test_bi_equivelance():
    hgf_dot = HgfBiEncoder(BI_ENCODER_CHECKPOINT)
    mechir_dot = Dot(BI_ENCODER_CHECKPOINT)

    hgf_scores = hgf_dot.transform(test_dataframe)
    mechir_scores = score_dot(mechir_dot, test_dataframe)
    query_id_doc_id_pairs = zip(test_dataframe.qid.to_list(), test_dataframe.docno.to_list())

    # check they are close
    for hgf, mechir, pair in zip(hgf_scores, mechir_scores, query_id_doc_id_pairs):
        assert abs(hgf - mechir) < 0.01, f"Pair {pair} is not close, {hgf} != {mechir}"
