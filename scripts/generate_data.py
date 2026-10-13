# Index-backed perturbations plus a CLI, so this needs two extras:
#
#     pip install "mechir[examples,axiomatic]"
#     # from a checkout: uv sync --extra examples --extra axiomatic
#
# TODO: mechir.perturb.axiom is unimportable as of 0.2.0 -- proximity.py does
# `from typing import Any, Path`, and typing has no Path. Fix that in
# src/mechir/perturb/axiom/proximity.py before running this script.
from functools import partial

from fire import Fire
import ir_datasets as irds
import pandas as pd
import pyterrier as pt
import pyterrier.java as java

java.init()
# Imported after java.init() on purpose: mechir.perturb.index resolves Terrier
# classes with pt.java.autoclass at import time, which needs a running JVM.
from mechir.perturb import FrequencyPerturbation, ProximityPerturbation  # noqa: E402
from tqdm import tqdm  # noqa: E402

DL19 = r"msmarco-passage/trec-dl-2019/judged"
DL20 = r"msmarco-passage/trec-dl-2020/judged"
MSMARCO = r"msmarco-passage/train/triples-small"
MSMARCO_TERRIER = r"msmarco_passage"

# The axioms the removed TFC1/TDC shorthands stood for, spelled out against
# the current classes. Each appends `num_additions` term(s) to the document.
PERTURBATIONS = {
    # TFC1: inject a query term, sampled by term frequency.
    "TFC1": partial(FrequencyPerturbation, frequency="tf", mode="random"),
    # TDC: inject the most discriminative (highest-IDF) query term.
    "TDC": partial(FrequencyPerturbation, frequency="idf", mode="max"),
    # Term proximity: inject the query term nearest the document's own terms.
    "proximity": partial(ProximityPerturbation, frequency="tf", mode="max"),
}


def generate(
    out_path: str,
    index_location: str = None,
    perturbation_type: str = "TFC1",
    stem: bool = True,
    stopwords: bool = False,
    exact_match: bool = False,
):
    """Write perturbed and unperturbed TREC DL judged pairs to a TSV.

    Args:
        out_path: Directory to write ``{perturbation_type}-data.tsv.gz`` into.
        index_location: A PyTerrier index or path; defaults to the MS MARCO
            passage index.
        perturbation_type: A key of ``PERTURBATIONS``.
        stem: Apply Porter stemming when matching and looking up terms.
        stopwords: Filter candidate terms through a stopword list.
        exact_match: Restrict candidates to terms present in both texts.
    """
    if index_location is None:
        index_location = pt.get_dataset(MSMARCO_TERRIER).get_index(
            "terrier_stemmed_text"
        )

    if perturbation_type not in PERTURBATIONS:
        raise ValueError(
            f"perturbation_type must be one of {sorted(PERTURBATIONS)}, "
            f"got {perturbation_type!r}"
        )

    perturbation = PERTURBATIONS[perturbation_type](
        index_location=index_location,
        num_additions=1,
        loc="end",
        stem=stem,
        stopwords=stopwords,
        exact_match=exact_match,
    )

    DL19_dataset = irds.load(DL19)
    DL20_dataset = irds.load(DL20)

    qrels = pd.concat(
        [
            pd.DataFrame(DL19_dataset.qrels_iter()),
            pd.DataFrame(DL20_dataset.qrels_iter()),
        ]
    )

    docs = pd.DataFrame(DL19_dataset.docs_iter()).set_index("doc_id").text.to_dict()
    queries = (
        pd.DataFrame(DL19_dataset.queries_iter()).set_index("query_id").text.to_dict()
    )
    queries.update(
        pd.DataFrame(DL20_dataset.queries_iter()).set_index("query_id").text.to_dict()
    )

    def convert_to_trec(df: pd.DataFrame):
        output = {
            "qid": [],
            "query": [],
            "docno": [],
            "text": [],
            "relevance": [],
            "perturbed": [],
        }

        for row in tqdm(df.itertuples(), desc="Converting to TREC format"):
            if queries[row.query_id] is None or len(queries[row.query_id]) == 0:
                print("ARGH")
            output["qid"].append(row.query_id)
            output["query"].append(queries[row.query_id])
            output["docno"].append(row.doc_id)
            output["text"].append(docs[row.doc_id])
            output["relevance"].append(row.relevance)
            output["perturbed"].append(False)

        output = pd.DataFrame(output)
        perturbed_output = output.copy()
        perturbed_output["perturbed"] = True
        perturbed_output["text"] = perturbed_output.apply(
            lambda x: perturbation(x.text, x.query), axis=1
        )

        output = pd.concat([output, perturbed_output])
        output["score"] = 0.0

        return output

    # Calculate all deltas
    all_data = convert_to_trec(qrels)
    output_file = f"{out_path}/{perturbation_type}-data.tsv.gz"

    all_data.to_csv(output_file, sep="\t", index=False)

    return 0


if __name__ == "__main__":
    Fire(generate)
