from fire import Fire
import ir_datasets as irds
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import mechir
from mechir import Cat, Dot
from mechir.data import CatDataCollator, DotDataCollator, MechDataset
import pyterrier as pt

if not pt.started():
    pt.init()

mechir.config("ignore-official", True)

DL19 = r"msmarco-passage/trec-dl-2019/judged"
DL20 = r"msmarco-passage/trec-dl-2020/judged"
MSMARCO = r"msmarco-passage/train/triples-small"
MSMARCO_TERRIER = r"msmarco_passage"


def load_bi(model_name_or_path: str):
    return Dot(model_name_or_path), DotDataCollator


def load_cross(model_name_or_path: str):
    return Cat(model_name_or_path), CatDataCollator


def process_frame(frame):

    output = {
        "qid": [],
        "query": [],
        "docno": [],
        "text": [],
        "perturbed": [],
    }

    for row in frame.itertuples():
        output["qid"].append(row.qid)
        output["query"].append(row.query)
        output["docno"].append(row.docno)
        output["text"].append(row.text)
        output["perturbed"].append(row.perturbed_text)

    return pd.DataFrame(output)


def patch(
    model_name_or_path: str,
    model_type: str,
    in_file: str,
    out_path: str,
    batch_size: int = 256,
    perturbation_type: str = "TFC1",
    k: int = 1000,
):
    if model_type == "bi":
        model, collator = load_bi(model_name_or_path)
    elif model_type == "cross":
        model, collator = load_cross(model_name_or_path)
    else:
        raise ValueError("model_type must be either 'bi' or 'cross'")

    DL19_dataset = irds.load(DL19)
    DL20_dataset = irds.load(DL20)

    queries = (
        pd.DataFrame(DL19_dataset.queries_iter()).set_index("query_id").text.to_dict()
    )
    queries.update(
        pd.DataFrame(DL20_dataset.queries_iter()).set_index("query_id").text.to_dict()
    )

    all_data = pd.read_csv(in_file, sep="\t")
    processed_frame = process_frame(all_data)

    dataset = MechDataset(processed_frame, pre_perturbed=True)
    collator = collator(model.tokenizer, pre_perturbed=True)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator
    )

    patching_head_outputs = []
    for batch in tqdm(dataloader):
        # Get the queries, documents, and perturbed documents from the batch

        if model_type == "bi":
            queries = batch["queries"]
            documents = batch["documents"]
            perturbed_documents = batch["perturbed_documents"]

            patch_head_out = model(
                queries, documents, perturbed_documents, patch_type="head_all"
            )
        else:
            sequences = batch["sequences"]
            perturbed_sequences = batch["perturbed_sequences"]

            patch_head_out = model(
                sequences, perturbed_sequences, patch_type="head_all"
            )

        patching_head_outputs.append(patch_head_out)

    output = torch.mean(torch.stack(patching_head_outputs), axis=0)
    # convert to numpy and dump
    output = output.cpu().detach().numpy()
    formatted_model_name = model_name_or_path.replace("/", "-")
    output_file = f"{out_path}/{formatted_model_name}_{model_type}_{perturbation_type}_{k}_patch_head.npy"
    np.save(output_file, output)

    return 0


if __name__ == "__main__":
    Fire(patch)
