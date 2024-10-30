from fire import Fire 
import ir_datasets as irds
import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
from mechir.perturb import TFC1, TDC
from tqdm import tqdm

DL19 = r"msmarco-passage/trec-dl-2019/judged"
DL20 = r"msmarco-passage/trec-dl-2020/judged"
MSMARCO = r"msmarco-passage/train/triples-small"
MSMARCO_TERRIER = r"msmarco_passage"

def generate(out_path : str, index_location : str = None, perturbation_type : str = 'TFC1', stopwords : bool = False, exact_match : bool = False):
    if index_location is None:
        index_location = pt.get_dataset(MSMARCO_TERRIER).get_index("terrier_stemmed_text")

    if perturbation_type == 'TFC1':
        perturbation = TFC1(index_location=index_location, stem=True, stopwords=stopwords, exact_match=exact_match)
    elif perturbation_type == 'TDC':
        perturbation = TDC(index_location=index_location)
    else:
        raise ValueError("perturbation must be either 'TFC1' or 'TDC'")
    
    DL19_dataset = irds.load(DL19)
    DL20_dataset = irds.load(DL20)

    qrels = pd.concat([pd.DataFrame(DL19_dataset.qrels_iter()), pd.DataFrame(DL20_dataset.qrels_iter())])

    docs = pd.DataFrame(DL19_dataset.docs_iter()).set_index("doc_id").text.to_dict()
    queries = pd.DataFrame(DL19_dataset.queries_iter()).set_index("query_id").text.to_dict()
    queries.update(pd.DataFrame(DL20_dataset.queries_iter()).set_index("query_id").text.to_dict())
    
    def convert_to_trec(df : pd.DataFrame):
        output = {
            'qid': [],
            'query': [],
            'docno': [],
            'text': [],
            'relevance': [],
            'perturbed': [],
        }

        for row in tqdm(df.itertuples(), desc="Converting to TREC format"):
            if queries[row.query_id] is None or len(queries[row.query_id]) == 0:
                print("ARGH")
            output['qid'].append(row.query_id)
            output['query'].append(queries[row.query_id])
            output['docno'].append(row.doc_id)
            output['text'].append(docs[row.doc_id])
            output['relevance'].append(row.relevance)
            output['perturbed'].append(False)

        output = pd.DataFrame(output)
        perturbed_output = output.copy()
        perturbed_output['perturbed'] = True
        perturbed_output['text'] = perturbed_output.apply(lambda x : perturbation(x.text, x.query), axis=1)
        
        output = pd.concat([output, perturbed_output])
        output['score'] = 0.

        return output
    
    # Calculate all deltas
    all_data = convert_to_trec(qrels)
    output_file = f"{out_path}/{perturbation_type}-data.tsv.gz"

    all_data.to_csv(output_file, sep="\t", index=False)

    return 0

if __name__ == "__main__":
    Fire(generate)