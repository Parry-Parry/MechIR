from fire import Fire 
import ir_datasets as irds
import pandas as pd
from mechir.peturb.axiom import TFC1, TDC

DL19 = r"msmarco-passage/trec-dl-2019/judged"
DL20 = r"msmarco-passage/trec-dl-2020/judged"
MSMARCO = r"msmarco-passage/train/triples-small"
MSMARCO_TERRIER = r"msmarco_passage"

def load_bi(model_name_or_path : str, batch_size : int = 256):
    from rankers import DotTransformer
    return DotTransformer.from_pretrained(model_name_or_path, batch_size=batch_size, verbose=True)

def load_cross(model_name_or_path : str, batch_size : int = 256):
    from rankers import CatTransformer
    return CatTransformer.from_pretrained(model_name_or_path, batch_size=batch_size, verbose=True)

def topk(model_name_or_path : str, model_type : str, out_path : str, index_location : str = None, k : int = 1000, batch_size : int = 256, perturbation_type : str = 'TFC1', max_rel : int = 3):
    if model_type == "bi":
        model = load_bi(model_name_or_path, batch_size)
    elif model_type == "cross":
        model = load_cross(model_name_or_path, batch_size)
    else:
        raise ValueError("model_type must be either 'bi' or 'cross'")
    
    MARCO = irds.load(MSMARCO)

    if index_location is None:
        index_location = pt.get_dataset(MSMARCO_TERRIER).get_index("terrier_stemmed_text")

    if perturbation_type == 'TFC1':
        perturbation = TFC1(index_location=index_location, dataset=MARCO)
    elif perturbation_type == 'TDC':
        perturbation = TDC(index_location=index_location, dataset=MARCO)
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
            'perturbed': [],
        }

        for row in df.itertuples():
            output['qid'].append(row.query_id)
            output['query'].append(queries[row.query_id])
            output['docno'].append(row.doc_id)
            output['text'].append(docs[row.doc_id])
            output['perturbed'].append(False)

        output = pd.DataFrame(output)

        perturbed_output = {
            'qid': [],
            'query': [],
            'docno': [],
            'text': [],
            'perturbed': [],
        }

        for row in df.itertuples():
            output['qid'].append(row.query_id)
            output['query'].append(queries[row.query_id])
            output['docno'].append(row.doc_id)
            output['text'].append(perturbation(docs[row.doc_id]))
            output['perturbed'].append(True)
        
        perturbed_output = pd.DataFrame(perturbed_output)
        output = pd.concat([output, perturbed_output])
        output['score'] = 0.

        return output
    
    # Calculate all deltas
    all_data = convert_to_trec(qrels)
    scored_data = model.transform(all_data)

    all_deltas = []
    for rel_grade in range(3):
        rel_data = scored_data[scored_data.relevance == rel_grade]
        original_scores = rel_data[~rel_data.perturbed].set_index(['qid', 'docno'])['score']
        perturbed_scores = rel_data[rel_data.perturbed].set_index(['qid', 'docno'])['score']
        
        score_deltas = (perturbed_scores - original_scores).reset_index()
        score_deltas.columns = ['qid', 'docno', 'score_delta']
        
        # Add back the other information from original data
        delta_info = score_deltas.merge(
            rel_data[~rel_data.perturbed],
            on=['qid', 'docno'],
            how='left'
        )
        all_deltas.append(delta_info)
    
    # Combine all deltas
    full_deltas = pd.concat(all_deltas, ignore_index=True)
    
    # Get top-k from each relevance grade
    topk_results = []
    for rel_grade in range(3):
        rel_deltas = full_deltas[full_deltas.relevance == rel_grade]
        top_k = rel_deltas.nlargest(k // 4, 'score_delta')
        topk_results.append(top_k)
    
    formatted_model_mame = model_name_or_path.replace("/", "-")
    output_k_file = f"{out_path}/{formatted_model_mame}_{model_type}_{perturbation_type}_topk_{k}.tsv"
    output_all_file = f"{out_path}/{formatted_model_mame}_{model_type}_{perturbation_type}_all.tsv"

    topk_df = pd.concat(topk_results)
    topk_df.to_csv(output_k_file, sep='\t', index=False)
    full_deltas.to_csv(output_all_file, sep='\t', index=False)

    print(f"Top-k results saved to {output_k_file}")
    print(f"Full results saved to {output_all_file}")

    return 0

if __name__ == "__main__":
    Fire(topk)