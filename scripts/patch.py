from fire import Fire 
import ir_datasets as irds
import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()

DL19 = r"msmarco-passage/trec-dl-2019/judged"
DL20 = r"msmarco-passage/trec-dl-2020/judged"
MSMARCO = r"msmarco-passage/train/triples-small"
MSMARCO_TERRIER = r"msmarco_passage"

def load_bi(model_name_or_path : str, batch_size : int = 256):
    from mechir import Dot
    return Dot(model_name_or_path)

def load_cross(model_name_or_path : str, batch_size : int = 256):
    from mechir import Cat
    return Cat(model_name_or_path)

def patch(model_name_or_path : str, model_type : str, in_file : str, out_path : str, batch_size : int = 256, perturbation_type : str = 'TFC1'):
    if model_type == "bi":
        model = load_bi(model_name_or_path, batch_size)
    elif model_type == "cross":
        model = load_cross(model_name_or_path, batch_size)
    else:
        raise ValueError("model_type must be either 'bi' or 'cross'")
    
    DL19_dataset = irds.load(DL19)
    DL20_dataset = irds.load(DL20)

    queries = pd.DataFrame(DL19_dataset.queries_iter()).set_index("query_id").text.to_dict()
    queries.update(pd.DataFrame(DL20_dataset.queries_iter()).set_index("query_id").text.to_dict())
    
    all_data = pd.read_csv(in_file, sep='\t')
    
    return 0

if __name__ == "__main__":
    Fire(topk)