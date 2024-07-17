from torch.utils.data import Dataset
import pandas as pd
import torch
import ir_datasets as irds

class PairDataset(Dataset):
    def __init__(self, 
                 pairs : pd.DataFrame, 
                 ir_dataset : str,
                 ) -> None:
        super().__init__()
        self.pairs = pairs
        for column in 'query_id', 'doc_id':
            if column not in self.pairs.columns: raise ValueError(f"Format not recognised, Column '{column}' not found in pairs dataframe")
        self.ir_dataset = irds.load(ir_dataset)
        self.docs = pd.DataFrame(self.ir_dataset.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.ir_dataset.queries_iter()).set_index("query_id")["text"].to_dict()

        self.qrels = pd.DataFrame(self.ir_dataset.qrels_iter()).set_index("query_id")

    '''
    @classmethod
    def from_irds_qrels(cls,
                    ir_dataset : str,
                    group_size : int = 2,
                    ) -> 'PairDataset':
            pairs = initialise_pairs(ir_dataset)
            return cls(pairs, ir_dataset, teacher_file, group_size)
    '''
    def __len__(self):
        return len(self.pairs)
    
    def _teacher(self, qid, doc_id, positive=False):
        assert self.labels, "No teacher file provided"
        try: return self.teacher[str(qid)][str(doc_id)] 
        except KeyError: return 0.

    def __getitem__(self, idx):
        item = self.pairs.iloc[idx]
        return item['query_id'], item['doc_id']