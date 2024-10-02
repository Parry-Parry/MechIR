from torch.utils.data import Dataset
import pandas as pd
import ir_datasets as irds

class MechIRDataset(Dataset):
    def __init__(self,  
                 ir_dataset : str,
                 pairs : pd.DataFrame = None,
                 ) -> None:
        super().__init__()
        self.ir_dataset = irds.load(ir_dataset)
        self.qrels = pd.DataFrame(self.ir_dataset.qrels_iter()).set_index("query_id")
        self.pairs = pairs if pairs is not None else pd.DataFrame(self.ir_dataset.qrels_iter())
        for column in 'query_id', 'doc_id':
            if column not in self.pairs.columns: raise ValueError(f"Format not recognised, Column '{column}' not found in pairs dataframe")
        self.docs = pd.DataFrame(self.ir_dataset.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.ir_dataset.queries_iter()).set_index("query_id")["text"].to_dict()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs.iloc[idx]
        return self.queries[item['query_id']], self.docs[item['doc_id']]

__all__ = ["MechIRDataset"]