from torch.utils.data import Dataset
import pandas as pd

class MechDataset(Dataset):
    def __init__(self,  
                 pairs : pd.DataFrame = None,
                 query_col : str = 'query',
                 doc_col : str = 'doc',
                 ) -> None:
        super().__init__()
        self.pairs = pairs if pairs is not None else pd.DataFrame(self.ir_dataset.qrels_iter())
        for column in [query_col, doc_col]:
            if column not in self.pairs.columns: raise ValueError(f"Format not recognised, Column '{column}' not found in pairs dataframe")
        self.query_col = query_col
        self.doc_col = doc_col

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs.iloc[idx]
        return item[self.query_col], item[self.doc_col]

__all__ = ["MechDataset"]