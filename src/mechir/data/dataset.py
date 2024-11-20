from torch.utils.data import Dataset
import pandas as pd


class MechDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        query_field: str = "query",
        text_field: str = "text",
        perturbed_field: str = "perturbed",
        pre_perturbed: bool = False,
    ) -> None:
        super().__init__()
        self.pairs = pairs
        required = [query_field, text_field]
        if pre_perturbed:
            required.append(perturbed_field)
        for column in required:
            if column not in self.pairs.columns:
                raise ValueError(
                    f"Format not recognised, Column '{column}' not found in pairs dataframe"
                )
        self.query_field = query_field
        self.text_field = text_field
        self.perturbed_field = perturbed_field
        self.pre_perturbed = pre_perturbed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs.iloc[idx]
        if self.pre_perturbed:
            return (
                item[self.query_field],
                item[self.text_field],
                item[self.perturbed_field],
            )
        return item[self.query_field], item[self.text_field]


__all__ = ["MechDataset"]
