from torch.utils.data import Dataset
import pandas as pd
import ir_datasets as irds


class MechIRDataset(Dataset):
    def __init__(
        self,
        ir_dataset: str,
        pairs: pd.DataFrame = None,
        lazy_load_docs: bool = False,
        text_field: str = "text",
        query_field: str = "text",
        perturbed_field: str = "perturbed",
        pre_perturbed: bool = False,
        query_id_subset: list = None,
    ) -> None:
        super().__init__()
        self.ir_dataset = irds.load(ir_dataset)
        self.qrels = pd.DataFrame(self.ir_dataset.qrels_iter()).set_index("query_id")
        self.pairs = (
            pairs if pairs is not None else pd.DataFrame(self.ir_dataset.qrels_iter())
        )
        for column in "query_id", "doc_id":
            if column not in self.pairs.columns:
                raise ValueError(
                    f"Format not recognised, Column '{column}' not found in pairs dataframe"
                )

        if query_id_subset is not None:
            self.pairs = self.pairs[self.pairs["query_id"].isin(query_id_subset)]

        self.lazy_load_docs = lazy_load_docs
        self._text_field = text_field
        self._query_field = query_field
        self._perturbed_field = perturbed_field
        self.pre_perturbed = pre_perturbed
        if self.lazy_load_docs:
            self.docs = self.ir_dataset.docs_store()
        else:
            self.docs = (
                pd.DataFrame(self.ir_dataset.docs_iter())
                .set_index("doc_id")[self._text_field]
                .to_dict()
            )
        self.queries = (
            pd.DataFrame(self.ir_dataset.queries_iter())
            .set_index("query_id")[self._query_field]
            .to_dict()
        )

    def _get_doc(self, doc_id):
        if self.lazy_load_docs:
            return getattr(self.docs.get(doc_id), self._text_field)
        else:
            return self.docs[doc_id]

    def _get_query(self, query_id):
        return self.queries[query_id]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs.iloc[idx]
        if self.pre_perturbed:
            return (
                self._get_query(item["query_id"]),
                self._get_doc(item["doc_id"]),
                item[self._perturbed_field],
            )
        return self._get_query(item["query_id"]), self._get_doc(item["doc_id"])


__all__ = ["MechIRDataset"]
