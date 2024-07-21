
class IndexPerturbation:
    def __init__(self, index_name_or_path : str) -> None:
        import pyterrier as pt
        self.index = pt.IndexFactory.of(index_name_or_path)
        self.meta_index = self.index.get_meta_index()