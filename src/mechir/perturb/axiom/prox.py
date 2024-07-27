from pathlib import Path
from functools import partial
from typing import Any, Callable, NamedTuple
from .. import AbstractPerturbation

class ProximityPerturbation(AbstractPerturbation):
    def __init__(self, 
                 index_location: Any | Path | str, 
                 dataset: Any | str | None = None, 
                 contents_accessor: str | Callable[[NamedTuple], str] | None = "text", 
                 tokeniser: Any | None = None, 
                 cache_dir: Path | None = None
                 ) -> None:
        super().__init__(index_location, dataset, contents_accessor, tokeniser, cache_dir)
    
    def _insert_terms(self, text : str, terms : list[str]) -> str:
        if self.loc == 'end':
            return f"{text} {' '.join(terms)}"
        elif self.loc == 'start':
            return f"{' '.join(terms)} {text}"
        else:
            raise ValueError(f"Invalid loc value {self.loc}")
    
    def _get_top_k_freq_terms(self, text : str) -> dict:  
        freq = self.get_freq_text(text)
        # Get the top num_additions terms with the highest term frequency
        return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:self.num_additions]

    def _get_max_freq_terms(self, text : str) -> int:
        freq = self.get_freq_text(text)
        term = max(freq, key=freq.get)
        return [freq[term]] * self.num_additions
    
    def _get_min_freq_terms(self, text : str) -> int:
        freq = self.get_freq_text(text)
        term = min(freq, key=freq.get)
        return [freq[term]] * self.num_additions

    def apply(self, document : str