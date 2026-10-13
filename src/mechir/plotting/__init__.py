"""Plotting helpers.

The plotting dependencies are an extra, not core:

    pip install "mechir[plotting]"

They used to be core, which meant ``pip install mechir`` pulled seaborn,
matplotlib, plotly *and* streamlit -- a web framework -- for four helper
functions.

Imports resolve lazily per function rather than eagerly for the whole
package. That matters beyond the extra: ``visualize_attn_patterns`` needs
streamlit while ``plotting_funcs`` does not, and the previous star-import
of both meant a missing streamlit broke ``plot_components`` too, even
though it only needs seaborn.
"""

from typing import Dict, Tuple

# name -> (submodule, the import it needs that might be missing)
_EXPORTS: Dict[str, Tuple[str, str]] = {
    "plot_blocks": ("plotting_funcs", "seaborn"),
    "plot_components": ("plotting_funcs", "seaborn"),
    "plot_attn_for_doc": ("visualize_attn_patterns", "streamlit"),
    "load_doc": ("visualize_attn_patterns", "streamlit"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import a plotting helper on first use (PEP 562)."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    submodule, needs = _EXPORTS[name]
    from importlib import import_module

    try:
        module = import_module(f".{submodule}", __name__)
    except ImportError as exc:
        raise ImportError(
            f"mechir.plotting.{name} needs the plotting extra "
            f"({needs} is missing). Install it with:\n\n"
            f'    pip install "mechir[plotting]"\n'
        ) from exc

    return getattr(module, name)


def __dir__():
    return __all__
