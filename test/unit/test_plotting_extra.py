"""Plotting is an optional extra, resolved lazily per function.

Two behaviours matter. Importing `mechir.plotting` must never require the
extra, and a missing dependency must say which extra to install rather than
surfacing a bare `ModuleNotFoundError: streamlit`.
"""

import builtins
import importlib
import sys

import pytest


@pytest.fixture
def fresh_plotting(monkeypatch):
    """Reimport mechir.plotting with a chosen dependency unavailable.

    Restores every mechir.plotting module afterwards. Without that, a
    partially-imported submodule is left in sys.modules and leaks into
    whatever runs next -- which hung the suite when unit and acceptance
    tests ran in one process.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("mechir.plotting")
    }

    def _reimport(missing: str):
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == missing or name.startswith(f"{missing}."):
                raise ImportError(f"No module named {missing!r}")
            return real_import(name, *args, **kwargs)

        for name in list(sys.modules):
            if name.startswith("mechir.plotting"):
                del sys.modules[name]
        monkeypatch.setattr(builtins, "__import__", guarded)
        return importlib.import_module("mechir.plotting")

    yield _reimport

    for name in list(sys.modules):
        if name.startswith("mechir.plotting"):
            del sys.modules[name]
    sys.modules.update(saved)


def test_importing_the_package_needs_no_plotting_dependency():
    """`import mechir.plotting` must not pull seaborn or streamlit."""
    import mechir.plotting as plotting

    assert plotting is not None


def test_public_names_are_advertised_without_importing_them():
    import mechir.plotting as plotting

    assert set(plotting.__all__) == {
        "plot_blocks",
        "plot_components",
        "plot_attn_for_doc",
        "load_doc",
    }
    assert set(dir(plotting)) == set(plotting.__all__)


def test_unknown_attribute_raises_attribute_error():
    import mechir.plotting as plotting

    with pytest.raises(AttributeError, match="has no attribute"):
        plotting.plot_nothing_at_all


def test_seaborn_helpers_do_not_require_streamlit(fresh_plotting):
    """Regression: the old star-import of both submodules meant a missing
    streamlit broke `plot_components`, which only needs seaborn.

    Needs the plotting extra to say anything, since the assertion is that
    plot_components *resolves*. Skips rather than fails without it, so a
    bare `pip install -e ".[dev]"` environment stays usable.
    """
    pytest.importorskip("seaborn", reason="needs mechir[plotting]")
    pytest.importorskip("matplotlib", reason="needs mechir[plotting]")
    plotting = fresh_plotting("streamlit")

    assert callable(plotting.plot_components)


def test_missing_dependency_names_the_extra(fresh_plotting):
    plotting = fresh_plotting("seaborn")

    with pytest.raises(ImportError, match=r"mechir\[plotting\]"):
        plotting.plot_components


def test_missing_dependency_error_says_which_package(fresh_plotting):
    plotting = fresh_plotting("streamlit")

    with pytest.raises(ImportError, match="streamlit is missing"):
        plotting.plot_attn_for_doc
