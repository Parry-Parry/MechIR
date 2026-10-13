
# MechIR Test Suite

| Directory          | What it covers                                           | Network |
| ------------------ | -------------------------------------------------------- | ------- |
| `test/unit/`       | Individual functions and classes.                        | no      |
| `test/acceptance/` | Compatibility with external models and APIs.             | small checkpoints |
| `test/integration/`| End-to-end workflows and model pipelines, nothing mocked.| ~1.4GB  |

Everything under `test/integration/` carries the `integration` marker,
applied automatically by `test/integration/conftest.py`. The split is a
marker rather than a path so there is one definition of "the fast suite".

## Running Tests

```bash
make test               # fast suite: pytest -m "not integration"
make test-integration   # the slow suite: pytest test/integration
make docker-test        # fast suite inside the dev image
```

`make test` is exactly what the per-push CI job runs
(`.github/workflows/python-package.yml`).

## The integration suite

A cold run downloads roughly 1.4GB — TAS-B, a MiniLM cross-encoder,
MonoT5-small, the HotpotQA distractor config and the `vaswani` collection.
It therefore runs nightly in `.github/workflows/integration.yml`, not on
every push, and that workflow is also `workflow_dispatch`-able on demand.

The heaviest cases are opt-in even there, behind an environment variable:

```bash
MECHIR_RUN_LARGE=1 make test-integration
```

That adds `Qwen/Qwen3-Reranker-0.6B` (~1.2GB) and the MS MARCO passage
corpus (8.8M passages) used by the TREC DL loaders. Without it those tests
skip.

## Dependencies

`pyproject.toml` is the single source of truth. Install the test environment
with `make install-dev`, which pulls the `dev`, `plotting` and `rag` extras —
all three matter, because the plotting and pipeline tests skip when their
extra is absent, which would hide a regression behind a green run.
