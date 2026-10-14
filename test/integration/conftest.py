"""Everything under ``test/integration`` is an integration test.

The marker is applied here rather than repeated as a ``pytestmark`` in every
module, so a new file in this directory cannot accidentally join the fast
suite -- `make test`, the Docker image's default command and the per-push CI
job all select on ``-m "not integration"``, and these tests download ~1.4GB
of models and datasets on a cold cache.

``pytest_collection_modifyitems`` is a session-wide hook even when declared
in a subdirectory conftest: it receives every collected item, not just this
directory's. Hence the explicit path check -- without it this marks the whole
suite as integration and ``-m "not integration"`` collects nothing.
"""

import pathlib

import pytest

INTEGRATION_DIR = pathlib.Path(__file__).parent


def pytest_collection_modifyitems(items):
    for item in items:
        path = pathlib.Path(str(getattr(item, "path", item.fspath)))
        if INTEGRATION_DIR == path.parent or INTEGRATION_DIR in path.parents:
            item.add_marker(pytest.mark.integration)
