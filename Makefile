.PHONY: format lint lint-fix test test-integration install-dev \
        docker-build docker-run docker-test

# Ruff is both the formatter and the linter. black and flake8 were dropped in
# favour of it: two tools with two config files (one of them an unpinned
# formatter that turned `main` red when its style changed under us) for what
# ruff does in one.
#
# `--no-sync` throughout: plain `uv run` re-syncs the environment first, which
# prunes anything not named in pyproject.toml -- including the extras you did
# not happen to pass on that invocation. `install-dev` is the one target
# allowed to change the environment.
UV = uv run --no-sync

format:
	$(UV) ruff format .

lint:
	$(UV) ruff check .

lint-fix:
	$(UV) ruff check . --fix

# The fast suite -- what CI runs on every push. No network, no model
# downloads.
test:
	$(UV) pytest -m "not integration"

# Downloads ~1.4GB of models and datasets on a cold cache. Run nightly in CI
# (.github/workflows/integration.yml), not on every push. Set
# MECHIR_RUN_LARGE=1 to additionally pull the Qwen checkpoints and the MS
# MARCO passage corpus.
test-integration:
	$(UV) pytest test/integration

# pyproject.toml is the single source of truth for dependencies.
install-dev:
	uv sync --extra dev --extra plotting --extra rag

docker-build:
	docker build -t mechir-dev .

docker-run:
	docker run --rm -it -v $(PWD):/workspace mechir-dev bash

docker-test:
	docker run --rm -v $(PWD):/workspace mechir-dev pytest -m "not integration"
