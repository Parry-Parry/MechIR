
FROM python:3.10-slim

WORKDIR /workspace

# pyproject.toml is the single source of truth for dependencies. This used to
# pip-compile requirements.in, which duplicated the pins and could drift from
# the package metadata that `pip install mechir` actually resolves.
#
# pyproject.toml alone is copied first so the dependency layer is cached and
# only reinstalls when the metadata changes, not on every source edit.
COPY pyproject.toml README.md LICENSE ./
COPY src/mechir/__init__.py src/mechir/__init__.py
RUN pip install --no-cache-dir -e ".[dev,plotting,rag]"

COPY . .

ENV PYTHONPATH=/workspace/src

CMD ["pytest", "-m", "not integration"]
