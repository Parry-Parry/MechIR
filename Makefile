.PHONY: format lint test install-dev docker-build docker-run docker-test

format:
	black src test

lint:
	flake8 src test

test:
	pytest

install-dev:
	pip install -r requirements.txt
	pip install black flake8 pytest

requirements:
	pip-compile requirements.in

docker-build:
	docker build -t mechir-dev .

docker-run:
	docker run --rm -it -v $(PWD):/workspace mechir-dev bash

docker-test:
	docker run --rm -v $(PWD):/workspace mechir-dev pytest