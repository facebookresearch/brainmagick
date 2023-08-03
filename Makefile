all: lint tests

lint:
	flake8 bm && mypy -p bm

tests:
	coverage run -m pytest bm || exit 1
	coverage report --include 'bm/*'

.PHONY: tests lint
