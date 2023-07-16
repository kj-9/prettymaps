SHELL=/bin/bash

.PHONY:
	pip-upgrade
	pip-install
	test
	pre-commit

pip-upgrade:
	python -m pip install --upgrade pip

pip-install: pip-upgrade
	pip install -e '.[test]'

test:
	python -m pytest tests/ -vv

pre-commit:
	pre-commit run --all-files --show-diff-on-failure

clean:
	rm -rf *cache

build:
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install build twine
	python -m pip list
	python -m build --outdir dist/ .
