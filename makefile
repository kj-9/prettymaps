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
	python -m pytest tests/

pre-commit:
	pre-commit run --all-files --show-diff-on-failure

clean:
	rm -rf *cache
