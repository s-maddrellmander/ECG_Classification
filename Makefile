# Add to PHONY target list so cmds always run even when nothing has changed
.PHONY: install lint test clean

install:
	pip3 install -r requirements.txt

lint:
	yapf --recursive --in-place .

test:
	python3 -m pytest -s -v --cache-clear --cov=. tests/

