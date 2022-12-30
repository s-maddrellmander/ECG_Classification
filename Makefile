# Add to PHONY target list so cmds always run even when nothing has changed
.PHONY: install lint test clean

install:
	pip3 install -r requirements.txt

lint:
	yapf --in-place *.py --verbose
	yapf --in-place tests --recursive --verbose
	yapf --in-place models --recursive --verbose

test:
	python3 -m pytest -s -v --cache-clear --cov=. 

