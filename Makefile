PYTHON ?= python
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/dev_small.yaml

.PHONY: install install-dev test lint format run-pipeline run-lint run-qa

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev]

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

format:
	$(PYTHON) -m black src tests scripts

run-pipeline:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_all --config $(CONFIG)

run-lint:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_lint --config $(CONFIG)

run-qa:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_qa --config $(CONFIG)
