PYTHON ?= python
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/default.yaml

.PHONY: install install-dev test lint format run-all run-ingest run-generate run-qa

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

run-all:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_all --config $(CONFIG)

run-ingest:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_ingest --config $(CONFIG)

run-generate:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_generate --config $(CONFIG)

run-qa:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_qa --config $(CONFIG)
