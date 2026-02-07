PYTHON ?= python
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/default.yaml

.PHONY: install install-dev install-service install-gpu test lint format run-all run-ingest run-generate run-qa run-api

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev]

install-service:
	$(PIP) install -e .[service]

install-gpu:
	$(PIP) install -e .[gpu]

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts services

format:
	$(PYTHON) -m black src tests scripts services

run-all:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_all --config $(CONFIG)

run-ingest:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_ingest --config $(CONFIG)

run-generate:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_generate --config $(CONFIG)

run-qa:
	$(PYTHON) -m image_edit_dataset_factory.scripts.run_qa --config $(CONFIG)

run-api:
	bash scripts/run_workflow_api_mode.sh $(CONFIG)
