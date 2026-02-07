#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-iedf-orchestrator}"
CONFIG_PATH="${1:-configs/server_qwen.yaml}"

conda run -n "${CONDA_ENV}" python -m image_edit_dataset_factory.scripts.run_all \
  --config "${CONFIG_PATH}" \
  --set services.api_mode=true \
  --set backends.layered_backend=api \
  --set backends.edit_backend=api
