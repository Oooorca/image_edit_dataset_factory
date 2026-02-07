#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-iedf-edit}"
EDIT_HOST="${EDIT_HOST:-0.0.0.0}"
EDIT_PORT="${EDIT_PORT:-8102}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export EDIT_BACKEND="${EDIT_BACKEND:-qwen}"
export EDIT_MODEL_DIR="${EDIT_MODEL_DIR:-Qwen/Qwen-Image-Edit}"
export EDIT_DEVICE="${EDIT_DEVICE:-cuda}"
export EDIT_PRELOAD="${EDIT_PRELOAD:-false}"
export EDIT_MAX_CONCURRENCY="${EDIT_MAX_CONCURRENCY:-1}"
export EDIT_MAX_QUEUE="${EDIT_MAX_QUEUE:-16}"
export EDIT_INFER_TIMEOUT_SEC="${EDIT_INFER_TIMEOUT_SEC:-600}"
export EDIT_CACHE_DIR="${EDIT_CACHE_DIR:-outputs/service_cache/edit}"

conda run -n "${CONDA_ENV}" \
  python -m uvicorn services.edit_service.app:app \
  --host "${EDIT_HOST}" --port "${EDIT_PORT}" --workers 1
