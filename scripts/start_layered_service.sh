#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-iedf-layered}"
LAYERED_HOST="${LAYERED_HOST:-0.0.0.0}"
LAYERED_PORT="${LAYERED_PORT:-8101}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export LAYERED_BACKEND="${LAYERED_BACKEND:-qwen}"
export LAYERED_MODEL_DIR="${LAYERED_MODEL_DIR:-qwen/Qwen-Image-Layered}"
export LAYERED_DEVICE="${LAYERED_DEVICE:-cuda}"
export LAYERED_PRELOAD="${LAYERED_PRELOAD:-false}"
export LAYERED_MAX_CONCURRENCY="${LAYERED_MAX_CONCURRENCY:-1}"
export LAYERED_MAX_QUEUE="${LAYERED_MAX_QUEUE:-16}"
export LAYERED_INFER_TIMEOUT_SEC="${LAYERED_INFER_TIMEOUT_SEC:-300}"
export LAYERED_CACHE_DIR="${LAYERED_CACHE_DIR:-outputs/service_cache/layered}"

conda run -n "${CONDA_ENV}" \
  python -m uvicorn services.layered_service.app:app \
  --host "${LAYERED_HOST}" --port "${LAYERED_PORT}" --workers 1
