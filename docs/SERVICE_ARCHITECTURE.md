# Service Architecture

## Overview

The project is split into three process types:

1. `Orchestrator` (existing pipeline in `src/image_edit_dataset_factory/pipeline`)
2. `Layered Service` (`services/layered_service/app.py`)
3. `Edit Service` (`services/edit_service/app.py`)

The orchestrator no longer needs to import heavy model runtime in API mode. It calls model services through HTTP clients.

## Process Diagram

```text
+--------------------+        HTTP /infer         +----------------------+
| Orchestrator       | -------------------------> | Layered Service      |
| run_all pipeline   | <------------------------- | qwen/mock backend    |
+--------------------+                            +----------------------+
          |
          | HTTP /infer
          v
+----------------------+
| Edit Service         |
| qwen/opencv/mock     |
+----------------------+
```

## Runtime Responsibilities

- Orchestrator:
  - ingest/filter/decompose/generate/export/qa orchestration
  - API retry/fallback policy from config
  - dataset and reports output
- Layered Service:
  - image decomposition into RGBA layers + alpha masks
  - optional local cache write (`outputs/service_cache/layered`)
- Edit Service:
  - image edit/inpaint
  - optional local cache write (`outputs/service_cache/edit`)

## API Mode Toggle

Use either:

- `services.api_mode=true` (global switch)
- or `backends.layered_backend=api` and `backends.edit_backend=api`

## Concurrency and Queue

Both services support:

- `*_MAX_CONCURRENCY`: concurrent in-flight inference workers
- `*_MAX_QUEUE`: max pending queue length
- `*_INFER_TIMEOUT_SEC`: request timeout (returns HTTP 504 on timeout)

## GPU Allocation

Recommended split for two services:

- Layered service: `CUDA_VISIBLE_DEVICES=0,1`
- Edit service: `CUDA_VISIBLE_DEVICES=2,3`

GPU binding is done at shell startup scripts level.
