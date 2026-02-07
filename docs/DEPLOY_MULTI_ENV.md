# Deploy Multi-Environment Workflow

## 1) Create isolated conda environments

```bash
conda env create -f envs/orchestrator_env.yml
conda env create -f envs/layered_env.yml
conda env create -f envs/edit_env.yml
```

## 2) Prepare local model directories

Do not auto-download in code. Download manually on server and set env vars.

Example paths:

- `/root/autodl-tmp/models/qwen/Qwen-Image-Layered`
- `/root/autodl-tmp/models/qwen/Qwen-Image-Edit`

## 3) Start services (recommended: tmux)

```bash
bash scripts/start_services_tmux.sh
```

Manual startup:

```bash
# terminal A
CUDA_VISIBLE_DEVICES=0,1 LAYERED_MODEL_DIR=/root/autodl-tmp/models/qwen/Qwen-Image-Layered \
  bash scripts/start_layered_service.sh

# terminal B
CUDA_VISIBLE_DEVICES=2,3 EDIT_MODEL_DIR=/root/autodl-tmp/models/qwen/Qwen-Image-Edit \
  bash scripts/start_edit_service.sh
```

## 4) Run orchestrator in API mode

```bash
bash scripts/run_workflow_api_mode.sh configs/server_qwen.yaml
```

Or explicit overrides:

```bash
conda run -n iedf-orchestrator python -m image_edit_dataset_factory.scripts.run_all \
  --config configs/default.yaml \
  --set services.api_mode=true \
  --set backends.layered_backend=api \
  --set backends.edit_backend=api \
  --set services.layered.endpoint=http://127.0.0.1:8101 \
  --set services.edit.endpoint=http://127.0.0.1:8102
```

## 5) Health checks

```bash
curl -s http://127.0.0.1:8101/healthz
curl -s http://127.0.0.1:8101/readyz
curl -s http://127.0.0.1:8102/healthz
curl -s http://127.0.0.1:8102/readyz
```

## 6) Logs and outputs

- pipeline logs: `logs/`
- service cache: `outputs/service_cache/`
- dataset output: `outputs/dataset/`
- manifests/reports: `outputs/manifests/`, `outputs/reports/`
