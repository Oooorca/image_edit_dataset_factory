# Image Edit Dataset Factory

End-to-end pipeline to ingest source images, filter by spec constraints, decompose layers, generate edit pairs, export to strict dataset layout, and run linter/QA checks.

## Install

```bash
make install-dev
```

## Quickstart (mock/fallback run)

1. Create demo inputs (if you do not already have images):

```bash
python scripts/create_demo_input.py --out data/demo_input --count 4 --size 512
```

2. Run full pipeline:

```bash
python -m image_edit_dataset_factory.scripts.run_all --config configs/dev_small.yaml
```

3. Inspect outputs:

- Dataset: `outputs/dataset/`
- Global index: `outputs/reports/index.csv`
- Lint report: `outputs/reports/lint_issues.json`
- QA report: `outputs/reports/qa/qa_scores.csv`

## CLI Steps

- `python -m image_edit_dataset_factory.scripts.run_ingest --config configs/default.yaml`
- `python -m image_edit_dataset_factory.scripts.run_filter --config configs/default.yaml`
- `python -m image_edit_dataset_factory.scripts.run_decompose --config configs/default.yaml`
- `python -m image_edit_dataset_factory.scripts.run_generate --config configs/default.yaml`
- `python -m image_edit_dataset_factory.scripts.run_export --config configs/default.yaml`
- `python -m image_edit_dataset_factory.scripts.run_lint --config configs/default.yaml`
- `python -m image_edit_dataset_factory.scripts.run_qa --config configs/default.yaml`

## Config Overrides

Use `--set key=value` with dotted keys:

```bash
python -m image_edit_dataset_factory.scripts.run_all \
  --config configs/dev_small.yaml \
  --set generate.dry_run=true \
  --set qa.max_mse_outside_region=8.0
```

## Qwen Backends

- Layered decomposition backend: `decompose.backend=qwen`
- Edit backend: `edit.backend=qwen`
- If model packages/weights are unavailable, switch to `mock`/`opencv` backends.

See `configs/gpu_qwen.yaml` for an example GPU config.
