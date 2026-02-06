# Spec Mapping

## Core requirements

- 5 categories + subtypes: `src/image_edit_dataset_factory/core/enums.py`
- Folder hierarchy `category/subtype/scene`: `src/image_edit_dataset_factory/pipeline/export.py`
- Naming (`00001.jpg`, `_CH`, `_EN`, `_mask`, `_mask-1`, `_result`): `src/image_edit_dataset_factory/utils/naming.py`, `src/image_edit_dataset_factory/pipeline/export.py`
- Structured configs + datamodels: `src/image_edit_dataset_factory/core/config.py`, `src/image_edit_dataset_factory/core/schema.py`

## Zero tolerance checks

- Corruption, naming, required files, orientation, shape checks: `src/image_edit_dataset_factory/qa/linter.py`
- Linter non-zero exit code in CLI: `src/image_edit_dataset_factory/scripts/run_lint.py`, `src/image_edit_dataset_factory/scripts/run_all.py`

## QA checks

- Non-edit region unchanged check: `src/image_edit_dataset_factory/qa/consistency.py`
- Allowed region = dilated mask: `src/image_edit_dataset_factory/utils/mask_ops.py`, `src/image_edit_dataset_factory/qa/consistency.py`
- QA CSV/JSON summaries: `src/image_edit_dataset_factory/qa/report.py`

## Semantic mask requirements

- White object on black mask + derived `mask-1`: `src/image_edit_dataset_factory/pipeline/generate/semantic.py`
- Mask morphology/refine: `src/image_edit_dataset_factory/utils/mask_ops.py`

## Backends

- Layered backend interface + Qwen skeleton + mock: `src/image_edit_dataset_factory/backends/`
- Edit backend interface + Qwen skeleton + OpenCV fallback + mock: `src/image_edit_dataset_factory/backends/`

## Resumable pipeline

- Stage skip on existing outputs (resume mode): `src/image_edit_dataset_factory/pipeline/orchestrator.py`
- Layer decompose cache skip unless overwrite: `src/image_edit_dataset_factory/pipeline/decompose.py`
