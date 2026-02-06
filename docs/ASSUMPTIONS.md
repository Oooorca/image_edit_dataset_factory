# Assumptions

1. `pyproject.toml` uses editable install (`pip install -e .`) so `python -m image_edit_dataset_factory.scripts.run_all` resolves package modules.
2. In `dev_small` config, filter thresholds are reduced to allow fast local validation with synthetic 512x512 images.
3. `semantic_edit` implemented minimally for `delete` subtype with OpenCV/mock inpainting.
4. `portrait_attribute` and `text_edit` are scaffolded placeholders pending dedicated model backends.
5. `mask-1` is treated as inverted silhouette mask derived from the primary binary mask.
6. Edge error for semantic masks is approximated via alpha-to-mask + morphology refinement utilities.
7. QA outside-region check uses either explicit `allowed_region_mask_path` or dilated primary mask fallback.
8. Linter is strict on naming, required files, shape/orientation consistency, and corruption checks.
9. Qwen backends are guarded skeletons and intentionally do not crash import-time when model deps are missing.
10. Resumability is stage-level (`pipeline.resume`) plus decomposition-level cache skipping (`decompose.overwrite`).
