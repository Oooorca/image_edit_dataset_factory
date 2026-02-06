# QA Guide

## Linter

`run_lint` enforces zero-tolerance checks:

- filename pattern correctness
- required files present
- semantic mask requirements
- readable/corruption checks
- source/result dimension and orientation consistency

Any lint issue returns non-zero exit.

## Non-edit region consistency

For each sample:

1. Build allowed edit region from `metadata.allowed_region_mask_path` or dilated mask.
2. Compare source vs result outside allowed region.
3. Compute:
   - `mse_outside_region`
   - `ssim_outside_region`
   - `changed_pixel_ratio_outside_region`
4. Mark fail if thresholds in config are violated.

Outputs:

- `outputs/reports/qa/qa_scores.csv`
- `outputs/reports/qa/qa_summary.json`

## Optional checks

- `qa/face_id.py`: guarded stub for face consistency
- `qa/ocr_check.py`: guarded stub for text OCR validation
