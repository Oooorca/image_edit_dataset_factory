# Data Format

## Export Root

`outputs/dataset/<category>/<subtype>/<scene>/`

## Required naming

Per sample ID `00001`:

- `00001.jpg` (source image)
- `00001_result.jpg` (edited image)
- `00001_CH.txt` (Chinese instruction)
- `00001_EN.txt` (English instruction)

Mask files:

- `00001_mask.png` (primary binary mask, white object on black)
- `00001_mask-1.png` (secondary/inverted silhouette mask)

For `semantic_edit`, mask and mask-1 are mandatory.

## Manifest/index files

- `outputs/reports/index.csv`
- `outputs/reports/index.jsonl`

Each row stores:

- `sample_id, category, subtype, scene`
- source/result paths
- mask path list
- CH/EN instructions
- metadata (JSON)
