from __future__ import annotations

import re
from pathlib import Path

from image_edit_dataset_factory.core.schema import LintIssue
from image_edit_dataset_factory.utils.image_io import image_shape, is_corrupted

SRC_RE = re.compile(r"^(\d{5})\.jpg$")
RES_RE = re.compile(r"^(\d{5})_result\.jpg$")
CH_RE = re.compile(r"^(\d{5})_CH\.txt$")
EN_RE = re.compile(r"^(\d{5})_EN\.txt$")
MASK_RE = re.compile(r"^(\d{5})_mask\.png$")
MASK1_RE = re.compile(r"^(\d{5})_mask-1\.png$")


REQUIRED = {"src", "result", "ch", "en"}


def _issue(path: Path, code: str, message: str) -> LintIssue:
    return LintIssue(path=str(path), code=code, message=message)


def lint_dataset(dataset_dir: str | Path) -> list[LintIssue]:
    root = Path(dataset_dir)
    if not root.exists():
        return [_issue(root, "missing_dataset", "dataset directory not found")]

    issues: list[LintIssue] = []
    for scene_dir in sorted(
        path for path in root.rglob("*") if path.is_dir() and len(path.parts) >= len(root.parts) + 3
    ):
        files = [p for p in scene_dir.iterdir() if p.is_file()]
        if not files:
            continue

        by_id: dict[str, set[str]] = {}
        for file in files:
            matched = False
            for regex, kind in [
                (SRC_RE, "src"),
                (RES_RE, "result"),
                (CH_RE, "ch"),
                (EN_RE, "en"),
                (MASK_RE, "mask"),
                (MASK1_RE, "mask1"),
            ]:
                m = regex.match(file.name)
                if m:
                    sid = m.group(1)
                    by_id.setdefault(sid, set()).add(kind)
                    matched = True
                    break
            if not matched:
                issues.append(_issue(file, "bad_name", "file name does not follow naming spec"))

            if file.suffix.lower() in {".jpg", ".jpeg", ".png"} and is_corrupted(file):
                issues.append(_issue(file, "corrupted", "corrupted image file"))

        for sid, kinds in by_id.items():
            missing = REQUIRED - kinds
            if missing:
                issues.append(
                    _issue(scene_dir / sid, "missing_required", f"missing files: {sorted(missing)}")
                )

            if "mask" in kinds and "mask1" not in kinds:
                issues.append(
                    _issue(scene_dir / sid, "missing_mask1", "mask exists but mask-1 is missing")
                )

            src = scene_dir / f"{sid}.jpg"
            res = scene_dir / f"{sid}_result.jpg"
            if src.exists() and res.exists():
                if image_shape(src) != image_shape(res):
                    issues.append(_issue(res, "shape_mismatch", "source and result shape mismatch"))

    return issues
