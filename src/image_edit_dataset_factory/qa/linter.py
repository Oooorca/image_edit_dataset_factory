from __future__ import annotations

import logging
import re
from pathlib import Path

from image_edit_dataset_factory.core.schema import LintIssue
from image_edit_dataset_factory.utils.image_io import image_shape, is_corrupted

LOGGER = logging.getLogger(__name__)

_SRC_RE = re.compile(r"^(\d{5})\.jpg$")
_RESULT_RE = re.compile(r"^(\d{5})_result\.jpg$")
_CH_RE = re.compile(r"^(\d{5})_CH\.txt$")
_EN_RE = re.compile(r"^(\d{5})_EN\.txt$")
_MASK_RE = re.compile(r"^(\d{5})_mask\.png$")
_MASK1_RE = re.compile(r"^(\d{5})_mask-1\.png$")


_REQUIRED_BASE = {"src", "result", "ch", "en"}


def _add_issue(issues: list[LintIssue], path: Path, code: str, message: str) -> None:
    issues.append(LintIssue(path=str(path), code=code, message=message))


def lint_dataset(dataset_root: str | Path) -> list[LintIssue]:
    root = Path(dataset_root)
    issues: list[LintIssue] = []

    if not root.exists():
        _add_issue(issues, root, "missing_dataset_root", "Dataset root does not exist")
        return issues

    for category_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        for subtype_dir in sorted([d for d in category_dir.iterdir() if d.is_dir()]):
            for scene_dir in sorted([d for d in subtype_dir.iterdir() if d.is_dir()]):
                file_map: dict[str, set[str]] = {}

                for file in sorted([f for f in scene_dir.iterdir() if f.is_file()]):
                    stem = file.name
                    sample_id = None
                    kind = None

                    for regex, label in [
                        (_SRC_RE, "src"),
                        (_RESULT_RE, "result"),
                        (_CH_RE, "ch"),
                        (_EN_RE, "en"),
                        (_MASK_RE, "mask"),
                        (_MASK1_RE, "mask1"),
                    ]:
                        match = regex.match(stem)
                        if match:
                            sample_id = match.group(1)
                            kind = label
                            break

                    if sample_id is None or kind is None:
                        _add_issue(
                            issues, file, "invalid_filename", "File name does not match spec"
                        )
                        continue

                    if sample_id not in file_map:
                        file_map[sample_id] = set()
                    file_map[sample_id].add(kind)

                    if file.suffix.lower() in {".jpg", ".jpeg", ".png"} and is_corrupted(file):
                        _add_issue(
                            issues, file, "corrupted_file", "Image file is corrupted or unreadable"
                        )

                for sample_id, kinds in file_map.items():
                    missing = _REQUIRED_BASE - kinds
                    if missing:
                        _add_issue(
                            issues,
                            scene_dir / sample_id,
                            "missing_required",
                            f"Missing required files: {sorted(missing)}",
                        )

                    if "mask" in kinds and "mask1" not in kinds:
                        _add_issue(
                            issues, scene_dir / sample_id, "missing_mask1", "mask-1 file missing"
                        )
                    if "mask1" in kinds and "mask" not in kinds:
                        _add_issue(
                            issues, scene_dir / sample_id, "missing_mask", "mask file missing"
                        )

                    if category_dir.name == "semantic_edit" and not {"mask", "mask1"}.issubset(
                        kinds
                    ):
                        _add_issue(
                            issues,
                            scene_dir / sample_id,
                            "semantic_mask_required",
                            "Semantic edit sample requires mask and mask-1 files",
                        )

                    src_path = scene_dir / f"{sample_id}.jpg"
                    result_path = scene_dir / f"{sample_id}_result.jpg"
                    if src_path.exists() and result_path.exists():
                        w1, h1 = image_shape(src_path)
                        w2, h2 = image_shape(result_path)
                        if (w1, h1) != (w2, h2):
                            _add_issue(
                                issues,
                                result_path,
                                "shape_mismatch",
                                "Source and result dimensions mismatch",
                            )
                        if (w1 > h1) != (w2 > h2):
                            _add_issue(
                                issues,
                                result_path,
                                "orientation_mismatch",
                                "Source and result orientation mismatch",
                            )

    LOGGER.info("linter_completed issues=%s", len(issues))
    return issues
