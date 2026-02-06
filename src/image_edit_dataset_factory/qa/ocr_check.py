from __future__ import annotations

from image_edit_dataset_factory.core.schema import SampleModel


def run_ocr_check(samples: list[SampleModel]) -> list[dict[str, object]]:
    """Optional OCR verification for text edits (guarded stub)."""
    try:
        import pytesseract  # type: ignore # noqa: F401
    except ImportError:
        return [
            {"sample_id": sample.sample_id, "skipped": True, "reason": "pytesseract_not_installed"}
            for sample in samples
        ]

    return [
        {"sample_id": sample.sample_id, "skipped": True, "reason": "not_implemented"}
        for sample in samples
    ]
