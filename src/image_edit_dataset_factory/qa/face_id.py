from __future__ import annotations

from image_edit_dataset_factory.core.schema import SampleModel


def run_face_id_check(samples: list[SampleModel]) -> list[dict[str, object]]:
    """Optional face-ID consistency check (guarded stub)."""
    try:
        import insightface  # type: ignore # noqa: F401
    except ImportError:
        return [
            {"sample_id": sample.sample_id, "skipped": True, "reason": "insightface_not_installed"}
            for sample in samples
        ]

    return [
        {"sample_id": sample.sample_id, "skipped": True, "reason": "not_implemented"}
        for sample in samples
    ]
