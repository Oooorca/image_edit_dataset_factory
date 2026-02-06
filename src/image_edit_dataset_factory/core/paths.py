from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from image_edit_dataset_factory.core.config import AppConfig


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    output_root: Path
    logs_root: Path

    @property
    def manifests_dir(self) -> Path:
        return self.output_root / "manifests"

    @property
    def cache_dir(self) -> Path:
        return self.output_root / "cache"

    @property
    def staging_dir(self) -> Path:
        return self.output_root / "staging"

    @property
    def dataset_dir(self) -> Path:
        return self.output_root / "dataset"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    def ensure_runtime_dirs(self) -> None:
        for path in [
            self.output_root,
            self.logs_root,
            self.manifests_dir,
            self.cache_dir,
            self.staging_dir,
            self.dataset_dir,
            self.reports_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def resolve_paths(cfg: AppConfig) -> ProjectPaths:
    project_root = Path(cfg.paths.project_root).resolve()
    data_root = (project_root / cfg.paths.data_root).resolve()
    output_root = (project_root / cfg.paths.output_root).resolve()
    logs_root = (project_root / cfg.paths.logs_root).resolve()
    return ProjectPaths(
        project_root=project_root,
        data_root=data_root,
        output_root=output_root,
        logs_root=logs_root,
    )
