from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def filtered(self) -> Path:
        return self.data / "filtered"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def layers_cache(self) -> Path:
        return self.outputs / "layers_cache"

    @property
    def dataset(self) -> Path:
        return self.outputs / "dataset"

    @property
    def reports(self) -> Path:
        return self.outputs / "reports"

    def ensure(self) -> None:
        for path in [
            self.data,
            self.raw,
            self.filtered,
            self.outputs,
            self.logs,
            self.layers_cache,
            self.dataset,
            self.reports,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def resolve_project_paths(root: str | Path | None = None) -> ProjectPaths:
    base = Path(root) if root else Path.cwd()
    return ProjectPaths(root=base.resolve())
