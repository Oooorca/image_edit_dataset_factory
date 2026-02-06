from __future__ import annotations

from pathlib import Path


def write_utf8_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.strip() + "\n")


def read_utf8_text(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as handle:
        return handle.read()
