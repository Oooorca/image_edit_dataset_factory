#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

IMPORT_TO_PACKAGE = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
}

OPTIONAL_IMPORT_PACKAGES = {"imagehash", "insightface", "pytesseract"}

SPECIAL_VERSION_SOURCES = {
    "opencv-python": ["opencv-python", "opencv-python-headless"],
}

DEFAULT_DEV_PACKAGES = ["pytest", "pytest-cov", "ruff", "black", "pre-commit"]

EXCLUDE_TOP_LEVEL = {
    "image_edit_dataset_factory",
    "__future__",
}


@dataclass
class SyncResult:
    runtime: list[str]
    dev: list[str]


def _normalize_pkg_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _load_conda_versions(conda_prefix: str | None) -> dict[str, str]:
    if not conda_prefix:
        return {}
    try:
        proc = subprocess.run(
            ["conda", "list", "--json", "-p", conda_prefix],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}

    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}

    versions: dict[str, str] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            version = row.get("version")
            if isinstance(name, str) and isinstance(version, str):
                versions[_normalize_pkg_name(name)] = version
    return versions


def _is_stdlib(name: str) -> bool:
    return name in sys.stdlib_module_names


def _extract_top_level_imports(py_file: Path) -> set[str]:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                imports.add(item.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".", 1)[0])

    return imports


def _discover_imports(paths: list[Path]) -> dict[str, set[str]]:
    bucket = {"runtime": set(), "dev": set()}

    for base in paths:
        if not base.exists():
            continue
        py_files = sorted(base.rglob("*.py"))
        for py_file in py_files:
            top_imports = _extract_top_level_imports(py_file)
            target = "dev" if "tests" in py_file.parts else "runtime"
            for name in top_imports:
                if name in EXCLUDE_TOP_LEVEL or _is_stdlib(name):
                    continue
                bucket[target].add(name)

    return bucket


def _map_to_package(import_name: str) -> str:
    return IMPORT_TO_PACKAGE.get(import_name, import_name)


def _short_version(version: str) -> str:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
    if not match:
        return version
    major, minor, patch = match.groups()
    if patch is not None:
        return f"{major}.{minor}.{patch}"
    if minor is not None:
        return f"{major}.{minor}"
    return f"{major}.0"


def _package_version(package: str, conda_versions: dict[str, str]) -> str | None:
    package_key = _normalize_pkg_name(package)
    if package_key in conda_versions:
        return conda_versions[package_key]

    names = SPECIAL_VERSION_SOURCES.get(package, [package])
    for dist_name in names:
        dist_key = _normalize_pkg_name(dist_name)
        if dist_key in conda_versions:
            return conda_versions[dist_key]
        try:
            return metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _format_dep(package: str, conda_versions: dict[str, str]) -> str:
    ver = _package_version(package, conda_versions)
    if ver is None:
        return package
    return f"{package}>={_short_version(ver)}"


def _format_block(items: list[str], indent: str = "  ") -> list[str]:
    lines = [f'{indent}"{item}",' for item in items]
    return lines


def _replace_array_block(lines: list[str], key: str, values: list[str]) -> list[str]:
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key} = ["):
            start_idx = i
            break

    if start_idx is None:
        raise RuntimeError(f"Cannot find array block: {key}")

    end_idx = start_idx
    while end_idx < len(lines):
        if lines[end_idx].strip() == "]":
            break
        end_idx += 1

    if end_idx >= len(lines):
        raise RuntimeError(f"Unclosed array block: {key}")

    new_lines = lines[:start_idx]
    new_lines.append(lines[start_idx].split("=")[0].rstrip() + " = [")
    new_lines.extend(_format_block(values))
    new_lines.append("]")
    new_lines.extend(lines[end_idx + 1 :])
    return new_lines


def _sync_dependencies(pyproject: Path, runtime_pkgs: list[str], dev_pkgs: list[str]) -> SyncResult:
    conda_versions = _load_conda_versions(os.getenv("CONDA_PREFIX"))
    runtime_deps = [_format_dep(pkg, conda_versions) for pkg in runtime_pkgs]
    dev_deps = [_format_dep(pkg, conda_versions) for pkg in dev_pkgs]

    lines = pyproject.read_text(encoding="utf-8").splitlines()
    lines = _replace_array_block(lines, "dependencies", runtime_deps)
    lines = _replace_array_block(lines, "dev", dev_deps)
    pyproject.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return SyncResult(runtime=runtime_deps, dev=dev_deps)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan imports and sync pyproject dependency arrays to installed versions."
    )
    parser.add_argument("--pyproject", default="pyproject.toml", help="Path to pyproject.toml")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["src", "scripts", "tests"],
        help="Paths to scan for Python imports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved deps without writing",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include known optional runtime packages (insightface/pytesseract/imagehash).",
    )
    parser.add_argument(
        "--show-env",
        action="store_true",
        help="Print the python executable and CONDA_PREFIX being used.",
    )
    parser.add_argument(
        "--conda-prefix",
        default=os.getenv("CONDA_PREFIX"),
        help="Override conda prefix used for version resolution.",
    )
    args = parser.parse_args()

    pyproject = Path(args.pyproject)
    scan_paths = [Path(p) for p in args.paths]

    discovered = _discover_imports(scan_paths)

    runtime_pkgs_set = {_map_to_package(x) for x in discovered["runtime"]}
    if not args.include_optional:
        runtime_pkgs_set -= OPTIONAL_IMPORT_PACKAGES
    runtime_pkgs = sorted(runtime_pkgs_set)

    dev_pkgs = sorted(
        ({_map_to_package(x) for x in discovered["dev"]} - set(runtime_pkgs))
        | set(DEFAULT_DEV_PACKAGES)
    )

    conda_prefix = args.conda_prefix
    conda_versions = _load_conda_versions(conda_prefix)

    runtime_deps = [_format_dep(pkg, conda_versions) for pkg in runtime_pkgs]
    dev_deps = [_format_dep(pkg, conda_versions) for pkg in dev_pkgs]

    if args.show_env:
        print(f"[env] python={sys.executable}")
        print(f"[env] conda_prefix={conda_prefix}")
        print(f"[env] conda_packages_loaded={len(conda_versions)}")

    if args.dry_run:
        print("[runtime dependencies]")
        for dep in runtime_deps:
            print(dep)
        print("\n[dev dependencies]")
        for dep in dev_deps:
            print(dep)
        return 0

    result = _sync_dependencies(pyproject, runtime_pkgs, dev_pkgs)
    print(f"Updated {pyproject}")
    print("runtime deps:")
    for dep in result.runtime:
        print(f"- {dep}")
    print("dev deps:")
    for dep in result.dev:
        print(f"- {dep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
