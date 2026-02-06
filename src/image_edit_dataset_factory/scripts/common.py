from __future__ import annotations

import argparse

from image_edit_dataset_factory.core.config import AppConfig, load_config
from image_edit_dataset_factory.core.logging import setup_logging
from image_edit_dataset_factory.core.paths import resolve_paths


def parse_common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config value by dotted path, e.g. backends.edit_backend=opencv",
    )
    parser.add_argument("--run-name", default="run", help="Log file run prefix")
    parser.add_argument("--no-json-logs", action="store_true", help="Enable plain text logs")
    return parser


def load_runtime_config(
    config_path: str,
    overrides: list[str],
    no_json_logs: bool,
    run_name: str,
) -> AppConfig:
    cfg = load_config(config_path, overrides)
    if no_json_logs:
        cfg.json_logs = False
    paths = resolve_paths(cfg)
    paths.ensure_runtime_dirs()
    setup_logging(paths.logs_root, run_name=run_name, json_logs=cfg.json_logs)
    return cfg
