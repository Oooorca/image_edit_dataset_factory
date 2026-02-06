from __future__ import annotations

import argparse
from pathlib import Path

from image_edit_dataset_factory.core.config import AppConfig, load_config
from image_edit_dataset_factory.core.logging import setup_logging


def parse_common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values with dotted keys: key=value",
    )
    parser.add_argument("--run-name", default="run", help="Log run name")
    parser.add_argument("--no-json-logs", action="store_true", help="Use human-readable logs")
    return parser


def load_runtime_config(
    config_path: str,
    overrides: list[str],
    no_json_logs: bool,
    run_name: str = "run",
) -> AppConfig:
    cfg = load_config(config_path, overrides=overrides)
    if no_json_logs:
        cfg.json_logs = False
    setup_logging(
        Path(cfg.paths.root) / cfg.paths.logs_dir, run_name=run_name, json_logs=cfg.json_logs
    )
    return cfg
