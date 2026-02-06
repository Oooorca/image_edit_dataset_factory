from __future__ import annotations

from image_edit_dataset_factory.pipeline.lint_step import run_lint
from image_edit_dataset_factory.scripts.common import load_runtime_config, parse_common_args


def main() -> int:
    parser = parse_common_args("Run dataset linter")
    args = parser.parse_args()
    cfg = load_runtime_config(args.config, args.set, args.no_json_logs, run_name=args.run_name)
    _, issue_count = run_lint(cfg)
    return 1 if issue_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
