from __future__ import annotations

from image_edit_dataset_factory.pipeline.filter import run_filter
from image_edit_dataset_factory.scripts.common import load_runtime_config, parse_common_args


def main() -> int:
    parser = parse_common_args("Run filter step")
    args = parser.parse_args()
    cfg = load_runtime_config(args.config, args.set, args.no_json_logs, run_name=args.run_name)
    run_filter(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
