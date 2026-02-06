from __future__ import annotations

import json

from image_edit_dataset_factory.pipeline.orchestrator import PipelineOrchestrator
from image_edit_dataset_factory.scripts.common import load_runtime_config, parse_common_args


def main() -> int:
    parser = parse_common_args("Run full pipeline")
    args = parser.parse_args()
    cfg = load_runtime_config(args.config, args.set, args.no_json_logs, args.run_name)

    summary = PipelineOrchestrator(cfg).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if int(summary.get("lint_issue_count", 0)) > 0:
        return 1
    if int(summary.get("qa_fail_count", 0)) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
