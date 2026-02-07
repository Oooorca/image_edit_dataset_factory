#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-iedf-services}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists"
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n layered "bash scripts/start_layered_service.sh"
tmux split-window -t "${SESSION}:0" -h "bash scripts/start_edit_service.sh"
tmux select-layout -t "${SESSION}:0" tiled

echo "tmux session started: ${SESSION}"
echo "attach with: tmux attach -t ${SESSION}"
