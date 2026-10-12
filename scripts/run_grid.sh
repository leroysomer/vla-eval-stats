#!/usr/bin/env bash
# Run a grid on the GPU of this machine. Arguments are passed on to scripts/run_grid.py.
#
#   scripts/run_grid.sh --grid configs/grid_throughput.yaml                        # here
#   scripts/tmux_run.sh grid scripts/run_grid.sh --grid configs/grid_throughput.yaml   # detached
#
# Idempotent: rerun the same command after a crash, a kill or a reboot.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

echo "== run_grid $* on $(hostname), MUJOCO_GL=$MUJOCO_GL"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

exec uv run scripts/run_grid.py "$@"
