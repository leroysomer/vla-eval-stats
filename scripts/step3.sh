#!/usr/bin/env bash
# Step 3 runs on the GPU, under an hour in all, then the tables that depend on them.
#
#   scripts/tmux_run.sh step3 scripts/step3.sh
#
# 1. times 20 episodes of each policy of the timing grid not stored yet;
# 2. runs them a second time, to check that each policy is reproducible for a fixed key;
# 3. runs the sanity check: 100 episodes per policy, stored with the main grid.
# Idempotent: rerun the same command if it is interrupted.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

scripts/run_grid.sh --grid configs/grid_throughput.yaml
scripts/run_grid.sh --grid configs/grid_throughput.yaml --out-dir results/raw/throughput_rerun
scripts/run_grid.sh --grid configs/grid_sanity.yaml

uv run scripts/analyze_throughput.py
uv run scripts/check_harness.py
uv run scripts/analyze_sanity.py
