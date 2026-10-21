#!/usr/bin/env bash
# Rebuild every table and figure of the study from results/raw/main/. CPU only, about 2 minutes
# (add --with-validation for the Step 5 simulation tables: 7 more minutes).
#
#   uv sync --no-group sim          # first time on a machine without the simulator
#   scripts/make_figures.sh [--with-validation]
#
# The rollout GIF is rebuilt too when the Step 7 videos are present (scripts/rerun_with_video.py,
# GPU).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run() { echo "== $*"; uv run --no-sync "$@"; }

run scripts/check_completeness.py
run scripts/analyze_sanity.py   # Step 3, on the 100 sanity keys of the main grid
if [ "${1:-}" = "--with-validation" ]; then
  run scripts/validate_stats.py
fi
run scripts/analyze_rq1.py
run scripts/analyze_rq2.py   # before RQ3, which reads its tables
run scripts/analyze_rq3.py
if ls figures/videos/*.mp4 >/dev/null 2>&1; then
  run scripts/make_rollout_gif.py
else
  echo "== no videos in figures/videos/: figures/rollout_pair.gif left as it is"
fi
echo "== done: results/tables/ and figures/"
