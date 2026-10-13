#!/usr/bin/env bash
# Prepare another lab machine to run this project.
#
#   scripts/setup_machine.sh <host>          e.g. scripts/setup_machine.sh bengali
#
# The repo and the results live in the home directory, which every lab machine mounts. The
# heavy, read-only part (Python environment, model weights, LIBERO assets: about 17 GB) lives
# on each machine's local /Data disk. This script copies that part from the current machine,
# then checks the setup on the remote machine without using its GPU. Safe to run again: only
# what is missing or changed is copied.
set -euo pipefail

if [ $# -ne 1 ]; then
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
host=$1

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
repo=$(pwd -P)

echo "== copying $DATA_ROOT to $host"
ssh "$host" "mkdir -p $DATA_ROOT && chmod 700 $DATA_ROOT"
rsync -a --info=stats1 \
  "$DATA_ROOT/huggingface" "$DATA_ROOT/python" "$DATA_ROOT/libero_assets" \
  "$DATA_ROOT/.libero" "$DATA_ROOT/.local" "$DATA_ROOT/venvs" \
  "$host:$DATA_ROOT/" | tail -2

echo "== checking $host"
ssh "$host" "
  set -e
  link=$DATA_ROOT/vla-eval-stats
  if [ -e \$link ] && [ ! -L \$link ]; then echo \"\$link exists and is not a link\" >&2; exit 1; fi
  ln -sfn $repo \$link
  cd $repo && source env.sh
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  uv run pytest -q 2>&1 | tail -1
  uv run scripts/run_grid.py --grid configs/grid_main.yaml --dry-run 2>&1 | grep INFO
"
echo "== $host is ready"
