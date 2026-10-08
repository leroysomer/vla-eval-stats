#!/usr/bin/env bash
# Run a command in a detached tmux session so it survives an SSH disconnect.
#
#   scripts/tmux_run.sh <session> <command> [args...]
#
# stdout and stderr go to logs/<session>_<timestamp>.log, whose last line is the exit code.
# The session closes when the command ends; the log stays.
#
#   tmux ls                    # sessions still running
#   tmux attach -t <session>   # watch live (detach again with Ctrl-b d)
#   tail -f logs/<session>_*.log
#   tmux kill-session -t <session>
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

session=$1
shift
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "tmux session '$session' is already running (tmux attach -t $session)" >&2
  exit 1
fi

mkdir -p logs
log=logs/${session}_$(date +%Y%m%d_%H%M%S).log
cmd=$(printf '%q ' "$@")

# `tmux wait-for <session>-done` blocks until the command has finished.
tmux new-session -d -s "$session" -c "$PWD" \
  "{ $cmd; echo \"exit code: \$?\"; } 2>&1 | tee $log; tmux wait-for -S $session-done"

echo "started tmux session '$session'"
echo "log: $log"
