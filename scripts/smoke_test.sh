#!/usr/bin/env bash
# Step 1 smoke test: SmolVLA on one LIBERO task, headless on the GPU, with a rollout video.
#
#   scripts/smoke_test.sh                      # run in the current terminal
#   scripts/tmux_run.sh smoke scripts/smoke_test.sh   # run detached, log in logs/
#
# Overridable through the environment: POLICY, SUITE, TASK_ID, N_EPISODES.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

POLICY=${POLICY:-lerobot/smolvla_libero}
SUITE=${SUITE:-libero_spatial}
TASK_ID=${TASK_ID:-0}
N_EPISODES=${N_EPISODES:-2}

out_dir=outputs/smoke_test/$(date +%Y%m%d_%H%M%S)

echo "== smoke test: $POLICY on $SUITE task $TASK_ID, $N_EPISODES episodes"
echo "== host $(hostname), MUJOCO_GL=$MUJOCO_GL, output $out_dir"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# lerobot/smolvla_libero names its cameras camera1/camera2/camera3, the LIBERO env
# defaults to image/image2, hence the mapping.
uv run lerobot-eval \
  --policy.path="$POLICY" \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.task_ids="[$TASK_ID]" \
  --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes="$N_EPISODES" \
  --output_dir="$out_dir"

video=$out_dir/videos/${SUITE}_${TASK_ID}/eval_episode_0.mp4
dest=figures/smoke_test_${POLICY##*/}_${SUITE}_task${TASK_ID}.mp4
cp "$video" "$dest"
echo "== results: $out_dir/eval_info.json"
echo "== video copied to $dest"
