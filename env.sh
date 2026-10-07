# Source this before working in the repo or inside Slurm jobs:  source env.sh
# Keeps every cache on the local /Data disk (the home quota is 30 GB and nearly full).
export DATA_ROOT=/Data/$USER
export PATH="$DATA_ROOT/.local/bin:$PATH"
export XDG_CACHE_HOME=$DATA_ROOT/.cache
export UV_CACHE_DIR=$DATA_ROOT/.cache/uv
export UV_PYTHON_INSTALL_DIR=$DATA_ROOT/python
export PIP_CACHE_DIR=$DATA_ROOT/.cache/pip
export HF_HOME=$DATA_ROOT/huggingface
export LIBERO_CONFIG_PATH=$DATA_ROOT/.libero
export MUJOCO_GL=egl
