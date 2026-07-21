#!/bin/bash
# The program HTCondor runs INSIDE the docker container for the full-scale job.
# It uses the uv environment + Hugging Face cache that cluster/setup_env.sh
# prepared once in your (mounted) home directory.
set -euo pipefail

# TODO: set this to where your repo lives in your home directory on the cluster.
REPO="$HOME/chemprop-tabpfn"
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$REPO/.hf_cache" # HF cache inside the (mounted) repo — no ambiguity
export HF_HUB_OFFLINE=1          # the checkpoint was cached here during setup
# Let the CUDA allocator grow segments instead of fragmenting — the long
# training loop repeatedly allocates/frees TabPFN activations of varying size.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ -f .env ]] && { set -a; source .env; set +a; }

echo "host: $(hostname)"
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || true
echo "start: $(date)"

# src/ on the path so 'from models.../from data_utils...' imports resolve.
# PYTHONUNBUFFERED=1 so prints appear in the log live instead of being buffered.
PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python src/scripts/run_experiment_gpu.py

echo "end: $(date)"
