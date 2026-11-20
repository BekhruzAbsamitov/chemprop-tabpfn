#!/bin/bash
# The program HTCondor runs INSIDE the docker container for the AL benchmark.
# It uses the uv environment + Hugging Face cache that cluster/setup_env.sh
# prepared once in your (mounted) home directory.
set -euo pipefail

# TODO: set this to where your repo lives in your home directory on the cluster.
REPO="$HOME/chemprop-tabpfn"
cd "$REPO"

# Keep a separate, timestamped copy of this start's output: if HTCondor requeues the job,
# the restart would otherwise overwrite the .out file of the first start.
mkdir -p runlogs
exec > >(tee -a "runlogs/benchmark_extended_${1}_${2}_${3}.$(date -u +%Y%m%dT%H%M%SZ).tee.log") 2>&1

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$REPO/.hf_cache" # HF cache inside the (mounted) repo — no ambiguity
export HF_HUB_OFFLINE=1          # the checkpoint was cached here during setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ -f .env ]] && { set -a; source .env; set +a; }

echo "host: $(hostname)"
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || true
echo "start: $(date)"

# $1 $2 $3 = experiment, target, representation, from `queue ... from (...)` in the .sub
PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python src/scripts/benchmark_extended.py "$1" "$2" "$3"

echo "end: $(date)"
