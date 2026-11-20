#!/bin/bash
# The program HTCondor runs INSIDE the docker container for the FOCUSED full_sp AL benchmark.
# It uses the uv environment + Hugging Face cache that cluster/setup_env.sh
# prepared once in your (mounted) home directory.
set -euo pipefail

# TODO: set this to where your repo lives in your home directory on the cluster.
REPO="$HOME/chemprop-tabpfn"
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$REPO/.hf_cache" # HF cache inside the (mounted) repo — no ambiguity
export HF_HUB_OFFLINE=1          # the checkpoint was cached here during setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ -f .env ]] && { set -a; source .env; set +a; }

echo "host: $(hostname)"
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || true
echo "start: $(date)"

# Fail fast if the supervisor's checkpoint was not synced: without it RIGREncoder falls
# back to random init and would benchmark the wrong thing under the name "pretrained".
for f in experiments/trained_encoder_full_sp_10ep.pt experiments/trained_encoder_full_sp_10ep_best.pt; do
  test -f "$f" || { echo "MISSING $f — pull it from the training run before submitting."; exit 1; }
done

# $1 is the target name supplied by `queue TARGET in (...)` in the .sub file, so
# the four targets run as four concurrent jobs instead of one long sequential one.
PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python src/scripts/benchmark_al.py "$1"

echo "end: $(date)"
