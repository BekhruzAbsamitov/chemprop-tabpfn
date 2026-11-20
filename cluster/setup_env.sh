#!/usr/bin/env bash
# Run ONCE, on a node that HAS INTERNET — the submit node (e.g. conduit2) is
# simplest, since your home is mounted into the GPU container. Builds the Python
# environment in your home and downloads the TabPFN checkpoint into the HF cache,
# so the offline batch GPU job finds both.
#
# Usage (from the repo root, on the submit node):
#   export HF_TOKEN=hf_...          # or put HF_TOKEN=... in a .env file
#   bash cluster/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
export HF_HOME="$(pwd)/.hf_cache"   # cache TabPFN inside the repo (mounted in the container)

# 1. Install uv (self-contained Python/package manager) if it's not already there.
#    Prefer pip (the container has Python); fall back to the standalone installer.
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv ..."
  python -m pip install --user uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Build the virtual environment on Python 3.13 (the cu124 torch wheels don't
#    exist for 3.14 yet). Downloads GPU torch + all deps.
echo ">> creating .venv (Python 3.13) and installing dependencies (a few minutes) ..."
uv sync --python 3.13

# 3. Make sure we have the HF token (needed to fetch the gated TabPFN checkpoint).
if [[ -z "${HF_TOKEN:-}" && -f .env ]]; then
  set -a; source .env; set +a
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "!! HF_TOKEN not set. Run 'export HF_TOKEN=hf_...' or add it to .env, then re-run." >&2
  exit 1
fi

# 4. Warm the cache: a tiny end-to-end predict downloads every file the GPU job needs.
echo ">> warming caches with a tiny Chemprop -> TabPFN predict ..."
PYTHONPATH=src .venv/bin/python - <<'PY'
import torch
from models.encoder import MoleculeEncoder
from models.tabpfn_regressor import TabPFNHead, _load_hf_token

_load_hf_token()
enc = MoleculeEncoder(hidden_size=32)
head = TabPFNHead(n_estimators=1)
emb = enc(["CCO", "CCN", "c1ccccc1", "CCC"])
out = head.predict(emb, torch.tensor([1., 2., 3., 4.]), enc(["CCC", "CCO"]))
print("warmup prediction shape:", tuple(out.shape), "| device:", head.device)
PY

echo ">> setup complete. Exit this interactive session, then submit the batch job:"
echo "   mkdir -p runlogs && condor_submit cluster/run_gpu.sub"
