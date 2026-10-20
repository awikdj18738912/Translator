#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
REPLAY_PYTHON="${REPLAY_PYTHON:-/home/aim0/data/conda/envs/agentic-asr/bin/python}"
if [[ ! -x "${REPLAY_PYTHON}" ]]; then
  echo "Set REPLAY_PYTHON to the Codex-web Python environment (Transformers 5)." >&2
  exit 1
fi
export PYTHONPATH="${PROJECT_DIR}/.runtime/lmt-deps${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/home/aim0/data/conda/envs/vllm/lib/python3.12/site-packages/nvidia/cublas/lib:/home/aim0/data/conda/envs/vllm/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${REPLAY_PYTHON}" scripts/replay_codexweb_translation.py "$@"
