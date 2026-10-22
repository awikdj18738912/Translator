#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
WEB_PYTHON="${WEB_PYTHON:-/home/aim0/data/conda/envs/agentic-asr/bin/python}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/home/aim0/data/conda/envs/vllm/lib/python3.12/site-packages/nvidia}"
export LD_LIBRARY_PATH="${CUDA_LIB_DIR}/cublas/lib:${CUDA_LIB_DIR}/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${WEB_PYTHON}" -u app.py "$@"
