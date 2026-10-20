#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
ASR_PYTHON="${ASR_PYTHON:-/home/aim0/data/conda/envs/qwen3-asr/bin/python}"
# Explicit cache pool for one streaming ASR sequence; keep the 8192-token context.
ASR_KV_CACHE_BYTES="${ASR_KV_CACHE_BYTES:-1073741824}"
export CUDA_VISIBLE_DEVICES="${ASR_GPU:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${ASR_PYTHON}" -u -m system.qwen_asr_stream_server \
  --model /home/aim0/data/models/ASR/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.30 --max-model-len 8192 \
  --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --kv-cache-memory-bytes "${ASR_KV_CACHE_BYTES}" \
  --vad silero --vad-model /home/aim0/data/models/ASR/silero_vad.onnx \
  --segmentation-mode vad_finalize --port 8766 "$@"
