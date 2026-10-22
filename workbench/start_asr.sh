#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
ASR_PYTHON="${ASR_PYTHON:-/home/aim0/data/conda/envs/qwen3-asr/bin/python}"
# Explicit cache pool for one streaming ASR sequence.
ASR_KV_CACHE_BYTES="${ASR_KV_CACHE_BYTES:-536870912}"
ASR_MAX_MODEL_LEN="${ASR_MAX_MODEL_LEN:-4096}"
ASR_MAX_BATCHED_TOKENS="${ASR_MAX_BATCHED_TOKENS:-1024}"
ASR_ENFORCE_EAGER="${ASR_ENFORCE_EAGER:-0}"
ASR_EXTRA_ARGS=()
if [[ "${ASR_ENFORCE_EAGER}" == "1" ]]; then
  ASR_EXTRA_ARGS+=(--enforce-eager)
fi
export CUDA_VISIBLE_DEVICES="${ASR_GPU:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${ASR_PYTHON}" -u -m system.qwen_asr_stream_server \
  --model /home/aim0/data/models/ASR/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.30 --max-model-len "${ASR_MAX_MODEL_LEN}" \
  --max-num-seqs 1 --max-num-batched-tokens "${ASR_MAX_BATCHED_TOKENS}" \
  --kv-cache-memory-bytes "${ASR_KV_CACHE_BYTES}" \
  --vad silero --vad-model /home/aim0/data/models/ASR/silero_vad.onnx \
  --segmentation-mode vad_finalize --port 8766 "${ASR_EXTRA_ARGS[@]}" "$@"
