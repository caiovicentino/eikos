#!/bin/bash
# Serve the model with vLLM (letter readout + hybrid prefix cache). REQUIRES vLLM >= 0.30.0:
# older builds return wrong answers when several requests with long inputs are batched together on this hybrid
# (Gated DeltaNet) architecture. Usage: serve_vllm.sh <model_dir> <port> [gpu_memory_utilization]
# Environment: MAX_NUM_SEQS (default 64), the most requests decoded at once; MAX_MODEL_LEN (default 16384), the longest
# prompt in tokens. Images count too: a 4K screenshot is about 8,300 tokens, so raise MAX_MODEL_LEN for several large
# images in one request (the model's own window is 262,144).
set -e
python - <<'PY'
import sys
import vllm
from packaging.version import Version
if Version(vllm.__version__.split("+")[0]) < Version("0.30.0"):
    sys.exit(f"vLLM {vllm.__version__} is too old for this model: please install vllm>=0.30.0")
PY
# --max-logprobs 600: every one of up to 588 option labels (A..Z, AA, AB, ...) is read in the same pass.
# --mamba-cache-mode align: what vLLM 0.30 uses for this hybrid model anyway ("all" falls back to it with a warning).
# --max-num-seqs: each request in flight holds one Mamba cache block. vLLM's default (1024 on large GPUs) needs
#   more blocks than a 27B build leaves free on one 80-96 GB GPU, and the server then refuses to start.
exec vllm serve "$1" --host 127.0.0.1 --port "$2" --served-model-name decider --dtype bfloat16 \
  --gpu-memory-utilization "${3:-0.85}" --max-model-len "${MAX_MODEL_LEN:-16384}" --enable-prefix-caching --mamba-cache-mode align \
  --logprobs-mode processed_logprobs --max-logprobs 600 --max-num-seqs "${MAX_NUM_SEQS:-64}"
