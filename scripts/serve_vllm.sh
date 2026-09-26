#!/bin/bash
# Serve the model with vLLM (letter readout + hybrid prefix cache). REQUIRES vLLM >= 0.30.0:
# older builds return wrong answers when several requests with long inputs are batched together on this hybrid
# (Gated DeltaNet) architecture. Usage: serve_vllm.sh <model_dir> <port> [gpu_memory_utilization]
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
exec vllm serve "$1" --host 127.0.0.1 --port "$2" --served-model-name decider --dtype bfloat16 \
  --gpu-memory-utilization "${3:-0.85}" --max-model-len 16384 --enable-prefix-caching --mamba-cache-mode align \
  --logprobs-mode processed_logprobs --max-logprobs 600
