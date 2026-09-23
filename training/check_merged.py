"""Checks that the merged model (on its own) gives the same probabilities as base+LoRA on the JevBench easy/original
items. Loads one model at a time (the 27B does not fit twice on a 97 GB GPU)."""
import gc
import json
import sys
import types

import torch

from letter_adapter import LetterAdapter

import os
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
base, adapter, merged, dev = sys.argv[1:5]
items = [json.loads(l) for t in ("easy", "original") for l in open(f"{JEVBENCH_DIR}/datasets/public/{t}.jsonl")][:40]
ts = [types.SimpleNamespace(id=it["id"], state=it["state"], question=it["question"], labels=it["labels"]) for it in items]


a = LetterAdapter(base, device=dev, adapter_path=adapter)
pa = [a.run(t).probs for t in ts]
del a
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
b = LetterAdapter(merged, device=dev)
pb = [b.run(t).probs for t in ts]
mx = max(max(abs(a[k] - b[k]) for k in a) for a, b in zip(pa, pb))
agree = sum(max(a, key=a.get) == max(b, key=b.get) for a, b in zip(pa, pb))
print(f"items={len(items)} same answer={agree}/{len(items)} max probability difference={mx:.4f}")
