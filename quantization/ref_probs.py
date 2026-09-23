"""bf16 (PyTorch) reference for validating quantized builds: the model's probabilities on the 231 public JevBench
items + the demo cases. Saves state, question, options and probs to a portable JSON (the comparison runs on a Mac).
usage: PROMPT_STYLE=semif python ref_probs.py <model> <device> <output.json>"""
import json
import sys
from decision_core import options_of  # noqa: E402
from letter_adapter import LetterAdapter  # noqa: E402
from local_demo_cases import CASES, MULTI  # noqa: E402

import os
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
M, dev, out = sys.argv[1:4]
ad = LetterAdapter(M, device=dev, calib=f"{M}/calib.json", max_tokens=40000)
ad.load()
items = []
for t in ("easy", "original", "hard"):
    for line in open(f"{JEVBENCH_DIR}/datasets/public/{t}.jsonl"):
        r = json.loads(line)
        items.append((r["id"], r["state"], r["question"], options_of(r["question"], list(r["labels"]))))
for name, st, q in CASES:
    items.append((f"demo:{name}", st, q, options_of(q)))
for i, q in enumerate(MULTI):
    items.append((f"demo_multi:{i}", CASES[0][1], q, options_of(q)))
res = []
for iid, st, q, opts in items:
    if len(opts) > 26:
        continue
    p, n = ad.dist(st, q, opts)
    res.append({"id": iid, "state": st, "question": q, "opts": opts, "probs": p, "n_tok": n})
json.dump(res, open(out, "w"), ensure_ascii=False)
print(f"{len(res)} reference items in {out}")
