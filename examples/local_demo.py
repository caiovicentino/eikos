"""Local demo (Mac, Apple Silicon/MPS): standalone decision model, with no LoRA, no API and no internet.
usage: python local_demo.py <model_dir> [mps|cpu]"""
import json
import os
import sys
import time

M = sys.argv[1]
DEV = sys.argv[2] if len(sys.argv) > 2 else "mps"
cfg = json.load(open(os.path.join(M, "decision_config.json")))
os.environ["PROMPT_STYLE"] = cfg["prompt_version"].rsplit("-", 1)[-1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from decision_core import options_of  # noqa: E402
from letter_adapter import LetterAdapter  # noqa: E402

t0 = time.perf_counter()
ad = LetterAdapter(M, device=DEV, calib=os.path.join(M, "calib.json"))
ad.load()
print(f"loaded in {time.perf_counter() - t0:.1f} s on device {DEV}"
      + (f" | MPS memory: {torch.mps.driver_allocated_memory() / 1e9:.1f} GB" if DEV == "mps" else ""), flush=True)

from local_demo_cases import CASES, MULTI  # noqa: E402

for name, state, q in CASES:
    opts = options_of(q)
    ad.dist(state, q, opts)  # warm-up
    t0 = time.perf_counter()
    p, n = ad.dist(state, q, opts)
    dt = time.perf_counter() - t0
    top = max(p, key=p.get)
    print(f"\n[{name}] {n} tokens | {dt * 1000:.0f} ms | answer: {top} ({p[top]:.0%}) | {json.dumps({k: round(v, 3) for k, v in p.items()})}",
          flush=True)
# several questions about the same state (prefix cache)
state = CASES[0][1]
qs = MULTI
items = [(q, options_of(q)) for q in qs]
ad.dist_many_cached(state, items)
t0 = time.perf_counter()
res = ad.dist_many_cached(state, items)
dt = time.perf_counter() - t0
print(f"\n[3 questions, same state, prefix cache] {dt * 1000:.0f} ms total")
for (q, o), (p, _) in zip(items, res):
    top = max(p, key=p.get)
    lab = q["criteria"][int(top)] if q["type"] == "score" and isinstance(q.get("criteria"), list) else top
    print(f"  {q['instructions'][:60]!r} → {lab} ({p[top]:.0%})")
t0 = time.perf_counter()
full = [ad.dist(state, q, o)[0] for q, o in items]
dt_full = time.perf_counter() - t0
diff = max(abs(full[i][k] - res[i][0][k]) for i in range(len(items)) for k in full[i])
print(f"  no cache (3 full prompts, one at a time): {dt_full * 1000:.0f} ms | max probability difference: {diff:.4f}")
