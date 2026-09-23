"""Validates an MLX build (e.g. 8-bit) against the PyTorch bf16 reference (ref_probs.py): same answer? how much do
the probabilities change? Also measures speed and memory on a Mac.
usage: python mlx_check.py <mlx_dir> <ref_probs_bf16.json> [n_shortest]"""
import json
import sys
import time

import mlx.core as mx

from mlx_decide import MLXDecider

M, ref_path = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
d = MLXDecider(M)
print(f"loaded in {time.perf_counter() - t0:.1f} s | active memory {mx.get_active_memory() / 1e9:.2f} GB", flush=True)
ref = json.load(open(ref_path))
if len(sys.argv) > 3:  # only the N shortest items (+ the demo cases): quick validation on CPU
    demo = [r for r in ref if r["id"].startswith("demo")]
    rest = sorted([r for r in ref if not r["id"].startswith("demo")], key=lambda r: r.get("n_tok", 0))
    ref = rest[:int(sys.argv[3])] + demo
agree, n, maxd, sumd, lat, n_tok = 0, 0, 0.0, 0.0, [], 0
worst = []
for r in ref:
    opts = [tuple(o) for o in r["opts"]]
    t1 = time.perf_counter()
    p, nt = d.dist(r["state"], r["question"], opts)
    lat.append(time.perf_counter() - t1)
    n_tok += nt
    q = r["probs"]
    diff = max(abs(p[k] - q[k]) for k in q)
    maxd = max(maxd, diff)
    sumd += diff
    agree += max(p, key=p.get) == max(q, key=q.get)
    n += 1
    worst.append((diff, r["id"]))
lat.sort()
print(json.dumps({"items": n, "same_answer": f"{agree}/{n}", "max_diff": round(maxd, 4), "mean_diff": round(sumd / n, 4),
                  "latency_p50_ms": round(1000 * lat[n // 2]), "latency_p90_ms": round(1000 * lat[int(n * 0.9)]),
                  "mean_tokens": round(n_tok / n), "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)}), flush=True)
print("largest differences:", [(i, round(x, 3)) for x, i in sorted(worst, reverse=True)[:5]])
multi = [r for r in ref if r["id"].startswith("demo_multi:")]
if multi:
    items = [(r["question"], [tuple(o) for o in r["opts"]]) for r in multi]
    d.dist_many_cached(multi[0]["state"], items)
    t1 = time.perf_counter()
    d.dist_many_cached(multi[0]["state"], items)
    print(f"{len(items)} questions about the same state (cache + batch): {1000 * (time.perf_counter() - t1):.0f} ms")
