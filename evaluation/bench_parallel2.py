"""Parallelism on a dedicated GPU, with the merged (standalone) model.
  hf     : in-process PyTorch — N questions sequentially vs. one batched pass
  sglang : N questions in one call (prefix cache) + K concurrent clients (throughput and latency)
usage: PROMPT_STYLE=semif python bench_parallel2.py hf <model> <gpu> <calib>
       PROMPT_STYLE=semif python bench_parallel2.py sglang <url> <model> <calib>
"""
import concurrent.futures as cf
import json
import statistics
import sys
import time

from decision_core import options_of
from letter_adapter import LetterAdapter

import os
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
MODE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("hf", "sglang") else "lib"
STATE = ("Order ticket #A-2231 (2026-09-22 14:05 UTC). Client: retail, risk profile = moderate, investment horizon "
         "4 years. Request: BUY 1,500 shares of XYZ at market. Account equity: USD 48,000; cash USD 9,500; margin "
         "enabled. Last XYZ price: USD 41.20; XYZ belongs to the technology sector; the client already holds "
         "USD 12,000 of technology stocks. Firm rule 4.2: for moderate-risk retail clients, a single order may not "
         "exceed 50% of account equity unless a supervisor approves it in writing. Rule 5.1: sector exposure above "
         "35% of equity requires a suitability note. Rule 6.3: orders above USD 25,000 must be routed to the block "
         "desk. Supervisor approvals on file: none. Suitability notes on file: none. ") * 3
QS = [
    {"type": "noul", "instructions": "Under rule 4.2, can this order be executed as submitted?",
     "criteria": {"true": "complies with rule 4.2", "false": "breaches rule 4.2"}},
    {"type": "choice", "instructions": "What should the desk do with this ticket?",
     "criteria": {"execute": "send as submitted", "request_approval": "hold and ask a supervisor",
                  "reduce_size": "cut to the allowed size", "reject": "refuse the order"}},
    {"type": "noul", "instructions": "Is a suitability note required under rule 5.1?",
     "criteria": {"true": "sector exposure would exceed 35% of equity", "false": "it would not"}},
    {"type": "noul", "instructions": "Must the order go to the block desk under rule 6.3?",
     "criteria": {"true": "the order value exceeds USD 25,000", "false": "it does not"}},
    {"type": "score", "instructions": "How risky is this ticket for the firm?",
     "criteria": ["low risk", "moderate risk", "high risk", "critical risk"]},
    {"type": "choice", "instructions": "Which rule is the main blocker?",
     "criteria": {"rule_4_2": "the order-size rule", "rule_5_1": "the sector-exposure rule",
                  "rule_6_3": "the routing rule", "none": "nothing blocks the order"}},
]
ITEMS = [(q, options_of(q)) for q in QS] * 2  # 12 questions


def timeit(f, reps=5):
    f()
    t0 = time.perf_counter()
    for _ in range(reps):
        r = f()
    return (time.perf_counter() - t0) / reps * 1000, r


if MODE == "lib":
    pass
elif MODE == "hf":
    import torch
    ad = LetterAdapter(sys.argv[2], device=sys.argv[3], calib=sys.argv[4])
    ad.load()
    for n in (1, 4, 12):
        it = ITEMS[:n]

        def seq():
            r = [ad.dist(STATE, q, o) for q, o in it]
            torch.cuda.synchronize()
            return r

        def par():
            r = ad.dist_many(STATE, it)
            torch.cuda.synchronize()
            return r
        ts, _ = timeit(seq)
        tp, rp = timeit(par)
        print(json.dumps({"backend": "hf", "questions": n, "state_tokens": rp[0][1], "seq_ms": round(ts, 1),
                          "batch_ms": round(tp, 1)}), flush=True)
    json.dump([p for p, _ in ad.dist_many(STATE, ITEMS)], open("/tmp/bench_hf_probs.json", "w"))
else:
    url, model_dir, calib = sys.argv[2], sys.argv[3], sys.argv[4]
    ad = LetterAdapter(f"sglang:{url}|{model_dir}", calib=calib)
    ad.load()
    for n in (1, 4, 12):
        it = ITEMS[:n]
        t, r = timeit(lambda: ad.dist_many(STATE, it) if n > 1 else [ad.dist(STATE, *it[0])])
        print(json.dumps({"backend": "sglang", "questions": n, "one_call_ms": round(t, 1)}), flush=True)
    try:
        hf = json.load(open("/tmp/bench_hf_probs.json"))
        sg = [p for p, _ in ad.dist_many(STATE, ITEMS)]
        same = sum(max(a, key=a.get) == max(b, key=b.get) for a, b in zip(hf, sg))
        dmax = max(abs(a[k] - b[k]) for a, b in zip(hf, sg) for k in a)
        print(json.dumps({"hf_vs_sglang": f"same answer {same}/{len(hf)}", "max_prob_diff": round(dmax, 4)}), flush=True)
    except FileNotFoundError:
        pass
    # throughput: K concurrent clients, each request = 1 public JevBench state with 1 question
    pub = [json.loads(l) for t in ("easy", "original", "hard") for l in open(f"{JEVBENCH_DIR}/datasets/public/{t}.jsonl")]
    reqs = [(r["state"], r["question"], options_of(r["question"], list(r["labels"]))) for r in pub][:200]
    for K in (1, 8, 32):
        lat = []

        def one(x):
            s, q, o = x
            t0 = time.perf_counter()
            ad.dist(s, q, o)
            return time.perf_counter() - t0
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(K) as ex:
            lat = list(ex.map(one, reqs))
        tot = time.perf_counter() - t0
        lat.sort()
        print(json.dumps({"backend": "sglang", "clients": K, "decisions_per_s": round(len(reqs) / tot, 1),
                          "p50_ms": round(1000 * lat[len(lat) // 2], 1),
                          "p95_ms": round(1000 * lat[int(0.95 * len(lat))], 1)}), flush=True)
