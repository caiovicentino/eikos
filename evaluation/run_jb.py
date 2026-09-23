"""Runs the LetterAdapter on the official JevBench harness (public items) and computes
an estimate of the JevBench Score (a proxy from the public items only — for guidance, not the leaderboard).

usage: python run_jb.py --model <path> --device cuda:0 --tag my_run [--sym] [--adapter <lora>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
sys.path.insert(0, JEVBENCH_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jevbench.budget import Ledger  # noqa: E402
from jevbench.runner import Runner  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402
from letter_adapter import LetterAdapter  # noqa: E402

PUB = f"{JEVBENCH_DIR}/datasets/public"
# $ per 1M input tokens per size class, read from the v1.2 leaderboard (est.)
PRICE_PER_M = {"0.8b": 0.0066, "2b": 0.021, "4b": 0.0236, "9b": 0.08}


def ece_top(conf, correct, bins=10):
    n = len(conf)
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n) if (conf[i] > lo or (b == 0 and conf[i] >= 0)) and conf[i] <= hi]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            c = sum(conf[i] for i in idx) / len(idx)
            tot += len(idx) / n * abs(acc - c)
    return tot


def speed_score(s):
    return max(0.0, min(100.0, 100 - 20 * math.log10(max(s, 1e-6) / 0.1)))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tiers", default="easy,original,hard")
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--verify-budget", type=int, default=0)
    ap.add_argument("--size-class", default="4b")
    ap.add_argument("--out", default=EIKOS_RUNS)
    a = ap.parse_args()

    ad = LetterAdapter(a.model, device=a.device, sym=a.sym, adapter_path=a.adapter, temp=a.temp, calib=a.calib, verify_budget=a.verify_budget)
    ad.price_input_per_m = 0.0
    ad.price_output_per_m = 0.0
    t0 = time.time()
    ad.load()
    print(f"[run_jb] load {time.time() - t0:.1f}s", flush=True)

    recs, tasks_by_id = {}, {}
    for tier in a.tiers.split(","):
        tasks = load_jsonl(f"{PUB}/{tier}.jsonl")
        for t in tasks:
            tasks_by_id[t.id] = t
        res_path = f"{a.out}/{a.tag}_{tier}.jsonl"
        if os.path.exists(res_path):
            os.remove(res_path)
        ledger = Ledger(f"{a.out}/ledger_{a.tag}_{tier}.jsonl", cap_usd=1000)
        runner = Runner(ad, ledger, raw_dir=f"{a.out}/raw_{a.tag}_{tier}_{int(time.time())}",
                        default_reserve_usd=0.0)
        recs[tier] = runner.run_all(tasks, results_path=res_path, progress_every=50)

    summ = {"tag": a.tag, "model": a.model, "adapter": a.adapter, "sym": a.sym, "temp": a.temp}
    lat, toks = [], []
    for tier, rs in recs.items():
        acc = sum(1 for r in rs if r["correct"]) / max(len(rs), 1)
        summ[f"acc_{tier}"] = round(acc, 4)
        summ[f"n_{tier}"] = len(rs)
        summ[f"fail_{tier}"] = sum(1 for r in rs if not r["ok"])
        lat += [r["latency_s"] for r in rs if r["ok"]]
        toks += [(r.get("usage") or {}).get("input_tokens", 0) for r in rs if r["ok"]]
    if "hard" in recs:
        hr = [r for r in recs["hard"] if r["ok"] and r.get("probs")]
        conf = [max(r["probs"].values()) for r in hr]
        corr = [1 if r["correct"] else 0 for r in hr]
        e = ece_top(conf, corr)
        tv = []
        for r in hr:
            gp = (tasks_by_id[r["task_id"]].provenance or {}).get("gold_probs")
            if gp:
                tv.append(0.5 * sum(abs(r["probs"].get(k, 0.0) - v) for k, v in gp.items()))
        summ["ece_hard"] = round(e, 4)
        summ["tv_prob_items"] = round(sum(tv) / len(tv), 4) if tv else None
        summ["n_prob_items"] = len(tv)
        cal_a = 100 * (1 - e / 0.5)
        cal_b = 100 * (1 - (sum(tv) / len(tv))) if tv else cal_a
        summ["Calibration"] = round((cal_a + cal_b) / 2, 1)
    w = {"hard": 0.30, "easy": 0.14, "original": 0.56}
    ws = sum(w[t] for t in recs)
    summ["Intelligence_pub"] = round(100 * sum(w[t] * summ[f"acc_{t}"] for t in recs) / ws, 1)
    p50, p95 = pct(lat, 0.5), pct(lat, 0.95)
    adj = lambda s: 2 * s + 0.15  # noqa: E731 — the leaderboard's adjustment for self-hosted endpoints
    summ["lat_p50"], summ["lat_p95"] = round(p50, 4), round(p95, 4)
    summ["Speed"] = round((speed_score(adj(p50)) + speed_score(adj(p95))) / 2, 1)
    mean_tok = sum(toks) / max(len(toks), 1)
    usd_1k = mean_tok * 1000 * PRICE_PER_M[a.size_class] / 1e6
    summ["usd_per_1k"] = round(usd_1k, 5)
    summ["Cost"] = round(max(0.0, min(100.0, 100 - 30 * math.log10(usd_1k / 0.001))), 1)
    axes = [summ["Intelligence_pub"], summ.get("Calibration", 1), summ["Speed"], summ["Cost"]]
    summ["Score_proxy"] = round(math.exp(sum(0.25 * math.log(max(x, 1)) for x in axes)), 1)
    print(json.dumps(summ, indent=1), flush=True)
    with open(f"{a.out}/summary_{a.tag}.json", "w") as f:
        json.dump(summ, f, indent=1)


if __name__ == "__main__":
    main()
