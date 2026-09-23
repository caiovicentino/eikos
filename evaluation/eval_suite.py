"""Runs the generalization suite (suite.jsonl) with the LetterAdapter and reports accuracy per task,
chance level per task and the macro average. Same readout/prompt as the rest of the project.
usage: python eval_suite.py --model <path> --device cuda:N --tag X [--adapter lora] [--temp T]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import types

import os
EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, suites and checkpoints
EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
sys.path.insert(0, JEVBENCH_DIR)
from letter_adapter import LetterAdapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--verify-budget", type=int, default=0)
    ap.add_argument("--think-base", action="store_true", help="hybrid verify mode: the base model thinks, the adapter reads the letter")
    ap.add_argument("--suite", default=f"{EIKOS_HOME}/suite.jsonl")
    a = ap.parse_args()
    ad = LetterAdapter(a.model, device=a.device, adapter_path=a.adapter, temp=a.temp, calib=a.calib, verify_budget=a.verify_budget, max_tokens=16000,
                       think_base=a.think_base)
    ad.load()
    rows = [json.loads(l) for l in open(a.suite)]
    per = collections.defaultdict(lambda: [0, 0, 0.0])
    kind = "fin" if "fin" in a.suite.split("/")[-1] else "gen"
    a.tag = a.tag if kind == "gen" else f"fin_{a.tag}"
    out = open(f"{EIKOS_RUNS}/suite_{a.tag}.jsonl", "w")
    for r in rows:
        t = types.SimpleNamespace(id=r["id"], state=r["state"], question=r["question"], labels=r["labels"])
        res = ad.run(t)
        pred = max(res.probs, key=res.probs.get) if res.ok and res.probs else None
        ok = pred == str(r["expected"])
        task = r["task"].split("_")[0] if r["task"].startswith("legalbench") else r["task"]
        per[task][0] += int(ok)
        per[task][1] += 1
        per[task][2] += 1.0 / len(r["labels"])
        out.write(json.dumps({"id": r["id"], "task": task, "pred": pred, "gold": r["expected"], "ok": ok,
                              "conf": max(res.probs.values()) if res.ok and res.probs else None,
                              "err": res.error}) + "\n")
    out.close()
    summ = {k: {"acc": round(v[0] / v[1], 4), "n": v[1], "chance": round(v[2] / v[1], 3)} for k, v in sorted(per.items())}
    summ["_macro"] = round(sum(v["acc"] for v in summ.values()) / len(summ), 4)
    summ["_tag"] = a.tag
    print(json.dumps(summ), flush=True)
    json.dump(summ, open(f"{EIKOS_RUNS}/suite_summary_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
