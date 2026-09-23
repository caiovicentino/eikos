"""DecisionBench (akhilaaa3/decision-bench, medium+hard) with the LetterAdapter — out-of-distribution
test (never trained on). All types: noul, choice, score.
usage: python eval_db.py --model <path> --device cuda:N --tag X [--adapter lora] [--sym] [--temp T]
"""
from __future__ import annotations

import argparse
import json
import sys
import types

import os
EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
sys.path.insert(0, JEVBENCH_DIR)
from datasets import load_dataset  # noqa: E402
from letter_adapter import LetterAdapter, options_of  # noqa: E402


def tasks_of(cfg):
    out = []
    for row in load_dataset("akhilaaa3/decision-bench", cfg)["test"]:
        qs = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        ans = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
        state = row["state"]
        for k, q in qs.items():
            if k not in ans:
                continue
            t = q.get("type")
            if t == "noul":
                labels, exp = ["no", "yes"], ("yes" if ans[k] else "no")
            elif t == "score":
                labels, exp = [str(i) for i in range(len(q["criteria"]))], str(int(ans[k]))
            else:
                labels, exp = list(q["criteria"].keys()), str(ans[k])
            out.append(types.SimpleNamespace(id=f"{row['id']}:{k}", state=state, question=q, labels=labels,
                                             expected=exp, cfg=cfg, qtype=t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--verify-budget", type=int, default=0)
    a = ap.parse_args()
    ad = LetterAdapter(a.model, device=a.device, sym=a.sym, adapter_path=a.adapter, temp=a.temp, calib=a.calib, verify_budget=a.verify_budget)
    ad.load()
    res = {}
    recs = []
    for cfg in ("medium", "hard"):
        for t in tasks_of(cfg):
            r = ad.run(t)
            ok = bool(r.ok and r.probs)
            pred = max(r.probs, key=r.probs.get) if ok else None
            recs.append({"id": t.id, "cfg": cfg, "type": t.qtype, "ok": ok, "pred": pred, "gold": t.expected,
                         "correct": pred == t.expected, "conf": max(r.probs.values()) if ok else None,
                         "err": r.error})
    for cfg in ("medium", "hard"):
        rs = [x for x in recs if x["cfg"] == cfg]
        res[cfg] = {"acc": round(sum(x["correct"] for x in rs) / len(rs), 4), "n": len(rs),
                    "fail": sum(not x["ok"] for x in rs)}
        for ty in ("noul", "choice", "score"):
            sub = [x for x in rs if x["type"] == ty]
            res[cfg][ty] = round(sum(x["correct"] for x in sub) / max(len(sub), 1), 4)
    res["tag"] = a.tag
    print(json.dumps(res), flush=True)
    with open(f"{EIKOS_RUNS}/db_{a.tag}.jsonl", "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    json.dump(res, open(f"{EIKOS_RUNS}/db_summary_{a.tag}.json", "w"))


if __name__ == "__main__":
    main()
