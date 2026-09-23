"""Runs Jev (TypeSafe via Vercel AI Gateway) on the generalization suites — EVALUATION ONLY (never
training: distilling from Jev is prohibited by its terms). Same question format as our model receives.
usage: python eval_jev_suite.py <suite.jsonl> <tag> [threads]
"""
from __future__ import annotations

import collections
import json
import sys
import threading
import time
import urllib.error
import urllib.request

import os
EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
SUITE, TAG = sys.argv[1], sys.argv[2]
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
KEY = os.environ["JEV_API_KEY"]
URL = os.environ["JEV_API_URL"]  # Jev decision endpoint (evaluation only)
LOCK = threading.Lock()


def ask(r):
    q = r["question"]
    t = q["type"]
    jq = {"type": "boolean" if t == "noul" else t, "instructions": q.get("instructions", ""),
          "criteria": q.get("criteria")}
    body = {"state": r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False),
            "model": "typesafe-ai/jev", "questions": {"d": jq}}
    err = "failed"
    for attempt in range(6):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                d = json.loads(resp.read())
            a = d["answers"]["d"]
            if t == "noul":
                p = float(a["probability"])
                probs = {"yes": p, "no": 1 - p}
            else:
                probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
            return probs, time.time() - t0, None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return None, time.time() - t0, f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            time.sleep(2 ** attempt)
            err = f"{type(e).__name__}: {str(e)[:80]}"
    return None, 0.0, err


def main():
    rows = [json.loads(l) for l in open(SUITE)]
    out = open(f"{EIKOS_RUNS}/jev_{TAG}.jsonl", "w")
    per = collections.defaultdict(lambda: [0, 0])
    lat = []
    idx = [0]

    def worker():
        while True:
            with LOCK:
                if idx[0] >= len(rows):
                    return
                r = rows[idx[0]]
                idx[0] += 1
            probs, dt, err = ask(r)
            pred = max(probs, key=probs.get) if probs else None
            ok = pred == str(r["expected"])
            task = r["task"].split("_")[0] if r["task"].startswith("legalbench") else r["task"]
            with LOCK:
                per[task][0] += int(ok)
                per[task][1] += 1
                lat.append(dt)
                out.write(json.dumps({"id": r["id"], "task": task, "pred": pred, "gold": r["expected"], "ok": ok,
                                      "conf": max(probs.values()) if probs else None, "err": err,
                                      "lat": round(dt, 3)}) + "\n")
                out.flush()
    ths = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    out.close()
    summ = {k: {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(per.items())}
    summ["_macro"] = round(sum(v["acc"] for v in summ.values()) / len(summ), 4)
    summ["_lat_p50"] = round(sorted(lat)[len(lat) // 2], 3) if lat else None
    summ["_tag"] = f"jev_{TAG}"
    print(json.dumps(summ), flush=True)
    json.dump(summ, open(f"{EIKOS_RUNS}/jev_summary_{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
