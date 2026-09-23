# Removes the style shortcut from prog_judge: balances yes/no within each format and strips the solution
# from the references (which were always correct and had their own style).
import json, random, re
import os
EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, evaluation suites and checkpoints
rng = random.Random(5)
rows = [json.loads(l) for l in open(f"{EIKOS_HOME}/data/prog_judge.jsonl")]
out = []
for r in rows:
    if r["source"].endswith("gsm8k_ref"):
        q = r["state"].split("\n\nProposed solution:")[0]
        ans = r["state"].rsplit("Proposed final answer:", 1)[-1].strip()
        r = dict(r, state=f"{q}\n\nProposed final answer: {ans}", source="prog_judge:gsm8k_ref_answer_only")
    r["_fmt"] = "sol" if "Proposed solution:" in r["state"] else "ans"
    ans = r["state"].rsplit("Proposed final answer:", 1)[-1].strip()
    if "." in ans:  # answers with a decimal point were almost always wrong (format shortcut): drop them
        continue
    out.append(r)
final = []
for fmt in ("sol", "ans"):
    ys = [r for r in out if r["_fmt"] == fmt and r["expected"] == "yes"]
    ns = [r for r in out if r["_fmt"] == fmt and r["expected"] == "no"]
    k = min(len(ys), len(ns))
    rng.shuffle(ys); rng.shuffle(ns)
    final += ys[:k] + ns[:k]
    print(fmt, "yes", len(ys), "no", len(ns), "-> kept", 2 * k)
rng.shuffle(final)
with open(f"{EIKOS_HOME}/data/prog_judge_bal.jsonl", "w") as f:
    for r in final:
        r.pop("_fmt", None)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("total", len(final))
