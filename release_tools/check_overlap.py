"""Overlap between the released dataset and EVERY evaluation suite we report (not only the public JevBench items).

- Natural-text suites (JevBench public, DecisionBench, general battery, finance, finance 2, RuleArena): word 8-grams,
  containment in both directions:
  - share of the dataset row found in the evaluation item;
  - share of the evaluation item found in the dataset row (catches an evaluation item embedded in a long dossier).
  Flags > 0.15 with at least 8 shared 8-grams. 8-grams that occur in more than 200 items of a suite are that suite's
  own boilerplate and are ignored.
- Programmatic suites (trade, rules, rulebooks): the same generator with other seeds repeats its templates on purpose,
  so only exact duplicates count (normalized state + instructions).
usage: python check_overlap.py <dataset_dir> <report.json>
"""
import collections
import glob
import json
import os
import re
import sys

J = os.environ.get("EIKOS_HOME", ".")
NATURAL = {"jevbench_public": "suite_jb_public.jsonl", "decisionbench": "suite_db.jsonl", "general": "suite.jsonl",
           "finance": "suite_fin.jsonl", "finance2": "suite_fin2.jsonl", "rulearena": "suite_rulearena.jsonl"}
TEMPLATED = {"trade": "suite_trade.jsonl", "rules": "suite_rules.jsonl", "rulebook": "suite_rulebook.jsonl"}
N, THRESHOLD, MIN_SHARED, STOP = 8, 0.15, 8, 200
PKG, OUT = sys.argv[1], sys.argv[2]


def text_of(r):
    st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
    q = r.get("question")
    ins = q.get("instructions", "") if isinstance(q, dict) else r.get("instructions", "")
    return f"{st} {ins}"


def grams(t):
    w = re.findall(r"\w+", t.lower())
    return {hash(" ".join(w[i:i + N])) for i in range(max(0, len(w) - N + 1))}


def norm(t):
    return " ".join(re.findall(r"\w+", t.lower()))


rows = []
for p in sorted(glob.glob(f"{PKG}/data/*/*.jsonl")):
    cfg = p.split("/")[-2] + "/" + os.path.basename(p)[:-6]
    for line in open(p):
        r = json.loads(line)
        rows.append((cfg, r["id"], text_of(r)))
print(f"dataset rows: {len(rows)}", flush=True)

rep = {"threshold": THRESHOLD, "n": N, "suites": {}, "flagged": []}
for name, fn in NATURAL.items():
    items = [json.loads(line) for line in open(f"{J}/{fn}")]
    g_items = [grams(text_of(x)) for x in items]
    df = collections.Counter(g for gs in g_items for g in gs)
    idx = collections.defaultdict(list)
    for k, gs in enumerate(g_items):
        g_items[k] = {g for g in gs if df[g] <= STOP}
        for g in g_items[k]:
            idx[g].append(k)
    worst, flagged = 0.0, 0
    for cfg, rid, t in rows:
        gr = {g for g in grams(t) if df.get(g, 0) <= STOP}
        if not gr:
            continue
        hits = collections.Counter()
        for g in gr:
            for k in idx.get(g, ()):
                hits[k] += 1
        for k, h in hits.items():
            if h < MIN_SHARED:
                continue
            c = max(h / len(gr), h / max(1, len(g_items[k])))
            worst = max(worst, c)
            if c > THRESHOLD:
                flagged += 1
                rep["flagged"].append({"row": rid, "file": cfg, "suite": name, "item": str(items[k].get("id")),
                                       "containment": round(c, 3), "shared": h})
    rep["suites"][name] = {"items": len(items), "rows_flagged": flagged, "max_containment": round(worst, 3)}
    print(name, rep["suites"][name], flush=True)

for name, fn in TEMPLATED.items():
    ev = {}
    for line in open(f"{J}/{fn}"):
        x = json.loads(line)
        ev[norm(text_of(x))] = str(x.get("id"))
    dup = [(cfg, rid, ev[norm(t)]) for cfg, rid, t in rows if norm(t) in ev]
    for cfg, rid, eid in dup:
        rep["flagged"].append({"row": rid, "file": cfg, "suite": name, "item": eid, "containment": 1.0,
                               "shared": "exact duplicate"})
    rep["suites"][name] = {"items": len(ev), "exact_duplicates": len(dup)}
    print(name, rep["suites"][name], flush=True)

rep["flagged_ids"] = sorted({s["row"] for s in rep["flagged"]})
json.dump(rep, open(OUT, "w"), indent=1, ensure_ascii=False)
print("flagged rows (unique):", len(rep["flagged_ids"]))
