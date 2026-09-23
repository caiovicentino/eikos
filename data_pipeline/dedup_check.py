"""Anti-contamination: word 8-gram overlap between training items and the 231
public JevBench items. Writes the list of ids to exclude (containment > LIMIAR threshold).
usage: python dedup_check.py out_exclude.json file1.jsonl [file2.jsonl ...]
"""
import json
import re
import sys

import os
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
N = 8
LIMIAR = 0.15


def text_of(r):
    st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
    return f"{st} {r['question'].get('instructions', '')}"


def grams(t):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + N]) for i in range(max(0, len(w) - N + 1))}


pub = []
for tier in ("easy", "original", "hard"):
    for line in open(f"{JEVBENCH_DIR}/datasets/public/{tier}.jsonl"):
        r = json.loads(line)
        pub.append((r["id"], grams(text_of(r))))
idx = {}
for pid, g in pub:
    for x in g:
        idx.setdefault(x, set()).add(pid)

excl, n_tot, worst = [], 0, []
for path in sys.argv[2:]:
    for line in open(path):
        r = json.loads(line)
        n_tot += 1
        g = grams(text_of(r))
        if not g:
            continue
        hits = {}
        for x in g:
            for pid in idx.get(x, ()):
                hits[pid] = hits.get(pid, 0) + 1
        if hits:
            pid, h = max(hits.items(), key=lambda kv: kv[1])
            c = h / len(g)
            worst.append((c, r.get("id"), pid))
            if c > LIMIAR:
                excl.append(r.get("id"))
worst.sort(reverse=True)
json.dump(excl, open(sys.argv[1], "w"))
print(f"items={n_tot} excluded={len(excl)} | highest containments: {[(round(c, 3), i, p) for c, i, p in worst[:5]]}")
