"""Long-context items for training: the original decision becomes "Document k" of a dossier with other cases
(states of items from other topics) and neutral records, totaling 6k-32k tokens (approx. 3.5 characters/token).
The question now cites the document ("Regarding Document k: ..."), so the teacher's gold answer remains valid.
Training items only (hash outside the dev split with DEV_FRAC_GEN=0.08).
usage: python make_long.py <output.jsonl> <n> <seed>"""
import hashlib
import json
import random
import sys

import os
EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, evaluation suites and checkpoints
OUT, N, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
SRC = [f"{EIKOS_HOME}/data/labeled.jsonl", f"{EIKOS_HOME}/data_q38/labeled.jsonl", f"{EIKOS_HOME}/data_q38b/labeled.jsonl"]
HOLD = {"topic": {"healthcare administration"}, "family": {"tradeoff"}, "lang": {"Spanish"}}
KINDS = ["ticket system export", "email archive", "compliance log", "chat transcript", "ERP record", "shared drive note",
         "call summary", "audit trail", "trading desk log", "vendor portal message"]
NEUTRAL = ["Routine log: queues reviewed, balances reconciled, checklist archived; no exception raised.",
           "Facilities note: the meeting room booking system was updated; no business decision was recorded.",
           "IT notice: nightly backup completed successfully; no data changes were made to client records.",
           "Calendar: the quarterly all-hands moved to Thursday; attendance optional for field staff."]


def is_dev(i, frac=0.08):
    return (int(hashlib.sha1(str(i).encode()).hexdigest()[:8], 16) % 10000) < frac * 10000


def st(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


rows = []
for p in SRC:
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not r.get("agree") or is_dev(r.get("id")) or r.get("topic") in HOLD["topic"] \
                or r.get("family") in HOLD["family"] or r.get("lang") in HOLD["lang"]:
            continue
        rows.append(r)
rng = random.Random(SEED)
rng.shuffle(rows)
targets, pool = rows[:N], rows[N:]
out = open(OUT, "w")
for t in targets:
    L = rng.choice([6000, 12000, 20000, 32000])
    budget = int(L * 3.5)
    docs, used = [], len(st(t["state"]))
    others = [x for x in rng.sample(pool, min(len(pool), 400)) if x.get("topic") != t.get("topic")]
    for x in others:
        s = st(x["state"])
        if used + len(s) > budget:
            break
        docs.append((rng.choice(KINDS), s))
        used += len(s)
        if rng.random() < 0.3:
            filler = " ".join(rng.choice(NEUTRAL) for _ in range(rng.randint(3, 12)))
            docs.append((rng.choice(KINDS), filler))
            used += len(filler)
    k = rng.randint(0, len(docs))
    docs.insert(k, (rng.choice(KINDS), st(t["state"])))
    body = "\n\n".join(f"--- Document {i + 1} ({kind}) ---\n{txt}" for i, (kind, txt) in enumerate(docs))
    q = dict(t["question"], instructions=f"Regarding Document {k + 1}: {t['question']['instructions']}")
    rec = dict(t, id=t["id"] + f"-long{L // 1000}k", state=body, question=q, source="long:" + (t.get("source") or "gen"),
               long_len=L, long_doc=k + 1, long_n_docs=len(docs))
    rec.pop("rationale", None)  # the rationale cites the short case; do not use it as a target in the dossier
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"long items: {len(targets)}", file=sys.stderr)
