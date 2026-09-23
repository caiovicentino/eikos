"""FinEntity (ODC-BY 1.0, attribution required: Tang et al. 2023, github.com/yixuantt/FinEntity) → per-entity
sentiment decisions on financial news (positive/negative/neutral; 53% neutral). Human label → soft target 0.85.
usage: python prog_finentity.py /tmp/FinEntity/data/FinEntity.json > data/real_finentity.jsonl"""
import hashlib
import json
import sys

CRIT = {"positive": "The news is favorable for this specific company (e.g., gains, beats, upgrades, wins).",
        "negative": "The news is unfavorable for this specific company (e.g., losses, misses, downgrades, penalties).",
        "neutral": "The news mentions this company without a clear favorable or unfavorable implication for it."}
d = json.load(open(sys.argv[1]))
n = 0
for x in d:
    seen = set()
    for a in x["annotations"]:
        ent, tag = a["value"].strip(), a["tag"].strip().lower()
        if tag not in CRIT or ent.lower() in seen:
            continue
        seen.add(ent.lower())
        labels = ["positive", "negative", "neutral"]
        tp = {l: (0.85 if l == tag else 0.075) for l in labels}
        rid = "real-finentity-" + hashlib.sha1((x["content"] + "|" + ent).encode()).hexdigest()[:12]
        print(json.dumps({"id": rid, "family": "entity_sentiment", "topic": "finance: capital markets",
                          "state": x["content"],
                          "question": {"type": "choice",
                                       "instructions": f"What is the implication of this news for {ent} specifically?",
                                       "criteria": CRIT},
                          "labels": labels, "expected": tag, "teacher_probs": tp, "teacher_top": tag, "agree": True,
                          "difficulty": "standard", "lang": "English", "source": "real_finentity:odc-by"},
                         ensure_ascii=False))
        n += 1
print(f"items: {n}", file=sys.stderr)
