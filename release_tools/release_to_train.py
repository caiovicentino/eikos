"""Converts the released dataset (Eikos Decisions) back to the trainer's format (training/train_dec.py).

It keeps the train and validation rows of the chosen model (field `used_in`), plus every held-out row; the trainer
separates the held-out rows itself with HOLDOUT_*. The conversion:
- rebuilds the criteria from the options;
- puts the target in teacher_probs (the trainer uses gold_probs or teacher_probs);
- keeps the original `source`, so the hash-based dev split comes out the same.
The snapshot gets an empty caps.json and excluir.json, because quotas and exclusions are already applied in the
dataset.
usage: python release_to_train.py <dataset_dir> <snapshot_dir> <eikos-4b|eikos-27b>
"""
import json
import os
import sys

SRC, OUT, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)


def criteria(qtype, opts):
    if qtype == "noul":
        d = {o["label"]: o["description"] for o in opts}
        return {"true": d["yes"], "false": d["no"]}
    if qtype == "score":
        return [o["description"] for o in opts]
    return {o["label"]: o["description"] for o in opts}


def to_train(r):
    opts = r["options"]
    q = {"type": r["question_type"], "instructions": r["instructions"], "criteria": criteria(r["question_type"], opts)}
    labels = ["no", "yes"] if r["question_type"] == "noul" else [o["label"] for o in opts]
    row = {"id": r["id"], "family": r["family"], "topic": r["topic"], "lang": r["lang"], "difficulty": r["difficulty"],
           "state": r["state"], "question": q, "labels": labels, "expected": r["expected"],
           "teacher_probs": {o["label"]: p for o, p in zip(opts, r["target_probs"])}, "agree": True}
    if r["source"] != "generated":  # generated items have no source in the training format ("gen" in the dev hash)
        row["source"] = r["source"]
    if r.get("rationale"):
        row["rationale"] = r["rationale"]
    return row


n = {}
for cfg in ("core", "long_context"):
    with open(f"{OUT}/{cfg}.jsonl", "w") as f:
        for split in ("train", "validation", "heldout"):
            p = f"{SRC}/data/{cfg}/{split}.jsonl"
            if not os.path.exists(p):
                continue
            for line in open(p):
                r = json.loads(line)
                if split == "heldout" or MODEL in r["used_in"]:
                    f.write(json.dumps(to_train(r), ensure_ascii=False) + "\n")
                    n[cfg] = n.get(cfg, 0) + 1
if MODEL == "eikos-4b":
    with open(f"{OUT}/views.jsonl.views", "w") as f:  # different extension: train_final.sh does not load it as DATA
        for line in open(f"{SRC}/data/views_pt_en/train.jsonl"):
            v = json.loads(line)
            view = {"state": v["state"], "instructions": v["instructions"],
                    "criteria": criteria(v["question_type"], v["options"]), "lang": v["lang"]}
            f.write(json.dumps({"id": v["id"], "view": view}, ensure_ascii=False) + "\n")
            n["views"] = n.get("views", 0) + 1
json.dump({}, open(f"{OUT}/caps.json", "w"))
json.dump([], open(f"{OUT}/excluir.json", "w"))
print("rows:", n)
