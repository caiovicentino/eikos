"""Builds the public Eikos Decisions dataset from the snapshots of the final training runs. It is included for
transparency: it needs our internal snapshots and run logs, which are not released.

It re-applies EXACTLY the selection of training/train_dec.py:
- filters: teacher agreement (`agree`), excluir.json, per-source quotas (per file);
- target: gold_probs or teacher_probs, normalized;
- the held-out family / topic / language and the hash-based dev split.

It then checks the result against the dev_ids.json files and the counts in the final training logs. The 4B used
snap_final1. The 27B used snap_final1_27b, which differs only in the long items (≤12k tokens).

Output:
- whitelisted fields, with options in the canonical training order and probabilities aligned to the options;
- `upstream` and `item_writer` on every row, for license attribution;
- PT↔EN views, only for the 4B training items (the 27B did not use views).

Items outside the training data are not included: teacher disagreement, contamination, FinQA, reserved trade rules,
and rows over quota.

Release procedure:
  1. python build_release_data.py full/                          (no drop list)
  2. python check_overlap.py full/ overlap.json                 (every evaluation suite)
  3. python pii_scan.py full/ drop_ids.json overlap.json known_label_errors.json
                                                                (credentials, personal data, overlap, label errors)
  4. python build_release_data.py release/ drop_ids.json
usage: python build_release_data.py <out_dir> [drop_ids.json]
"""
import collections
import hashlib
import json
import os
import sys

J = os.environ.get("EIKOS_HOME", ".")
SNAP = {"eikos-4b": f"{J}/snap_final1", "eikos-27b": f"{J}/snap_final1_27b"}
DEV_IDS = {"eikos-4b": f"{J}/ckpt/final_4b/dev_ids.json", "eikos-27b": f"{J}/ckpt/final_27b/dev_ids.json"}
EXPECT = {"eikos-4b": (23168, 1296), "eikos-27b": (22216, 1266)}  # train/dev counts from the final training logs
VIEWS = [f"{J}/data/views_pt.jsonl", f"{J}/data/views_pt_b.jsonl"]  # only the 4B (recipes B and E) used views
VIEWS_EXPECT = 1637  # "training items with a view" in the final_4b log
HOLD = {"family": {"tradeoff"}, "topic": {"healthcare administration"}, "lang": {"Spanish"}}
HOLD_TRADE = {"prog_trade:t_incoterm", "prog_trade:t_lc_presentation", "prog_trade:t_vat"}
DEV_FRAC, DEV_FRAC_GEN = 0.04, 0.08
OUT = sys.argv[1]
DROP = set(json.load(open(sys.argv[2]))) if len(sys.argv) > 2 else set()
WRITER = {"glm-5.3-flash": "GLM-5.3-Flash", "q38": "Qwen3.8-27B"}
UPSTREAM = {"prog_finjudge:tatqa": "TAT-QA (CC BY 4.0)", "prog_judge:gsm8k_model": "GSM8K train (MIT)",
            "prog_judge:gsm8k_ref_answer_only": "GSM8K train (MIT)", "real_finentity:odc-by": "FinEntity (ODC-BY 1.0)"}


def select(snap):
    """Same logic as load_rows() + the held-out rule + is_dev() in train_dec.py."""
    ex = set(json.load(open(f"{snap}/excluir.json")))
    caps = json.load(open(f"{snap}/caps.json"))
    sel, why = [], collections.Counter()
    for fn in sorted(f for f in os.listdir(snap) if f.endswith(".jsonl")):
        n_src = {}
        for line in open(f"{snap}/{fn}"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                why["invalid_json"] += 1
                continue
            if not r.get("agree", True):
                why["teacher_disagreed"] += 1
                continue
            if r.get("id") in ex:
                rid, s = str(r.get("id")), str(r.get("source"))
                why["excluded:finqa" if "-finqa-" in rid else "excluded:reserved_trade_rules" if s in HOLD_TRADE
                    else "excluded:jevbench_contamination"] += 1
                continue
            src = (r.get("source") or "gen").split(":")[0]
            if src in caps and n_src.get(src, 0) >= caps[src]:
                why["over_quota"] += 1
                continue
            n_src[src] = n_src.get(src, 0) + 1
            tgt = r.get("gold_probs") or r.get("teacher_probs")
            if not tgt:
                why["no_target"] += 1
                continue
            tgt = {str(k): max(0.0, float(v)) for k, v in tgt.items()}
            s = sum(tgt.values())
            if s <= 0:
                why["no_target"] += 1
                continue
            tgt = {k: v / s for k, v in tgt.items()}
            if r.get("family") in HOLD["family"] or r.get("topic") in HOLD["topic"] or r.get("lang") in HOLD["lang"]:
                split = "heldout"
            else:
                h = int(hashlib.sha1(str(r.get("id")).encode()).hexdigest()[:8], 16)
                split = "validation" if (h % 10000) < (DEV_FRAC_GEN if src == "gen" else DEV_FRAC) * 10000 else "train"
            sel.append({"split": split, "file": fn, "row": r, "src": src, "tgt": tgt})
    return sel, why


def canonical_options(q, labels, crit=None):
    """(label, description) in the canonical training order (same as canonical_opts in train_dec.py)."""
    crit = (q.get("criteria") if crit is None else crit) or {}
    if q["type"] == "noul":
        return [("yes", str(crit.get("true", "yes"))), ("no", str(crit.get("false", "no")))]
    if q["type"] == "score":
        if isinstance(crit, dict):
            return [(str(l), str(crit.get(str(l), l))) for l in labels]
        return [(str(i), str(c)) for i, c in enumerate(crit)]
    return [(str(l), str(crit.get(l, l)) if isinstance(crit, dict) else str(l)) for l in labels]


sels, whys, dups = {}, {}, collections.Counter()
for m, snap in SNAP.items():
    lst, whys[m] = select(snap)
    tr = sum(1 for v in lst if v["split"] == "train")
    dv = sorted(str(v["row"]["id"]) for v in lst if v["split"] == "validation")
    assert (tr, len(dv)) == EXPECT[m], f"{m}: train/dev {tr}/{len(dv)} != log {EXPECT[m]}"
    assert dv == sorted(json.load(open(DEV_IDS[m]))), f"{m}: dev split differs from dev_ids.json"
    print(f"[{m}] matches training: train={tr} dev={len(dv)} heldout="
          f"{sum(1 for v in lst if v['split'] == 'heldout')} | not used: {dict(whys[m])}")
    # FinEntity repeats some news (same text + entity = same id): training saw both copies; the release keeps only
    # the first one (if the labels disagree, the row is left out)
    sels[m] = {}
    for v in lst:
        i = v["row"]["id"]
        if i in sels[m]:
            dups[m] += 1
            if str(sels[m][i]["row"]["expected"]) != str(v["row"]["expected"]):
                sels[m][i]["conflict"] = True
            continue
        sels[m][i] = v
print("duplicate ids (extra copies left out):", dict(dups))

# writer of generated items: the `generator` field; the 69 oldest rows of labeled.jsonl (and the long items built
# from them) lack it, and the gen_pipeline.py default applies (GLM-5.3-Flash)
base_writer = {}
for fn in ("labeled.jsonl", "labeled_q38.jsonl", "labeled_q38b.jsonl"):
    for line in open(f"{SNAP['eikos-4b']}/{fn}"):
        r = json.loads(line)
        base_writer[r["id"]] = WRITER.get(r.get("generator"), "GLM-5.3-Flash" if fn == "labeled.jsonl" else None)

ids = sorted(set(sels["eikos-4b"]) | set(sels["eikos-27b"]))
out_rows = collections.defaultdict(list)
stats = collections.defaultdict(collections.Counter)
drop_hits = 0
for i in ids:
    e = sels["eikos-4b"].get(i) or sels["eikos-27b"][i]
    for m in sels:
        if i in sels[m]:
            assert sels[m][i]["split"] == e["split"], f"split differs between models: {i}"
    if i in DROP:
        drop_hits += 1
        continue
    if e.get("conflict"):
        stats["conflicts"]["divergent_labels_removed"] += 1
        continue
    r, fn, split = e["row"], e["file"], e["split"]
    q = r["question"]
    opts = canonical_options(q, r["labels"])
    assert {o for o, _ in opts} == {str(l) for l in r["labels"]}, f"options != labels: {i}"
    tgt = [e["tgt"].get(o, 0.0) for o, _ in opts]
    assert abs(sum(tgt) - 1) < 1e-6, f"target mass outside the options: {i}"
    generated = fn.startswith("labeled") or fn == "long_items.jsonl"
    if generated:
        base = i.rsplit("-long", 1)[0] if fn == "long_items.jsonl" else i
        writer = WRITER.get(r.get("generator")) or base_writer.get(base) or "unknown"
        label_source = "writer_exact_probs" if r.get("gold_probs") else "teacher"
        tp = {str(k): max(0.0, float(v)) for k, v in (r.get("teacher_probs") or {}).items()}
        s = sum(tp.values())
        teacher = [round(tp.get(o, 0.0) / s, 6) for o, _ in opts] if s > 0 else None
    else:
        writer = "FinEntity (human annotators)" if fn == "real_finentity.jsonl" else "program"
        label_source = {"prog_prob.jsonl": "exact_probability", "real_finentity.jsonl": "human"}.get(fn, "program")
        teacher = None
    used = [m for m in sels if i in sels[m] and split != "heldout"]
    rec = {
        "id": i, "split": split, "family": r.get("family"), "topic": r.get("topic"), "lang": r.get("lang"),
        "difficulty": r.get("difficulty"), "format": r.get("fmt"),
        "source": r.get("source") or "generated", "upstream": UPSTREAM.get(r.get("source")),
        "item_writer": writer, "label_source": label_source,
        "state": r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False),
        "question_type": q["type"], "instructions": q.get("instructions") or "",
        "options": [{"label": o, "description": d} for o, d in opts],
        "expected": str(r["expected"]),
        "target_probs": [round(x, 6) for x in tgt], "teacher_probs": teacher,
        "rationale": r.get("rationale") or None, "used_in": used,
    }
    assert rec["expected"] in {o for o, _ in opts}, f"expected label not among the options: {i}"
    cfg = "long_context" if fn == "long_items.jsonl" else "core"
    if cfg == "long_context":
        rec.update(long_len=r.get("long_len"), long_n_docs=r.get("long_n_docs"), long_answer_doc=r.get("long_doc"))
    out_rows[(cfg, split)].append(rec)
    for k in ("family", "lang", "label_source", "item_writer", "question_type", "upstream", "difficulty"):
        stats[f"{cfg}:{split}:{k}"][str(rec[k])] += 1
    stats[f"{cfg}:{split}:used_in"][",".join(used) or "none"] += 1

# PT<->EN views (the last view per id wins, as in the trainer's VIEW_MAP), only for 4B training items
view_map = {}
for p in VIEWS:
    for line in open(p):
        try:
            v = json.loads(line)
            view_map[v["id"]] = v["view"]
        except Exception:  # noqa: BLE001
            pass
train4 = {k for k, v in sels["eikos-4b"].items() if v["split"] == "train"}
n_views_train = sum(1 for k in view_map if k in train4)
assert n_views_train == VIEWS_EXPECT, f"training views {n_views_train} != log {VIEWS_EXPECT}"
for k in sorted(view_map):
    if k not in train4 or k in DROP:
        continue
    r, v = sels["eikos-4b"][k]["row"], view_map[k]
    opts = canonical_options(r["question"], r["labels"], crit=v.get("criteria"))
    out_rows[("views_pt_en", "train")].append({
        "id": k, "lang": v.get("lang"), "original_lang": r.get("lang"),
        "state": v["state"] if isinstance(v["state"], str) else json.dumps(v["state"], ensure_ascii=False),
        "question_type": r["question"]["type"], "instructions": v.get("instructions") or "",
        "options": [{"label": o, "description": d} for o, d in opts], "expected": str(r["expected"]),
        "target_probs": [round(sels["eikos-4b"][k]["tgt"].get(o, 0.0), 6) for o, _ in opts],
        "writer": "Qwen3.8-27B (translation)", "used_in": ["eikos-4b"]})

summary = {}
for (cfg, split), rows in sorted(out_rows.items()):
    d = f"{OUT}/data/{cfg}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{split}.jsonl", "w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary[f"{cfg}/{split}"] = len(rows)
json.dump({"files": summary, "dropped_rows": drop_hits, "duplicate_ids": dict(dups),
           "not_used_in_training": {m: dict(w) for m, w in whys.items()},
           "distributions": {k: dict(c.most_common()) for k, c in sorted(stats.items())}},
          open(f"{OUT}/stats.json", "w"), indent=1, ensure_ascii=False)
print("files:", summary, "| dropped rows (credentials, personal data, evaluation overlap):", drop_hits)
