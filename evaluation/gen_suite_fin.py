"""Financial generalization suite with human labels (never trained on) — to measure financial skill
and compare with Jev. Tasks: financial news sentiment, judging a numerical answer about a report
(FinQA), presence of a clause in a real contract (CUAD).
usage: python gen_suite_fin.py suite_fin.jsonl [n_per_task]
"""
from __future__ import annotations

import json
import random
import sys

from datasets import load_dataset

OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rng = random.Random(77)
items = []


def add(task, idx, state, qtype, instructions, criteria, labels, expected):
    items.append({"id": f"{task}-{idx}", "task": task, "family": task, "lang": "English", "state": state,
                  "question": {"type": qtype, "instructions": instructions, "criteria": criteria},
                  "labels": labels, "expected": expected})


def sample(ds, n):
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    return [ds[i] for i in idx[:n]]


def safe(name, fn):
    try:
        before = len(items)
        fn()
        print(f"{name}: {len(items) - before}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: SKIPPED ({type(e).__name__}: {str(e)[:150]})", flush=True)


def fin_sentiment():
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")["validation"]
    labs = ["bearish", "bullish", "neutral"]
    crit = {"bearish": "The news is negative for the asset/market (price likely pressured down).",
            "bullish": "The news is positive for the asset/market.",
            "neutral": "The news is neither clearly positive nor negative."}
    for i, r in enumerate(sample(ds, N)):
        add("fin_sentiment", i, f"Financial news post: {r['text']}", "choice",
            "What is the market sentiment of this financial news post?", crit, labs, labs[r["label"]])


def finqa_judge():
    ds = None
    for repo, split in (("ChanceFocus/flare-finqa", "test"), ("dreamerdeo/finqa", "test"), ("ibm/finqa", "test")):
        try:
            ds = load_dataset(repo)[split]
            break
        except Exception:  # noqa: BLE001
            continue
    if ds is None:
        raise RuntimeError("no FinQA source available")
    cols = ds.column_names
    for i, r in enumerate(sample(ds, N)):
        if "query" in cols:  # FLARE format: query already contains context + question
            ctx, ans = r["query"], str(r["answer"])
        else:
            ctx = "\n".join(str(r.get(k, "")) for k in ("pre_text", "table", "post_text", "question"))
            ans = str(r.get("answer", ""))
        try:
            g = float(ans.replace(",", "").replace("%", "").replace("$", ""))
        except ValueError:
            continue
        ok = rng.random() < 0.5
        raw = ans.strip().rstrip("%")
        d = min(len(raw.split(".")[1]) if "." in raw else 0, 6)
        if ok:
            val = g
        else:  # same precision as the correct answer: the format must not give away the label
            pert = rng.choice([1.1, 0.9, 1.25, 0.8, 2.0, 0.5, -1.0])
            val = g * pert
            if round(val, d) == round(g, d):
                continue
        prop = f"{val:.{d}f}"
        add("finqa_judge", i, f"{ctx}\n\nProposed answer: {prop}", "noul",
            "Is the proposed answer to the financial question correct (allow rounding to 2 decimals)?",
            {"true": "The proposed answer matches the correct result.",
             "false": "The proposed answer does not match the correct result."},
            ["no", "yes"], "yes" if ok else "no")


def cuad_clause():
    tasks = ["cuad_change_of_control", "cuad_termination_for_convenience", "cuad_non-compete",
             "cuad_revenue-profit_sharing", "cuad_most_favored_nation", "cuad_liquidated_damages",
             "cuad_minimum_commitment", "cuad_price_restrictions"]
    per = max(20, N // len(tasks))
    for t in tasks:
        try:
            ds = load_dataset("nguha/legalbench", t)["test"]
        except Exception as e:  # noqa: BLE001
            print(f"  {t}: skipped ({type(e).__name__})", flush=True)
            continue
        cat = t.replace("cuad_", "").replace("_", " ").replace("-", " ")
        for i, r in enumerate(sample(ds, per)):
            add("cuad_clause", f"{t}-{i}", f"Contract clause:\n{r['text']}", "noul",
                f"Is this contract clause a '{cat}' clause?",
                {"true": f"The clause is a '{cat}' clause.", "false": f"The clause is not a '{cat}' clause."},
                ["no", "yes"], str(r["answer"]).strip().lower())


safe("fin_sentiment", fin_sentiment)
safe("finqa_judge", finqa_judge)
safe("cuad_clause", cuad_clause)
with open(OUT, "w") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("TOTAL", len(items))
