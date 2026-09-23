"""Financial answer verification with exact labels, built from the training split of FinQA (flare-finqa) and
TAT-QA (next-tat). "Is the proposed answer correct?" with varied realistic errors and reformatted correct
answers (rounding, %, $). The FinQA test split is used only for evaluation.
usage: python prog_finjudge.py out.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys

from datasets import load_dataset

OUT = sys.argv[1]
rng = random.Random(31)
INSTR = ["Is the proposed answer to the financial question correct (allow rounding to 2 decimals)?",
         "Does the proposed answer correctly answer the question, based on the report above?",
         "Check the proposed answer against the figures in the document: is it correct?",
         "Verify the proposed figure: does it match the correct answer (rounding differences are fine)?"]
CRIT = {"true": "The proposed answer matches the correct result (up to rounding).",
        "false": "The proposed answer does not match the correct result."}


def fmt_num(x, style):
    if style == "plain":
        return f"{x:g}" if abs(x) >= 1e-3 else f"{x:.6f}"
    if style == "r2":
        return f"{x:.2f}"
    if style == "comma":
        return f"{x:,.2f}"
    if style == "pct":
        return f"{x:.2f}%"
    if style == "usd":
        return f"${x:,.2f}"
    return str(x)


def wrong_value(g):
    kind = rng.choice(["mul", "mul", "add", "sign", "scale", "swap", "scale"])
    if kind == "mul":
        return g * rng.choice([1.1, 0.9, 1.25, 0.8, 2.0, 0.5, 1.5, 0.75, 1.05, 0.95])
    if kind == "add":
        return g + rng.choice([-1, 1]) * max(abs(g) * rng.choice([0.02, 0.04, 0.07]), 0.5)
    if kind == "sign":
        return -g if g != 0 else 1.0
    if kind == "scale":
        return g * rng.choice([10, 0.1, 100, 0.01])
    s = f"{abs(g):.2f}".replace(".", "")
    if len(s) >= 3:  # swap two adjacent digits
        i = rng.randrange(len(s) - 1)
        t = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        v = float(t[:-2] + "." + t[-2:]) * (1 if g >= 0 else -1)
        if abs(v - g) > 1e-6:
            return v
    return g * 1.2


def decimals_of(raw):
    raw = str(raw).strip().rstrip("%")
    return len(raw.split(".")[1]) if "." in raw else 0


def fmt_same(x, style, d):
    """Same format for correct and wrong answers (the style is drawn before knowing whether the answer is correct)."""
    if style == "source":           # natural precision of the dataset's answer
        return f"{x:.{d}f}"
    if style == "r2":
        return f"{x:.2f}"
    if style == "comma":
        return f"{x:,.2f}"
    return f"{x:.2f}%"              # pct


def items_from(task, idx, ctx, gold_raw, gold, src):
    out = []
    ok = rng.random() < 0.5
    d = min(decimals_of(gold_raw), 6)
    style = rng.choices(["source", "r2", "comma", "pct"], weights=[50, 25, 15, 10])[0]
    if style == "pct" and ("%" not in str(gold_raw) and abs(gold) > 100):
        style = "r2"
    if ok:
        prop = fmt_same(gold, style, d)
    else:
        w = wrong_value(gold)
        if abs(round(w, 2) - round(gold, 2)) < 0.006:
            return out
        prop = fmt_same(w, style, d)
        if prop == fmt_same(gold, style, d):  # at this precision the wrong value would print exactly like the gold
            return out
    out.append({"id": f"prog-finjudge-{src}-{idx}", "family": "judge", "topic": "finance",
                "state": f"{ctx}\n\nProposed answer: {prop}",
                "question": {"type": "noul", "instructions": rng.choice(INSTR), "criteria": CRIT},
                "labels": ["no", "yes"], "expected": "yes" if ok else "no",
                "teacher_probs": {"yes": 0.97 if ok else 0.03, "no": 0.03 if ok else 0.97},
                "teacher_top": "yes" if ok else "no", "agree": True, "difficulty": "hard",
                "lang": "English", "source": f"prog_finjudge:{src}"})
    return out


def to_float(s):
    try:
        return float(str(s).replace(",", "").replace("%", "").replace("$", "").strip())
    except ValueError:
        return None


recs = []
ds = load_dataset("ChanceFocus/flare-finqa")["train"]
for i, r in enumerate(ds):
    g = to_float(r["answer"])
    if g is not None:
        recs += items_from("finqa", i, r["query"], r["answer"], g, "finqa")

tat = load_dataset("next-tat/TAT-QA")["train"]
for di, doc in enumerate(tat):
    table = doc["table"]["table"] if isinstance(doc["table"], dict) else doc["table"]
    ttxt = "\n".join(" | ".join(str(c) for c in row) for row in table)
    ptxt = "\n".join(p["text"] for p in sorted(doc["paragraphs"], key=lambda p: p["order"]))
    for q in doc["questions"]:
        if q.get("answer_type") != "arithmetic":
            continue
        g = to_float(q["answer"])
        if g is None:
            continue
        scale = q.get("scale") or ""
        ctx = f"{ptxt}\n\nTable:\n{ttxt}\n\nQuestion: {q['question']}" + (f" (answer in {scale})" if scale else "")
        recs += items_from("tatqa", f"{di}-{q['order']}", ctx, q["answer"], g, "tatqa")

rng.shuffle(recs)
with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("items:", len(recs), "| yes:", sum(r["expected"] == "yes" for r in recs),
      "| finqa:", sum(r["source"].endswith("finqa") for r in recs), "| tatqa:", sum(r["source"].endswith("tatqa") for r in recs))
