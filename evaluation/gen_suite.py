"""Generalization suite with human labels (never used in training): real tasks converted to the
typed-decision format. Measures whether the model generalizes — and whether training does not erase
what the base model already knew (MMLU-Pro).

usage: python gen_suite.py suite.jsonl [n_per_task]   (a task that fails to download is skipped)
"""
from __future__ import annotations

import json
import random
import re
import sys

from datasets import load_dataset

OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rng = random.Random(2026)
items = []


def add(task, idx, state, qtype, instructions, criteria, labels, expected, lang="English"):
    items.append({"id": f"{task}-{idx}", "task": task, "family": task, "lang": lang, "state": state,
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
        print(f"{name}: SKIPPED ({type(e).__name__}: {str(e)[:120]})", flush=True)


NOUL_YN = {"true": "Yes.", "false": "No."}


def boolq():
    for i, r in enumerate(sample(load_dataset("google/boolq")["validation"], N)):
        add("boolq", i, f"Passage:\n{r['passage']}", "noul",
            f"Based only on the passage, is the answer to this question yes? Question: {r['question']}?",
            NOUL_YN, ["no", "yes"], "yes" if r["answer"] else "no")


def intents(name, repo, cfg, split, text_f, label_f):
    ds = load_dataset(repo, cfg)[split] if cfg else load_dataset(repo)[split]
    names = ds.features[label_f].names
    crit = {re.sub(r"[^a-z0-9_]", "_", n.lower()): n.replace("_", " ") for n in names}
    labs = list(crit.keys())
    for i, r in enumerate(sample(ds, N)):
        add(name, i, f"Customer message: {r[text_f]}", "choice",
            "Which intent does the customer message express?", crit, labs, labs[r[label_f]])


def gsm8k_judge():
    # answers: half correct (gold answer), half with a plausible error (gold ± a small number, doubled or halved)
    ds = load_dataset("openai/gsm8k", "main")["test"]
    for i, r in enumerate(sample(ds, N)):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        body = re.sub(r"<<[^>]*>>", "", r["answer"].split("####")[0]).strip()
        ok = rng.random() < 0.5
        if ok:
            ans = gold
        else:
            g = float(gold)
            cands = [g + rng.choice([1, 2, 5, 10]), g - rng.choice([1, 2, 5]), g * 2, g / 2 if g % 2 == 0 else g + 3]
            ans = str(int(rng.choice([c for c in cands if c != g and c >= 0] or [g + 1])))
        add("gsm8k_judge", i, f"Problem:\n{r['question']}\n\nProposed final answer: {ans}",
            "noul", "Is the proposed final answer correct?",
            {"true": "The proposed final answer is the correct answer to the problem.",
             "false": "The proposed final answer is not the correct answer to the problem."},
            ["no", "yes"], "yes" if ok else "no")


def mmlu_pro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
    for i, r in enumerate(sample(ds, N)):
        opts = r["options"]
        labs = [f"option_{chr(97 + k)}" for k in range(len(opts))]
        add("mmlu_pro", i, f"Question ({r['category']}):\n{r['question']}", "choice",
            "Which option correctly answers the question?", dict(zip(labs, [str(o) for o in opts])),
            labs, labs[r["answer_index"]])


def fields(r):
    return "\n".join(f"{k}: {v}" for k, v in r.items() if k not in ("answer", "index", "idx") and isinstance(v, str))


def legalbench():
    tasks = ["hearsay", "personal_jurisdiction", "ucc_v_common_law", "consumer_contracts_qa", "contract_qa",
             "diversity_3", "overruling", "privacy_policy_qa"]
    per = max(20, N // len(tasks))
    for t in tasks:
        ds = load_dataset("nguha/legalbench", t)["test"]
        ans_f = "answer"
        vals = sorted({str(x[ans_f]).strip() for x in ds})
        if set(v.lower() for v in vals) <= {"yes", "no"}:
            for i, r in enumerate(sample(ds, per)):
                add(f"legalbench_{t}", i, fields(r), "noul",
                    f"Legal task ({t.replace('_', ' ')}): is the correct answer yes?",
                    NOUL_YN, ["no", "yes"], str(r[ans_f]).strip().lower())
        else:
            labs = [re.sub(r"[^a-z0-9_]", "_", v.lower())[:40] for v in vals]
            crit = dict(zip(labs, vals))
            for i, r in enumerate(sample(ds, per)):
                add(f"legalbench_{t}", i, fields(r), "choice", f"Legal task ({t.replace('_', ' ')}): pick the correct label.",
                    crit, labs, labs[vals.index(str(r[ans_f]).strip())])


def reward_bench():
    ds = load_dataset("allenai/reward-bench")["filtered"]
    for i, r in enumerate(sample(ds, N)):
        a_first = rng.random() < 0.5
        ra, rb = (r["chosen"], r["rejected"]) if a_first else (r["rejected"], r["chosen"])
        add("rewardbench", i, f"User request:\n{r['prompt']}\n\nResponse A:\n{ra}\n\nResponse B:\n{rb}", "choice",
            "Which response better fulfils the user's request (helpful, correct, safe)?",
            {"response_a": "Response A is better.", "response_b": "Response B is better."},
            ["response_a", "response_b"], "response_a" if a_first else "response_b")


def assin2():
    ds = load_dataset("nilc-nlp/assin2")["test"]
    for i, r in enumerate(sample(ds, N)):
        add("assin2_pt", i, f"Frase A: {r['premise']}\nFrase B: {r['hypothesis']}", "noul",
            "A frase A implica a frase B (se A é verdadeira, B também é)?",
            {"true": "Sim, A implica B.", "false": "Não, A não implica B."}, ["no", "yes"],
            "yes" if r["entailment_judgment"] == 1 else "no", lang="Brazilian Portuguese")


def xnli_es():
    ds = load_dataset("facebook/xnli", "es")["test"]
    labs = ["implicacion", "neutral", "contradiccion"]
    crit = {"implicacion": "La premisa implica la hipótesis.", "neutral": "La premisa no implica ni contradice la hipótesis.",
            "contradiccion": "La premisa contradice la hipótesis."}
    for i, r in enumerate(sample(ds, N)):
        add("xnli_es", i, f"Premisa: {r['premise']}\nHipótesis: {r['hypothesis']}", "choice",
            "¿Qué relación hay entre la premisa y la hipótesis?", crit, labs, labs[r["label"]], lang="Spanish")


safe("boolq", boolq)
def banking77():
    ds = load_dataset("mteb/banking77")["test"]
    names = {}
    for r in ds:
        names[r["label"]] = r["label_text"]
    ordered = [names[k] for k in sorted(names)]
    crit = {re.sub(r"[^a-z0-9_]", "_", n.lower()): n.replace("_", " ") for n in ordered}
    labs = list(crit.keys())
    for i, r in enumerate(sample(ds, N)):
        add("banking77", i, f"Customer message: {r['text']}", "choice",
            "Which intent does the customer message express?", crit, labs, labs[sorted(names).index(r["label"])])


safe("banking77", banking77)
safe("clinc150", lambda: intents("clinc150", "clinc/clinc_oos", "plus", "test", "text", "intent"))
safe("gsm8k_judge", gsm8k_judge)
safe("mmlu_pro", mmlu_pro)
safe("legalbench", legalbench)
safe("rewardbench", reward_bench)
safe("assin2_pt", assin2)
safe("xnli_es", xnli_es)
with open(OUT, "w") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("TOTAL", len(items))
