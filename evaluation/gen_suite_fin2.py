"""Financial suite 2 (EVALUATION ONLY, never train on it):
- wcb_stance: monetary-policy stance in sentences from central banks around the world (WCB, gtfintechlab;
  CC BY-NC-SA → evaluation only), banks outside the US (Brazil, Mexico, Chile, Colombia, Peru, ECB, England, India,
  Japan, Canada).
- findver: is the statement supported by the excerpt of the 10-K/10-Q report? (FinDVer testmini, MIT).
usage: python gen_suite_fin2.py > suite_fin2.jsonl
"""
import collections
import json
import random

from datasets import load_dataset

import os
rng = random.Random(23)
out = []
BANKS = ["central_bank_of_brazil", "bank_of_mexico", "central_bank_of_chile", "bank_of_the_republic_colombia",
         "central_reserve_bank_of_peru", "european_central_bank", "bank_of_england", "reserve_bank_of_india",
         "bank_of_japan", "bank_of_canada"]
DEF = {"hawkish": "The sentence signals tighter monetary policy (e.g., higher rates, inflation-fighting concern).",
       "dovish": "The sentence signals looser monetary policy (e.g., lower rates, support for growth or employment).",
       "neutral": "The sentence is about monetary policy or the economy but signals neither tightening nor loosening.",
       "irrelevant": "The sentence is not about monetary policy stance at all."}
for b in BANKS:
    try:
        d = load_dataset(f"gtfintechlab/{b}", "5768")["test"]
    except Exception as e:  # noqa: BLE001
        print("failed", b, repr(e)[:100], file=__import__("sys").stderr)
        continue
    labs = sorted({str(x).lower() for x in d["stance_label"]})
    idx = list(range(len(d)))
    rng.shuffle(idx)
    for i in idx[:40]:
        r = d[i]
        gold = str(r["stance_label"]).lower()
        crit = {l: DEF.get(l, l) for l in labs}
        out.append({"id": f"wcb-{b}-{i}", "task": "wcb_stance",
                    "state": f"Central bank: {b.replace('_', ' ')}. Sentence from an official communication ({r['year']}):\n{r['sentences']}",
                    "question": {"type": "choice", "instructions": "What monetary-policy stance does this sentence express?",
                                 "criteria": crit},
                    "labels": labs, "expected": gold})

fd = json.load(open(os.path.join(os.environ.get("FINDVER_DIR", "FinDVer"), "data/testmini.json")))
pos = [r for r in fd if r["entailment_label"] is True]
neg = [r for r in fd if r["entailment_label"] is False]
rng.shuffle(pos)
rng.shuffle(neg)
for r in pos[:150] + neg[:150]:
    rep = json.load(open(os.path.join(os.environ.get("FINDVER_DIR", "FinDVer"), "financial_reports", r["report"])))
    paras = {c["id"]: c["context"] for c in rep["context"]}
    ctx = "\n\n".join(paras[i] for i in r["relevant_context"] if i in paras)
    if not ctx or len(ctx) > 24000:
        continue
    out.append({"id": f"findver-{r['example_id']}", "task": "findver",
                "state": f"Excerpts from a company report ({r.get('report', '')}):\n{ctx}\n\nStatement to verify:\n{r['statement']}",
                "question": {"type": "noul", "instructions": "Is the statement fully supported by the report excerpts?",
                             "criteria": {"true": "Every fact and number in the statement is supported by the excerpts.",
                                          "false": "At least one fact or number in the statement is contradicted or unsupported."}},
                "labels": ["no", "yes"], "expected": "yes" if r["entailment_label"] else "no"})
for r in out:
    print(json.dumps(r, ensure_ascii=False))
c = collections.Counter((r["task"], r["expected"]) for r in out)
print(dict(c), file=__import__("sys").stderr)
