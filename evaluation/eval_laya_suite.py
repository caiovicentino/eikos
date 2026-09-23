"""Runs Laya (official `laya` 0.3.7 package, Convai Innovations, Apache-2.0) on a suite_*.jsonl suite — evaluation only.
Modes:
  router : the one recommended by the model card (Router: English or multilingual checkpoint depending on the
           writing system of the state)
  typed  : typed-decisions checkpoint (fine-tuned on typed-decision workflows)
Choice questions with too many options for the model's head (head_max_len) → pre-selection (shortlist) by the package
itself (top-k by encoder embedding, k = 20 → 10 → 5), then the decision. Output in the same per-item format as our
evaluations, with the probabilities (for the confidence analysis) and whether the item fit entirely in Laya's context.
usage: python eval_laya_suite.py <suite.jsonl> <tag> <router|typed> [device]"""
import collections
import json
import sys
import time
from laya import Router  # noqa: E402
from laya.common import render_options, serialize_state  # noqa: E402
from laya.shortlist import embed_fn_from_agent, predict_shortlist  # noqa: E402

import os
EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
SUITE, TAG, MODE = sys.argv[1:4]
DEV = sys.argv[4] if len(sys.argv) > 4 else "cuda"
rows = [json.loads(line) for line in open(SUITE)]
router = Router(device=DEV, max_loaded=3)
router.preload(["typed-decisions"] if MODE == "typed" else ["english", "multilingual"])
KW = {"model": "typed-decisions"} if MODE == "typed" else {}


def to_laya(q):
    t = "noul" if q["type"] in ("noul", "boolean") else q["type"]
    lq = {"type": t, "instructions": q.get("instructions") or ""}
    if q.get("criteria") is not None:
        lq["criteria"] = q["criteria"]
    return lq


def fits(agent, state, lq):
    """Does the whole item fit in Laya's window? (state + instructions + options + special tokens <= max_len)"""
    tok = agent.tok
    n_state = len(tok(serialize_state(state), add_special_tokens=False)["input_ids"])
    internal = agent._to_internal(lq)
    n_head = len(tok(internal["ins"], add_special_tokens=False)["input_ids"]) + 4
    n_head += sum(len(tok(o, add_special_tokens=False)["input_ids"]) + 1 for o in render_options(internal))
    return n_state + n_head <= agent.cfg.get("max_len", 512), n_state + n_head


def predict(state, lq):
    name = "typed-decisions" if MODE == "typed" else router.route(state, {"d": lq}).model
    agent = router.load(name)
    try:
        return router.predict(state, {"d": lq}, **KW), name, agent, None
    except ValueError as e:
        if "head_max_len" not in str(e) or lq["type"] != "choice":
            raise
        for k in (20, 10, 5):
            try:
                res = predict_shortlist(agent, state, {"d": lq}, embed_fn_from_agent(agent), k=k)
                return res, name, agent, f"shortlist_k{k}"
            except ValueError as e2:
                if "head_max_len" not in str(e2):
                    raise
        raise


out, t0 = [], time.time()
for r in rows:
    lq = to_laya(r["question"])
    rec = {"id": r["id"], "task": r.get("task"), "gold": str(r["expected"]), "pred": None, "ok": False, "conf": None,
           "probs": None, "err": None, "route": None, "fits": None, "n_tok_full": None, "shortlist": None}
    try:
        res, name, agent, sl = predict(r["state"], lq)
        a = res["answers"]["d"]
        if lq["type"] == "noul":
            p_true = float(a["noul"])
            labs = [str(x) for x in r["labels"]]
            yes = "yes" if "yes" in labs else ("true" if "true" in labs else labs[-1])
            no = "no" if "no" in labs else ("false" if "false" in labs else labs[0])
            probs = {yes: p_true, no: 1.0 - p_true}
        else:
            probs = {str(k): float(v) for k, v in a["probabilities"].items()}
        pred = max(probs, key=probs.get)
        ok_fit, n_full = fits(agent, r["state"], lq)
        rec.update(pred=pred, ok=(pred == rec["gold"]), conf=probs[pred], probs=probs, route=name, fits=ok_fit,
                   n_tok_full=n_full, shortlist=sl)
    except Exception as e:  # noqa: BLE001  (a failure counts as an error, as on the official leaderboard)
        rec["err"] = f"{type(e).__name__}: {str(e)[:200]}"
    out.append(rec)
with open(f"{EIKOS_RUNS}/suite_{TAG}.jsonl", "w") as f:
    for rec in out:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
by = collections.defaultdict(list)
for rec in out:
    by[rec["task"]].append(rec)
summ = {t: {"acc": sum(x["ok"] for x in v) / len(v), "n": len(v), "err": sum(1 for x in v if x["err"]),
            "fits": sum(1 for x in v if x["fits"]) / len(v)} for t, v in by.items()}
summ["_macro"] = sum(v["acc"] for v in summ.values()) / len(by)
summ["_tag"] = TAG
summ["_seconds"] = round(time.time() - t0, 1)
json.dump(summ, open(f"{EIKOS_RUNS}/suite_summary_{TAG}.json", "w"), indent=1)
print(json.dumps({k: (round(v["acc"], 3) if isinstance(v, dict) else v) for k, v in summ.items()}))
