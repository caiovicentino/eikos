"""Evaluates an MLX build (8-bit, 4-bit...) on a suite_*.jsonl evaluation suite, in the same per-item format as
eval_vllm_suite (id, task, pred, gold, ok, conf), to compare quantizations item by item against bf16.
Runs on a Mac (Metal) or on a Linux GPU server (in an MLX environment with CUDA).
Up to 588 options are read in one pass (labels A..Z, AA, AB, ...), as in eval_vllm_suite; more via the tournament
(decision_core.tournament). TOURNAMENT_26=1 reproduces v1.0/v1.1 (tournament above 26 options).
usage: [MLX_ITEMS=all|big|small] python eval_mlx_suite.py <mlx_dir> <suite.jsonl> <tag>
       (big = only the questions with more than 26 options; small = only the others; default all)"""
import collections
import json
import sys
import time

from mlx_decide import MLXDecider

import mlx.core as mx  # noqa: E402

import os
EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
M, SUITE, TAG = sys.argv[1:4]
mx.set_cache_limit(4 * 1024 ** 3)  # without a limit, the MLX (CUDA) cache grows until it fills the whole GPU
d = MLXDecider(M)
import decision_core  # noqa: E402  (after MLXDecider sets PROMPT_STYLE and the model's one-pass limit)
from decision_core import options_of, tournament  # noqa: E402

ITEMS = os.environ.get("MLX_ITEMS", "all")
assert ITEMS in ("all", "big", "small"), ITEMS
OLD = dict(chunk=20, keep=2, limit=26) if os.environ.get("TOURNAMENT_26") == "1" else {}  # v1.0/v1.1 readout

BUCKET = 128


def dist_bucketed(state, q, opts):
    """Same computation as MLXDecider.dist, with right padding up to a multiple of 128 tokens: real positions never
    see the padding (causal attention and the recurrence only look back), so the readout at the last real position is
    identical. This lets MLX-CUDA compile only a few kernels (one per length bucket) instead of one per length."""
    ids = d._ids(state, q, opts)
    n = len(ids)
    pad = d.tok.pad_token_id if d.tok.pad_token_id is not None else d.tok.eos_token_id
    L = ((n + BUCKET - 1) // BUCKET) * BUCKET
    h = d.inner(mx.array([ids + [pad] * (L - n)]), d.lm.make_cache())[:, n - 1, :]
    lg = d._letter_logits(h)[0]
    mx.eval(lg)
    return d._probs(lg, n, opts, q), n


rows = [json.loads(line) for line in open(SUITE)]
per = collections.defaultdict(list)
t0 = time.time()
with open(f"{EIKOS_RUNS}/suite_{TAG}.jsonl", "w") as fo:
    for r in rows:
        o = options_of(r["question"], list(r["labels"]))
        big = len(o) > 26
        if (ITEMS == "big" and not big) or (ITEMS == "small" and big):
            continue
        if big:  # one pass up to 588 options, tournament beyond (or above 26 with TOURNAMENT_26=1)
            p, n = tournament(lambda oo, r=r: dist_bucketed(r["state"], r["question"], oo), o, **OLD)
        else:
            p, n = dist_bucketed(r["state"], r["question"], o)
        if n > 4000:
            mx.clear_cache()
        pred = max(p, key=p.get)
        ok = pred == str(r["expected"])
        per[r["task"]].append(ok)
        fo.write(json.dumps({"id": r["id"], "task": r["task"], "pred": pred, "gold": r["expected"], "ok": ok,
                             "conf": p[pred], "n_tok": n,
                             "tournament": len(o) > (OLD.get("limit") or decision_core.MAX_ONE_PASS)}) + "\n")
summ = {t: {"acc": round(sum(v) / len(v), 4), "n": len(v)} for t, v in sorted(per.items())}
summ["_macro"] = round(sum(x["acc"] for x in summ.values()) / len(summ), 4)
summ["_tag"] = TAG
summ["_seconds"] = round(time.time() - t0, 1)
json.dump(summ, open(f"{EIKOS_RUNS}/suite_summary_{TAG}.json", "w"), indent=1)
print(json.dumps({"_macro": summ["_macro"], "_tag": TAG, "_seconds": summ["_seconds"]}), flush=True)
