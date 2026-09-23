"""Temperature conditioned on observable signals of the question (calibration robust to distribution shift):
T(x) = exp(b + w·f(x)), f = [log(n_tok), log(n_opts), is_noul, is_score]. Fit by NLL on the internal dev set,
with extra weight on the hard items. Compares against a single global temperature. Saves calib.json.
usage: python calib_fit.py <dev_preds.jsonl> <out_calib.json> [hard_weight]
"""
import json
import math
import sys

import torch

import os
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if "n_tok" in r]
# Calibrate on the decisions that look real (generated items). The programmatic ones are easy and have an exact gold
# answer: if they are included, the temperature drops (<1) and the model becomes overconfident on the hard decisions
# (ECE on hard items 0.096 → 0.141 in arm A, Sep 23). CALIB_SRC=all restores the old behavior.
import os  # noqa: E402
_src = os.environ.get("CALIB_SRC", "gen")
if _src != "all":
    _sel = [r for r in rows if (r.get("src") or "gen") == _src]
    if len(_sel) >= 80:
        rows = _sel
print(f"calibration items: {len(rows)} (source={_src})")
W_HARD = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
FEATS = ["log_ntok", "log_nopts", "noul", "score"]


def feats(r):
    return [math.log(max(r["n_tok"], 1)) - 6.5, math.log(r["n_opts"]) - 1.2,
            float(r["qtype"] == "noul"), float(r["qtype"] == "score")]


X = torch.tensor([feats(r) for r in rows])
wt = torch.tensor([W_HARD if r.get("difficulty") == "hard" else 1.0 for r in rows])
gold = [max(range(len(r["target"])), key=lambda k: r["target"][k]) for r in rows]


def nll_and_ece(T, subset=None):
    tot, n, confs, corr = 0.0, 0, [], []
    for i, r in enumerate(rows):
        if subset and not subset(r):
            continue
        lg = torch.tensor(r["logits"]) / float(T[i])
        p = torch.softmax(lg, -1)
        tot -= math.log(max(float(p[gold[i]]), 1e-9))
        k = int(p.argmax())
        confs.append(float(p[k])); corr.append(int(k == gold[i])); n += 1
    ece = 0.0
    for b in range(10):
        idx = [j for j, c in enumerate(confs) if b / 10 < c <= (b + 1) / 10]
        if idx:
            ece += len(idx) / n * abs(sum(corr[j] for j in idx) / len(idx) - sum(confs[j] for j in idx) / len(idx))
    return tot / max(n, 1), ece, n


def fit(use_feats):
    b = torch.zeros(1, requires_grad=True)
    w = torch.zeros(len(FEATS), requires_grad=True)
    opt = torch.optim.Adam([b, w] if use_feats else [b], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        T = torch.exp(b + (X @ w if use_feats else torch.zeros(len(rows))))
        loss = 0.0
        for i, r in enumerate(rows):
            lp = torch.log_softmax(torch.tensor(r["logits"]) / T[i], -1)
            loss = loss - wt[i] * lp[gold[i]]
        (loss / wt.sum()).backward()
        opt.step()
    with torch.no_grad():
        T = torch.exp(b + (X @ w if use_feats else torch.zeros(len(rows))))
    return b.item(), w.tolist(), T


hard = lambda r: r.get("difficulty") == "hard"  # noqa: E731
one = torch.ones(len(rows))
for name, T in (("T=1", one),):
    print(f"{name:12s} overall nll/ece={nll_and_ece(T)[:2]} | hard={nll_and_ece(T, hard)[:2]}")
b0, _, T0 = fit(False)
print(f"{'single T':12s} T={math.exp(b0):.3f} overall={nll_and_ece(T0)[:2]} | hard={nll_and_ece(T0, hard)[:2]}")
b1, w1, T1 = fit(True)
print(f"{'T(x)':12s} b={b1:.3f} w={[round(x, 3) for x in w1]} overall={nll_and_ece(T1)[:2]} | hard={nll_and_ece(T1, hard)[:2]}")
print(f"  T(x) ranges from {float(T1.min()):.2f} to {float(T1.max()):.2f} (median {float(T1.median()):.2f})")
# Default = a single global temperature: on the human-authored out-of-distribution suites (Sep 22, run2 4B), T(x)
# worsened ECE from 0.055 to 0.110 (it extrapolates poorly on tasks with many options: MMLU-Pro, banking77,
# clinc150), while the single T gave 0.034 with the same JevBench leaderboard score. T(x) is kept in "tx" for
# reference only.
json.dump({"b": b0, "w": [0.0] * len(FEATS), "features": FEATS, "t_global": math.exp(b0), "mode": "global",
           "tx": {"b": b1, "w": w1}}, open(sys.argv[2], "w"), indent=1)
print("saved", sys.argv[2])
