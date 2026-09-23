"""Compares quantized variants against bf16 on the same items: accuracy, answer agreement, |Δconfidence|,
ECE, and the fixed-threshold policy (≥0.90). Approval criterion for releasing a variant: accuracy ≥ bf16 − 1 pp,
ECE ≤ bf16 + 0.01 and agreement ≥ 97%.
usage: python compare_quant.py <bf16_tag> <variant1_tag> [<variant2_tag> ...]   (files $EIKOS_RUNS/suite_<tag>.jsonl)"""
import json
import sys

import os
EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
R = EIKOS_RUNS


def load(tag):
    return {r["id"]: r for r in (json.loads(line) for line in open(f"{R}/suite_{tag}.jsonl"))}


def ece(rows):
    n, e = len(rows), 0.0
    for b in range(10):
        idx = [r for r in rows if b / 10 < r["conf"] <= (b + 1) / 10]
        if idx:
            e += len(idx) / n * abs(sum(bool(r["ok"]) for r in idx) / len(idx) - sum(r["conf"] for r in idx) / len(idx))
    return e


def policy(rows, th=0.9):
    sel = [r for r in rows if r["conf"] >= th]
    return len(sel) / len(rows), (1 - sum(bool(r["ok"]) for r in sel) / len(sel)) if sel else None


def pct(x):
    return f"{100 * x:4.1f}%" if x is not None else "   - "


base_tag, variants = sys.argv[1], sys.argv[2:]
base = load(base_tag)
print(f"{'variant':22s}{'n':>6s}{'acc':>8s}{'Δacc':>7s}{'agree':>10s}{'|Δconf|':>9s}{'ECE':>7s}{'≥0.90 decide/error':>20s}  approved?")
b_rows = list(base.values())
b_acc = sum(bool(r["ok"]) for r in b_rows) / len(b_rows)
b_ece = ece(b_rows)
cov, err = policy(b_rows)
print(f"{base_tag:22s}{len(b_rows):6d}{100 * b_acc:8.1f}{'':>7s}{'':>10s}{'':>9s}{b_ece:7.3f}{100 * cov:11.1f}/{pct(err)}")
for t in variants:
    v = load(t)
    ids = [i for i in base if i in v]
    vr = [v[i] for i in ids]
    br = [base[i] for i in ids]
    acc = sum(bool(r["ok"]) for r in vr) / len(vr)
    bacc = sum(bool(r["ok"]) for r in br) / len(br)
    agree = sum(v[i]["pred"] == base[i]["pred"] for i in ids) / len(ids)
    dconf = sum(abs(v[i]["conf"] - base[i]["conf"]) for i in ids) / len(ids)
    e = ece(vr)
    cov, err = policy(vr)
    ok = acc >= bacc - 0.01 and e <= ece(br) + 0.01 and agree >= 0.97
    print(f"{t:22s}{len(ids):6d}{100 * acc:8.1f}{100 * (acc - bacc):+7.1f}{100 * agree:9.1f}%{dconf:9.3f}{e:7.3f}"
          f"{100 * cov:11.1f}/{pct(err)}  {'YES' if ok else 'NO'}")
    # (complementary analysis, done after seeing the results) where do the answers change?
    for th in (0.7, 0.9):
        conf_ids = [i for i in ids if base[i]["conf"] >= th]
        if conf_ids:
            ag = sum(v[i]["pred"] == base[i]["pred"] for i in conf_ids) / len(conf_ids)
            print(f"{'':22s}   agreement when bf16 confidence is ≥{th}: {100 * ag:.1f}% (n={len(conf_ids)})")
    low = [i for i in ids if base[i]["conf"] < 0.7]
    flips = [i for i in ids if v[i]["pred"] != base[i]["pred"]]
    if flips:
        print(f"{'':22s}   of the {len(flips)} changed answers, {100 * sum(1 for i in flips if i in set(low)) / len(flips):.0f}% "
              f"were on items where bf16 confidence was < 0.7")
