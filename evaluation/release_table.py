"""Final table of the full evaluation of the release versions (vLLM, all 7 suites in full): accuracy per suite in the
same format as the leaderboard, ECE and the ≥0.90 threshold policy (6 suites, without JevBench), and the delta of each
quantization against the bf16 model of the same size. usage: python release_table.py [prefix]   (default: rel)"""
import collections
import json
import os

EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, suites and checkpoints
EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
R = EIKOS_RUNS
HOLD = {"trade_incoterm", "trade_lc_presentation", "trade_vat"}
PREFIX = __import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "rel"
VARS = [("27b", "bf16"), ("27b", "fp8"), ("27b", "int4"), ("4b", "bf16"), ("4b", "fp8"), ("4b", "int4")]


def load(tag):
    p = f"{R}/suite_{tag}.jsonl"
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else None


def acc(rows, pred=lambda r: True):
    sel = [bool(r["ok"]) for r in rows if pred(r)]
    return sum(sel) / len(sel) if sel else None


def macro(rows, key=lambda r: r["task"]):
    by = collections.defaultdict(list)
    for r in rows:
        by[key(r)].append(bool(r["ok"]))
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def bal(rows, task):
    by = collections.defaultdict(list)
    for r in rows:
        if r["task"] == task:
            by[str(r["gold"])].append(bool(r["ok"]))
    return sum(sum(v) / len(v) for v in by.values()) / len(by) if by else None


def ece(rows):
    n, e = len(rows), 0.0
    for b in range(10):
        idx = [r for r in rows if b / 10 < r["conf"] <= (b + 1) / 10]
        if idx:
            e += len(idx) / n * abs(sum(bool(r["ok"]) for r in idx) / len(idx) - sum(r["conf"] for r in idx) / len(idx))
    return e


out = {}
for m, v in VARS:
    t = f"{PREFIX}_{m}_{v}"
    d = {b: load(f"{t}_{b}") for b in ("jb", "db", "gen", "fin", "fin2", "trade", "rules")}
    if any(x is None for x in d.values()):
        out[t] = None
        continue
    pooled = [r for b in ("db", "gen", "fin", "fin2", "trade", "rules") for r in d[b]]
    sel = [r for r in pooled if r["conf"] >= 0.9]
    out[t] = {
        "jb_easy": acc(d["jb"], lambda r: r["task"] == "jb_easy"), "jb_orig": acc(d["jb"], lambda r: r["task"] == "jb_original"),
        "jb_hard": acc(d["jb"], lambda r: r["task"] == "jb_hard"),
        "db_med": acc(d["db"], lambda r: r["task"] == "db_medium"), "db_hard": acc(d["db"], lambda r: r["task"] == "db_hard"),
        "general": macro(d["gen"], key=lambda r: "legalbench" if r["task"].startswith("legalbench") else r["task"]),
        "fin": macro(d["fin"]), "wcb_bal": bal(d["fin2"], "wcb_stance"),
        "findver": acc(d["fin2"], lambda r: r["task"] == "findver"),
        "trade_seen": acc(d["trade"], lambda r: r["task"] not in HOLD), "trade_unseen": acc(d["trade"], lambda r: r["task"] in HOLD),
        "rules": acc(d["rules"], lambda r: r["task"] == "rules_seen"),
        "rules_new_domain": acc(d["rules"], lambda r: r["task"] == "rules_holdout"),
        "rulebooks": acc(d["rules"], lambda r: r["task"] == "rules_book"),
        "ece": ece(pooled), "decide_090": len(sel) / len(pooled), "error_090": 1 - sum(bool(r["ok"]) for r in sel) / len(sel),
        "n": len(pooled) + len(d["jb"]), "tournament": sum(1 for b in d.values() for r in b if r.get("tournament")),
    }
json.dump(out, open(f"{EIKOS_HOME}/release_table_{PREFIX}.json", "w"), indent=1)
cols = [("jb_easy", "JB easy"), ("jb_orig", "JB orig"), ("jb_hard", "JB hard"), ("db_med", "DB med"), ("db_hard", "DB hard"),
        ("general", "General"), ("fin", "Fin"), ("wcb_bal", "WCB bal"), ("findver", "FinDVer"), ("trade_seen", "Trade seen"),
        ("trade_unseen", "Trade unseen"), ("rules", "Rules"), ("rules_new_domain", "Rules new"), ("rulebooks", "Rulebooks"),
        ("ece", "ECE"), ("decide_090", "≥0.90 dec"), ("error_090", "≥0.90 err")]
print(f"{'build':16s}" + "".join(f"{c[1]:>11s}" for c in cols) + f"{'items':>8s}")
for t, v in out.items():
    if v is None:
        print(f"{t:16s} (incomplete)")
        continue
    cells = [f"{v[k]:.3f}" if k == "ece" else f"{100 * v[k]:.1f}" for k, _ in cols]
    print(f"{t.replace(PREFIX + '_', ''):16s}" + "".join(f"{c:>11s}" for c in cells) + f"{v['n']:8d}")
print("\nDelta vs. the bf16 model of the same size (percentage points; ECE in absolute terms):")
for m in ("27b", "4b"):
    b = out.get(f"{PREFIX}_{m}_bf16")
    for q in ("fp8", "int4"):
        v = out.get(f"{PREFIX}_{m}_{q}")
        if b and v:
            print(f"  {m} {q:5s}" + "".join(f"  {name}: {(v[k] - b[k]) if k == 'ece' else 100 * (v[k] - b[k]):+.{3 if k == 'ece' else 1}f}"
                                         for k, name in cols if k in ("jb_hard", "db_hard", "general", "fin", "trade_unseen", "ece", "error_090")))
