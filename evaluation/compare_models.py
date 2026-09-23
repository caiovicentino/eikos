"""Tests 1 and 2 — Eikos × Jev × Laya on the same items.
Test 1: accuracy per suite (public JevBench from the official leaderboard for Jev/Laya + our Laya run as validation).
Test 2: usable confidence — risk-coverage curve, coverage at 95%/99% accuracy, AURC, ECE, and the policy with a
threshold fixed in advance (decide on its own if confidence >= 0.90/0.95: how many cases and how much actual error).
Laya is also "recalibrated" as its model card recommends (one temperature per question type × number of options),
with 2-fold cross-fitting (fit on one half, measure on the other) — generous to the competitor.
usage: python compare_models.py  (reads $EIKOS_RUNS; writes compare_models.json)"""
import collections
import hashlib
import json
import math
import os

EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, suites and checkpoints
EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
R = EIKOS_RUNS
J = EIKOS_HOME


def rd(name):
    p = os.path.join(R, name)
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else None


def truthy(x):
    return x is True or x == "True" or x == "true" or x == 1


def norm(rows, task_key="task"):
    """-> {id: (task, ok, conf, probs)}"""
    out = {}
    for r in rows or []:
        iid = r.get("id") or r.get("task_id")
        # "correct" = correct answer (run_jb, eval_db); in those files "ok" only means "valid answer".
        # In the suites, "ok" = correct answer.
        ok = truthy(r.get("correct")) if "correct" in r else truthy(r.get("ok"))
        conf = r.get("conf")
        probs = r.get("probs")
        if isinstance(probs, str):
            probs = None
        if conf is None and isinstance(probs, dict) and probs:
            conf = max(probs.values())
        task = r.get(task_key) or r.get("cfg") or r.get("family")
        if r.get("err") not in (None, "None", "") and conf is None:
            ok, conf = False, None
        out[iid] = (task, ok, None if conf is None else float(conf), probs)
    return out


SUITES = {"gen": "suite.jsonl", "fin": "suite_fin.jsonl", "fin2": "suite_fin2.jsonl", "trade": "suite_trade.jsonl",
          "rules": "suite_rules.jsonl", "db": "suite_db.jsonl", "jb": "suite_jb_public.jsonl"}
META = {}  # id -> (qtype, n_opts, task)
for b, f in SUITES.items():
    for line in open(os.path.join(J, f)):
        r = json.loads(line)
        t = r.get("task")
        if b == "gen" and str(t).startswith("legalbench"):
            t = "legalbench"  # general battery: 9 families (8 LegalBench subtasks count as one, as on the leaderboard)
        META[r["id"]] = (r["question"]["type"], len(r["labels"]), t)

MODELS = {
    "Eikos-4B": {"gen": "suite_final_4b_soup.jsonl", "fin": "suite_fin_final_4b_soup.jsonl",
                 "fin2": "suite_fin_final_4b_soup_fin2.jsonl", "trade": "suite_final_4b_soup_trade.jsonl",
                 "rules": "suite_final_4b_soup_rules.jsonl", "db": "db_final_4b_soup.jsonl",
                 "jb": ["final_4b_soup_easy.jsonl", "final_4b_soup_original.jsonl", "final_4b_soup_hard.jsonl"]},
    "Eikos-27B": {"gen": "suite_final_27b.jsonl", "fin": "suite_fin_final_27b.jsonl",
                  "fin2": "suite_fin_final_27b_fin2.jsonl", "trade": "suite_final_27b_trade.jsonl",
                  "rules": "suite_final_27b_rules.jsonl", "db": "db_final_27b.jsonl",
                  "jb": ["final_27b_easy.jsonl", "final_27b_original.jsonl", "final_27b_hard.jsonl"]},
    "Jev": {"gen": "jev_gen.jsonl", "fin": "jev_fin2.jsonl", "fin2": "jev_fin2x.jsonl", "trade": "jev_tradex.jsonl",
            "rules": ["jev_rulesx.jsonl", "jev_bookx.jsonl"], "db": "jev_db.jsonl"},
    "Laya": {b: f"suite_laya_router_{b}.jsonl" for b in SUITES},
    "Laya-typed": {b: f"suite_laya_typed_{b}.jsonl" for b in SUITES},
}
DATA = {m: {} for m in MODELS}
for m, spec in MODELS.items():
    for b, f in spec.items():
        rows = []
        for ff in (f if isinstance(f, list) else [f]):
            rows += rd(ff) or []
        if rows:
            d = norm(rows)
            for iid, v in d.items():  # canonical task = the suite's task
                task = META.get(iid, (None, None, v[0]))[2] or v[0]
                d[iid] = (task, v[1], v[2], v[3])
            DATA[m][b] = d

# Public JevBench from the official leaderboard (Jev and Laya): per-item correctness
off = json.load(open(f"{JEVBENCH_DIR}/results/v1.2/jevbench-v1.2-per-task.json"))
OFFICIAL = {}
for name, key in (("Jev", "jev-1.13.0"), ("Laya", "laya")):
    pt = off["systems"][key]["public_tasks"]
    OFFICIAL[name] = {iid: (META[iid][2], v[0] == "c") for iid, v in pt.items() if iid in META}


def acc(d, ids, task=None):
    sel = [d[i][1] for i in ids if i in d and (task is None or d[i][0] == task)]
    return (sum(sel) / len(sel), len(sel)) if sel else (None, 0)


def bal(d, ids, task, suite_rows):
    gold = {r["id"]: str(r["expected"]) for r in suite_rows if r.get("task") == task}
    by = collections.defaultdict(list)
    for i in ids:
        if i in gold and i in d:
            by[gold[i]].append(d[i][1])
    return sum(sum(v) / len(v) for v in by.values()) / len(by) if by else None


# ---------------- TEST 1
res1 = {}
common = {}
for b in SUITES:
    if b == "jb":
        continue
    have = [m for m in MODELS if b in DATA[m]]
    ids = set.intersection(*[set(DATA[m][b]) for m in have]) if have else set()
    common[b] = ids
    tasks = sorted({DATA[have[0]][b][i][0] for i in ids}) if have else []
    rows_b = [json.loads(line) for line in open(os.path.join(J, SUITES[b]))]
    for m in have:
        d = DATA[m][b]
        per = {t: acc(d, ids, t)[0] for t in tasks}
        res1.setdefault(m, {})[b] = {"macro": sum(per.values()) / len(per), "n": len(ids), "tarefas": per}
        if b == "fin2":
            res1[m][b]["wcb_bal"] = bal(d, ids, "wcb_stance", rows_b)
# JevBench: Eikos (our harness), Jev/Laya (official) and Laya (our run = validation)
for m in ("Eikos-4B", "Eikos-27B"):
    d = DATA[m].get("jb", {})
    res1.setdefault(m, {})["jb"] = {t: acc(d, d.keys(), t)[0] for t in ("jb_easy", "jb_original", "jb_hard")}
for m, d in OFFICIAL.items():
    dd = {i: (t, ok, None, None) for i, (t, ok) in d.items()}
    res1.setdefault(m, {})["jb_official"] = {t: acc(dd, dd.keys(), t)[0] for t in ("jb_easy", "jb_original", "jb_hard")}
for m in ("Laya", "Laya-typed"):
    d = DATA[m].get("jb", {})
    if d:
        res1[m]["jb_our_run"] = {t: acc(d, d.keys(), t)[0] for t in ("jb_easy", "jb_original", "jb_hard")}

# Subset that fits in Laya's context (router): accuracy of all models on those items
fit_ids = set()
for b in SUITES:
    if b == "jb":
        continue
    for r in rd(f"suite_laya_router_{b}.jsonl") or []:
        if r.get("fits"):
            fit_ids.add(r["id"])
res_fit = {}
for m in MODELS:
    ok = [DATA[m][b][i][1] for b in common for i in common[b] if i in fit_ids and b in DATA[m] and i in DATA[m][b]]
    allv = [DATA[m][b][i][1] for b in common for i in common[b] if b in DATA[m] and i in DATA[m][b]]
    res_fit[m] = {"cabe_no_laya": (sum(ok) / len(ok), len(ok)) if ok else None,
                  "todos_comuns": (sum(allv) / len(allv), len(allv)) if allv else None}


# ---------------- TEST 2
def pooled(m, recal=None):
    out = []
    for b in common:
        if b not in DATA[m]:
            continue
        for i in common[b]:
            t, ok, conf, probs = DATA[m][b][i]
            if recal and probs:
                conf = recal(i, probs)
            out.append((conf if conf is not None else 0.0, ok))
    return out


def rc_metrics(pairs):
    n = len(pairs)
    # Items with the same confidence form one group (Jev reports steps of 0.01; bf16 logits also tie).
    # accs[k-1] = accuracy among the k most confident items, in expectation over the order of tied items,
    # so the result does not depend on input order. ends = the k where a confidence threshold can cut.
    groups = collections.defaultdict(lambda: [0, 0])  # confidence -> [items, correct]
    for c, ok in pairs:
        groups[c][0] += 1
        groups[c][1] += ok
    accs, ends, k, cum = [], [], 0, 0
    for c in sorted(groups, reverse=True):
        g, gok = groups[c]
        accs += [(cum + j * gok / g) / (k + j) for j in range(1, g + 1)]
        k, cum = k + g, cum + gok
        ends.append(k)
    aurc = sum(1 - a for a in accs) / n

    def cov_at(target):
        return max((kk for kk in ends if accs[kk - 1] >= target), default=0) / n

    def acc_at(cov):
        k = max(1, int(round(cov * n)))
        return accs[k - 1]

    ece = 0.0
    for bb in range(10):
        idx = [(c, ok) for c, ok in pairs if bb / 10 < c <= (bb + 1) / 10 or (bb == 0 and c == 0)]
        if idx:
            ece += len(idx) / n * abs(sum(ok for _, ok in idx) / len(idx) - sum(c for c, _ in idx) / len(idx))

    def policy(th):
        sel = [ok for c, ok in pairs if c >= th]
        return {"decide_sozinho": round(len(sel) / n, 3), "erro_real": round(1 - sum(sel) / len(sel), 3) if sel else None}
    return {"n": n, "acc_100": round(accs[-1], 4), "acc_80": round(acc_at(0.8), 4), "acc_50": round(acc_at(0.5), 4),
            "cobertura_95": round(cov_at(0.95), 3), "cobertura_99": round(cov_at(0.99), 3), "aurc": round(aurc, 4),
            "ece": round(ece, 4), "limiar_0.90": policy(0.90), "limiar_0.95": policy(0.95)}


def laya_recal(m):
    """Temperature per (type, number of options), cross-fitted on 2 halves split by id hash
    (what the model card recommends)."""
    items = []
    for b in common:
        for i in common[b]:
            if b in DATA[m] and i in DATA[m][b] and DATA[m][b][i][3]:
                qt, k, _ = META[i]
                items.append((i, (qt, k), DATA[m][b][i][3], DATA[m][b][i][1]))
    fold = {i: int(hashlib.sha1(i.encode()).hexdigest(), 16) % 2 for i, *_ in items}
    temps = {}
    grid = [0.25 * x for x in range(1, 41)]  # 0.25 .. 10
    for f in (0, 1):
        byb = collections.defaultdict(list)
        for i, bk, probs, ok in items:
            if fold[i] == f:
                byb[bk].append((probs, ok))
        for bk, lst in byb.items():
            def nll(T):
                s = 0.0
                for probs, ok in lst:
                    lg = {k: math.log(max(v, 1e-9)) / T for k, v in probs.items()}
                    z = max(lg.values())
                    den = sum(math.exp(v - z) for v in lg.values())
                    pred = max(probs, key=probs.get)
                    p = math.exp(lg[pred] - z) / den
                    s -= math.log(max(p if ok else 1 - p, 1e-9))
                return s
            temps[(f, bk)] = min(grid, key=nll)

    def recal(i, probs):
        qt, k, _ = META[i]
        T = temps.get((1 - fold.get(i, 0), (qt, k)), 1.0)  # temperature fitted on the other half
        lg = {kk: math.log(max(v, 1e-9)) / T for kk, v in probs.items()}
        z = max(lg.values())
        den = sum(math.exp(v - z) for v in lg.values())
        return max(math.exp(v - z) / den for v in lg.values())
    return recal


res2 = {m: rc_metrics(pooled(m)) for m in MODELS if any(b in DATA[m] for b in common)}
for m in ("Laya", "Laya-typed"):
    if any(b in DATA[m] for b in common):
        res2[m + " (recalibrated)"] = rc_metrics(pooled(m, laya_recal(m)))
json.dump({"teste1": res1, "cabe_no_laya": res_fit, "teste2": res2, "itens_comuns": {b: len(v) for b, v in common.items()}},
          open(os.path.join(J, "compare_models.json"), "w"), indent=1, default=str)

f = lambda x: "—" if x is None else f"{100 * x:5.1f}"  # noqa: E731
print("common items per suite:", {b: len(v) for b, v in common.items()})
print("\n== TEST 1: accuracy (common items; macro over tasks)")
hdr = ["DB", "General", "Fin", "WCB bal", "FinDVer", "Trade", "Rules"]
print(f"{'model':22s}" + "".join(f"{h:>9s}" for h in hdr))
for m in MODELS:
    r = res1.get(m, {})
    g = lambda b: r.get(b, {}).get("macro") if b in r else None  # noqa: E731
    print(f"{m:22s}" + "".join(f"{f(x):>9s}" for x in (g("db"), g("gen"), g("fin"), r.get("fin2", {}).get("wcb_bal"),
                                                       r.get("fin2", {}).get("tarefas", {}).get("findver"), g("trade"),
                                                       g("rules"))))
print("\nPublic JevBench (easy / original / hard):")
for m in res1:
    for k in ("jb", "jb_official", "jb_our_run"):
        if k in res1[m]:
            v = res1[m][k]
            print(f"  {m:14s} {k:16s} {f(v.get('jb_easy'))} / {f(v.get('jb_original'))} / {f(v.get('jb_hard'))}")
print("\nOnly on the items that fit in Laya's context (accuracy, n):")
for m, v in res_fit.items():
    c, t = v["cabe_no_laya"], v["todos_comuns"]
    print(f"  {m:14s} fits: {f(c[0]) if c else '—'} (n={c[1] if c else 0}) | all: {f(t[0]) if t else '—'} (n={t[1] if t else 0})")
print("\n== TEST 2: usable confidence (common items of the 6 suites)")
print(f"{'model':28s}{'acc':>7s}{'acc@80%':>9s}{'acc@50%':>9s}{'cov@95%':>9s}{'cov@99%':>9s}{'AURC':>8s}{'ECE':>7s}"
      f"{'  ≥0.90: decide / error':>24s}{'  ≥0.95: decide / error':>24s}")
for m, v in res2.items():
    p9, p95 = v["limiar_0.90"], v["limiar_0.95"]
    print(f"{m:28s}{f(v['acc_100']):>7s}{f(v['acc_80']):>9s}{f(v['acc_50']):>9s}{f(v['cobertura_95']):>9s}"
          f"{f(v['cobertura_99']):>9s}{v['aurc']:8.3f}{v['ece']:7.3f}"
          f"{f(p9['decide_sozinho']):>12s} / {f(p9['erro_real']):>6s}{f(p95['decide_sozinho']):>13s} / {f(p95['erro_real']):>6s}")
