"""Generates typed-decision items in the JevBench style and labels them blindly with the teacher.

Item writer: GLM-5.3-Flash (reasoning "high"), writes N_PER_CALL items per call (default 3).
Teacher: GLM-5.3-Flash at maximum effort (the default), via JevBench's own openai_compat adapter
(same prompt and probability JSON as the leaderboard), without seeing the gold answer.
An item counts only if the teacher's argmax matches the item writer's gold answer.

Never shows public JevBench items to the item writer (anti-contamination).
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import random
import re
import sys
import threading
import time
import urllib.request

EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, evaluation suites and checkpoints
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
sys.path.insert(0, JEVBENCH_DIR)
from jevbench.adapters.openai_compat import OpenAICompatAdapter  # noqa: E402
from jevbench.tasks import Task  # noqa: E402

API = os.environ["TEACHER_API_URL"]  # teacher: GLM-5.3-Flash on an OpenAI-API-compatible endpoint
MODEL = os.environ.get("TEACHER_MODEL", "glm-5.3-flash")
GEN_API = os.environ.get("GEN_API", API)  # item writer (default: the same GLM; or a local SGLang server)
GEN_MODEL = os.environ.get("GEN_MODEL", MODEL)
GEN_EXTRA = json.loads(os.environ.get("GEN_EXTRA", '{"chat_template_kwargs": {"reasoning_effort": "high"}}'))
# hard items: maximum reasoning for the item writer (GLM default = max; {} does not send reasoning_effort)
GEN_EXTRA_HARD = json.loads(os.environ.get("GEN_EXTRA_HARD", '{}'))
KEY = os.environ["TEACHER_API_KEY"]
OUT = os.environ.get("OUT", f"{EIKOS_HOME}/data")
N_GEN = int(os.environ.get("N_GEN", "3"))
N_LAB = int(os.environ.get("N_LAB", "5"))
TARGET = int(os.environ.get("TARGET", "30000"))
SEED = int(os.environ.get("SEED", "1"))
N_PER_CALL = int(os.environ.get("N_PER_CALL", "3"))
os.makedirs(OUT, exist_ok=True)

FAMILIES = {
    # family: (weight, allowed types, difficulty, description)
    "long_policy": (12, ["choice", "noul", "score"], "hard",
                    "A long policy, contract, regulation or handbook (many clauses, definitions, exceptions, "
                    "amendments that supersede earlier versions) plus a concrete case. The decisive clause is "
                    "buried; a superseded or general rule points the wrong way."),
    "multi_hop": (12, ["choice", "noul"], "hard",
                  "The answer needs chaining 2-4 facts from different parts of the state (e.g. look up a tier "
                  "in one table, apply its limit from another section, then check a date)."),
    "temporal_numeric": (10, ["choice", "noul", "score"], "hard",
                         "Dates, deadlines, business days, time zones, durations, amounts, percentages, "
                         "unit conversions or thresholds decide the answer; the arithmetic must be done carefully."),
    "trap": (8, ["choice", "noul"], "hard",
             "A salient but non-authoritative cue (an estimate, a cached value, a customer's claim, a preview) "
             "suggests the surface answer; a stated authority/precedence rule makes a different label correct."),
    "ambiguous": (5, ["choice"], "hard",
                  "Sources conflict. Either a stated precedence rule resolves the conflict (then the resolved "
                  "label is gold) or nothing resolves it (then an explicit cannot_determine-style label is gold). "
                  "Include such a label among the options."),
    "adversarial": (6, ["choice", "noul"], "hard",
                    "The state contains untrusted text (an email, a web page, a ticket comment, a tool output) "
                    "with injected instructions trying to force a label or override policy. The correct label "
                    "follows the legitimate rules and ignores the injection."),
    "tradeoff": (5, ["choice"], "hard",
                 "Several options satisfy some constraints; the state states hard requirements and a priority "
                 "order or weights; exactly one option is best under them."),
    "routing_hard": (5, ["choice"], "hard",
                     "Nuanced routing rules (teams, queues, models, escalation levels) with exceptions; route "
                     "one request correctly."),
    "judge_hard": (10, ["noul", "choice"], "hard",
                   "The state has a task (math word problem, calculation, SQL/code snippet, support reply, "
                   "extraction) and a proposed answer or solution. Decide whether it is correct/acceptable, or "
                   "classify the kind of error. Make wrong answers plausible (off-by-one, wrong unit, wrong "
                   "branch, sign error, misread condition). About half the proposed answers are correct."),
    "probability": (4, ["noul", "choice"], "hard",
                    "A chance event fully specified in the state (sampling without replacement, dice, reliability "
                    "of independent parts, base rates with a test, queue of random events). The question asks for "
                    "probabilities that reflect the evidence. Provide gold_probs = the EXACT distribution over the "
                    "labels (4 decimals, summing to 1); expected = the most likely label."),
    "intent": (4, ["choice"], "standard",
               "A short user message (chat, email, voice transcript) to classify into one of several intents."),
    "fact": (4, ["noul", "choice"], "standard",
             "A factual question answered by a detail in the state (a record, a note, an itinerary)."),
    "extraction": (5, ["choice"], "standard",
                   "Pick which value/field in the state satisfies the question (e.g. which address is the "
                   "shipping one, which date is the renewal date)."),
    "tool_selection": (4, ["choice"], "standard",
                       "Given available tools/APIs with descriptions and a request, choose the right tool."),
    "policy": (6, ["noul"], "standard",
               "A short policy and a request: is the requested action permitted? Unproved required "
               "conditions count as not satisfied."),
    "ordinal": (5, ["score"], "standard",
                "Rate something (urgency, severity, quality, risk, sentiment) on an ordinal rubric whose levels "
                "are defined precisely enough that one level is right."),
    "adequacy": (4, ["noul", "score"], "standard",
                 "A request and a response (support reply, summary, answer): does the response adequately "
                 "satisfy the stated requirements?"),
    "routing": (3, ["choice"], "standard",
                "Route a request to the right team/category/model given short routing rules."),
    "news_signal": (8, ["choice"], "standard",
                    "Classify the market impact or stance of a news item, social post, earnings-call line, central-bank "
                    "sentence, filing excerpt or analyst note, with a 3-way or 4-way label set that ALWAYS includes a "
                    "neutral option (e.g. bullish/bearish/neutral, hawkish/dovish/neutral, upgrade/downgrade/no_change, "
                    "material/immaterial/unclear). Most real texts are informational: make the neutral label correct in "
                    "about 40% of items (routine announcements, mixed signals, facts already priced in, questions, "
                    "reposted headlines), and never let tone words alone decide."),
    "large_choice": (5, ["choice"], "standard",
                     "Classify or route into ONE of MANY options: this family uses 10-26 labels instead of 3-7 "
                     "(intents, teams, tools/APIs, incident or product categories). Criteria define every option "
                     "precisely and several options are near-misses of the right one."),
}
TOPICS = ["math & numbers", "support & operations", "everyday language", "rules, policy & law",
          "coding & software", "safety & security", "healthcare administration",
          "HR & people ops", "logistics & travel", "education & research admin", "public sector services",
          # finance (8 of the 19 topics): the model's specialty, global + Brazil
          "finance: banking & payments (cards, chargebacks, ACH, SEPA, wires, instant payments; any country)",
          "finance: credit & lending (underwriting, limits, collections; US/EU/UK/Asia/LatAm rules)",
          "finance: AML/KYC, fraud & sanctions screening (FATF-style rules, OFAC/EU/UN lists)",
          "finance: insurance claims & underwriting",
          "finance: accounting, tax & invoices (reconciliation, VAT/GST/sales tax, withholding, IFRS/GAAP)",
          "finance: capital markets & crypto (orders, suitability, custody, disclosures, on-chain risk)",
          "finance: trading & markets (orders, risk limits, margin/liquidation, stop rules, pre-trade compliance, "
          "market-abuse surveillance, settlement T+1/T+2; apply stated rules, never predict prices)",
          "finance: trade finance & international trade (letters of credit, Incoterms, customs, document checks)"]
FORMATS = ["plain prose", "an email thread", "a chat log", "a JSON record", "a support ticket with comments",
           "a table in text", "log lines", "a contract or policy excerpt", "a form with filled fields",
           "a code snippet with context"]
LANGS = [("English", 75), ("Brazilian Portuguese", 20), ("Spanish", 5)]
LEN = {"hard": ["400-900 tokens", "900-1800 tokens", "1500-2500 tokens"],
       "standard": ["80-250 tokens", "250-600 tokens"]}

SPEC = """You write evaluation items for typed-decision models. A typed-decision model reads a piece of STATE (text or a JSON object) and a QUESTION with a bounded rubric, and returns a probability for every allowed label in a single pass, without explanation.

Write {n} INDEPENDENT items (different scenarios, companies, names) as a JSON object {{"items": [...]}}. Each item:
{{"family": "{family}", "topic": "<topic>", "state": <string or JSON object>,
 "question": {{"type": "noul"|"choice"|"score", "instructions": "<the decision to make, 1-3 sentences>", "criteria": <see types>}},
 "labels": [...], "expected": <gold>, "rationale": "<2-5 sentences citing the decisive facts>",
 "surface_answer": <the tempting wrong label, or null>, "gold_probs": <ONLY for family probability, else null>}}

Question types (exactly):
- noul: labels exactly ["no", "yes"]; expected "no" or "yes"; criteria = {{"true": "<when yes>", "false": "<when no>"}}.
- choice: 3-7 labels in lowercase snake_case; expected is one of them; criteria = {{"<label>": "<one-line definition>", ...}} covering every label.
- score: labels ["0","1",...,"k"] with 3-6 levels; expected is an INTEGER level index; criteria = ["<level 0 description>", ..., "<level k description>"].

Rules:
- NEUTRAL OPTIONS (most important): labels are short neutral names of the answer options (the category, value or action, e.g. "approve_refund", "eur_1250", "escalate_to_tier2"). Never put hints about correctness in a label (no "correct", "final", "stale", "old", "wrong", "valid", "confirmed").
- NEUTRAL CRITERIA: each criterion defines WHEN that label applies in general (a condition), never how THIS state relates to it. Never write "the one the group agreed on", "cancelled, so not it", "this is the stale amount", or anything that tells which option matches the state.
- BALANCED YES/NO: a yes/no item is as likely to be "yes" as "no". When the slot says "yes", write a case where the action IS allowed / the claim IS true, with tempting details that look like problems but do not change the outcome.
- DEFAULT OPTIONS MATTER: when the choice labels include a neutral/default option (neutral, no_action, none, keep_current, not_applicable, cannot_determine), make it the correct answer in about 1 of every 4 such items, i.e. whenever the evidence does not support a decisive label. Never make the decisive label correct by default.
- Everything needed to decide is in the state; no outside knowledge beyond common sense and arithmetic.
- Exactly one defensible gold label that a careful expert with time would pick; the other labels are plausible but wrong.
- Difficulty ({difficulty}) must come from reasoning over the state, never from vague wording or trick grammar.
- Never put the answer, a rationale, or keys named expected/label/answer/gold inside the state.
- Realistic, specific details (IDs, timestamps, amounts, names). Vary industries, cultures and formats.

This call: family = {family}: {fdesc}
{slots}
Labels are always English snake_case (noul: "no"/"yes"). Output only the JSON object."""


def chat(messages, extra=None, timeout=600):
    body = {"model": GEN_MODEL, "messages": messages, "temperature": 0.9,
            "max_tokens": int(os.environ.get("GEN_MAX_TOKENS", "32000"))}
    if extra:
        body.update(extra)
    req = urllib.request.Request(f"{GEN_API}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {})


def wpick(pairs, rng):
    tot = sum(w for _, w in pairs)
    x = rng.uniform(0, tot)
    for v, w in pairs:
        x -= w
        if x <= 0:
            return v
    return pairs[-1][0]


def make_job(rng):
    fam = wpick([(f, v[0]) for f, v in FAMILIES.items()], rng)
    _, types, diff, desc = FAMILIES[fam]
    slots = []
    for _ in range(N_PER_CALL):
        qt = rng.choice(types)
        slots.append(dict(topic=rng.choice(TOPICS), qtype=qt, fmt=rng.choice(FORMATS),
                          len=rng.choice(LEN[diff]), lang=wpick(LANGS, rng),
                          target=rng.choice(["no", "yes"]) if qt == "noul" else None))
    return fam, diff, desc, slots


def validate(it, fam):
    q = it.get("question") or {}
    t = q.get("type")
    labels = it.get("labels")
    exp = it.get("expected")
    crit = q.get("criteria")
    if t not in ("noul", "choice", "score") or not isinstance(labels, list) or not q.get("instructions"):
        return "type/labels"
    if t == "noul":
        if labels != ["no", "yes"] or exp not in ("no", "yes") or not isinstance(crit, dict) \
                or set(crit) != {"true", "false"}:
            return "noul"
    elif t == "choice":
        lo, hi = (10, 26) if fam == "large_choice" else (3, 7)
        if not (lo <= len(labels) <= hi) or len(set(labels)) != len(labels) or exp not in labels \
                or not isinstance(crit, dict) or set(crit) != set(labels) \
                or not all(re.fullmatch(r"[a-z0-9_]+", l or "") for l in labels):
            return "choice"
    else:
        if not isinstance(crit, list) or not (3 <= len(crit) <= 6) or labels != [str(i) for i in range(len(crit))]:
            return "score"
        try:
            exp = int(exp)
        except Exception:  # noqa: BLE001
            return "score-exp"
        if not 0 <= exp < len(crit):
            return "score-range"
        it["expected"] = exp
    st = it.get("state")
    s = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
    if not s or len(s) < 80:
        return "short-state"
    if isinstance(st, dict) and any(k in st for k in ("expected", "label", "ground_truth", "answer_key", "gold")):
        return "leak"
    if fam == "probability":
        gp = it.get("gold_probs")
        if not isinstance(gp, dict) or set(gp) != set(labels) or abs(sum(float(v) for v in gp.values()) - 1) > 0.02:
            return "gold_probs"
    return None


def parse_items(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    return obj.get("items") or []


def iid(it):
    s = it["state"] if isinstance(it["state"], str) else json.dumps(it["state"], sort_keys=True)
    return hashlib.sha1((s + json.dumps(it["question"], sort_keys=True)).encode()).hexdigest()[:16]


LOCK = threading.Lock()
STATS = {"gen_calls": 0, "gen_items": 0, "gen_rejects": {}, "lab": 0, "agree": 0, "lab_fail": 0, "t0": time.time()}
lab_q: queue.Queue = queue.Queue(maxsize=6000)
STOP = threading.Event()


def append(path, obj):
    with LOCK:
        with open(path, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def gen_worker(wid):
    rng = random.Random(SEED * 1000 + wid)
    while not STOP.is_set():
        fam, diff, desc, slots = make_job(rng)
        slot_txt = "\n".join(
            f"Item {i + 1}: topic = {sl['topic']}; question type = {sl['qtype']}; state format = {sl['fmt']}; "
            f"state length ≈ {sl['len']}; language of state and question = {sl['lang']}."
            + (f' Its correct answer must be "{sl["target"]}"' + (
                " (the action is allowed / the statement holds; details that look like problems, if any, do not change "
                "the outcome)." if sl["target"] == "yes" else
                " (the action is not allowed / the statement does not hold, for a reason checkable in the state).")
               if sl.get("target") else "")
            for i, sl in enumerate(slots))
        prompt = SPEC.format(n=len(slots), family=fam, fdesc=desc, difficulty=diff, slots=slot_txt)
        try:
            txt, usage = chat([{"role": "user", "content": prompt}],
                              extra=GEN_EXTRA_HARD if diff == "hard" else GEN_EXTRA, timeout=1200)
        except Exception as e:  # noqa: BLE001
            with LOCK:
                STATS["gen_rejects"]["http"] = STATS["gen_rejects"].get("http", 0) + 1
                if STATS["gen_rejects"]["http"] <= 3:
                    body = getattr(e, "read", lambda: b"")()
                    print("ERROR (item writer):", repr(e)[:200], body[:300], flush=True)
            time.sleep(5)
            continue
        with LOCK:
            STATS["gen_calls"] += 1
        for it, slot in zip(parse_items(txt), slots):
            it["family"] = fam
            err = validate(it, fam)
            if not err and slot.get("target") and (it.get("question") or {}).get("type") == "noul" \
                    and str(it.get("expected")) != slot["target"]:
                err = "yes_no_target"
            if err:
                with LOCK:
                    STATS["gen_rejects"][err] = STATS["gen_rejects"].get(err, 0) + 1
                continue
            it["id"] = iid(it)
            it["difficulty"] = diff
            it["lang"] = slot["lang"]
            it["fmt"] = slot["fmt"]
            it["generator"] = GEN_MODEL
            append(f"{OUT}/gen_raw.jsonl", it)
            with LOCK:
                STATS["gen_items"] += 1
            lab_q.put(it)


def lab_worker(wid):
    ad = OpenAICompatAdapter(endpoint=API, model=MODEL, key_env="TEACHER_API_KEY")
    ad.timeout_s = 900
    ad.request_options = {"max_tokens": 24000}  # the default (4096) cuts off maximum-effort reasoning on long items
    while not STOP.is_set():
        try:
            it = lab_q.get(timeout=5)
        except queue.Empty:
            continue
        task = Task(id=it["id"], family=it["family"], state=it["state"], question=it["question"],
                    labels=it["labels"], expected=str(it["expected"]) if it["question"]["type"] == "score"
                    else it["expected"], split="train", group=it["id"], provenance={})
        t0 = time.time()
        err = None
        try:
            r = ad.run(task)
        except Exception as e:  # noqa: BLE001
            r, err = None, repr(e)[:200]
        dt = time.time() - t0
        if r is None or not r.ok or not r.probs:
            if err is None and r is not None:
                err = str(getattr(r, "error", None) or "no probabilities")[:200]
            append(f"{OUT}/lab_fail.jsonl", {"id": it["id"], "s": round(dt, 1), "err": err})
            with LOCK:
                STATS["lab_fail"] += 1
            continue
        probs = {str(k): float(v) for k, v in r.probs.items()}
        s = sum(probs.values()) or 1.0
        probs = {k: v / s for k, v in probs.items()}
        top = max(probs, key=probs.get)
        agree = top == str(it["expected"])
        rec = dict(it, teacher_probs=probs, teacher_top=top, agree=agree, teacher_s=round(dt, 1),
                   teacher_usage=r.usage)
        append(f"{OUT}/labeled.jsonl", rec)
        with LOCK:
            STATS["lab"] += 1
            STATS["agree"] += int(agree)
            n_ok = STATS["agree"]
        if n_ok >= TARGET:
            STOP.set()


def reporter():
    while not STOP.is_set():
        time.sleep(60)
        with LOCK:
            el = (time.time() - STATS["t0"]) / 60
            line = dict(STATS, t0=None, min=round(el, 1),
                        agree_rate=round(STATS["agree"] / max(STATS["lab"], 1), 3),
                        agree_per_min=round(STATS["agree"] / max(el, 1e-6), 2), qsize=lab_q.qsize())
        print(json.dumps(line, ensure_ascii=False), flush=True)


def requeue_pending():
    """Lossless restart: items already generated but not yet labeled go back into the queue."""
    raw, done = f"{OUT}/gen_raw.jsonl", f"{OUT}/labeled.jsonl"
    if not os.path.exists(raw):
        return 0
    seen = set()
    try:  # items that already failed labeling are not re-queued (they used up GLM slots on every restart)
        for line in open(f"{OUT}/lab_fail.jsonl"):
            try:
                seen.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    except FileNotFoundError:
        pass
    if os.path.exists(done):
        for line in open(done):
            try:
                seen.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    n = 0
    for line in open(raw):
        try:
            it = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if it["id"] not in seen:
            seen.add(it["id"])
            lab_q.put(it)
            n += 1
    return n


if __name__ == "__main__":
    print("pending items re-queued:", requeue_pending(), flush=True)
    ths = [threading.Thread(target=gen_worker, args=(i,), daemon=True) for i in range(N_GEN)]
    ths += [threading.Thread(target=lab_worker, args=(i,), daemon=True) for i in range(N_LAB)]
    ths.append(threading.Thread(target=reporter, daemon=True))
    for t in ths:
        t.start()
    try:
        while not STOP.is_set():
            time.sleep(5)
    except KeyboardInterrupt:
        STOP.set()
    print("DONE", json.dumps(STATS), flush=True)
