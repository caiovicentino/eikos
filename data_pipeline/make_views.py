"""Views for JEPA training: the same decision in another language (EN→PT and PT→EN; Spanish stays held out), same facts, same labels.
Generated with a local Qwen3.8-27B (SGLang, thinking disabled), output as validated JSON. Training items only
(never dev, held-out or benchmark items).
usage: python make_views.py <output.jsonl> <n_max> <api1[,api2]> [threads]
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import random
import sys
import threading
import urllib.request

import os
EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, evaluation suites and checkpoints
OUT, NMAX = sys.argv[1], int(sys.argv[2])
APIS = [f"{u.rstrip('/')}/v1/chat/completions" for u in sys.argv[3].split(",")]
TH = int(sys.argv[4]) if len(sys.argv) > 4 else 16
SRC = [f"{EIKOS_HOME}/data/labeled.jsonl", f"{EIKOS_HOME}/data_q38/labeled.jsonl", f"{EIKOS_HOME}/data_q38b/labeled.jsonl"]
HOLD = {"topic": {"healthcare administration"}, "family": {"tradeoff"}, "lang": {"Spanish"}}
PROMPT = """Translate this decision item into {lang}. Keep EVERY fact, number, date, amount, currency, name, ID, code, rule and condition exactly; keep JSON keys, label names and code unchanged; translate only natural-language text. Do not add, drop, soften or explain anything.
Return ONLY a JSON object with exactly these keys: "state" (same type as the input state: a string, or a JSON object with the same keys), "instructions" (string), "criteria" (same keys/structure as the input criteria, values translated).

INPUT:
{item}"""


def is_dev(i, frac=0.04):  # same hash as train_dec.py (DEV_FRAC=0.04): views only from training items
    return (int(hashlib.sha1(str(i).encode()).hexdigest()[:8], 16) % 10000) < frac * 10000


def target_lang(r, rng):
    if r.get("lang") in ("Brazilian Portuguese", "Spanish"):
        return "English"
    return "Brazilian Portuguese"  # Spanish is held out of training (unseen-language test)


def ask(r, lang, api):
    q = r["question"]
    item = {"state": r["state"], "instructions": q["instructions"], "criteria": q.get("criteria")}
    body = {"model": os.environ.get("VIEWS_MODEL", "q38"), "temperature": 0.2, "max_tokens": 9000,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": PROMPT.format(lang=lang, item=json.dumps(item, ensure_ascii=False, indent=1))}]}
    req = urllib.request.Request(api, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        txt = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    v = json.loads(txt)
    ok_state = isinstance(v.get("state"), type(r["state"])) and (
        not isinstance(r["state"], dict) or set(v["state"]) == set(r["state"]))
    crit = q.get("criteria")
    ok_crit = (isinstance(crit, dict) and isinstance(v.get("criteria"), dict) and set(v["criteria"]) == set(crit)) or \
              (isinstance(crit, list) and isinstance(v.get("criteria"), list) and len(v["criteria"]) == len(crit))
    if not (ok_state and ok_crit and isinstance(v.get("instructions"), str) and v["instructions"].strip()):
        return None
    # numbers preserved: at least 90% of the 3+-digit numbers in the original must appear in the translation
    import re
    src_nums = set(re.findall(r"\d{3,}", json.dumps(r["state"], ensure_ascii=False)))
    dst = json.dumps(v["state"], ensure_ascii=False)
    if src_nums and sum(n in dst for n in src_nums) < 0.9 * len(src_nums):
        return None
    return {"state": v["state"], "instructions": v["instructions"], "criteria": v["criteria"], "lang": lang}


def main():
    rows = []
    for p in SRC:
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if not r.get("agree") or is_dev(r.get("id")):
                continue
            if r.get("topic") in HOLD["topic"] or r.get("family") in HOLD["family"] or r.get("lang") in HOLD["lang"]:
                continue
            rows.append(r)
    rng = random.Random(17)
    rng.shuffle(rows)
    if len(sys.argv) > 5 and sys.argv[5] == "rev":  # 2nd server: same list in reverse order (merged later by id)
        rows = rows[::-1]
    done = set()
    try:
        done = {json.loads(l)["id"] for l in open(OUT)}
    except FileNotFoundError:
        pass
    rows = [r for r in rows if r["id"] not in done][:NMAX]
    lock, st = threading.Lock(), {"ok": 0, "bad": 0, "error": 0}
    out = open(OUT, "a")

    def work(i_r):
        i, r = i_r
        lang = target_lang(r, random.Random(i))
        try:
            v = ask(r, lang, APIS[i % len(APIS)])
            key = "ok" if v else "bad"
        except Exception:  # noqa: BLE001
            v, key = None, "error"
        with lock:
            st[key] += 1
            if v:
                out.write(json.dumps({"id": r["id"], "view": v}, ensure_ascii=False) + "\n")
                out.flush()
            if sum(st.values()) % 100 == 0:
                print(json.dumps(st), flush=True)

    with cf.ThreadPoolExecutor(TH) as ex:
        list(ex.map(work, enumerate(rows)))
    print("DONE", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
