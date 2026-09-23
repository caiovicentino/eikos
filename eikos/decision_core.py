"""Shared core for the decision format and readout, used by training (train_dec.py), evaluation
(letter_adapter.py) and the server (serve.py), so that the released model answers exactly as it was
trained and evaluated.

- Prompt style: PROMPT_STYLE=semif (JSON evidence/criterion/options, yes/no as true/false;
  the SemIf format, MIT) or "ours" (plain text).
- Temperature: fixed, or conditioned on observable signals (calib_fit.py).
- Verify mode (optional): short reasoning with a token budget before reading the letter.
License: MIT.
"""
from __future__ import annotations

import json
import math
import os

LETTERS = [chr(65 + i) for i in range(26)]
STYLE = os.environ.get("PROMPT_STYLE", "semif")
if STYLE == "semif":
    SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
              "Respond with only its uppercase letter, with no explanation or reasoning.")
else:
    SYSTEM = ("You are a decision model. Read the state and the question, then answer "
              "with the letter of the single best option. Use only the information in the state.")
PROMPT_VERSION = f"letter-v1-{STYLE}"
VERIFY_SYSTEM = ("Apply the supplied criterion to the supplied evidence. Think briefly and check any numbers, "
                 "then respond with only the uppercase letter of the single correct option.")


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def options_of(question: dict, labels: list | None = None) -> list[tuple[str, str]]:
    """(label, description). With `labels`, follows the item's order; otherwise derives them from the criteria (API)."""
    qtype = question.get("type")
    crit = question.get("criteria")
    if qtype in ("noul", "boolean"):
        crit = crit or {}
        return [("yes", str(crit.get("true", "yes"))), ("no", str(crit.get("false", "no")))]
    if qtype == "score":
        if isinstance(crit, dict):
            keys = [str(l) for l in labels] if labels else list(crit.keys())
            return [(k, str(crit.get(k, k))) for k in keys]
        return [(str(i), str(c)) for i, c in enumerate(crit or [])]
    if isinstance(crit, dict):
        keys = list(labels) if labels else list(crit.keys())
        return [(k, str(crit.get(k, k))) for k in keys]
    keys = list(labels) if labels else [str(o) for o in (question.get("options") or crit or [])]
    return [(k, k) for k in keys]


def render_user(state, question: dict, opts: list[tuple[str, str]]) -> str:
    if STYLE == "semif":
        noul = question.get("type") in ("noul", "boolean")
        shown = (lambda lab: {"yes": "true", "no": "false"}.get(lab, lab)) if noul else (lambda lab: lab)
        payload = {"evidence": state, "criterion": question.get("instructions", ""),
                   "options": [{"letter": LETTERS[i], "description": f"{shown(lab)}: {desc}"}
                               for i, (lab, desc) in enumerate(opts)]}
        return json.dumps(payload, ensure_ascii=False)
    ol = "\n".join(f"{LETTERS[i]}. {lab}: {desc}" for i, (lab, desc) in enumerate(opts))
    return (f"State:\n{state_text(state)}\n\nQuestion: {question.get('instructions', '')}\n\n"
            f"Options:\n{ol}\n\nAnswer with the letter of the correct option.")


def messages(state, question, opts, verify: bool = False) -> list[dict]:
    return [{"role": "system", "content": VERIFY_SYSTEM if verify else SYSTEM},
            {"role": "user", "content": render_user(state, question, opts)}]


def temp_for(calib: dict | None, temp: float, n_tok: int, n_opts: int, qtype: str) -> float:
    """Fixed temperature, or T(x) = exp(b + w·f) with exactly the same features as calib_fit.py."""
    if not calib:
        return temp
    f = [math.log(max(n_tok, 1)) - 6.5, math.log(n_opts) - 1.2,
         float(qtype in ("noul", "boolean")), float(qtype == "score")]
    return math.exp(calib["b"] + sum(w * x for w, x in zip(calib["w"], f)))


def tournament(dist_fn, opts, chunk: int = 20, keep: int = 2):
    """More than 26 options: blocks of `chunk`; the top-`keep` of each block go to the final round (≤26)."""
    if len(opts) <= 26:
        return dist_fn(opts)
    finalists, n_tok = [], 0
    for i in range(0, len(opts), chunk):
        block = opts[i:i + chunk]
        p, n = dist_fn(block)
        n_tok += n
        finalists += sorted(block, key=lambda o: -p[o[0]])[:keep]
    pf, n = dist_fn(finalists[:26])
    probs = {lab: 1e-4 for lab, _ in opts}
    probs.update(pf)
    z = sum(probs.values())
    return {k: v / z for k, v in probs.items()}, n_tok + n


def option_spans(text: str, user: str, opts: list[tuple[str, str]], question: dict, offsets) -> list | None:
    """Token spans of each option's description in the semif prompt, used by the energy readout (JEPA)."""
    noul = question.get("type") in ("noul", "boolean")
    shown = (lambda lab: {"yes": "true", "no": "false"}.get(lab, lab)) if noul else (lambda lab: lab)  # noqa: E731
    base = text.find(user)
    k = user.find('"options"')
    if base < 0 or k < 0:
        return None
    spans = []
    for lab, desc in opts:
        needle = json.dumps(f"{shown(lab)}: {desc}", ensure_ascii=False)
        pos = user.find(needle, k)
        if pos < 0:
            return None
        a, b = base + pos + 1, base + pos + len(needle) - 1
        toks = [t for t, (s0, e0) in enumerate(offsets) if e0 > a and s0 < b]
        if not toks:
            return None
        spans.append((toks[0], toks[-1] + 1))
        k = pos + len(needle)
    return spans
