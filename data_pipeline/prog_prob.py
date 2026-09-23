"""Items of the "probability" family with an exact gold distribution (no teacher).

Each item describes a random event fully specified in the state, with
distractors (superseded plan, irrelevant numbers, unverified claims).
Output in the same format as labeled.jsonl (teacher_probs = gold_probs).
usage: python prog_prob.py N SEED > prog_prob.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from fractions import Fraction
from math import comb

COMPANIES = ["Ardent Circuits", "Nimbus Components", "Norte Plásticos", "Halvard Foods", "Pinecrest Labs",
             "Aurora Textiles", "Brightwater Pumps", "Solace Medical", "Tarn Logistics", "Quill & Sons",
             "Mirante Energia", "Oakridge Tools", "Cobalt Bikes", "Lumen Optics", "Sierra Paper"]
PEOPLE = ["Ana", "Ravi", "Mei", "Lucas", "Fatima", "Jonas", "Priya", "Tomás", "Chloe", "Kwame", "Sofia", "Hiro"]


def r4(p):
    return round(float(p), 4)


def noul(p_yes):
    p = r4(p_yes)
    gp = {"no": r4(1 - p), "yes": p}
    return gp, ("yes" if p >= 0.5 else "no")


def hyper_at_least(N, D, n, c):
    tot = comb(N, n)
    return sum(Fraction(comb(D, k) * comb(N - D, n - k), tot) for k in range(c, min(D, n) + 1))


def t_inspection(rng):
    N = rng.randint(8, 30)
    D = rng.randint(1, max(1, N // 3))
    old_n, new_n = rng.sample(range(2, min(7, N)), 2)
    c = rng.choice([1, 1, 1, 2])
    in_force_new = rng.random() < 0.7
    n = new_n if in_force_new else old_n
    p = hyper_at_least(N, D, n, c)
    co = rng.choice(COMPANIES)
    claimed = rng.choice([5, 8, 10, 12, 15])
    batch = f"B{rng.randint(100, 999)}/{rng.randint(1, 12):02d}"
    when = "applies to this batch" if in_force_new else "starts with the next batch, not this one"
    state = (f"Receiving QA log, {co}, pallet batch {batch}: {N} identical cartons arrived. A recount by the "
             f"quality lead confirmed that exactly {D} cartons are water-damaged inside; the damage is invisible "
             f"from outside. The carrier's claim form estimates damage at 'about {claimed}%', which the QA manual "
             f"says must be ignored once a recount exists.\n"
             f"QA manual, rule R7 (current): open {old_n} cartons chosen at random (no carton opened twice).\n"
             f"QA manual, rule R7 (amended): open {new_n} cartons chosen at random (no carton opened twice); "
             f"the amendment {when}.\n"
             f"The batch is sent back if at least {c} opened carton{'s are' if c > 1 else ' is'} damaged.")
    q = {"type": "noul",
         "instructions": f"Will batch {batch} be sent back? Give probabilities that reflect the evidence in the state.",
         "criteria": {"true": f"At least {c} opened carton{'s are' if c > 1 else ' is'} damaged, so the batch is sent back.",
                      "false": "Fewer opened cartons are damaged than the threshold, so the batch is kept."}}
    gp, exp = noul(p)
    return state, q, ["no", "yes"], exp, gp


def t_reliability(rng):
    k = rng.randint(2, 4)
    ps = [rng.choice([0.9, 0.95, 0.97, 0.98, 0.99, 0.8, 0.85]) for _ in range(k)]
    series = rng.random() < 0.6
    p = 1.0
    if series:
        for x in ps:
            p *= x
    else:
        q = 1.0
        for x in ps:
            q *= (1 - x)
        p = 1 - q
    names = rng.sample(["payment gateway", "auth service", "database primary", "message queue", "CDN edge",
                        "search index", "cache cluster", "DNS resolver"], k)
    co = rng.choice(COMPANIES)
    comp = "\n".join(f"- {n}: available with probability {x} during the window, independently of the others"
                     for n, x in zip(names, ps))
    rule = ("The checkout flow works only if ALL of the components below are available."
            if series else "The status page stays reachable if AT LEAST ONE of the redundant mirrors below is available.")
    state = (f"{co} — RELEASE READINESS NOTE\n{rule}\n{comp}\n"
             f"An older note estimated overall availability at {rng.choice([90, 95, 99])}% but used a different "
             f"architecture and is obsolete. Engineer {rng.choice(PEOPLE)} believes the window will 'probably be fine'.")
    q = {"type": "noul",
         "instructions": "Will the service be available for the whole window? Give probabilities that reflect the evidence in the state.",
         "criteria": {"true": "The service is available for the whole window.",
                      "false": "The service is unavailable at some point in the window."}}
    gp, exp = noul(p)
    return state, q, ["no", "yes"], exp, gp


def t_raffle(rng):
    T = rng.randint(20, 200)
    h = rng.randint(1, max(1, T // 8))
    w = rng.randint(1, 5)
    p = 1 - Fraction(comb(T - h, w), comb(T, w))
    who = rng.choice(PEOPLE)
    state = (f"Office raffle rules: {T} tickets were sold in total. {w} winning tickets will be drawn at random "
             f"without replacement; a person can win at most once per ticket held. {who} holds {h} of the tickets. "
             f"Last year {who} won twice, which has no effect on this draw. Tickets bought after the cutoff "
             f"({rng.randint(3, 15)} late requests) were refunded and are not in the draw.")
    q = {"type": "noul",
         "instructions": f"Will {who} win at least one prize? Give probabilities that reflect the evidence in the state.",
         "criteria": {"true": f"At least one of {who}'s tickets is drawn.", "false": f"None of {who}'s tickets is drawn."}}
    gp, exp = noul(p)
    return state, q, ["no", "yes"], exp, gp


def t_bayes(rng):
    base = rng.choice([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    tpr = rng.choice([0.8, 0.9, 0.95, 0.99])
    fpr = rng.choice([0.01, 0.02, 0.05, 0.1])
    post = base * tpr / (base * tpr + (1 - base) * fpr)
    co = rng.choice(COMPANIES)
    state = (f"{co} fraud desk memo. Historical data: {base * 100:g}% of transactions of this type are fraudulent. "
             f"The screening model flags {tpr * 100:g}% of fraudulent transactions and {fpr * 100:g}% of legitimate "
             f"ones. Transaction TX-{rng.randint(100000, 999999)} was flagged. The customer's message says the "
             f"purchase is 'definitely mine', which the desk policy says must not change the estimate. "
             f"No other evidence is available yet.")
    q = {"type": "noul",
         "instructions": "Is the flagged transaction actually fraudulent? Give probabilities that reflect the evidence in the state.",
         "criteria": {"true": "The transaction is fraudulent.", "false": "The transaction is legitimate."}}
    gp, exp = noul(post)
    return state, q, ["no", "yes"], exp, gp


def t_dice(rng):
    n = rng.randint(2, 4)
    face = rng.choice([4, 6, 8])
    kind = rng.choice(["sum", "any"])
    from itertools import product
    outs = list(product(range(1, face + 1), repeat=n))
    if kind == "sum":
        t = rng.randint(n + 2, n * face - 2)
        fav = sum(1 for o in outs if sum(o) >= t)
        desc = f"the total of the {n} dice is at least {t}"
    else:
        v = face
        fav = sum(1 for o in outs if v in o)
        desc = f"at least one die shows {v}"
    p = Fraction(fav, len(outs))
    game = rng.choice(["a board-game tie-breaker", "a team-building game", "a classroom demo"])
    state = (f"Rules of {game}: roll {n} fair {face}-sided dice (faces 1 to {face}) once. The player advances "
             f"if {desc}. A house rule proposed last week (re-roll on doubles) was voted down and does not apply.")
    q = {"type": "noul",
         "instructions": "Will the player advance on this roll? Give probabilities that reflect the evidence in the state.",
         "criteria": {"true": f"Yes: {desc}.", "false": "No: the condition is not met."}}
    gp, exp = noul(p)
    return state, q, ["no", "yes"], exp, gp


def t_count_choice(rng):
    N = rng.randint(10, 25)
    D = rng.randint(2, 5)
    n = rng.randint(3, 5)
    labs = ["none_defective", "exactly_one_defective", "two_or_more_defective"]
    p0 = Fraction(comb(N - D, n), comb(N, n))
    p1 = Fraction(comb(D, 1) * comb(N - D, n - 1), comb(N, n))
    p2 = 1 - p0 - p1
    gp = {labs[0]: r4(p0), labs[1]: r4(p1), labs[2]: r4(p2)}
    gp[labs[2]] = r4(1 - gp[labs[0]] - gp[labs[1]])
    exp = max(gp, key=gp.get)
    co = rng.choice(COMPANIES)
    state = (f"{co} audit: a box holds {N} sealed kits, of which exactly {D} are known to be mislabeled (count "
             f"confirmed by a full recount). An auditor opens {n} kits chosen at random without replacement. "
             f"The warehouse app shows a 'low risk' badge for this box, but the badge ignores recounts.")
    q = {"type": "choice",
         "instructions": "How many mislabeled kits will the auditor find? Give probabilities that reflect the evidence in the state.",
         "criteria": {labs[0]: "No opened kit is mislabeled.", labs[1]: "Exactly one opened kit is mislabeled.",
                      labs[2]: "Two or more opened kits are mislabeled."}}
    return state, q, labs, exp, gp


TEMPLATES = [t_inspection, t_reliability, t_raffle, t_bayes, t_dice, t_count_choice]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
    for i in range(n):
        tpl = rng.choice(TEMPLATES)
        state, q, labels, exp, gp = tpl(rng)
        rec = {"id": f"prog-prob-{i:05d}", "family": "probability", "topic": "math & numbers",
               "state": state, "question": q, "labels": labels, "expected": exp, "gold_probs": gp,
               "teacher_probs": gp, "teacher_top": exp, "agree": True, "difficulty": "hard",
               "lang": "English", "source": f"prog_prob:{tpl.__name__}"}
        print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
