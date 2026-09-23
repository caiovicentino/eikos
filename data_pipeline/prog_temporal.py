"""Items of the "temporal_numeric" family with exact gold answers: deadlines in business days, SLA in business
hours, renewal notice periods, prorated refunds, late-payment fees and time zones.
Each item includes distractors (superseded rule, irrelevant numbers) and neutral labels.
usage: python prog_temporal.py N SEED > prog_temporal.jsonl
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys

CO = ["Harborgate Foods", "Verratero Logistics", "Pinecrest Labs", "Mirante Energia", "Tarn Freight",
      "Solace Clinics", "Lumen Optics", "Quill Insurance", "Oakridge Utilities", "Cobalt Telecom"]
D = lambda d: d.strftime("%A %d %B %Y")  # noqa: E731


def lab_date(d):
    return "d_" + d.strftime("%Y_%m_%d")


def soft(gold, labels, p=0.95):
    o = (1 - p) / (len(labels) - 1)
    return {l: (p if l == gold else o) for l in labels}


def add_business_days(start, n, holidays):
    d = start
    k = 0
    while k < n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d not in holidays:
            k += 1
    return d


def t_filing_deadline(rng):
    start = dt.date(2026, rng.randint(1, 11), rng.randint(1, 28))
    n = rng.choice([5, 7, 10, 15])
    hol = set()
    for _ in range(rng.randint(1, 3)):
        h = start + dt.timedelta(days=rng.randint(1, n + 6))
        if h.weekday() < 5:
            hol.add(h)
    dl = add_business_days(start, n, hol)
    cal_dl = start + dt.timedelta(days=n)
    no_hol = add_business_days(start, n, set())
    wrong = {cal_dl, no_hol, dl + dt.timedelta(days=1), dl - dt.timedelta(days=1)} - {dl}
    opts = [dl] + rng.sample(sorted(wrong), min(3, len(wrong)))
    rng.shuffle(opts)
    labels = [lab_date(o) for o in opts]
    co = rng.choice(CO)
    hol_txt = ", ".join(D(h) for h in sorted(hol)) or "none"
    state = (f"{co} — Claims Procedure Manual, section 4.2 (revised)\n"
             f"An appeal must be filed within {n} business days after the date of the decision letter. "
             f"Business days exclude Saturdays, Sundays and company holidays. The decision date itself is not counted. "
             f"(The pre-revision manual allowed {n} calendar days; it was withdrawn on 1 January 2026.)\n"
             f"Company holidays in the period: {hol_txt}.\n\n"
             f"Case file: decision letter dated {D(start)}; the claimant called on {D(start + dt.timedelta(days=2))} "
             f"asking about the deadline.")
    q = {"type": "choice", "instructions": "What is the last day on which the appeal can be filed on time?",
         "criteria": {lab_date(o): f"The deadline is {D(o)}." for o in opts}}
    return "choice", state, q, labels, lab_date(dl)


def t_sla(rng):
    open_h = rng.randint(8, 18)
    open_day = dt.datetime(2026, rng.randint(1, 11), rng.randint(1, 26), open_h, rng.choice([0, 15, 30, 45]))
    sla_h = rng.choice([4, 8, 12, 16])
    bh = (9, 17)
    # advance the clock only during business hours (Mon-Fri, 9-17)
    t = open_day
    rem = sla_h * 60
    while rem > 0:
        if t.weekday() >= 5 or t.hour >= bh[1]:
            t = dt.datetime(t.year, t.month, t.day, bh[0]) + dt.timedelta(days=1)
            continue
        if t.hour < bh[0]:
            t = dt.datetime(t.year, t.month, t.day, bh[0])
            continue
        end_today = dt.datetime(t.year, t.month, t.day, bh[1])
        step = min(rem, int((end_today - t).total_seconds() // 60))
        t += dt.timedelta(minutes=step)
        rem -= step
    deadline = t
    delta = rng.choice([-150, -40, -5, 5, 40, 150])
    resolved = deadline + dt.timedelta(minutes=delta)
    breached = resolved > deadline
    co = rng.choice(CO)
    fmt = "%a %d %b %Y %H:%M"
    state = (f"{co} Support SLA policy (v5): priority-2 tickets must be resolved within {sla_h} business hours. "
             f"Business hours are Monday to Friday, 09:00-17:00 local time; the clock pauses outside them. "
             f"The clock starts when the ticket is opened, not when it is first viewed. "
             f"(Policy v4 counted wall-clock hours; superseded.)\n\n"
             f"Ticket P2-{rng.randint(10000, 99999)}: opened {open_day.strftime(fmt)}; first viewed "
             f"{(open_day + dt.timedelta(minutes=rng.randint(5, 90))).strftime(fmt)}; resolved {resolved.strftime(fmt)}. "
             f"The customer rated the support 'fast'.")
    q = {"type": "noul", "instructions": "Did this ticket breach the SLA?",
         "criteria": {"true": "Resolution happened after the SLA deadline.",
                      "false": "Resolution happened at or before the SLA deadline."}}
    return "noul", state, q, ["no", "yes"], ("yes" if breached else "no")


def t_renewal(rng):
    renew = dt.date(2026, rng.randint(2, 12), rng.randint(1, 28))
    notice = rng.choice([30, 45, 60, 90])
    sent = renew - dt.timedelta(days=notice + rng.choice([-9, -3, -1, 0, 1, 4, 12]))
    ok = (renew - sent).days >= notice
    co = rng.choice(CO)
    state = (f"Master Services Agreement between {co} and its client, clause 12.3: the agreement renews "
             f"automatically for one year on {D(renew)} unless either party gives written notice of non-renewal at "
             f"least {notice} days before the renewal date. Notice is effective on the day it is sent by email to "
             f"legal@{co.split()[0].lower()}.com. Clause 12.4 (deleted by Amendment 2) had required notice by "
             f"registered mail.\n\nThe client emailed its non-renewal notice to legal@{co.split()[0].lower()}.com on "
             f"{D(sent)} and followed up by phone a week later.")
    q = {"type": "noul", "instructions": "Did the client's notice prevent the automatic renewal?",
         "criteria": {"true": "The notice meets the clause 12.3 deadline, so the agreement does not renew.",
                      "false": "The notice misses the deadline, so the agreement renews."}}
    return "noul", state, q, ["no", "yes"], ("yes" if ok else "no")


def t_prorated(rng):
    price = rng.choice([480, 600, 720, 960, 1200, 1440])
    used_days = rng.randint(20, 300)
    months_used = -(-used_days // 30)  # started months count as used
    fee = rng.choice([0, 25, 40, 50])
    refund = max(0, round(price * (12 - months_used) / 12 - fee, 2))
    wrong = {round(price * (365 - used_days) / 365 - fee, 2), round(price * (12 - months_used) / 12, 2),
             round(price * (12 - used_days // 30) / 12 - fee, 2), round(refund + fee * 2, 2)} - {refund}
    vals = [refund] + rng.sample(sorted(wrong), min(3, len(wrong)))
    rng.shuffle(vals)
    lab = lambda v: "usd_" + f"{v:.2f}".replace(".", "_")  # noqa: E731
    co = rng.choice(CO)
    state = (f"{co} annual plan terms: price ${price} per year, paid upfront. On cancellation, the refund is the "
             f"price times the number of whole months not yet started, divided by 12, minus a ${fee} processing fee "
             f"(never below zero). Any started month counts as used; a month is 30 days from the start date.\n"
             f"The customer cancelled {used_days} days after the start date. A chat agent had promised 'a full "
             f"refund of unused days', which the terms say agents cannot promise.")
    q = {"type": "choice", "instructions": "How much must be refunded under the terms?",
         "criteria": {lab(v): f"The refund is ${v:.2f}." for v in vals}}
    return "choice", state, q, [lab(v) for v in vals], lab(refund)


def t_tz(rng):
    base_off = rng.choice([-5, -3, 0, 1, 2, 5.5, 8, 9])
    other_off = rng.choice([o for o in [-8, -5, -3, 0, 1, 3, 5.5, 8, 10] if o != base_off])
    h = rng.randint(7, 18)
    mt = dt.datetime(2026, 5, 12, h, rng.choice([0, 30]))
    local = mt + dt.timedelta(hours=other_off - base_off)
    fmtt = lambda x: x.strftime("%a %H:%M")  # noqa: E731
    cands = {local, local + dt.timedelta(hours=1), local - dt.timedelta(hours=1),
             mt + dt.timedelta(hours=base_off - other_off)}
    cands = sorted(cands)
    rng.shuffle(cands)
    lab = lambda x: "t_" + x.strftime("%a_%H%M").lower()  # noqa: E731
    uniq = []
    for c in cands:
        if lab(c) not in [lab(u) for u in uniq]:
            uniq.append(c)
    state = (f"Meeting invite: 'Quarterly review', {mt.strftime('%A %d %B %Y, %H:%M')} at UTC{base_off:+g} "
             f"(organizer's office). The attendee works at UTC{other_off:+g}; neither office changes clocks this "
             f"month. The calendar preview shows the organizer's time only.")
    q = {"type": "choice", "instructions": "At what local time does the meeting start for the attendee?",
         "criteria": {lab(u): f"It starts at {fmtt(u)} attendee local time." for u in uniq}}
    return "choice", state, q, [lab(u) for u in uniq], lab(local)


def t_latefee(rng):
    due = dt.date(2026, rng.randint(1, 11), rng.randint(1, 28))
    late_days = rng.choice([0, 3, 7, 8, 14, 15, 22, 40])
    paid = due + dt.timedelta(days=late_days)
    amount = rng.choice([1200, 2500, 4800, 10000])
    rate = rng.choice([1, 1.5, 2])
    cap = rng.choice([4, 5, 6])
    weeks = -(-late_days // 7) if late_days > 0 else 0
    fee = round(amount * min(weeks * rate, cap) / 100, 2)
    wrong = {round(amount * min((late_days // 7) * rate, cap) / 100, 2), round(amount * weeks * rate / 100, 2),
             round(amount * min((weeks + 1) * rate, cap) / 100, 2), 0.0} - {fee}
    vals = [fee] + rng.sample(sorted(wrong), min(3, len(wrong)))
    rng.shuffle(vals)
    lab = lambda v: "eur_" + f"{v:.2f}".replace(".", "_")  # noqa: E731
    state = (f"Invoice for EUR {amount} due {D(due)}, paid {D(paid)}. Late-payment clause: {rate}% of the invoice "
             f"amount for each started week of delay after the due date, capped at {cap}% in total. Payment on the "
             f"due date is not late. The customer says the bank 'took two extra days', which the clause does not excuse.")
    q = {"type": "choice", "instructions": "What late fee applies?",
         "criteria": {lab(v): f"The late fee is EUR {v:.2f}." for v in vals}}
    return "choice", state, q, [lab(v) for v in vals], lab(fee)


TEMPLATES = [t_filing_deadline, t_sla, t_renewal, t_prorated, t_tz, t_latefee]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    out = 0
    while out < n:
        tpl = rng.choice(TEMPLATES)
        qtype, state, q, labels, gold = tpl(rng)
        if len(set(labels)) != len(labels) or gold not in labels or len(labels) < 2:
            continue
        rec = {"id": f"prog-temp-{out:05d}", "family": "temporal_numeric", "topic": "math & numbers",
               "state": state, "question": q, "labels": labels, "expected": gold,
               "teacher_probs": soft(gold, labels), "teacher_top": gold, "agree": True,
               "difficulty": "hard", "lang": "English", "source": f"prog_temp:{tpl.__name__}"}
        print(json.dumps(rec, ensure_ascii=False))
        out += 1


if __name__ == "__main__":
    main()
