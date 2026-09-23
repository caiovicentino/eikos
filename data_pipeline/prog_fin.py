"""Finance and trading items with exact gold answers (calculations done in code), with distractors that
reproduce common mistakes. Neutral labels. Global (USD/EUR/GBP/JPY/BRL).
usage: python prog_fin.py N SEED > prog_fin.jsonl
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys

CUR = [("USD", "$"), ("EUR", "€"), ("GBP", "£"), ("BRL", "R$"), ("JPY", "¥"), ("SGD", "S$")]
BANKS = ["Northgate Bank", "Arcadia Credit Union", "Banco Serra Azul", "Kestrel Capital", "Helix Brokerage",
         "Tidewater Payments", "Marlow & Finch", "Orion Exchange", "Sakura Trust", "Delta Clearing"]


def money(v, sym):
    return f"{sym}{v:,.2f}"


def lab_amt(v, code):
    return f"{code.lower()}_{v:.2f}".replace(".", "_").replace("-", "neg_")


def soft(gold, labels, p=0.95):
    o = (1 - p) / (len(labels) - 1)
    return {l: (p if l == gold else o) for l in labels}


def choice_amounts(rng, gold, wrongs, code, sym, what):
    vals = [round(gold, 2)] + [round(w, 2) for w in wrongs if abs(round(w, 2) - round(gold, 2)) > 0.004]
    uniq = []
    for v in vals:
        if all(abs(v - u) > 0.004 for u in uniq):
            uniq.append(v)
    uniq = uniq[:5]
    rng.shuffle(uniq)
    labels = [lab_amt(v, code) for v in uniq]
    crit = {lab_amt(v, code): f"The {what} is {money(v, sym)}." for v in uniq}
    return labels, crit, lab_amt(round(gold, 2), code)


def t_amortization(rng):
    code, sym = rng.choice(CUR[:4])
    P = rng.choice([5000, 12000, 25000, 40000, 80000, 150000])
    r_a = rng.choice([0.06, 0.09, 0.12, 0.18, 0.24])
    n = rng.choice([12, 24, 36, 48, 60])
    i = r_a / 12
    pmt = P * i / (1 - (1 + i) ** -n)
    wrongs = [P * r_a / (1 - (1 + r_a) ** -n),          # annual rate not divided by 12
              P / n + P * i,                            # 1st installment under constant amortization (SAC)
              P * (1 + r_a * n / 12) / n,               # simple interest
              P * i / (1 - (1 + i) ** -(n / 12))]       # term in years
    labels, crit, gold = choice_amounts(rng, pmt, wrongs, code, sym, "monthly installment")
    state = (f"{rng.choice(BANKS)} — loan offer. Principal {money(P, sym)}; nominal annual rate {r_a * 100:g}% "
             f"compounded monthly; {n} equal monthly installments (French/annuity system), first due one month "
             f"after disbursement. No fees. A marketing flyer mentions 'from {money(P / n, sym)} per month', which "
             f"ignores interest.")
    q = {"type": "choice", "instructions": "What is the monthly installment of this loan?", "criteria": crit}
    return "choice", state, q, labels, gold


def t_fx(rng):
    (c1, s1), (c2, s2) = rng.sample(CUR[:5], 2)
    amt = rng.choice([1000, 2500, 7800, 15000, 42000])
    mid = round(rng.uniform(0.2, 6.5), 4)
    spread = rng.choice([0.5, 1.0, 1.5, 2.0])
    fee = rng.choice([0, 5, 15, 25])
    rate = mid * (1 - spread / 100)
    recv = (amt - fee) * rate
    wrongs = [amt * mid, (amt - fee) * mid, amt * rate, (amt - fee) * mid * (1 + spread / 100)]
    labels, crit, gold = choice_amounts(rng, recv, wrongs, c2, s2, "amount credited")
    state = (f"Cross-border transfer at {rng.choice(BANKS)}: the customer sends {money(amt, s1)} ({c1}). The flat "
             f"fee of {money(fee, s1)} is deducted from the amount before conversion. Mid-market rate: 1 {c1} = "
             f"{mid} {c2}. The bank applies a {spread}% margin below the mid-market rate. The app preview showed the "
             f"mid-market conversion only.")
    q = {"type": "choice", "instructions": f"How much will the beneficiary receive in {c2}?", "criteria": crit}
    return "choice", state, q, labels, gold


def t_dti(rng):
    inc = rng.choice([3000, 4500, 6000, 8000, 12000])
    debts = rng.choice([400, 900, 1500, 2200, 3100])
    new_pmt = rng.choice([250, 400, 650, 900])
    max_dti = rng.choice([35, 40, 43, 45])
    score = rng.randint(560, 820)
    min_score = rng.choice([620, 660, 680, 700])
    late = rng.choice([0, 0, 0, 1, 2])
    dti = (debts + new_pmt) / inc * 100
    ok = dti <= max_dti and score >= min_score and late == 0
    code, sym = rng.choice(CUR[:4])
    state = (f"Underwriting policy UW-7: approve only if (a) debt-to-income, including the new payment, is at most "
             f"{max_dti}%; (b) credit score is at least {min_score}; (c) no payment 60+ days late in the last 12 "
             f"months.\nApplicant: gross monthly income {money(inc, sym)}; existing monthly debt payments "
             f"{money(debts, sym)}; requested loan payment {money(new_pmt, sym)}; credit score {score}; "
             f"60+ day late payments in last 12 months: {late}. The branch manager recommended approval.")
    q = {"type": "noul", "instructions": "Does the application meet policy UW-7?",
         "criteria": {"true": "All three conditions of UW-7 are met.", "false": "At least one condition fails."}}
    return "noul", state, q, ["no", "yes"], ("yes" if ok else "no")


def t_three_way(rng):
    code, sym = rng.choice(CUR[:4])
    qty_po = rng.choice([100, 250, 400, 1000])
    price_po = rng.choice([4.5, 12.0, 37.25, 89.9])
    qty_rcv = qty_po - rng.choice([0, 0, 0, 5, 20])
    qty_inv = rng.choice([qty_rcv, qty_rcv, qty_po])
    tol = rng.choice([1, 2, 5])
    price_inv = round(price_po * (1 + rng.choice([0, 0, 0.005, 0.015, 0.03, 0.08])), 2)
    var = (price_inv - price_po) / price_po * 100
    if qty_inv != qty_rcv:
        gold = "hold_quantity_mismatch"
    elif var > tol:
        gold = "hold_price_variance"
    else:
        gold = "approve_payment"
    labels = ["approve_payment", "hold_quantity_mismatch", "hold_price_variance", "reject_invoice"]
    crit = {"approve_payment": "Invoice matches the goods receipt quantity and the PO price within tolerance.",
            "hold_quantity_mismatch": "Invoiced quantity differs from the quantity received.",
            "hold_price_variance": "Unit price exceeds the PO price by more than the tolerance.",
            "reject_invoice": "The invoice does not reference a valid purchase order."}
    state = (f"Accounts payable three-way match (policy AP-3). Price tolerance: +{tol}% over PO unit price; "
             f"quantity must equal the goods-receipt quantity; quantity checks come before price checks.\n"
             f"PO-{rng.randint(10000, 99999)}: {qty_po} units at {money(price_po, sym)}.\n"
             f"Goods receipt: {qty_rcv} units received.\nInvoice (references the PO): {qty_inv} units at "
             f"{money(price_inv, sym)} per unit. The supplier's cover email says 'all as ordered'.")
    q = {"type": "choice", "instructions": "What should accounts payable do with this invoice?", "criteria": crit}
    return "choice", state, q, labels, gold


def t_structuring(rng):
    thr = rng.choice([10000, 15000])
    lo = int(thr * 0.8)
    days = rng.choice([5, 7, 10])
    base = dt.date(2026, rng.randint(1, 11), rng.randint(1, 20))
    k = rng.randint(2, 5)
    deps = []
    for _ in range(k):
        d = base + dt.timedelta(days=rng.randint(0, days + 6))
        a = rng.choice([rng.randint(lo, thr - 1), rng.randint(lo, thr - 1), rng.randint(1000, lo - 1)])
        deps.append((d, a))
    deps.sort()
    flagged = False
    for i, (d0, _) in enumerate(deps):
        win = [a for d, a in deps if 0 <= (d - d0).days < days and lo <= a < thr]
        if len(win) >= 3:
            flagged = True
    code, sym = rng.choice(CUR[:4])
    dep_txt = "\n".join(f"- {d.isoformat()}: cash deposit {money(a, sym)}" for d, a in deps)
    state = (f"AML rule R-12 (structuring): open an alert when an account receives at least 3 cash deposits, each "
             f"between {money(lo, sym)} and {money(thr - 0.01, sym)}, within any window of {days} consecutive "
             f"days. Deposits outside that band do not count.\nAccount activity:\n{dep_txt}\nThe customer says the "
             f"deposits are 'weekly sales from the shop'.")
    q = {"type": "noul", "instructions": "Does rule R-12 require an alert?",
         "criteria": {"true": "The deposits satisfy rule R-12.", "false": "The deposits do not satisfy rule R-12."}}
    return "noul", state, q, ["no", "yes"], ("yes" if flagged else "no")


def t_liquidation(rng):
    side = rng.choice(["long", "short"])
    entry = rng.choice([100.0, 250.0, 1800.0, 27000.0, 64000.0])
    lev = rng.choice([2, 3, 5, 10, 20])
    mmr = rng.choice([0.005, 0.01, 0.02])
    # liquidation price (isolated margin, no fees): long = entry*(1-1/lev+mmr); short = entry*(1+1/lev-mmr)
    liq = entry * (1 - 1 / lev + mmr) if side == "long" else entry * (1 + 1 / lev - mmr)
    path_ext = liq * (1 + rng.choice([-0.02, -0.005, 0.005, 0.03])) if side == "long" else \
        liq * (1 + rng.choice([0.02, 0.005, -0.005, -0.03]))
    hit = path_ext <= liq if side == "long" else path_ext >= liq
    word = "low" if side == "long" else "high"
    op, cl = entry * (1 + rng.uniform(-0.01, 0.01)), entry * (1 + rng.uniform(-0.02, 0.02))
    if side == "long":  # consistency: the day's low cannot be above the open/close
        op, cl = max(op, path_ext), max(cl, path_ext)
    else:
        op, cl = min(op, path_ext), min(cl, path_ext)
    state = (f"{rng.choice(BANKS)} perpetual futures, isolated margin. A trader opened a {side} position at "
             f"{entry:,.2f} with {lev}x leverage. Maintenance margin rate: {mmr * 100:g}%. Ignore fees and "
             f"funding. Liquidation happens when the mark price reaches the liquidation price, computed as "
             f"entry × (1 − 1/leverage + MMR) for longs and entry × (1 + 1/leverage − MMR) for shorts.\n"
             f"Mark price path today: open {op:,.2f}, intraday {word} "
             f"{path_ext:,.2f}, close {cl:,.2f}. A community channel claimed "
             f"'liquidations only happen at the close', which is false for this venue.")
    q = {"type": "noul", "instructions": "Was the position liquidated today?",
         "criteria": {"true": "The mark price reached the liquidation price.", "false": "It never reached it."}}
    return "noul", state, q, ["no", "yes"], ("yes" if hit else "no")


def t_pretrade(rng):
    sym_ = rng.choice(["ACME", "NVX", "PETR4", "7203.T", "BARC.L", "SAP.DE"])
    pos = rng.choice([0, 2000, 8000, 15000])
    qty = rng.choice([500, 1000, 5000, 9000])
    px = rng.choice([12.5, 48.0, 150.0, 910.0])
    pos_lim = rng.choice([10000, 20000])
    notional_lim = rng.choice([500000, 1000000, 2000000])
    restricted = rng.random() < 0.2
    if restricted:
        gold = "reject_restricted_list"
    elif pos + qty > pos_lim:
        gold = "reject_position_limit"
    elif qty * px > notional_lim:
        gold = "reject_notional_limit"
    else:
        gold = "accept_order"
    labels = ["accept_order", "reject_restricted_list", "reject_position_limit", "reject_notional_limit"]
    crit = {"accept_order": "The order passes every pre-trade check.",
            "reject_restricted_list": "The instrument is on the restricted list.",
            "reject_position_limit": "Resulting position would exceed the position limit.",
            "reject_notional_limit": "The order's notional value exceeds the per-order notional limit."}
    state = (f"Pre-trade risk engine, checks in this order: (1) restricted list, (2) position limit of {pos_lim:,} "
             f"shares after the order, (3) per-order notional limit of {notional_lim:,}.\n"
             f"Restricted list today: {sym_ + ', ' if restricted else ''}ZZTOP, QXR.\n"
             f"Buy order: {qty:,} {sym_} at limit {px}; current position {pos:,} shares. The trader notes the "
             f"desk 'has always traded this name'.")
    q = {"type": "choice", "instructions": "What is the pre-trade decision for this order?", "criteria": crit}
    return "choice", state, q, labels, gold


def t_pnl(rng):
    code, sym = rng.choice(CUR[:4])
    q1, p1 = rng.choice([100, 300, 1000]), rng.choice([20.0, 55.5, 120.0])
    p2 = round(p1 * (1 + rng.choice([-0.08, -0.03, 0.02, 0.06, 0.12])), 2)
    fee_bps = rng.choice([5, 10, 25])
    fees = (q1 * p1 + q1 * p2) * fee_bps / 10000
    pnl = (p2 - p1) * q1 - fees
    wrongs = [(p2 - p1) * q1, (p2 - p1) * q1 - q1 * p1 * fee_bps / 10000, (p2 - p1) * q1 + fees,
              (p2 - p1) * q1 - fees * 2]
    labels, crit, gold = choice_amounts(rng, pnl, wrongs, code, sym, "realized P&L")
    state = (f"Trade blotter: bought {q1} shares at {money(p1, sym)}, later sold all {q1} at {money(p2, sym)}. "
             f"Commission: {fee_bps} basis points charged on the notional of EACH execution (buy and sell). "
             f"No other costs. The broker's summary screen shows gross P&L only.")
    q = {"type": "choice", "instructions": "What is the realized P&L net of commissions?", "criteria": crit}
    return "choice", state, q, labels, gold


def t_settlement(rng):
    trade = dt.date(2026, rng.randint(1, 11), rng.randint(1, 26))
    cycle = rng.choice([1, 2])
    hol = set()
    for _ in range(rng.randint(0, 2)):
        h = trade + dt.timedelta(days=rng.randint(1, 5))
        if h.weekday() < 5:
            hol.add(h)
    d, k = trade, 0
    while k < cycle:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            k += 1
    cands = {d, trade + dt.timedelta(days=cycle), d + dt.timedelta(days=1), d - dt.timedelta(days=1)} - {trade}
    opts = sorted(cands)
    rng.shuffle(opts)
    lab = lambda x: "d_" + x.strftime("%Y_%m_%d")  # noqa: E731
    crit = {lab(o): f"Settlement on {o.strftime('%A %d %B %Y')}." for o in opts}
    htxt = ", ".join(h.strftime("%A %d %B %Y") for h in sorted(hol)) or "none"
    state = (f"Equity trade executed {trade.strftime('%A %d %B %Y')}. Market settlement cycle: T+{cycle} business "
             f"days; business days exclude weekends and exchange holidays. Exchange holidays in the period: {htxt}. "
             f"(The venue moved from T+{cycle + 1} to T+{cycle} last year.)")
    q = {"type": "choice", "instructions": "On which date does the trade settle?", "criteria": crit}
    return "choice", state, q, [lab(o) for o in opts], lab(d)


def t_option_expiry(rng):
    kind = rng.choice(["call", "put"])
    strike = rng.choice([50, 100, 250, 1000])
    settle = round(strike * (1 + rng.choice([-0.05, -0.01, -0.0001, 0.0, 0.00005, 0.0002, 0.01, 0.04])), 4)
    itm = settle - strike if kind == "call" else strike - settle
    auto = itm >= 0.01 - 1e-9  # floating-point tolerance at the $0.01 boundary
    state = (f"Clearing rule: at expiration, equity options that are in the money by at least $0.01 based on the "
             f"official settlement price are automatically exercised unless the holder files a contrary "
             f"instruction. No contrary instruction was filed.\nPosition: long 1 {kind}, strike {strike}. "
             f"Official settlement price: {settle}. The last trade printed at {round(settle * 1.002, 4)}, but only "
             f"the official settlement price counts.")
    q = {"type": "noul", "instructions": "Is the option automatically exercised?",
         "criteria": {"true": "It is in the money by at least $0.01 at the settlement price.",
                      "false": "It is not in the money by at least $0.01."}}
    return "noul", state, q, ["no", "yes"], ("yes" if auto else "no")


TEMPLATES = [t_amortization, t_fx, t_dti, t_three_way, t_structuring, t_liquidation, t_pretrade, t_pnl,
             t_settlement, t_option_expiry]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 9)
    out = 0
    cnt = {"yes": 0, "no": 0}
    while out < n:
        tpl = rng.choice(TEMPLATES)
        qtype, state, q, labels, gold = tpl(rng)
        if len(labels) < 2 or len(set(labels)) != len(labels) or gold not in labels:
            continue
        if qtype == "noul":  # balance yes/no (without this, ~2:1 in favor of "no")
            if cnt[gold] > cnt["yes" if gold == "no" else "no"] + 5:
                continue
            cnt[gold] += 1
        rec = {"id": f"prog-fin-{out:05d}", "family": "finance_exact", "topic": "finance",
               "state": state, "question": q, "labels": labels, "expected": gold,
               "teacher_probs": soft(gold, labels), "teacher_top": gold, "agree": True,
               "difficulty": "hard", "lang": "English", "source": f"prog_fin:{tpl.__name__}"}
        print(json.dumps(rec, ensure_ascii=False))
        out += 1


if __name__ == "__main__":
    main()
