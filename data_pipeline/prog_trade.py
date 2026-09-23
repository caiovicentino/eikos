"""Trading, markets and foreign trade with exact gold answers (calculations and rules applied in code).
The rule is always written in the state, in our own words: the model learns to apply given rules, not to
memorize laws (which change by country and by year). Distractors reproduce common mistakes (calendar vs business
day, forgetting a fee, wrong day-count basis, wrong side of the Incoterm...). Yes/no balanced by construction.
Languages: English and Portuguese (Spanish is held out of training to measure language generalization).
usage: python prog_trade.py N SEED > prog_trade.jsonl
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys

CUR = [("USD", "$"), ("EUR", "€"), ("GBP", "£"), ("JPY", "¥"), ("SGD", "S$"), ("INR", "₹"), ("HKD", "HK$")]
FIRMS = ["Northgate Securities", "Helix Brokerage", "Kestrel Capital Markets", "Orion Exchange", "Sakura Trust",
         "Marlow & Finch", "Tidewater Clearing", "Banyan Wealth", "Aurora Prime", "Meridian Trade Bank"]
PEOPLE = ["A. Okafor", "L. Moreau", "K. Tanaka", "R. Mehta", "S. Lindqvist", "M. Oliveira", "D. Chen", "F. Rossi",
          "J. Novak", "P. Adeyemi", "H. Kim", "T. Walsh"]


def money(v, sym):
    return f"{sym}{v:,.2f}"


def lab_amt(v, code):
    return f"{code.lower()}_{v:.2f}".replace(".", "_").replace("-", "neg_")


def soft(gold, labels, p=0.95):
    o = (1 - p) / (len(labels) - 1)
    return {l: (p if l == gold else o) for l in labels}


def choice_amounts(rng, gold, wrongs, code, sym, what):
    uniq = []
    for v in [round(gold, 2)] + [round(w, 2) for w in wrongs]:
        if all(abs(v - u) > 0.004 for u in uniq):
            uniq.append(v)
    uniq = uniq[:5]
    rng.shuffle(uniq)
    crit = {lab_amt(v, code): f"The {what} is {money(v, sym)}." for v in uniq}
    return [lab_amt(v, code) for v in uniq], crit, lab_amt(round(gold, 2), code)


def noul(true_txt, false_txt, instr):
    return {"type": "noul", "instructions": instr, "criteria": {"true": true_txt, "false": false_txt}}


def bdays_after(d, n, hol):
    k = 0
    while k < n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            k += 1
    return d


def fmt_d(d):
    return d.strftime("%a %d %b %Y")


# ---------------------------------------------------------------- markets
def t_maintenance_call(rng, want):
    code, sym = rng.choice(CUR[:3])
    qty = rng.choice([200, 500, 1000, 2500])
    buy = rng.choice([40.0, 85.0, 120.0, 310.0])
    loan = round(qty * buy * rng.choice([0.4, 0.5]), 2)
    req = rng.choice([0.25, 0.30, 0.35])
    # cutoff price: equity/market = req  ->  (P*qty - loan)/(P*qty) = req  ->  P = loan / (qty*(1-req))
    cut = loan / (qty * (1 - req))
    px = round(cut * (rng.uniform(0.90, 0.99) if want == "yes" else rng.uniform(1.01, 1.12)), 2)
    mv = px * qty
    eq = mv - loan
    state = (f"{rng.choice(FIRMS)} margin policy: a maintenance call is issued when account equity falls BELOW "
             f"{req * 100:g}% of the current market value of the long position (equity = market value - debit "
             f"balance). Account: long {qty} shares bought at {money(buy, sym)}; debit balance {money(loan, sym)}; "
             f"no other positions or cash. Closing price today: {money(px, sym)}. The client notes the position is "
             f"'only down {abs(px / buy - 1) * 100:.1f}% from cost'.")
    q = noul("Equity is below the maintenance percentage, so a call is issued.",
             "Equity is at or above the maintenance percentage.", "Is a maintenance margin call issued today?")
    return "noul", state, q, ["no", "yes"], ("yes" if eq < req * mv else "no")


def t_pdt(rng, want):
    days = [d for d in (dt.date(2026, 3, 2) + dt.timedelta(days=i) for i in range(10)) if d.weekday() < 5][:5]
    if want == "yes":
        n_dt = rng.choice([4, 5, 6])
        total = rng.randint(n_dt + 5, int(n_dt / 0.061))
    elif rng.random() < 0.5:
        n_dt = rng.choice([2, 3])
        total = rng.randint(n_dt + 5, 60)
    else:  # 4+ day trades, but no more than 6% of the total
        n_dt = rng.choice([4, 5])
        total = rng.randint(int(n_dt / 0.06) + 1, 140)
    per = [0] * 5
    for _ in range(n_dt):
        per[rng.randrange(5)] += 1
    log = "\n".join(f"{fmt_d(d)}: {k} day trade(s)" for d, k in zip(days, per))
    flagged = n_dt >= 4 and n_dt / total > 0.06
    state = (f"Broker rule (margin accounts): the account is flagged as a pattern day trader if it makes FOUR or more "
             f"day trades within five consecutive business days AND those day trades are more than 6% of all its "
             f"trades in that period. Activity over the five business days below: {total} trades in total, of which "
             f"the day trades were:\n{log}")
    q = noul("The account meets both conditions and is flagged.", "At least one condition is not met.",
             "Is the account flagged as a pattern day trader?")
    return "noul", state, q, ["no", "yes"], ("yes" if flagged else "no")


def t_wash_sale(rng, want):
    code, sym = rng.choice(CUR[:3])
    sell = dt.date(2026, rng.randint(2, 10), rng.randint(1, 25))
    gap = rng.randint(3, 29) if want == "yes" else rng.randint(31, 45)
    before = rng.random() < 0.35
    rebuy = sell - dt.timedelta(days=gap) if before else sell + dt.timedelta(days=gap)
    loss = rng.choice([850, 2300, 4100, 12500])
    state = (f"Tax rule used by the firm's tax report: a loss on a sale is disallowed (wash sale) if substantially "
             f"identical shares are bought within 30 calendar days BEFORE or AFTER the sale date. Client "
             f"{rng.choice(PEOPLE)} sold 300 shares of QRX on {fmt_d(sell)} at a loss of {money(loss, sym)} and "
             f"bought 300 QRX shares on {fmt_d(rebuy)}. The client says 'the repurchase was in a different month'.")
    q = noul("The repurchase falls within 30 calendar days of the sale, so the loss is disallowed.",
             "No repurchase within 30 calendar days before or after the sale.", "Is the loss disallowed as a wash sale?")
    return "noul", state, q, ["no", "yes"], ("yes" if gap <= 30 else "no")


def t_price_band(rng, want):
    code, sym = rng.choice(CUR[:3])
    ref = rng.choice([4.2, 18.5, 64.0, 150.0, 920.0])
    band = 0.20 if ref < 3 else (0.10 if ref <= 75 else 0.05)
    lim = ref * band
    move = lim * (rng.uniform(0.5, 0.97) if want == "yes" else rng.uniform(1.03, 1.6))
    px = round(ref + move * rng.choice([1, -1]), 2)
    state = (f"Venue price-band rule: an order may only execute within ±{band * 100:g}% of the reference price "
             f"(bands: 20% for reference prices up to {sym}3.00, 10% from {sym}3.00 to {sym}75.00, 5% above "
             f"{sym}75.00). Reference price: {money(ref, sym)}. Incoming order: limit {money(px, sym)}, marketable "
             f"at that price.")
    ok = abs(px - ref) <= lim + 1e-9
    q = noul("The price is inside the band, so it may execute.", "The price is outside the band.",
             "May the order execute at its limit price?")
    return "noul", state, q, ["no", "yes"], ("yes" if ok else "no")


def t_stop_fill(rng):
    code, sym = rng.choice(CUR[:3])
    entry = rng.choice([50.0, 120.0, 310.0])
    stop = round(entry * rng.choice([0.95, 0.97, 0.98]), 2)
    gap_open = rng.random() < 0.5
    bars = []
    p = entry
    for i in range(3):
        o = max(p, round(stop * 1.012, 2))
        lo = max(round(o * rng.uniform(0.985, 0.998), 2), round(stop * 1.005, 2))
        hi = round(o * rng.uniform(1.002, 1.01), 2)
        c = round(rng.uniform(lo, hi), 2)
        bars.append([o, hi, lo, c])
        p = c
    if gap_open:
        o = round(stop * rng.uniform(0.95, 0.985), 2)
        bars.append([o, round(o * 1.01, 2), round(o * 0.99, 2), round(o * 1.004, 2)])
        fill = o
    else:
        o = round(stop * 1.01, 2)
        bars.append([o, round(o * 1.004, 2), round(stop * 0.99, 2), round(stop * 0.995, 2)])
        fill = stop
    rows = "\n".join(f"Bar {i + 1}: open {b[0]}, high {b[1]}, low {b[2]}, close {b[3]}" for i, b in enumerate(bars))
    state = (f"Long position entered at {money(entry, sym)} with a sell STOP (market) order at {money(stop, sym)}. "
             f"Execution model: a stop triggers when a bar trades at or below the stop; if a bar OPENS below the stop, "
             f"the fill is that bar's open price; otherwise the fill is the stop price. No slippage beyond that.\n{rows}")
    labels, crit, gold = choice_amounts(rng, fill, [stop, bars[-1][2], bars[-1][3], entry], code, sym, "exit fill price")
    q = {"type": "choice", "instructions": "At what price is the position closed?", "criteria": crit}
    return "choice", state, q, labels, gold


def t_accrued(rng):
    code, sym = rng.choice(CUR[:3])
    face = rng.choice([100000, 250000, 1000000])
    cpn = rng.choice([0.035, 0.045, 0.0525, 0.06])
    last = dt.date(2026, rng.choice([1, 3, 4, 7]), rng.choice([15, 1, 30]))
    settle = last + dt.timedelta(days=rng.randint(20, 150))
    conv = rng.choice(["ACT/360", "ACT/365", "30/360"])

    def d30(a, b):  # 30E/360, as the item text states: any day 31 counts as day 30
        d1, d2 = min(a.day, 30), min(b.day, 30)
        return 360 * (b.year - a.year) + 30 * (b.month - a.month) + (d2 - d1)
    act = (settle - last).days
    days = {"ACT/360": act, "ACT/365": act, "30/360": d30(last, settle)}
    base = {"ACT/360": 360, "ACT/365": 365, "30/360": 360}
    ai = face * cpn * days[conv] / base[conv]
    wrongs = [face * cpn * days[c] / base[c] for c in base if c != conv] + [face * cpn * (act + 1) / base[conv],
                                                                         face * cpn / 2 * days[conv] / base[conv]]
    labels, crit, gold = choice_amounts(rng, ai, wrongs, code, sym, "accrued interest")
    state = (f"Bond trade: face amount {money(face, sym)}, annual coupon {cpn * 100:g}% (paid annually), day-count "
             f"convention {conv}. Last coupon date {fmt_d(last)}; settlement date {fmt_d(settle)}. Accrued interest = "
             f"face × coupon × (days from last coupon to settlement under the convention) / (year basis of the "
             f"convention: 360 for ACT/360 and 30/360, 365 for ACT/365). ACT counts actual calendar days; 30/360 counts every "
             f"month as 30 days (a day 31 is treated as day 30).")
    q = {"type": "choice", "instructions": "How much accrued interest does the buyer pay?", "criteria": crit}
    return "choice", state, q, labels, gold


def t_concentration(rng, want):
    code, sym = rng.choice(CUR[:4])
    total = rng.choice([400000, 1200000, 5000000])
    lim = rng.choice([0.05, 0.10, 0.15])
    cur = total * lim * rng.uniform(0.3, 0.8)
    room = total * lim - cur
    add = room * (rng.uniform(0.4, 0.95) if want == "yes" else rng.uniform(1.05, 1.8))
    add = round(add, -2) if add > 1000 else round(add, 2)
    after = (cur + add) / (total + 0)  # purchase paid with the portfolio's own cash: total unchanged
    state = (f"Investment mandate: no single issuer may exceed {lim * 100:g}% of total portfolio value AFTER a trade "
             f"(the purchase is paid from cash already in the portfolio, so total value does not change). Portfolio "
             f"total: {money(total, sym)}. Current holding of issuer Veltrix: {money(cur, sym)}. Proposed purchase of "
             f"Veltrix: {money(add, sym)}.")
    q = noul("After the trade the issuer weight is within the limit.", "The issuer weight would exceed the limit.",
             "Does the proposed purchase comply with the concentration limit?")
    return "noul", state, q, ["no", "yes"], ("yes" if after <= lim + 1e-12 else "no")


def t_limit_fill(rng):
    code, sym = rng.choice(CUR[:3])
    lim = rng.choice([25.0, 48.5, 102.0])
    side = rng.choice(["buy", "sell"])
    asks = [round(lim * (1 + rng.choice([-0.02, -0.01, 0.005, 0.01, 0.02])), 2) for _ in range(5)]
    ok = [a for a in asks if (a <= lim if side == "buy" else a >= lim)]
    gold_lab = "not_filled" if not ok else f"fill_at_quote_{asks.index(ok[0]) + 1}"
    labels = ["not_filled"] + [f"fill_at_quote_{i + 1}" for i in range(5)]
    crit = {"not_filled": "No quote in the sequence satisfies the limit, so the order does not fill."}
    word = "ask" if side == "buy" else "bid"
    for i in range(5):
        crit[f"fill_at_quote_{i + 1}"] = f"The order fills at quote {i + 1} ({word} {money(asks[i], sym)})."
    seq = "\n".join(f"Quote {i + 1}: best {word} {money(a, sym)}" for i, a in enumerate(asks))
    state = (f"A {side} LIMIT order for 100 shares at {money(lim, sym)} is resting from before Quote 1. A {side} "
             f"limit fills at the FIRST quote whose best {word} is {'at or below' if side == 'buy' else 'at or above'} "
             f"the limit (enough size at every quote). Quotes, in time order:\n{seq}")
    q = {"type": "choice", "instructions": "When does the order fill?", "criteria": crit}
    return "choice", state, q, labels, gold_lab


# ---------------------------------------------------------------- foreign trade / trade finance
INCOTERMS = {  # risk transfer point (summary in our own words)
    "EXW": "when the goods are placed at the buyer's disposal at the seller's premises",
    "FCA": "when the goods are delivered to the carrier nominated by the buyer",
    "CPT": "when the goods are handed to the first carrier",
    "CIP": "when the goods are handed to the first carrier",
    "FAS": "when the goods are placed alongside the ship at the port of shipment",
    "FOB": "when the goods are on board the ship at the port of shipment",
    "CFR": "when the goods are on board the ship at the port of shipment",
    "CIF": "when the goods are on board the ship at the port of shipment",
    "DAP": "when the goods arrive at the named destination, ready for unloading",
    "DPU": "when the goods are unloaded at the named destination",
    "DDP": "when the goods arrive at the named destination, ready for unloading, duties paid",
}
EVENTS = [  # (description, stage). The buyer bears the loss iff stage >= STAGE[term]
    ("while stored in the seller's warehouse, before being placed at the buyer's disposal", 0),
    ("at the seller's premises, after being placed at the buyer's disposal and before any carrier collected them", 1),
    ("in the depot of the first carrier, after the seller had handed the goods over to that carrier{fca}", 2),
    ("on the quay alongside the vessel at the port of shipment, before loading", 3),
    ("at sea, after loading on board", 4),
    ("at the named destination, still on the arriving means of transport, ready for unloading", 5),
    ("at the named destination, after unloading", 6),
]
STAGE = {"EXW": 1, "FCA": 2, "CPT": 2, "CIP": 2, "FAS": 3, "FOB": 4, "CFR": 4, "CIF": 4, "DAP": 5, "DDP": 5, "DPU": 6}


def t_incoterm(rng):
    term = rng.choice(list(INCOTERMS))
    ev, st = rng.choice(EVENTS)
    ev = ev.format(fca=" (the carrier nominated by the buyer)" if term == "FCA" else "")
    table = "; ".join(f"{k}: risk passes {v}" for k, v in INCOTERMS.items())
    state = (f"Sales contract between {rng.choice(FIRMS)} (seller) and Qamar Trading (buyer) on {term} terms. Risk "
             f"transfer points used in the contract (summary): {table}. Loss event: the goods were damaged {ev}. "
             f"Who paid the freight or the insurance does not change who bears the risk.")
    labels = ["buyer", "seller"]
    crit = {"buyer": "Risk had already passed to the buyer when the loss happened.",
            "seller": "Risk was still with the seller when the loss happened."}
    q = {"type": "choice", "instructions": "Who bears the risk of this loss under the contract terms?", "criteria": crit}
    return "choice", state, q, labels, ("buyer" if st >= STAGE[term] else "seller")


def t_lc_presentation(rng, want):
    ship = dt.date(2026, rng.randint(1, 10), rng.randint(1, 20))
    days = rng.choice([14, 21])
    expiry = ship + dt.timedelta(days=rng.randint(10, 40))
    deadline = min(ship + dt.timedelta(days=days), expiry)
    pres = deadline - dt.timedelta(days=rng.randint(0, 6)) if want == "yes" else deadline + dt.timedelta(days=rng.randint(1, 5))
    state = (f"Documentary credit terms (summary): documents must be presented no later than {days} calendar days "
             f"after the date of shipment AND in any case not later than the expiry date of the credit. Date of "
             f"shipment on the bill of lading: {fmt_d(ship)}. Credit expiry: {fmt_d(expiry)}. Documents presented to "
             f"the nominated bank on: {fmt_d(pres)}. The bank then takes up to 5 banking days to examine them, which "
             f"does not change the presentation deadline.")
    q = noul("The presentation date meets both deadlines.", "The presentation is late for at least one deadline.",
             "Was the presentation made on time?")
    return "noul", state, q, ["no", "yes"], ("yes" if pres <= deadline else "no")


def t_vat(rng):
    code, sym = ("EUR", "€")
    net = rng.choice([1200, 8400, 15600, 42000])
    rate = rng.choice([0.19, 0.20, 0.21, 0.23])
    case = rng.choice(["b2b_cross", "b2c_cross", "domestic"])
    seller_c, buyer_c = rng.sample(["Germany", "France", "Netherlands", "Portugal", "Italy", "Ireland"], 2)
    if case == "domestic":
        buyer_c = seller_c
    gold = {"b2b_cross": "reverse_charge_no_vat", "b2c_cross": "charge_vat", "domestic": "charge_vat"}[case]
    who = "a VAT-registered business (valid VAT number verified)" if case != "b2c_cross" else "a private individual"
    state = (f"Invoice rule used by the ERP: for goods sold between two EU member states to a VAT-registered business "
             f"with a verified VAT number, the seller issues the invoice WITHOUT VAT and marks it 'reverse charge'; "
             f"in all other cases covered here the seller charges VAT at the seller-country rate. Seller established "
             f"in {seller_c}; buyer is {who} in {buyer_c}; goods shipped from {seller_c} to {buyer_c}. Net amount "
             f"{money(net, sym)}; seller-country VAT rate {rate * 100:g}%.")
    labels = ["reverse_charge_no_vat", "charge_vat", "exempt_export"]
    crit = {"reverse_charge_no_vat": "Invoice without VAT, marked reverse charge.",
            "charge_vat": "Invoice with VAT at the seller-country rate.",
            "exempt_export": "Exempt export outside the EU."}
    q = {"type": "choice", "instructions": "How must this invoice be issued?", "criteria": crit}
    return "choice", state, q, labels, gold


def t_gst(rng):
    code, sym = ("INR", "₹")
    val = rng.choice([48000, 125000, 560000])
    rate = rng.choice([0.05, 0.12, 0.18, 0.28])
    states = ["Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat", "Delhi", "Telangana"]
    s1 = rng.choice(states)
    s2 = rng.choice(states) if rng.random() < 0.5 else s1
    inter = s1 != s2
    tax = val * rate
    lab = {"igst": f"igst_{int(round(tax))}", "split": f"cgst_sgst_{int(round(tax / 2))}_each",
           "igst_half": f"igst_{int(round(tax / 2))}", "split_full": f"cgst_sgst_{int(round(tax))}_each"}
    labels = list(dict.fromkeys(lab.values()))
    rng.shuffle(labels)
    crit = {lab["igst"]: f"Integrated tax (IGST) of {money(tax, sym)}.",
            lab["split"]: f"Central + state tax (CGST + SGST) of {money(tax / 2, sym)} each.",
            lab["igst_half"]: f"Integrated tax (IGST) of {money(tax / 2, sym)}.",
            lab["split_full"]: f"Central + state tax (CGST + SGST) of {money(tax, sym)} each."}
    state = (f"Invoice rule (summary): when supplier and place of supply are in DIFFERENT states, charge integrated tax "
             f"(IGST) at the full rate; when they are in the SAME state, split the rate equally into central tax "
             f"(CGST) and state tax (SGST). Supplier registered in {s1}; place of supply {s2}. Taxable value "
             f"{money(val, sym)}; applicable rate {rate * 100:g}%.")
    q = {"type": "choice", "instructions": "Which tax lines must the invoice show?", "criteria": crit}
    return "choice", state, q, labels, (lab["igst"] if inter else lab["split"])


def t_sanctions(rng):
    listed = rng.choice([("Viktor Aleksandrovich Morozov", "1971-04-12", "Russia"),
                         ("Omar Farouk Al-Tamimi", "1980-09-03", "Iraq"),
                         ("Chen Weiliang", "1966-01-27", "China"),
                         ("Maria Esperanza Duarte", "1975-11-30", "Venezuela")])
    name, dob, nat = listed
    case = rng.choice(["exact", "variant_same_dob", "variant_diff_dob", "different"])
    if case == "exact":
        cand = (name, dob, nat)
    elif case == "variant_same_dob":
        parts = name.split()
        cand = (f"{parts[0]} {parts[-1]}", dob, nat)
    elif case == "variant_diff_dob":
        parts = name.split()
        cand = (f"{parts[0]} {parts[-1]}", f"{int(dob[:4]) + rng.choice([9, 14, 21])}{dob[4:]}", nat)
    else:
        cand = (rng.choice(PEOPLE) + " Jr.", "1990-06-01", "Canada")
    gold = {"exact": "confirmed_match", "variant_same_dob": "confirmed_match",
            "variant_diff_dob": "false_positive", "different": "no_match"}[case]
    state = (f"Screening procedure: a hit is a CONFIRMED MATCH if the name matches the listed entry (full name or the "
             f"listed first + last name) AND the date of birth matches; it is a FALSE POSITIVE if the name matches but "
             f"the date of birth clearly differs; NO MATCH if the name does not match any listed entry.\nListed entry: "
             f"{name}, born {dob}, nationality {nat}.\nCustomer being onboarded: {cand[0]}, born {cand[1]}, "
             f"nationality {cand[2]}.")
    labels = ["confirmed_match", "false_positive", "no_match"]
    crit = {"confirmed_match": "Name and date of birth match the listed entry.",
            "false_positive": "The name matches but the date of birth clearly differs.",
            "no_match": "The name does not match the listed entry."}
    q = {"type": "choice", "instructions": "How should the screening hit be resolved?", "criteria": crit}
    return "choice", state, q, labels, gold


def t_suitability(rng, want):
    prof = rng.randint(1, 5)
    need = rng.choice([1, 3, 5])
    mode = "ok" if want == "yes" else rng.choice(["risk", "horizon", "both"])
    prod = prof - rng.randint(0, prof - 1) if mode in ("ok", "horizon") else min(7, prof + rng.randint(1, 3))
    horizon = need + rng.randint(0, 4) if mode in ("ok", "risk") else rng.randint(0, need - 1)
    ok = prod <= prof and horizon >= need
    state = (f"Suitability rule: a product may be recommended only if its risk class (1 = lowest, 7 = highest) is NOT "
             f"above the client's risk profile score (profiles 1-5 use the same scale) AND the client's investment "
             f"horizon is at least the product's minimum holding period. Client {rng.choice(PEOPLE)}: risk profile "
             f"{prof}, investment horizon {horizon} years. Product: risk class {prod}, minimum holding period {need} "
             f"years. The client asked for 'the highest-yield option'.")
    q = noul("Both conditions hold, so the product may be recommended.", "At least one condition fails.",
             "May the adviser recommend this product?")
    return "noul", state, q, ["no", "yes"], ("yes" if ok else "no")


def brl(v):
    return "R$" + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def t_pix_noturno(rng, want):
    lim = rng.choice([1000, 500, 2000])
    noturno = rng.random() < 0.8
    hora = rng.choice(["20:15", "21:40", "23:05", "02:30", "05:50"]) if noturno else rng.choice(["19:55", "06:05", "14:30"])
    ja = round(lim * rng.choice([0, 0.3, 0.6, 0.9]), 2) if noturno else 0.0
    if noturno:
        valor = round((lim - ja) * rng.uniform(0.3, 0.98), 2) if want == "yes" else round(lim - ja + rng.uniform(20, 900), 2)
    else:
        valor = round(rng.uniform(lim * 1.2, 20000), 2)  # above the night limit, but by day the daytime limit applies
    bloqueia = noturno and (ja + valor > lim)
    state = (f"Regra do banco para pessoa física: entre 20h00 e 6h00 as transferências Pix somam no máximo "
             f"{brl(lim)} por período noturno (a soma inclui o que já foi enviado nesse período); fora desse horário "
             f"vale o limite diurno de {brl(50000)}. O cliente já enviou {brl(ja)} no período noturno atual. "
             f"Novo Pix de {brl(valor)} solicitado às {hora}.")
    q = noul("O Pix respeita o limite aplicável e é autorizado.", "O Pix ultrapassa o limite aplicável e é bloqueado.",
             "O novo Pix é autorizado?")
    return "noul", state, q, ["no", "yes"], ("no" if bloqueia else "yes")


def t_chargeback(rng, want):
    code, sym = rng.choice(CUR[:3])
    win = rng.choice([60, 90, 120])
    txn = dt.date(2026, rng.randint(1, 6), rng.randint(1, 28))
    deliv = txn + dt.timedelta(days=rng.randint(2, 20))
    base_is_delivery = rng.random() < 0.6
    base = deliv if base_is_delivery else txn
    filed = base + dt.timedelta(days=rng.randint(10, win) if want == "yes" else rng.randint(win + 1, win + 30))
    state = (f"Card scheme rule for 'goods not as described': the cardholder may dispute within {win} calendar days "
             f"from the {'delivery date' if base_is_delivery else 'transaction date'}. Transaction date {fmt_d(txn)}; "
             f"delivery date {fmt_d(deliv)}; dispute filed {fmt_d(filed)}. Amount {money(rng.choice([89, 450, 1299]), sym)}.")
    q = noul("The dispute was filed within the window.", "The dispute was filed after the window closed.",
             "Is the dispute within the time limit?")
    return "noul", state, q, ["no", "yes"], ("yes" if (filed - base).days <= win else "no")


NOUL = [t_maintenance_call, t_pdt, t_wash_sale, t_price_band, t_concentration, t_lc_presentation, t_suitability,
        t_pix_noturno, t_chargeback]
CHOICE = [t_stop_fill, t_accrued, t_limit_fill, t_incoterm, t_vat, t_gst, t_sanctions]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 11)
    out, fails = 0, 0
    while out < n:
        if rng.random() < 0.5:
            tpl = rng.choice(NOUL)
            want = "yes" if rng.random() < 0.5 else "no"
            qtype, state, q, labels, gold = tpl(rng, want)
            if gold != want:  # construction failed to force the target: discard (keeps the balance exact)
                fails += 1
                continue
        else:
            tpl = rng.choice(CHOICE)
            qtype, state, q, labels, gold = tpl(rng)
        if len(labels) < 2 or len(set(labels)) != len(labels) or gold not in labels:
            fails += 1
            continue
        lang = "Brazilian Portuguese" if tpl is t_pix_noturno else "English"
        rec = {"id": f"prog-trade-{out:05d}", "family": "trade_exact", "topic": "finance",
               "state": state, "question": q, "labels": labels, "expected": gold,
               "teacher_probs": soft(gold, labels), "teacher_top": gold, "agree": True,
               "difficulty": "hard", "lang": lang, "source": f"prog_trade:{tpl.__name__}"}
        print(json.dumps(rec, ensure_ascii=False))
        out += 1
    print(f"discarded: {fails}", file=sys.stderr)


if __name__ == "__main__":
    main()
