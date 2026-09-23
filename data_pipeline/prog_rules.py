"""Compositional rules with exact gold answers — each item carries a new rule, assembled from pieces:
numeric conditions, derived quantities (notional, utilization, tolerance), date windows (calendar and
business days), lists, booleans, "all / at least k / one of", exceptions with precedence, first-matching-rule
routing, tier tables and factor-based scoring. The gold answer comes from executing the rule in code.
Goal: teach the model to apply never-seen rules (generalization), not to memorize templates.
Languages: English and Portuguese (Spanish is held out of training). The "insurance" domain can be held out (test).
usage: python prog_rules.py N SEED [--holdout] [--book] > prog_rules.jsonl
     --holdout: generates only what training never sees (insurance domain + exception × business-day
                combination in permit rules), for evaluation. Note: the tier rules can also produce an exception
                over a business-day window (6 of ~6,000 rule rows in the released training data).
     --book:    multi-section rulebooks instead of single rules (prog_rules_book.jsonl, suite_rulebook.jsonl)
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys

# ------------------------------------------------------------------ per-domain vocabulary
DOMAINS = {
    "trading": {"entity": ("order", "a ordem"), "action": ("be executed", "ser executada"), "attrs": [
        ("quantity", "quantity", "quantidade", "int", (10, 20000)),
        ("price", "price per unit (USD)", "preço por unidade (USD)", "money", (1, 900)),
        ("client_risk", "client risk profile (1-5)", "perfil de risco do cliente (1-5)", "int", (1, 5)),
        ("product_risk", "product risk class (1-7)", "classe de risco do produto (1-7)", "int", (1, 7)),
        ("asset_class", "asset class", "classe de ativo", "enum", ["equity", "bond", "fx", "crypto", "commodity", "fund"]),
        ("venue", "execution venue", "local de execução", "enum", ["lit_exchange", "dark_pool", "otc", "internalizer"]),
        ("approvals", "number of approvals on file", "número de aprovações registradas", "int", (0, 3)),
        ("pretrade_check", "pre-trade check passed", "checagem pré-negociação aprovada", "bool", None),
        ("leverage", "leverage (x)", "alavancagem (x)", "float", (1, 20)),
        ("account_age", "account age (days)", "idade da conta (dias)", "int", (1, 2000)),
        ("trade_date", "trade date", "data da negociação", "date", None),
    ], "derived": [("notional", "order notional (quantity x price)", "nocional da ordem (quantidade x preço)",
                    lambda c: c["quantity"] * c["price"], "money")],
        "outcomes": ["execute", "route_to_desk", "require_approval", "reject", "hold_for_review"]},
    "payments": {"entity": ("payment", "o pagamento"), "action": ("be released", "ser liberado"), "attrs": [
        ("amount", "amount (EUR)", "valor (EUR)", "money", (5, 250000)),
        ("country_from", "origin country", "país de origem", "enum", ["DE", "FR", "BR", "US", "SG", "IN", "AE", "MX"]),
        ("country_to", "destination country", "país de destino", "enum", ["DE", "FR", "BR", "US", "SG", "IN", "AE", "MX"]),
        ("sca_done", "strong customer authentication completed", "autenticação forte concluída", "bool", None),
        ("daily_total", "amount already sent today (EUR)", "valor já enviado hoje (EUR)", "money", (0, 200000)),
        ("customer_type", "customer type", "tipo de cliente", "enum", ["retail", "sme", "corporate", "government"]),
        ("screening_hits", "sanctions screening hits", "alertas de triagem de sanções", "int", (0, 3)),
        ("request_date", "request date", "data do pedido", "date", None),
    ], "derived": [("day_total_after", "total sent today including this payment (EUR)",
                    "total enviado hoje incluindo este pagamento (EUR)", lambda c: c["amount"] + c["daily_total"], "money")],
        "outcomes": ["release", "step_up_auth", "manual_review", "block", "queue_next_day"]},
    "lending": {"entity": ("loan application", "o pedido de crédito"), "action": ("be approved", "ser aprovado"), "attrs": [
        ("income", "monthly income (USD)", "renda mensal (USD)", "money", (800, 40000)),
        ("debt", "existing monthly debt payments (USD)", "parcelas mensais de dívidas existentes (USD)", "money", (0, 15000)),
        ("installment", "requested monthly installment (USD)", "parcela mensal pedida (USD)", "money", (50, 9000)),
        ("score", "credit score", "score de crédito", "int", (300, 850)),
        ("employment_months", "months in current job", "meses no emprego atual", "int", (0, 240)),
        ("collateral", "collateral type", "tipo de garantia", "enum", ["none", "vehicle", "property", "deposit"]),
        ("prior_default", "prior default on record", "inadimplência anterior registrada", "bool", None),
        ("application_date", "application date", "data do pedido", "date", None),
    ], "derived": [("dti", "debt-to-income ratio after the new installment (%)",
                    "comprometimento de renda com a nova parcela (%)",
                    lambda c: round(100 * (c["debt"] + c["installment"]) / c["income"], 2), "pct")],
        "outcomes": ["approve", "approve_with_conditions", "refer_to_underwriter", "decline"]},
    "trade_finance": {"entity": ("documentary presentation", "a apresentação de documentos"),
                      "action": ("be accepted", "ser aceita"), "attrs": [
        ("invoice_amount", "invoice amount (USD)", "valor da fatura (USD)", "money", (5000, 900000)),
        ("credit_amount", "credit amount (USD)", "valor do crédito (USD)", "money", (5000, 900000)),
        ("shipment_date", "shipment date", "data de embarque", "date", None),
        ("presentation_date", "presentation date", "data de apresentação", "date", None),
        ("expiry_date", "credit expiry date", "data de validade do crédito", "date", None),
        ("partial_shipment", "partial shipment", "embarque parcial", "bool", None),
        ("port", "port of loading", "porto de embarque", "enum", ["Santos", "Rotterdam", "Shanghai", "Singapore", "Houston", "Mumbai"]),
        ("docs_missing", "number of required documents missing", "número de documentos exigidos faltando", "int", (0, 3)),
        ("insured_pct", "insurance cover (% of invoice)", "cobertura do seguro (% da fatura)", "pct", (90, 130)),
    ], "derived": [("tolerance", "difference between invoice and credit amount (% of credit)",
                    "diferença entre fatura e crédito (% do crédito)",
                    lambda c: round(100 * abs(c["invoice_amount"] - c["credit_amount"]) / c["credit_amount"], 2), "pct")],
        "outcomes": ["accept", "accept_with_discrepancy_waiver", "refuse", "request_amendment"]},
    "kyc_aml": {"entity": ("customer file", "o cadastro do cliente"), "action": ("be approved", "ser aprovado"), "attrs": [
        ("risk_rating", "customer risk rating", "classificação de risco do cliente", "enum", ["low", "medium", "high"]),
        ("pep", "politically exposed person", "pessoa politicamente exposta", "bool", None),
        ("cash_deposits_30d", "cash deposits in the last 30 days (USD)", "depósitos em espécie nos últimos 30 dias (USD)", "money", (0, 90000)),
        ("country", "country of residence", "país de residência", "enum", ["BR", "US", "DE", "PA", "KY", "AE", "SG", "NG"]),
        ("id_verified", "identity document verified", "documento de identidade verificado", "bool", None),
        ("ubo_known", "ultimate beneficial owner identified", "beneficiário final identificado", "bool", None),
        ("alerts_open", "open monitoring alerts", "alertas de monitoramento abertos", "int", (0, 5)),
        ("last_review", "date of last periodic review", "data da última revisão periódica", "date", None),
    ], "derived": [], "outcomes": ["approve", "enhanced_due_diligence", "escalate_to_mlro", "reject"]},
    "insurance": {"entity": ("claim", "o sinistro"), "action": ("be paid", "ser pago"), "attrs": [
        ("claimed", "claimed amount (USD)", "valor reclamado (USD)", "money", (100, 120000)),
        ("deductible", "deductible (USD)", "franquia (USD)", "money", (0, 5000)),
        ("policy_limit", "policy limit (USD)", "limite da apólice (USD)", "money", (5000, 100000)),
        ("incident_date", "incident date", "data do incidente", "date", None),
        ("report_date", "report date", "data do aviso", "date", None),
        ("police_report", "police report attached", "boletim de ocorrência anexado", "bool", None),
        ("cause", "cause of loss", "causa da perda", "enum", ["fire", "theft", "flood", "collision", "wear_and_tear"]),
        ("prior_claims", "claims in the last 12 months", "sinistros nos últimos 12 meses", "int", (0, 4)),
    ], "derived": [("payable", "amount above the deductible (USD)", "valor acima da franquia (USD)",
                    lambda c: max(0, c["claimed"] - c["deductible"]), "money")],
        "outcomes": ["pay", "pay_partial", "investigate", "deny"]},
}
TRAIN_DOMAINS = [d for d in DOMAINS if d != "insurance"]
BASE_DATE = dt.date(2026, 3, 2)


def money(v, pt):
    s = f"{v:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") if pt else s


def fmt_date(d, pt):
    return d.strftime("%d/%m/%Y") if pt else d.strftime("%d %b %Y")


def bdays(a, b):  # business days from a (exclusive) to b (inclusive), Mon-Fri
    step = 1 if b >= a else -1
    n, d = 0, a
    while d != b:
        d += dt.timedelta(days=step)
        if d.weekday() < 5:
            n += step
    return n


def sample_case(rng, dom):
    c = {}
    for key, _, _, typ, par in DOMAINS[dom]["attrs"]:
        if typ in ("int",):
            c[key] = rng.randint(*par)
        elif typ == "money":
            c[key] = round(rng.uniform(*par), 2)
        elif typ == "float":
            c[key] = round(rng.uniform(*par), 1)
        elif typ == "pct":
            c[key] = round(rng.uniform(*par), 1)
        elif typ == "enum":
            c[key] = rng.choice(par)
        elif typ == "bool":
            c[key] = rng.random() < 0.5
        elif typ == "date":
            c[key] = BASE_DATE + dt.timedelta(days=rng.randint(0, 120))
    for key, _, _, fn, _ in DOMAINS[dom]["derived"]:
        c[key] = fn(c)
    return c


# ------------------------------------------------------------------ conditions (evaluate the case; self-describing)
class Cond:
    def __init__(self, fn, en, pt, attrs, tr=None):
        self.fn, self.en, self.pt, self.attrs, self.tr = fn, en, pt, attrs, tr

    def __call__(self, c):
        return bool(self.fn(c))

    def trace(self, c, pt):
        """Explicit evaluation with the case's values (becomes the exact training rationale)."""
        ok = self(c)
        body = self.tr(c, pt) if self.tr else (self.pt if pt else self.en)
        return f"{body} → {('sim' if ok else 'não') if pt else ('true' if ok else 'false')}"


DERIVED_CALC = {  # explicit computation of the derived quantities, with the case's values
    "notional": lambda c, f: f"{c['quantity']} x {f(c['price'])} = {f(c['quantity'] * c['price'])}",
    "day_total_after": lambda c, f: f"{f(c['amount'])} + {f(c['daily_total'])} = {f(c['amount'] + c['daily_total'])}",
    "dti": lambda c, f: f"({f(c['debt'])} + {f(c['installment'])}) / {f(c['income'])} x 100 = {c['dti']:g}%",
    "tolerance": lambda c, f: (f"|{f(c['invoice_amount'])} - {f(c['credit_amount'])}| / {f(c['credit_amount'])} x 100 = "
                               f"{c['tolerance']:g}%"),
    "payable": lambda c, f: f"max(0, {f(c['claimed'])} - {f(c['deductible'])}) = {f(c['payable'])}",
}


def attr_info(dom, key):
    for k, en, pt, typ, par in DOMAINS[dom]["attrs"]:
        if k == key:
            return en, pt, typ, par
    for k, en, pt, fn, typ in DOMAINS[dom]["derived"]:
        if k == key:
            return en, pt, typ, None
    raise KeyError(key)


DERIVED_BASE = {"notional": ["quantity", "price"], "day_total_after": ["amount", "daily_total"],
                "dti": ["income", "debt", "installment"], "tolerance": ["invoice_amount", "credit_amount"],
                "payable": ["claimed", "deductible"]}


def _num_fmt(typ, x, pt):
    if typ == "money":
        return money(x, pt)
    s = f"{x:g}"
    return s.replace(".", ",") if pt else s


def make_cond(rng, dom, c, want=None, allow_bdays=True):
    """Random condition over the case. want=True/False: rejection sampling until the requested truth value,
    with thresholds close to the actual value (hard cases)."""
    keys = [k for k, *_ in DOMAINS[dom]["attrs"]] + [k for k, *_ in DOMAINS[dom]["derived"]]
    for _ in range(80):
        key = rng.choice(keys)
        en, pt, typ, par = attr_info(dom, key)
        v = c[key]
        cond = None
        if typ in ("int", "money", "float", "pct"):
            op = rng.choice([">=", "<=", ">", "<"])
            if typ == "int":
                thr = max(0, v + rng.randint(-2, 2))
            else:
                thr = round(v * rng.choice([0.85, 0.93, 0.98, 1.0, 1.02, 1.07, 1.15]), 2 if typ == "money" else 1)
            ops_en = {">=": "at least", "<=": "at most", ">": "more than", "<": "less than"}
            ops_pt = {">=": "no mínimo", "<=": "no máximo", ">": "maior que", "<": "menor que"}
            fn = {">=": lambda c, k=key, t=thr: c[k] >= t, "<=": lambda c, k=key, t=thr: c[k] <= t,
                  ">": lambda c, k=key, t=thr: c[k] > t, "<": lambda c, k=key, t=thr: c[k] < t}[op]
            def tr(c, ptl, key=key, en=en, pt_=pt, typ=typ, thr=thr, op=op):
                f = (lambda x: _num_fmt(typ, x, ptl))
                lab = pt_ if ptl else en
                calc = DERIVED_CALC[key](c, lambda x: _num_fmt("money" if isinstance(x, float) else typ, x, ptl)) \
                    if key in DERIVED_CALC else f(c[key])
                return f"{lab} = {calc} {op} {f(thr)}?"
            cond = Cond(fn, f"the {en} is {ops_en[op]} {_num_fmt(typ, thr, False)}",
                        f"o campo \"{pt}\" é {ops_pt[op]} {_num_fmt(typ, thr, True)}", [key], tr)
        elif typ == "enum":
            vals = list(par)
            s = set(rng.sample(vals, rng.randint(1, max(1, len(vals) // 2))))
            neg = rng.random() < 0.3
            lst = ", ".join(sorted(s))
            fn = (lambda c, k=key, s=frozenset(s): c[k] not in s) if neg else (lambda c, k=key, s=frozenset(s): c[k] in s)
            cond = Cond(fn, f"the {en} is {'not ' if neg else ''}one of: {lst}",
                        f"o campo \"{pt}\" {'não ' if neg else ''}é um destes: {lst}", [key],
                        lambda c, ptl, key=key, en=en, pt_=pt, lst=lst, neg=neg:
                        f"{pt_ if ptl else en} = {c[key]} {'∉' if neg else '∈'} {{{lst}}}?")
        elif typ == "bool":
            pos = rng.random() < 0.5
            fn = (lambda c, k=key: c[k]) if pos else (lambda c, k=key: not c[k])
            cond = Cond(fn, f"the field \"{en}\" is {'yes' if pos else 'no'}",
                        f"o campo \"{pt}\" é {'sim' if pos else 'não'}", [key],
                        lambda c, ptl, key=key, en=en, pt_=pt, pos=pos:
                        (f"{pt_} = {'sim' if c[key] else 'não'} (exigido: {'sim' if pos else 'não'})?" if ptl else
                         f"{en} = {'yes' if c[key] else 'no'} (required: {'yes' if pos else 'no'})?"))
        elif typ == "date":
            dates = [k for k in keys if attr_info(dom, k)[2] == "date" and k != key]
            n = rng.randint(3, 45)
            if dates and rng.random() < 0.6:
                k2 = rng.choice(dates)
                en2, pt2, *_ = attr_info(dom, k2)
                if allow_bdays and rng.random() < 0.4:
                    fn = lambda c, a=key, b=k2, n=n: 0 <= bdays(c[a], c[b]) <= n  # noqa: E731
                    cond = Cond(fn, f"the {en2} is no more than {n} business days (Mon-Fri) after the {en}",
                                f"o campo \"{pt2}\" é no máximo {n} dias úteis (seg-sex) depois do campo \"{pt}\"",
                                [key, k2], lambda c, ptl, a=key, b=k2, n=n:
                                (f"dias úteis de {fmt_date(c[a], True)} até {fmt_date(c[b], True)} = {bdays(c[a], c[b])}; "
                                 f"entre 0 e {n}?" if ptl else
                                 f"business days from {fmt_date(c[a], False)} to {fmt_date(c[b], False)} = "
                                 f"{bdays(c[a], c[b])}; between 0 and {n}?"))
                else:
                    fn = lambda c, a=key, b=k2, n=n: 0 <= (c[b] - c[a]).days <= n  # noqa: E731
                    cond = Cond(fn, f"the {en2} is no more than {n} calendar days after the {en}",
                                f"o campo \"{pt2}\" é no máximo {n} dias corridos depois do campo \"{pt}\"",
                                [key, k2], lambda c, ptl, a=key, b=k2, n=n:
                                (f"dias corridos de {fmt_date(c[a], True)} até {fmt_date(c[b], True)} = "
                                 f"{(c[b] - c[a]).days}; entre 0 e {n}?" if ptl else
                                 f"calendar days from {fmt_date(c[a], False)} to {fmt_date(c[b], False)} = "
                                 f"{(c[b] - c[a]).days}; between 0 and {n}?"))
            else:
                ref = v + dt.timedelta(days=rng.randint(-6, 6))
                before = rng.random() < 0.5
                fn = (lambda c, k=key, r=ref: c[k] <= r) if before else (lambda c, k=key, r=ref: c[k] >= r)
                cond = Cond(fn, f"the {en} is on or {'before' if before else 'after'} {fmt_date(ref, False)}",
                            f"o campo \"{pt}\" é em ou {'antes de' if before else 'depois de'} {fmt_date(ref, True)}", [key],
                            lambda c, ptl, key=key, en=en, pt_=pt, ref=ref, before=before:
                            f"{pt_ if ptl else en} = {fmt_date(c[key], ptl)} {'≤' if before else '≥'} {fmt_date(ref, ptl)}?")
        if cond is not None and (want is None or cond(c) == want):
            return cond
    return None


def distinct_conds(rng, dom, c, n, allow_bdays=True):
    """n conditions over different variables (avoids trivial pairs such as 'before X' and 'after X')."""
    out, seen = [], set()
    for _ in range(40):
        x = make_cond(rng, dom, c, allow_bdays=allow_bdays)
        if x is None or any(a in seen for a in x.attrs):
            continue
        out.append(x)
        seen.update(x.attrs)
        if len(out) == n:
            return out
    return None


# ------------------------------------------------------------------ rule types
def conj_text(conds, mode, k, pt):
    items = [(x.pt if pt else x.en) for x in conds]
    letters = "abcdefg"
    lst = "; ".join(f"({letters[i]}) {t}" for i, t in enumerate(items))
    if mode == "all":
        return (f"todas as condições a seguir forem verdadeiras: {lst}" if pt else f"ALL of the following hold: {lst}")
    if mode == "any":
        return (f"pelo menos uma das condições a seguir for verdadeira: {lst}" if pt
                else f"AT LEAST ONE of the following holds: {lst}")
    return (f"pelo menos {k} das condições a seguir forem verdadeiras: {lst}" if pt
            else f"AT LEAST {k} of the following hold: {lst}")


def eval_conj(conds, mode, k, c):
    s = sum(x(c) for x in conds)
    return s == len(conds) if mode == "all" else (s >= 1 if mode == "any" else s >= k)


def rule_permit(rng, dom, c, pt, want, holdout):
    ent_en, ent_pt = DOMAINS[dom]["entity"]
    act_en, act_pt = DOMAINS[dom]["action"]
    for _ in range(60):
        n = rng.randint(2, 4)
        mode = rng.choice(["all", "all", "any", "k"])
        k = rng.randint(2, n) if mode == "k" else 0
        conds = distinct_conds(rng, dom, c, n)
        if conds is None:
            continue
        exc = None
        if rng.random() < (0.9 if holdout else 0.45):  # exception with precedence
            exc = make_cond(rng, dom, c, allow_bdays=holdout or rng.random() < 0.5)
            if exc is None or any(a in {k for x in conds for k in x.attrs} for a in exc.attrs):
                continue
        base = eval_conj(conds, mode, k, c)
        gold = base and not (exc(c) if exc else False)
        if gold != want:
            continue
        if holdout and not (exc and any("business" in x.en for x in conds + [exc])):
            continue  # combination reserved for the held-out test: exception + business-day window
        if not holdout and exc and any("business" in x.en for x in conds + [exc]):
            continue
        body = conj_text(conds, mode, k, pt)
        if pt:
            txt = f"REGRA: {ent_pt} só pode {act_pt} se {body}."
            if exc:
                txt += f" EXCEÇÃO (prevalece sobre o resto): {ent_pt} nunca pode {act_pt} se {exc.pt}."
            instr = f"De acordo com a regra, {ent_pt} pode {act_pt}?"
            crit = {"true": "sim, a regra permite", "false": "não, a regra não permite"}
        else:
            txt = f"RULE: the {ent_en} may {act_en} only if {body}."
            if exc:
                txt += f" EXCEPTION (overrides everything above): the {ent_en} may never {act_en} if {exc.en}."
            instr = f"Under the rule, may the {ent_en} {act_en}?"
            crit = {"true": "yes, the rule permits it", "false": "no, the rule does not permit it"}
        letters = "abcdefg"
        parts = [f"({letters[i]}) {x.trace(c, pt)}" for i, x in enumerate(conds)]
        need = {"all": ("todas" if pt else "all"), "any": ("pelo menos uma" if pt else "at least one"),
                "k": (f"pelo menos {k}" if pt else f"at least {k}")}[mode]
        sat = sum(x(c) for x in conds)
        rat = ("; ".join(parts) + (f". Satisfeitas: {sat} de {len(conds)}; exigidas: {need} → {'cumpre' if base else 'não cumpre'}."
                                    if pt else f". Satisfied: {sat} of {len(conds)}; required: {need} → {'met' if base else 'not met'}."))
        if exc:
            rat += (f" Exceção: {exc.trace(c, pt)}." if pt else f" Exception: {exc.trace(c, pt)}.")
        rat += (f" Resultado: {'sim' if gold else 'não'}." if pt else f" Result: {'yes' if gold else 'no'}.")
        return "noul", txt, {"type": "noul", "instructions": instr, "criteria": crit}, ["no", "yes"], ("yes" if gold else "no"), conds + ([exc] if exc else []), rat
    return None


def rule_route(rng, dom, c, pt, holdout):
    outs = DOMAINS[dom]["outcomes"][:]
    rng.shuffle(outs)
    n = rng.randint(2, min(4, len(outs) - 1))
    labels, default = outs[:n], outs[n]
    target = rng.randint(0, n)  # index of the first rule that matches (n = falls through to the default)
    for _ in range(60):
        conds = []
        ok = True
        for i in range(n):
            x = make_cond(rng, dom, c, want=(True if i == target else (False if i < target else None)))
            if x is None:
                ok = False
                break
            conds.append(x)
        if not ok:
            continue
        gold = default
        for x, lab in zip(conds, labels):
            if x(c):
                gold = lab
                break
        exp = labels[target] if target < n else default
        if gold != exp:
            continue
        lines = []
        for i, (x, lab) in enumerate(zip(conds, labels)):
            lines.append(f"{i + 1}. {'Se' if pt else 'If'} {x.pt if pt else x.en} → {lab}")
        lines.append(f"{n + 1}. {'Caso contrário' if pt else 'Otherwise'} → {default}")
        head = ("ROTEAMENTO (aplique a PRIMEIRA linha que se encaixa, na ordem):" if pt
                else "ROUTING POLICY (apply the FIRST line that matches, in order):")
        txt = head + "\n" + "\n".join(lines)
        allr = labels + [default]
        crit = {l: (f"a política leva a {l}" if pt else f"the policy leads to {l}") for l in allr}
        instr = "Qual é o resultado da política para este caso?" if pt else "What outcome does the policy give for this case?"
        steps = []
        for i, (x, lab) in enumerate(zip(conds, labels)):
            steps.append(f"{'Linha' if pt else 'Line'} {i + 1}: {x.trace(c, pt)}")
            if x(c):
                break
        else:
            steps.append(f"{'nenhuma linha se aplica → padrão' if pt else 'no line matches → default'}")
        rat = "; ".join(steps) + f" → {gold}."
        return "choice", txt, {"type": "choice", "instructions": instr, "criteria": crit}, allr, gold, conds, rat
    return None


def rule_tier(rng, dom, c, pt, holdout):
    nums = [(k, *attr_info(dom, k)[:3]) for k in list(c) if attr_info(dom, k)[2] in ("money", "pct", "int", "float")]
    if not nums:
        return None
    key, en, ptl, typ = rng.choice(nums)
    v = c[key]
    cuts = sorted({round(v * f, 2) for f in rng.sample([0.4, 0.7, 0.85, 1.1, 1.3, 1.8, 2.5], 3)})
    names = ["tier_a", "tier_b", "tier_c", "tier_d"]
    gold = names[sum(v > t for t in cuts)]
    exc = make_cond(rng, dom, c) if rng.random() < 0.4 else None
    if exc and exc(c):
        gold = "exempt"
    fm = (lambda x: money(x, pt)) if typ == "money" else (lambda x: (f"{x:g}".replace(".", ",") if pt else f"{x:g}"))
    rows = [f"{'até' if pt else 'up to'} {fm(cuts[0])} → {names[0]}"]
    for i in range(1, len(cuts)):
        rows.append(f"{'acima de' if pt else 'above'} {fm(cuts[i - 1])} {'e até' if pt else 'and up to'} {fm(cuts[i])} → {names[i]}")
    rows.append(f"{'acima de' if pt else 'above'} {fm(cuts[-1])} → {names[len(cuts)]}")
    if pt:
        txt = f"TABELA DE FAIXAS por {ptl}:\n" + "\n".join(rows)
        if exc:
            txt += f"\nEXCEÇÃO: se {exc.pt}, a faixa é exempt (isento), qualquer que seja o valor."
    else:
        txt = f"TIER TABLE by {en}:\n" + "\n".join(rows)
        if exc:
            txt += f"\nEXCEPTION: if {exc.en}, the tier is exempt regardless of the value."
    labs = names[:len(cuts) + 1] + ["exempt"]
    crit = {l: (f"faixa {l}" if pt else f"{l} applies") for l in labs}
    instr = "Em qual faixa este caso se enquadra?" if pt else "Which tier applies to this case?"
    keycond = Cond(lambda c: True, "", "", [key])  # ensures the table's field (or its base fields) appears in the case
    fmv = (lambda x: _num_fmt("money" if typ == "money" else typ, x, pt))
    calc = DERIVED_CALC[key](c, fmv) if key in DERIVED_CALC else fmv(v)
    base_t = names[sum(v > t for t in cuts)]
    rat = f"{ptl if pt else en} = {calc} → {base_t}"
    if exc:
        rat += (f"; exceção: {exc.trace(c, pt)}" if pt else f"; exception: {exc.trace(c, pt)}")
    rat += f" → {gold}."
    return "choice", txt, {"type": "choice", "instructions": instr, "criteria": crit}, labs, gold, [keycond] + ([exc] if exc else []), rat


def rule_score(rng, dom, c, pt, holdout):
    conds = distinct_conds(rng, dom, c, rng.randint(3, 5))
    if conds is None:
        return None
    crit_c = make_cond(rng, dom, c) if rng.random() < 0.35 else None
    if crit_c is not None and any(a in {k for x in conds for k in x.attrs} for a in crit_c.attrs):
        crit_c = None
    s = sum(x(c) for x in conds)
    lvl = 0 if s == 0 else (1 if s == 1 else 2)
    if crit_c and crit_c(c):
        lvl = 3
    letters = "abcde"
    lst = "; ".join(f"({letters[i]}) {(x.pt if pt else x.en)}" for i, x in enumerate(conds))
    if pt:
        txt = (f"MATRIZ DE RISCO: conte quantos fatores se aplicam: {lst}. Nenhum → nível 0; exatamente um → nível 1; "
               f"dois ou mais → nível 2.")
        if crit_c:
            txt += f" Se {crit_c.pt}, o nível é 3 (crítico), independentemente da contagem."
        levels = ["baixo", "médio", "alto", "crítico"]
    else:
        txt = (f"RISK MATRIX: count how many factors apply: {lst}. None → level 0; exactly one → level 1; "
               f"two or more → level 2.")
        if crit_c:
            txt += f" If {crit_c.en}, the level is 3 (critical) regardless of the count."
        levels = ["low", "medium", "high", "critical"]
    instr = "Qual é o nível de risco do caso?" if pt else "What is the risk level of this case?"
    parts = [f"({letters[i]}) {x.trace(c, pt)}" for i, x in enumerate(conds)]
    rat = "; ".join(parts) + (f". Fatores presentes: {s} → nível {min(s, 2)}" if pt else f". Factors present: {s} → level {min(s, 2)}")
    if crit_c:
        rat += (f"; crítico: {crit_c.trace(c, pt)}" if pt else f"; critical: {crit_c.trace(c, pt)}")
    rat += f" → {lvl}."
    return "score", txt, {"type": "score", "instructions": instr, "criteria": levels}, ["0", "1", "2", "3"], str(lvl), conds, rat


def rule_book(rng, dom, c, pt, want, holdout):
    """Rulebook: 6-12 numbered clauses (most irrelevant; some similar); the question cites one
    clause (find and apply it) or asks for the effect of the first applicable clause (precedence)."""
    ent_en, ent_pt = DOMAINS[dom]["entity"]
    act_en, act_pt = DOMAINS[dom]["action"]
    n = rng.randint(6, 12)
    clauses = []
    for i in range(n):
        x = make_cond(rng, dom, c)
        if x is None:
            return None
        effect = rng.choice(["permit", "forbid"])
        clauses.append((x, effect))
    mode = rng.choice(["cite", "first"])
    labels = ["no", "yes"]
    if mode == "cite":
        k = rng.randrange(n)
        x, eff = clauses[k]
        applies = x(c)
        gold = ("yes" if eff == "permit" else "no") if applies else ("no" if eff == "permit" else "yes")
        instr = (f"Aplicando SÓ a cláusula {k + 1} (cláusula de permissão só permite quando a condição vale; cláusula de "
                 f"proibição só proíbe quando a condição vale e, fora disso, permite), {ent_pt} pode {act_pt}?" if pt else
                 f"Applying ONLY clause {k + 1} (a 'may' clause allows only when its condition holds; a 'may NOT' clause "
                 f"forbids only when its condition holds and otherwise allows), may the {ent_en} {act_en}?")
        rat = (f"Cláusula {k + 1}: {x.trace(c, pt)} → {'permite' if (eff == 'permit') == applies else 'não permite'}."
               if pt else f"Clause {k + 1}: {x.trace(c, pt)} → {'permits' if (eff == 'permit') == applies else 'does not permit'}.")
    else:
        first = next((i for i, (x, _) in enumerate(clauses) if x(c)), None)
        gold = "yes" if first is None else ("yes" if clauses[first][1] == "permit" else "no")
        instr = (f"Pelas cláusulas (a PRIMEIRA aplicável decide; se nenhuma se aplica, é permitido), {ent_pt} pode {act_pt}?"
                 if pt else f"Under the clauses (the FIRST applicable clause decides; if none applies, it is permitted), "
                 f"may the {ent_en} {act_en}?")
        steps = []
        for i, (x, eff) in enumerate(clauses):
            steps.append(f"{i + 1}: {x.trace(c, pt)}")
            if x(c):
                break
        rat = "; ".join(steps) + f" → {gold}."
    if (gold == "yes") != want:
        return None
    lines = []
    for i, (x, eff) in enumerate(clauses):
        if pt:
            verb = "pode" if eff == "permit" else "NÃO pode"
            lines.append(f"Cláusula {i + 1}. Se {x.pt}, {ent_pt} {verb} {act_pt}.")
        else:
            verb = "may" if eff == "permit" else "may NOT"
            lines.append(f"Clause {i + 1}. If {x.en}, the {ent_en} {verb} {act_en}.")
    head = "REGULAMENTO INTERNO" if pt else "INTERNAL RULEBOOK"
    txt = head + "\n" + "\n".join(lines)
    crit = ({"true": "sim, pode", "false": "não pode"} if pt else {"true": "yes, it may", "false": "no, it may not"})
    return "noul", txt, {"type": "noul", "instructions": instr, "criteria": crit}, labels, gold, [x for x, _ in clauses], rat


def render_case(rng, dom, c, pt, used):
    derived = {k for k, *_ in DOMAINS[dom]["derived"]}
    used = {b for k in used for b in (DERIVED_BASE.get(k, [k]) if k in derived else [k])}
    keys = [k for k in c if k not in derived]
    extra = [k for k in keys if k not in used]
    rng.shuffle(extra)
    show = [k for k in keys if k in used] + extra[:rng.randint(1, 3)]  # relevant facts + distractors
    rng.shuffle(show)
    fmt = rng.choice(["json", "prose", "table"])
    vals = {}
    for k in show:
        en, ptl, typ, _ = attr_info(dom, k)
        v = c[k]
        if typ == "date":
            s = fmt_date(v, pt)
        elif typ == "bool":
            s = ("sim" if v else "não") if pt else ("yes" if v else "no")
        elif typ == "money":
            s = money(v, pt)
        elif typ in ("pct", "float"):
            s = f"{v:g}".replace(".", ",") if pt else f"{v:g}"
        else:
            s = str(v)
        vals[ptl if pt else en] = s
    ent = DOMAINS[dom]["entity"][1 if pt else 0]
    if fmt == "json":
        return json.dumps({("caso" if pt else "case"): ent, **vals}, ensure_ascii=False, indent=1)
    if fmt == "table":
        return "\n".join(f"| {k} | {v} |" for k, v in vals.items())
    return (f"Dados d{'o' if not ent.startswith('a ') else 'a'} {ent.split(' ', 1)[-1]}: " if pt else f"Facts about the {ent}: ") + \
        "; ".join(f"{k}: {v}" for k, v in vals.items()) + "."


def soft(gold, labels, p=0.95):
    o = (1 - p) / (len(labels) - 1)
    return {l: (p if l == gold else o) for l in labels}


def main():
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    holdout = "--holdout" in sys.argv
    out, fails, yn = 0, 0, {"yes": 0, "no": 0}
    while out < n:
        dom = "insurance" if (holdout and rng.random() < 0.5) else rng.choice(TRAIN_DOMAINS)
        if holdout and dom != "insurance":
            kind = "permit"  # outside insurance, the held-out test is the exception x business-day combination
        else:
            kind = rng.choice(["permit", "permit", "route", "tier", "score"]) if "--book" not in sys.argv else "book"
        pt = rng.random() < 0.2
        c = sample_case(rng, dom)
        if kind == "permit":
            want = yn["yes"] <= yn["no"]
            r = rule_permit(rng, dom, c, pt, want, holdout and dom != "insurance")
        elif kind == "book":
            want = yn["yes"] <= yn["no"]
            r = rule_book(rng, dom, c, pt, want, holdout)
        elif kind == "route":
            r = rule_route(rng, dom, c, pt, holdout)
        elif kind == "tier":
            r = rule_tier(rng, dom, c, pt, holdout)
        else:
            r = rule_score(rng, dom, c, pt, holdout)
        if not r:
            fails += 1
            continue
        qtype, rule_txt, q, labels, gold, conds, rationale = r
        if gold not in labels or len(set(labels)) != len(labels):
            fails += 1
            continue
        used = {a for x in conds if x is not None for a in x.attrs}
        case = render_case(rng, dom, c, pt, used)
        state = (f"{rule_txt}\n\n{'CASO' if pt else 'CASE'}:\n{case}")
        if qtype == "noul":
            yn[gold] += 1
        tag = "holdout" if holdout else "train"
        rec = {"id": f"prog-rules-{tag}-{out:06d}", "family": f"rules_{kind}", "topic": f"rules: {dom}",
               "state": state, "question": q, "labels": labels, "expected": gold,
               "teacher_probs": soft(gold, labels), "teacher_top": gold, "agree": True, "difficulty": "hard",
               "lang": "Brazilian Portuguese" if pt else "English",
               "source": f"{'prog_rulebook' if kind == 'book' else 'prog_rules'}:{kind}:{dom}",
               "rationale": rationale}
        print(json.dumps(rec, ensure_ascii=False))
        out += 1
    print(f"failures={fails} yes/no={yn}", file=sys.stderr)


if __name__ == "__main__":
    main()
