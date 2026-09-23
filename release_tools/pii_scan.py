"""Privacy scan of the dataset before release. It finds two kinds of rows:
- rows with personal-data-shaped values that could belong to real people;
- rows with credential-shaped strings.

It prints only counts and writes the ids of the rows to drop. A flagged row is removed, never edited.

What is dropped:
- credential-shaped strings: AWS / OpenAI / HF / GitHub / Vercel-style keys, bearer tokens, `api_key = "..."`
  assignments, private-key blocks;
- e-mail addresses at consumer webmail domains;
- checksum-valid payment card numbers (Luhn; well-known test numbers excluded) and SSN-format numbers with a valid
  area, group and serial;
- checksum-valid IBANs and Brazilian CPFs, except well-known documentation examples.

Checksum-valid CNPJs are reported but not dropped, because they are public company-registry identifiers.

Extra JSON files are merged into the drop list: check_overlap.py reports (their "flagged_ids") or plain lists of
ids, such as known_label_errors.json.
usage: python pii_scan.py <dataset_dir> <drop_ids.json> [overlap_report.json] [known_label_errors.json ...]
"""
import collections
import glob
import json
import re
import sys

CREDENTIALS = [r"\bAKIA[0-9A-Z]{16}\b", r"Bearer\s+[A-Za-z0-9._~+/-]{12,}",
               r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]",
               r"vck_[A-Za-z0-9]{10,}", r"\bbk-[A-Za-z0-9]{10,}", r"\bsk-[A-Za-z0-9_-]{16,}",
               r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{8,}", r"\bhf_[A-Za-z0-9]{20,}", r"\bgh[pousr]_[A-Za-z0-9]{30,}",
               r"-----BEGIN [A-Z ]*PRIVATE KEY-----"]
TEST_CARDS = {"4111111111111111", "4242424242424242", "5555555555554444", "378282246310005", "4012888888881881",
              "5105105105105100", "6011111111111117", "4000056655665556", "371449635398431", "30569309025904"}
CONSUMER_MAIL = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "proton.me", "protonmail.com", "aol.com",
    "live.com", "uol.com.br", "bol.com.br", "terra.com.br", "yahoo.com.br", "hotmail.com.br", "gmx.de", "mail.ru",
    "qq.com", "163.com", "mail.com", "email.com", "usa.com", "post.com", "europe.com", "consultant.com", "myself.com",
    "dr.com", "engineer.com", "ymail.com", "rocketmail.com", "googlemail.com", "me.com", "mac.com", "msn.com",
    "zoho.com", "yandex.ru", "yandex.com", "tutanota.com", "fastmail.com", "hey.com", "pm.me", "ig.com.br",
    "globo.com", "r7.com", "zipmail.com.br", "outlook.com.br", "hotmail.co.uk", "yahoo.co.uk", "web.de",
    "t-online.de", "orange.fr", "free.fr", "libero.it", "gmx.com", "gmx.net", "inbox.com"}
# documentation examples (published in bank / standards / government docs); they may stay
KNOWN_IBAN = {"DE89370400440532013000", "GB82WEST12345698765432", "GB29NWBK60161331926819",
              "FR1420041010050500013M02606", "NL91ABNA0417164300", "ES9121000418450200051332",
              "IT60X0542811101000000123456", "BE68539007547034", "CH9300762011623852957", "AT611904300234573201",
              "GB33BUKB20201555555555", "DE75512108001245126199", "NL02ABNA0123456789", "ES7921000813610123456789",
              "FR7630006000011234567890189", "PT50000201231234567890154", "GB94BARC10201530093459",
              "GB15MIDL40051512345678", "DE44500105175407324931", "IE29AIBK93115212345678",
              "BR1800360305000010009795493C1", "BR9700360305000010009795493P1"}
KNOWN_CPF = {"12345678909", "11144477735", "52998224725", "98765432100", "11122233396"}


def cpf_ok(d):
    if len(set(d)) == 1:
        return False
    for n in (9, 10):
        c = (sum(int(d[i]) * (n + 1 - i) for i in range(n)) * 10) % 11 % 10
        if c != int(d[n]):
            return False
    return True


def cnpj_ok(d):
    if len(set(d)) == 1:
        return False
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    for w, n in ((w1, 12), ([6] + w1, 13)):
        s = sum(int(d[i]) * w[i] for i in range(n))
        if (0 if s % 11 < 2 else 11 - s % 11) != int(d[n]):
            return False
    return True


def luhn(d):
    t = 0
    for i, ch in enumerate(reversed(d)):
        x = int(ch)
        if i % 2:
            x = x * 2 - 9 if x > 4 else x * 2
        t += x
    return t % 10 == 0


def iban_ok(s):
    s = s.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    return int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1


def findings(t):
    """(category, value, drop?) for every match in the serialized row."""
    for rx in CREDENTIALS:
        for m in re.finditer(rx, t):
            yield "credential", None, True
    for m in re.finditer(r"(?<!\d)(\d{3})\.?(\d{3})\.?(\d{3})-?(\d{2})(?!\d)", t):
        d = "".join(m.groups())
        if cpf_ok(d) and ("." in m.group(0) or "-" in m.group(0)):
            yield "cpf", d, d not in KNOWN_CPF
    for m in re.finditer(r"(?<!\d)(\d{2})\.?(\d{3})\.?(\d{3})/?(\d{4})-?(\d{2})(?!\d)", t):
        d = "".join(m.groups())
        if cnpj_ok(d) and "/" in m.group(0):
            yield "cnpj", d, False
    for m in re.finditer(r"(?<![\d-])((?:\d[ -]?){12,18}\d)(?![\d-])", t):
        d = re.sub(r"\D", "", m.group(1))
        if 13 <= len(d) <= 19 and d[0] in "3456" and luhn(d) and d not in TEST_CARDS:
            yield "card", d, True
    for m in re.finditer(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)", t):
        a, g, s = m.groups()
        if a not in ("000", "666") and not a.startswith("9") and g != "00" and s != "0000" \
                and (a, g, s) != ("123", "45", "6789"):
            yield "ssn", a + g + s, True
    for m in re.finditer(r"\b([A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?)\b", t):
        v = m.group(1).replace(" ", "")
        if iban_ok(v):
            yield "iban", v, v not in KNOWN_IBAN
    for m in re.finditer(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", t):
        if m.group(1).lower() in CONSUMER_MAIL:
            yield "consumer_email", m.group(0).lower(), True


occ, rows, distinct, drop_rows = collections.Counter(), collections.defaultdict(set), \
    collections.defaultdict(set), collections.defaultdict(set)
for p in sorted(glob.glob(f"{sys.argv[1]}/data/*/*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        t, rid = json.dumps(r, ensure_ascii=False), r["id"]
        for cat, val, drop in findings(t):
            occ[cat] += 1
            rows[cat].add(rid)
            if val is not None:
                distinct[cat].add(val)
            if drop:
                drop_rows[cat].add(rid)
drop = set().union(*drop_rows.values()) if drop_rows else set()
print(f"{'category':15s} {'matches':>8s} {'rows':>6s} {'distinct':>9s} {'rows to drop':>13s}")
for cat in ("credential", "consumer_email", "card", "ssn", "iban", "cpf", "cnpj"):
    print(f"{cat:15s} {occ[cat]:8d} {len(rows[cat]):6d} {len(distinct[cat]) if cat != 'credential' else '-':>9} "
          f"{len(drop_rows[cat]):13d}")
for extra in sys.argv[3:]:
    x = json.load(open(extra))
    ids = set(x["flagged_ids"] if isinstance(x, dict) else x)
    print(f"{extra}: {len(ids)} rows added")
    drop |= ids
json.dump(sorted(drop), open(sys.argv[2], "w"))
print(f"rows to drop: {len(drop)} -> {sys.argv[2]}")
