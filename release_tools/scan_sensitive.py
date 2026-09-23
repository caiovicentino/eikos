"""Sensitive-data scan of a release directory: prints ONLY counts per pattern and file, never the matched values.

- Real needles (keys, IPs, endpoints, e-mail addresses): read from a file kept OUTSIDE the release, one per line.
  The output only says "needle k: n occurrences".
- Generic patterns:
  - key prefixes, bearer tokens, api_key/secret/password assignments;
  - IPv4 addresses by class (private, documentation, loopback, public);
  - /root, /Users, /Volumes and /home paths;
  - e-mail addresses (with the most common domains, which are not sensitive).
usage: python scan_sensitive.py <dir> <needles_file>
"""
import collections
import ipaddress
import os
import re
import sys

ROOT, NEEDLES = sys.argv[1], sys.argv[2]
needles = [x.strip() for x in open(NEEDLES) if x.strip()]
PAT = {
    "vercel_key(vck_)": r"vck_[A-Za-z0-9]{10,}",
    "key(bk-)": r"\bbk-[A-Za-z0-9]{10,}",
    "openai_key(sk-)": r"\bsk-[A-Za-z0-9_-]{16,}",
    "hf_token(hf_)": r"\bhf_[A-Za-z0-9]{20,}",
    "aws(AKIA)": r"\bAKIA[0-9A-Z]{16}\b",
    "github(ghp_)": r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    "slack(xox)": r"\bxox[abpr]-[A-Za-z0-9-]{10,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "bearer_literal": r"Bearer\s+[A-Za-z0-9._~+/-]{12,}",
    "assigned_secret": r"(?i)\b(api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]",
    "path_/root": r"/root/",
    "path_/Users": r"/Users/",
    "path_/Volumes": r"/Volumes/",
    "path_/home": r"/home/[a-z_][a-z0-9_-]*/",
}
cnt = collections.defaultdict(collections.Counter)
ipc = collections.Counter()
dom = collections.Counter()
nd = collections.Counter()
for dp, _, fs in os.walk(ROOT):
    for f in fs:
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, ROOT)
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for k, n in enumerate(needles):
            c = t.count(n)
            if c:
                nd[(k, rel)] += c
        for name, rx in PAT.items():
            c = len(re.findall(rx, t))
            if c:
                cnt[name][rel] += c
        for m in re.findall(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])", t):
            try:
                ip = ipaddress.ip_address(m)
            except ValueError:
                continue
            doc = any(ip in ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"))
            ipc["documentation" if doc else "loopback" if ip.is_loopback else "private" if ip.is_private
                else "public"] += 1
        for d in re.findall(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", t):
            dom[d.lower()] += 1
print(f"== {ROOT}")
print("REAL NEEDLES:", "no occurrences" if not nd else {f"needle {k} in {rel}": c for (k, rel), c in nd.items()})
for name in PAT:
    tot = sum(cnt[name].values())
    print(f"{name:22s} {tot:6d}" + (f"  in {len(cnt[name])} file(s): {dict(cnt[name].most_common(4))}" if tot else ""))
print("IPv4 by class:", dict(ipc))
print(f"e-mails: {sum(dom.values())} | most common domains: {dom.most_common(8)}")
