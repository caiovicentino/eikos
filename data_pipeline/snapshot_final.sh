#!/bin/bash
# Data snapshot for the final training run: generated + programmatic (quotas ~0.6x generated) + new sources
# (FinEntity, compositional rules, regulations, long dossiers). No FinQA in training; contamination exclusions.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
S=$1
mkdir -p $S
D=${EIKOS_HOME:-.}/data
cp $D/labeled.jsonl $S/labeled.jsonl
cp ${EIKOS_HOME:-.}/data_q38/labeled.jsonl $S/labeled_q38.jsonl
cp ${EIKOS_HOME:-.}/data_q38b/labeled.jsonl $S/labeled_q38b.jsonl
for f in prog_prob prog_temporal prog_judge_bal prog_fin prog_finjudge_clean prog_trade real_finentity prog_rules prog_rules_book long_items; do
  cp $D/$f.jsonl $S/
done
NGEN=$(cat $S/labeled.jsonl $S/labeled_q38.jsonl $S/labeled_q38b.jsonl | grep -c '"agree": true')
qq() { python -c "print(max(300, int($NGEN * $1)))"; }
echo "{\"prog_prob\": $(qq 0.06), \"prog_temp\": $(qq 0.05), \"prog_judge\": $(qq 0.12), \"prog_fin\": $(qq 0.1), \"prog_finjudge\": $(qq 0.1), \"prog_trade\": $(qq 0.15), \"real_finentity\": 2100, \"prog_rules\": $(qq 0.4), \"prog_rulebook\": $(qq 0.2), \"long\": $(qq 0.2)}" > $S/caps.json
${PYTHON:-python} "$HERE/dedup_check.py" $S/excluir.json $S/*.jsonl | tail -1 | cut -c1-120
HOLD_TRADE=${HOLD_TRADE:-1} python - "$S" <<'PYEOF'
import json, os, sys
S = sys.argv[1]
ex = set(json.load(open(f"{S}/excluir.json")))
hold = {"prog_trade:t_incoterm", "prog_trade:t_lc_presentation", "prog_trade:t_vat"} if os.environ.get("HOLD_TRADE", "1") == "1" else set()
n = 0
for l in open(f"{S}/prog_trade.jsonl"):
    r = json.loads(l)
    if r["source"] in hold:
        ex.add(r["id"]); n += 1
for l in open(f"{S}/prog_finjudge_clean.jsonl"):
    r = json.loads(l)
    if "-finqa-" in r["id"]:
        ex.add(r["id"]); n += 1
json.dump(sorted(ex), open(f"{S}/excluir.json", "w"))
print("excluded (held-out trade rules + FinQA):", n)
PYEOF
echo "final snapshot $S: approved generated items=$NGEN quotas=$(cat $S/caps.json)"
