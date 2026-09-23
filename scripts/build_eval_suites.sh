#!/bin/bash
# Rebuilds the third-party evaluation suites used in our tables, downloading each dataset from its source.
# Nothing here is ever used for training. Our own trade and rules suites ship with the dataset (eval/).
# Needs JEVBENCH_DIR, FINDVER_DIR and RULEARENA_DIR (see .env.example); writes to $EIKOS_HOME (default: .).
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
H=$(cd "${EIKOS_HOME:-.}" && pwd)
cd "$HERE/evaluation"
"$PY" jb_public_to_suite.py "$H/suite_jb_public.jsonl"        # JevBench public items (easy / original / hard)
"$PY" db_to_suite.py "$H/suite_db.jsonl"                      # DecisionBench, medium + hard
"$PY" gen_suite.py "$H/suite.jsonl"                           # general battery (9 human-labeled tasks)
"$PY" gen_suite_fin.py "$H/suite_fin.jsonl"                   # CUAD, financial sentiment, FinQA-judge
"$PY" gen_suite_fin2.py > "$H/suite_fin2.jsonl"               # WCB stance (CC BY-NC-SA: evaluation only), FinDVer
"$PY" gen_suite_rulearena.py > "$H/suite_rulearena.jsonl"     # RuleArena NBA
echo "suites in $H (copy eval/suite_trade.jsonl and eval/suite_rules.jsonl from the dataset next to them)"
