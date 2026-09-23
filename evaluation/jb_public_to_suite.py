"""Public JevBench (easy / original / hard) in the suite format (task = jb_<tier>), to run through the same
harness as the other suites. usage: python jb_public_to_suite.py suite_jb_public.jsonl"""
import json
import os
import sys

JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
with open(sys.argv[1], "w") as f:
    for tier in ("easy", "original", "hard"):
        for line in open(f"{JEVBENCH_DIR}/datasets/public/{tier}.jsonl"):
            r = json.loads(line)
            f.write(json.dumps({"id": r["id"], "task": f"jb_{tier}", "family": r.get("family"), "state": r["state"],
                                "question": r["question"], "labels": r.get("labels"), "expected": r.get("expected")},
                               ensure_ascii=False) + "\n")
print("ok")
