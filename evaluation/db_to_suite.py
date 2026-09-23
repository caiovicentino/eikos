# DecisionBench (medium+hard) in the suite format, to run Jev and our models with the same code.
import json, sys
from eval_db import tasks_of
with open(sys.argv[1], "w") as f:
    for cfg in ("medium", "hard"):
        for t in tasks_of(cfg):
            f.write(json.dumps({"id": t.id, "task": f"db_{cfg}", "family": "db", "lang": "English", "state": t.state,
                                "question": t.question, "labels": t.labels, "expected": t.expected},
                               ensure_ascii=False) + "\n")
print("ok")
