"""RuleArena NBA (MIT, github.com/SkyRiver-2000/RuleArena) → evaluation-only suite: the entire NBA salary-cap
rulebook (~25k tokens) + team/player situations + operations; question: is any operation illegal?
usage: python gen_suite_rulearena.py > suite_rulearena.jsonl"""
import json

import os
rules = open(os.path.join(os.environ.get("RULEARENA_DIR", "RuleArena"), "nba/reference_rules.txt")).read()
for comp in (0, 1, 2):
    for i, x in enumerate(json.load(open(os.path.join(os.environ.get("RULEARENA_DIR", "RuleArena"), f"nba/annotated_problems/comp_{comp}.json")))):
        case = ("Team situations:\n" + "\n".join(x["team_situations"]) + "\n\nPlayer situations:\n" +
                "\n".join(x["player_situations"]) + "\n\nProposed operations:\n" + "\n".join(x["operations"]))
        state = f"NBA SALARY CAP RULES (reference):\n{rules}\n\n=== CASE ===\n{case}"
        print(json.dumps({"id": f"rulearena-nba-{comp}-{i}", "task": f"rulearena_nba_c{comp}", "state": state,
                          "question": {"type": "noul", "instructions": "Under the rules above, is at least one of the proposed operations illegal?",
                                       "criteria": {"true": "at least one operation violates the rules",
                                                    "false": "all operations comply with the rules"}},
                          "labels": ["no", "yes"], "expected": "yes" if x["answer"] else "no"}, ensure_ascii=False))
