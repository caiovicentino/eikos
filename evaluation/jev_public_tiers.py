# Accuracy of official leaderboard entrants on the public items (easy/original/hard) of JevBench v1.2,
# and the official by_tier breakdown (all 534 items, incl. held-out).
import json
import os
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
d = json.load(open(f"{JEVBENCH_DIR}/results/v1.2/jevbench-v1.2-per-task.json"))
tier_of = {}
for t in ("easy", "original", "hard"):
    for l in open(f"{JEVBENCH_DIR}/datasets/public/{t}.jsonl"):
        tier_of[json.loads(l)["id"]] = t
want = ["jev-1.13.0", "semif-qwen3.5-4b", "djev", "reflex-4b", "winnow-12b", "qwen3.8-27b", "jqv", "reflex-27b",
        "openjev-verdict-1.4", "laya", "kev-0.6b", "gpt-5.6-luna", "deepseek-flash"]
print(f"{'system':22s} {'pub easy':>9s} {'pub orig':>9s} {'pub hard':>9s} | official (534): easy standard judge hard")
for name in want:
    v = d["systems"].get(name)
    if not v:
        continue
    pt = v.get("public_tasks") or {}
    acc = {}
    for tier in ("easy", "original", "hard"):
        ids = [i for i, t in tier_of.items() if t == tier]
        c = sum(1 for i in ids if (pt.get(i) or ["n"])[0] == "c")
        acc[tier] = c / len(ids)
    bt = v.get("by_tier", {})
    off = " ".join(f"{bt.get(k, {}).get('accuracy', float('nan')):.3f}" for k in ("easy", "standard", "judge", "hard"))
    print(f"{name:22s} {acc['easy']:9.3f} {acc['original']:9.3f} {acc['hard']:9.3f} | {off}")
