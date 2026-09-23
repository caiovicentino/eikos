# Drops every financial answer-verification training item whose document appears in the test set (suite_fin).
import json, re
import os
EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, evaluation suites and checkpoints
def doc_key(state):
    s = state.split("Proposed answer:")[0]
    s = s.split("Question:")[0]                     # only the context (without the question)
    s = re.sub(r"Please answer the given financial question based on the context\.?", "", s)
    return re.sub(r"\s+", " ", s).strip()[:600]
test_docs = {doc_key(json.loads(l)["state"]) for l in open(f"{EIKOS_HOME}/suite_fin.jsonl") if '"finqa_judge"' in l}
rows = [json.loads(l) for l in open(f"{EIKOS_HOME}/data/prog_finjudge.jsonl")]
keep = [r for r in rows if doc_key(r["state"]) not in test_docs]
with open(f"{EIKOS_HOME}/data/prog_finjudge_clean.jsonl", "w") as f:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"test docs: {len(test_docs)} | training before: {len(rows)} | removed: {len(rows) - len(keep)} | after: {len(keep)}")
keep_docs = {doc_key(r["state"]) for r in keep}
left = sum(doc_key(json.loads(l)["state"]) in keep_docs
           for l in open(f"{EIKOS_HOME}/suite_fin.jsonl") if '"finqa_judge"' in l)
print("test items whose document is still in training:", left)
