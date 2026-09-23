"""Long-context probe of our model (vLLM): public JevBench decisions with the state hidden inside L tokens of
neutral text, at 3 depths. Measures accuracy and ECE per (L, depth). Evaluation only.
usage: python probe_long_ctx.py <release_model> <gpu_frac> <n_items> [tag]"""
import json
import math
import os
import random
import sys
import time
EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
os.environ.setdefault("PROMPT_STYLE", "semif")

FILL = ["Operations log {i}: the desk reviewed settlement queues, reconciled cash against the custodian report and "
        "archived the checklist; no exception was raised. ",
        "Minutes {i}: the committee discussed office logistics, the cafeteria schedule and the quarterly training "
        "calendar; no decision relevant to clients or transactions was taken. ",
        "System notice {i}: scheduled maintenance of the internal wiki completed; search indexes were rebuilt and "
        "archived pages were moved to cold storage without content changes. "]


def main():
    from vllm import LLM, SamplingParams
    from decision_core import LETTERS, messages, options_of, temp_for
    M, frac, n_items = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    calib = json.load(open(f"{M}/calib.json"))
    # PREFIX=0: no prefix cache (the "all" mode gives wrong results with shared prefixes)
    pc = os.environ.get("PREFIX", "1") == "1"
    cache = dict(enable_prefix_caching=True, mamba_cache_mode="all") if pc else dict(enable_prefix_caching=False)
    llm = LLM(model=M, dtype="bfloat16", gpu_memory_utilization=frac, max_model_len=131072,
              logprobs_mode="processed_logprobs", max_logprobs=32, max_num_seqs=int(os.environ.get("MAX_SEQS", "4")),
              **cache)  # MAX_SEQS=1: one sequence at a time (vLLM batches give wrong results on the hybrid model)
    tok = llm.get_tokenizer()
    let = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    pub = [json.loads(l) for t in ("original", "hard") for l in open(f"{JEVBENCH_DIR}/datasets/public/{t}.jsonl")]
    rng = random.Random(4)
    rng.shuffle(pub)
    pub = [r for r in pub if len(options_of(r["question"], list(r["labels"]))) <= 26][:n_items]
    filler_tok = len(tok(FILL[0].format(i=0))["input_ids"])

    def haystack(state, L, depth):
        st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
        n = max(0, (L - len(tok(st)["input_ids"])) // filler_tok)
        parts = [FILL[i % 3].format(i=i) for i in range(n)]
        k = int(len(parts) * depth)
        return "".join(parts[:k]) + "\n=== CASE FILE (relevant) ===\n" + st + "\n=== END OF CASE FILE ===\n" + "".join(parts[k:])

    res = {}
    for L in (0, 4000, 16000, 64000):
        for depth in ((0.5,) if L == 0 else (0.1, 0.5, 0.9)):
            texts, meta = [], []
            for r in pub:
                opts = options_of(r["question"], list(r["labels"]))
                st = r["state"] if L == 0 else haystack(r["state"], L, depth)
                texts.append(tok.apply_chat_template(messages(st, r["question"], opts), tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False))
                meta.append((r, opts))
            sps = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(o), allowed_token_ids=let[:len(o)])
                   for _, o in meta]
            t0 = time.perf_counter()
            outs = llm.generate(texts, sps, use_tqdm=False)
            dt = time.perf_counter() - t0
            hits, confs, corr, ntok = 0, [], [], []
            for out, (r, o) in zip(outs, meta):
                lp = out.outputs[0].logprobs[0]
                lg = [lp[t].logprob if t in lp else -1e9 for t in let[:len(o)]]
                T = temp_for(calib, 1.0, len(out.prompt_token_ids), len(o), r["question"].get("type", "choice"))
                m = max(lg)
                e = [math.exp((x - m) / T) for x in lg]
                s = sum(e)
                p = [x / s for x in e]
                k = max(range(len(p)), key=lambda j: p[j])
                ok = o[k][0] == str(r["expected"])
                hits += ok
                confs.append(p[k])
                corr.append(int(ok))
                ntok.append(len(out.prompt_token_ids))
            n = len(corr)
            ece = sum(len(idx) / n * abs(sum(corr[j] for j in idx) / len(idx) - sum(confs[j] for j in idx) / len(idx))
                      for b in range(10) for idx in [[j for j, c in enumerate(confs) if b / 10 < c <= (b + 1) / 10]] if idx)
            row = {"target_tokens": L, "depth": depth, "mean_tokens": int(sum(ntok) / n), "acc": round(hits / n, 3),
                   "ece": round(ece, 3), "n": n, "seconds": round(dt, 1)}
            print(json.dumps(row), flush=True)
            res[f"{L}_{depth}"] = row
    json.dump(res, open(f"{EIKOS_RUNS}/long_ctx_probe{('_' + sys.argv[4]) if len(sys.argv) > 4 else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
