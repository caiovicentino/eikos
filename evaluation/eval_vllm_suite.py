"""Evaluates a suite (suite_*.jsonl format) with offline vLLM using the exported model (official format): exact
letter readout (processed logprobs over the letters only), the model's calibration T, prefix cache (long rules that
are identical across items come almost for free). Reports accuracy and balanced accuracy per task.
usage: python eval_vllm_suite.py <model> <suite.jsonl|ALL> <tag> <gpu_frac> [max_len]
       ALL = the 7 suites (tags <tag>_<suite>), loading the model only once. >26 options: tournament."""
import collections
import json
import math
import os
import sys
EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, suites and checkpoints
EIKOS_RUNS = os.environ.get("EIKOS_RUNS", "runs")  # evaluation outputs
os.environ.setdefault("PROMPT_STYLE", "semif")


def main():
    from vllm import LLM, SamplingParams
    from decision_core import LETTERS, messages, options_of, temp_for
    M, suite, tag, frac = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    max_len = int(sys.argv[5]) if len(sys.argv) > 5 else 40960
    calib = json.load(open(f"{M}/calib.json")) if os.path.exists(f"{M}/calib.json") else None
    extra = {"quantization": os.environ["QUANT"]} if os.environ.get("QUANT") else {}  # e.g. QUANT=fp8 (validation)
    pc = os.environ.get("PREFIX", "1") == "1"  # diagnostics: PREFIX=0 disables the prefix cache
    if pc:
        extra.update(enable_prefix_caching=True, mamba_cache_mode=os.environ.get("MAMBA_MODE", "all"))
    else:
        extra.update(enable_prefix_caching=False)
    if os.environ.get("MAX_BATCHED"):  # diagnostics: token budget per step (avoids splitting long prompts into chunks)
        extra["max_num_batched_tokens"] = int(os.environ["MAX_BATCHED"])
    if os.environ.get("EAGER") == "1":  # diagnostics: no CUDA graphs
        extra["enforce_eager"] = True
    if os.environ.get("NO_CHUNKED") == "1":
        extra["enable_chunked_prefill"] = False
    llm = LLM(model=M, dtype="bfloat16", gpu_memory_utilization=frac, max_model_len=max_len, logprobs_mode="processed_logprobs",
              max_logprobs=32, max_num_seqs=int(os.environ.get("MAX_SEQS", "4")), **extra)
    tok = llm.get_tokenizer()
    let = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]

    def run(batch):
        """One vLLM pass over a batch of (item, options) → [(probs per label, n_tokens)], with the model's
        calibration. WARM=1 (fix for vLLM's hybrid cache): requests that share a prefix must not read a cache block
        while it is still being computed (race → wrong state). So: phase 1 = 1 request per initial prefix (first 1024
        tokens); phase 2 = 1 per exact state; phase 3 = the rest, all reading blocks that are already complete."""
        texts = [tok.apply_chat_template(messages(r["state"], r["question"], o), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for r, o in batch]
        sps = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(o), allowed_token_ids=let[:len(o)])
               for _, o in batch]
        if os.environ.get("WARM") == "1" and len(batch) > 1:
            ids = [tok.encode(tx, add_special_tokens=False) for tx in texts]
            left = list(range(len(batch)))
            outs = [None] * len(batch)
            for keyf in (lambda j: tuple(ids[j][:1024]),
                         lambda j: json.dumps(batch[j][0]["state"], sort_keys=True, ensure_ascii=False)):
                seen, phase = set(), []
                for j in left:
                    k = keyf(j)
                    if k not in seen:
                        seen.add(k)
                        phase.append(j)
                for j, out in zip(phase, llm.generate([texts[j] for j in phase], [sps[j] for j in phase], use_tqdm=False)):
                    outs[j] = out
                left = [j for j in left if outs[j] is None]
            for j, out in zip(left, llm.generate([texts[j] for j in left], [sps[j] for j in left], use_tqdm=False)):
                outs[j] = out
        else:
            outs = llm.generate(texts, sps, use_tqdm=False)
        res = []
        for out, (r, o) in zip(outs, batch):
            lp = out.outputs[0].logprobs[0]
            lg = [lp[t].logprob if t in lp else -1e9 for t in let[:len(o)]]
            T = temp_for(calib, 1.0, len(out.prompt_token_ids), len(o), r["question"].get("type", "choice")) if calib else 1.0
            m = max(lg)
            e = [math.exp((x - m) / T) for x in lg]
            z = sum(e)
            res.append(({o[j][0]: e[j] / z for j in range(len(o))}, len(out.prompt_token_ids)))
        return res

    suites = ({b: f"{EIKOS_HOME}/{f}" for b, f in ALL.items()} if suite == "ALL" else {None: suite})
    for b, path in suites.items():
        t = f"{tag}_{b}" if b else tag
        rows = [json.loads(line) for line in open(path)]
        items = [(r, options_of(r["question"], list(r["labels"]))) for r in rows]
        small = [i for i, (_, o) in enumerate(items) if len(o) <= 26]
        big = [i for i, (_, o) in enumerate(items) if len(o) > 26]
        final = dict(zip(small, run([items[i] for i in small])))
        if big:  # same tournament as decision_core.tournament: blocks of 20, top 2 per block advance to the final (≤26)
            blocks = [(i, items[i][1][k:k + 20]) for i in big for k in range(0, len(items[i][1]), 20)]
            fin, ntok = collections.defaultdict(list), collections.defaultdict(int)
            for (i, blk), (p, n) in zip(blocks, run([(items[i][0], blk) for i, blk in blocks])):
                fin[i] += sorted(blk, key=lambda x: -p[x[0]])[:2]
                ntok[i] += n
            for i, (pf, n) in zip(big, run([(items[i][0], fin[i][:26]) for i in big])):
                probs = {lab: 1e-4 for lab, _ in items[i][1]}
                probs.update(pf)
                z = sum(probs.values())
                final[i] = ({k: v / z for k, v in probs.items()}, ntok[i] + n)
        per = collections.defaultdict(list)
        with open(f"{EIKOS_RUNS}/suite_{t}.jsonl", "w") as fo:
            for i, (r, o) in enumerate(items):
                p, n = final[i]
                pred = max(p, key=p.get)
                ok = pred == str(r["expected"])
                per[r["task"]].append((str(r["expected"]), pred, ok))
                fo.write(json.dumps({"id": r["id"], "task": r["task"], "pred": pred, "gold": r["expected"], "ok": ok,
                                     "conf": p[pred], "n_tok": n, "tournament": len(o) > 26}) + "\n")
        summ = {}
        for tk, v in sorted(per.items()):
            acc = sum(x[2] for x in v) / len(v)
            by = collections.defaultdict(list)
            for g, _, ok in v:
                by[g].append(ok)
            bal = sum(sum(x) / len(x) for x in by.values()) / len(by)
            summ[tk] = {"acc": round(acc, 4), "balanced_acc": round(bal, 4), "n": len(v)}
        summ["_macro"] = round(sum(x["acc"] for x in summ.values()) / len(summ), 4)
        summ["_tag"] = t
        json.dump(summ, open(f"{EIKOS_RUNS}/suite_summary_{t}.json", "w"), indent=1)
        print(json.dumps({"_tag": t, "_macro": summ["_macro"], "n": len(items), "tournament": len(big)}), flush=True)


ALL = {"jb": "suite_jb_public.jsonl", "db": "suite_db.jsonl", "gen": "suite.jsonl", "fin": "suite_fin.jsonl",
       "fin2": "suite_fin2.jsonl", "trade": "suite_trade.jsonl", "rules": "suite_rules.jsonl"}


if __name__ == "__main__":
    main()
