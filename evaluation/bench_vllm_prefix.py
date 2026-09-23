"""vLLM prefix cache on the hybrid model: modes none/align/all, short and long state.
Measures 200 questions about the same state after 1 warm-up question (agent pattern).
usage: python bench_vllm_prefix.py <release_model> <mode> <gpu_frac> [bf16]"""
import json
import os
import random
import sys
import time
os.environ.setdefault("PROMPT_STYLE", "semif")


def main():
    from vllm import LLM, SamplingParams
    from decision_core import LETTERS, messages
    from bench_parallel2 import ITEMS, STATE
    M, mode, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
    kw = dict(model=M, dtype="bfloat16", gpu_memory_utilization=frac, max_model_len=12288, disable_log_stats=False,
              logprobs_mode="processed_logprobs", max_logprobs=32)
    if mode == "none":
        kw["enable_prefix_caching"] = False
    else:
        kw["enable_prefix_caching"] = True
        kw["mamba_cache_mode"] = mode
    if os.environ.get("MAX_SEQS"):  # large models (27B): the sequence count cannot exceed the Mamba cache block count
        kw["max_num_seqs"] = int(os.environ["MAX_SEQS"])
    if len(sys.argv) > 4 and sys.argv[4] == "bf16":  # recurrent state in bf16 = smaller page = smaller cache blocks
        kw["mamba_ssm_cache_dtype"] = "bfloat16"
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    let = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    rng = random.Random(0)
    for label, state in (("curto", STATE), ("longo", STATE * 5)):
        qs = []
        for i in range(201):
            q, o = ITEMS[i % 6]
            o = o[:]
            rng.shuffle(o)
            qs.append((dict(q, instructions=q["instructions"] + f" (check {i} {label})"), o))
        texts = [tok.apply_chat_template(messages(state, q, o), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for q, o in qs]
        sps = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(o), allowed_token_ids=let[:len(o)]) for _, o in qs]
        n_tok = len(tok(texts[0])["input_ids"])
        t0 = time.perf_counter()
        llm.generate(texts[:1], sps[:1], use_tqdm=False)
        t1 = time.perf_counter()
        outs = llm.generate(texts[1:], sps[1:], use_tqdm=False)
        t2 = time.perf_counter()
        cached = [getattr(o, "num_cached_tokens", None) for o in outs]
        cached = [c for c in cached if c is not None]
        print(json.dumps({"mode": mode + ("+bf16" if len(sys.argv) > 4 else ""), "state": label, "tokens_prompt": n_tok, "first_ms": round((t1 - t0) * 1000, 1),
                          "200_questions_s": round(t2 - t1, 2), "decisions_per_s": round(200 / (t2 - t1), 1),
                          "mean_cached_tokens": (round(sum(cached) / len(cached)) if cached else None)}), flush=True)


if __name__ == "__main__":
    main()
