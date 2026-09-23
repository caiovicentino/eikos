"""Latency in a game loop (Snake-style): every move brings a new state (a 10x10 board as text), one choice question
with 4 options (up/down/left/right), one decision at a time. Measures p50/p90 on vLLM (offline engine, no HTTP) and
on PyTorch (LetterAdapter). usage: PROMPT_STYLE=semif python bench_game.py <model> <vllm|torch> [n]"""
import json
import os
import random
import sys
import time
os.environ.setdefault("PROMPT_STYLE", "semif")
from decision_core import LETTERS, messages, options_of  # noqa: E402

M, mode = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
rng = random.Random(0)
Q = {"type": "choice", "instructions": "Which move should the snake make next to reach the food safely?",
     "criteria": {"up": "move up", "down": "move down", "left": "move left", "right": "move right"}}


def board():
    g = [["." for _ in range(10)] for _ in range(10)]
    x, y = rng.randrange(10), rng.randrange(10)
    body = [(x, y)]
    for _ in range(rng.randint(2, 8)):
        dx, dy = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        nx, ny = body[-1][0] + dx, body[-1][1] + dy
        if 0 <= nx < 10 and 0 <= ny < 10 and (nx, ny) not in body:
            body.append((nx, ny))
    for i, (bx, by) in enumerate(body):
        g[by][bx] = "H" if i == 0 else "S"
    while True:
        fx, fy = rng.randrange(10), rng.randrange(10)
        if g[fy][fx] == ".":
            g[fy][fx] = "F"
            break
    return ("Snake game, 10x10 grid. H = snake head, S = body, F = food, . = empty. Row 0 is the top.\n" +
            "\n".join(f"{r:>2} " + " ".join(row) for r, row in enumerate(g)) +
            f"\nCurrent direction: {rng.choice(['up', 'down', 'left', 'right'])}. Moving into a wall or the body loses.")


states = [board() for _ in range(N + 10)]
opts = options_of(Q)
lat = []
if mode == "vllm":
    def main():
        from vllm import LLM, SamplingParams
        llm = LLM(model=M, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=4096, enable_prefix_caching=True,
                  mamba_cache_mode="all", logprobs_mode="processed_logprobs", max_logprobs=32)
        tok = llm.get_tokenizer()
        let = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS[:len(opts)]]
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(let), allowed_token_ids=let)
        n_tok = []
        for i, st in enumerate(states):
            text = tok.apply_chat_template(messages(st, Q, opts), tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            t0 = time.perf_counter()
            llm.generate([text], sp, use_tqdm=False)
            if i >= 10:
                lat.append(time.perf_counter() - t0)
                n_tok.append(len(tok.encode(text)))
        report(sum(n_tok) / len(n_tok))
else:
    def main():
        import torch
        from letter_adapter import LetterAdapter
        ad = LetterAdapter(M, device="cuda:0", calib=f"{M}/calib.json")
        ad.load()
        n_tok = []
        for i, st in enumerate(states):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, n = ad.dist(st, Q, opts)
            torch.cuda.synchronize()
            if i >= 10:
                lat.append(time.perf_counter() - t0)
                n_tok.append(n)
        report(sum(n_tok) / len(n_tok))


def report(avg_tok):
    lat.sort()
    n = len(lat)
    print(json.dumps({"mode": mode, "moves": n, "mean_tokens": round(avg_tok), "p50_ms": round(1000 * lat[n // 2], 1),
                      "p90_ms": round(1000 * lat[int(n * 0.9)], 1), "decisions_per_s": round(n / sum(lat), 1)}), flush=True)


if __name__ == "__main__":
    main()
