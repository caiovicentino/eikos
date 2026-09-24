"""MLX backend (Apple Silicon) for the decision model: same letter readout, same prompt and same calibration as the
PyTorch backend (letter_adapter), with no LoRA, no API and no internet. The common prefix (system + state) goes
through the model once; the hybrid cache (attention KV + Gated DeltaNet states) is replicated and the N questions run
together in one right-padded batch (real positions never see the padding: causal attention and the recurrence only
look back). Questions with more than 26 options use the same multi-round tournament as the GPU backends
(decision_core.tournament).
usage: python mlx_decide.py <model_dir>        (demo; compares against PyTorch if COMPARE_TORCH=1)"""
import copy
import json
import math
import os
import sys
import time

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import ArraysCache, KVCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MLXDecider:
    def __init__(self, path):
        cfg = json.load(open(os.path.join(path, "decision_config.json")))
        os.environ["PROMPT_STYLE"] = cfg["prompt_version"].rsplit("-", 1)[-1]
        global LETTERS, messages, options_of, temp_for, tournament
        from decision_core import LETTERS, messages, options_of, temp_for, tournament  # noqa: F401  (after PROMPT_STYLE is set)
        cp = os.path.join(path, cfg.get("calib") or "calib.json")
        self.calib = json.load(open(cp)) if os.path.exists(cp) else None
        self.model, self.tok = mlx_load(path)
        lm = self.model.language_model if hasattr(self.model, "language_model") else self.model
        self.inner, self.lm = lm.model, lm
        self.let = []
        for L in LETTERS:
            t = self.tok.encode(L, add_special_tokens=False)
            assert len(t) == 1, f"letter {L} is not a single token: {t}"
            self.let.append(t[0])
        self.let_idx = mx.array(self.let)

    def _ids(self, state, q, opts):
        text = self.tok.apply_chat_template(messages(state, q, opts), tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        return self.tok.encode(text, add_special_tokens=False)

    def _letter_logits(self, h):
        """h: (B, d) → logits (B, 26) over the letters only (tied weights: the embedding used as a linear layer)."""
        if getattr(self.lm.args, "tie_word_embeddings", False):
            out = self.inner.embed_tokens.as_linear(h)
        else:
            out = self.lm.lm_head(h)
        return out[:, self.let_idx].astype(mx.float32)

    def _probs(self, lg, n_tok, opts, q):
        T = temp_for(self.calib, 1.0, n_tok, len(opts), q.get("type", "choice"))
        z = lg[: len(opts)] / T
        p = mx.softmax(z, axis=-1).tolist()
        return {lab: p[i] for i, (lab, _) in enumerate(opts)}

    def dist(self, state, q, opts):
        if len(opts) > 26:  # only 26 letters: blocks of 20, the top 2 of each block go to a final round
            return tournament(lambda o: self.dist(state, q, o), opts)
        ids = self._ids(state, q, opts)
        cache = self.lm.make_cache()
        h = self.inner(mx.array([ids]), cache)[:, -1, :]
        lg = self._letter_logits(h)[0]
        mx.eval(lg)
        return self._probs(lg, len(ids), opts, q), len(ids)

    @staticmethod
    def _expand(cache, n):
        for c in cache:
            if isinstance(c, KVCache) and c.keys is not None:
                k, v = c.state
                c.state = (mx.repeat(k, n, axis=0), mx.repeat(v, n, axis=0))
            elif isinstance(c, ArraysCache):
                c.cache = [mx.repeat(x, n, axis=0) if x is not None else None for x in c.cache]
        return cache

    def dist_many_cached(self, state, items, chunk=32):
        """Several questions about the same state: the prefix once, then the questions in a batch from the cache.
        Questions with more than 26 options go through dist (tournament), one at a time."""
        if any(len(o) > 26 for _, o in items):
            small = [(q, o) for q, o in items if len(o) <= 26]
            done = iter(self.dist_many_cached(state, small, chunk) if small else [])
            return [self.dist(state, q, o) if len(o) > 26 else next(done) for q, o in items]
        seqs = [self._ids(state, q, o) for q, o in items]
        P = 0
        while all(len(x) > P + 1 for x in seqs) and len({x[P] for x in seqs}) == 1:
            P += 1
        base = self.lm.make_cache()
        self.inner(mx.array([seqs[0][:P]]), base)
        mx.eval([c.state for c in base])
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        out = []
        for c0 in range(0, len(seqs), chunk):
            part = seqs[c0:c0 + chunk]
            n = len(part)
            cache = self._expand(copy.deepcopy(base), n)
            suf = [x[P:] for x in part]
            S = max(len(x) for x in suf)
            ids = mx.array([x + [pad] * (S - len(x)) for x in suf])
            H = self.inner(ids, cache)
            last = mx.array([len(x) - 1 for x in suf])
            h = H[mx.arange(n), last]
            lg = self._letter_logits(h)
            mx.eval(lg)
            for i, (q, o) in enumerate(items[c0:c0 + chunk]):
                out.append((self._probs(lg[i], len(part[i]), o, q), len(part[i])))
        return out

    # ---- parity shims so serve.py's Decider can drive the MLX backend like letter_adapter ----
    def load(self):
        return self.model, self.tok, self.let_idx

    def dist_many(self, state, items):
        return self.dist_many_cached(state, items)

    def dist_any(self, state, question, opts, **kw):
        return self.dist(state, question, opts)


def _demo(M):
    t0 = time.perf_counter()
    d = MLXDecider(M)
    print(f"MLX: loaded in {time.perf_counter() - t0:.1f} s | active memory {mx.get_active_memory() / 1e9:.1f} GB",
          flush=True)
    from local_demo_cases import CASES, MULTI
    res_single = []
    for name, state, q in CASES:
        opts = options_of(q)
        d.dist(state, q, opts)
        t0 = time.perf_counter()
        p, n = d.dist(state, q, opts)
        dt = time.perf_counter() - t0
        top = max(p, key=p.get)
        res_single.append(p)
        print(f"[{name}] {n} tokens | {dt * 1000:.0f} ms | {top} ({p[top]:.0%}) | "
              f"{json.dumps({k: round(v, 3) for k, v in p.items()})}", flush=True)
    state = CASES[0][1]
    items = [(q, options_of(q)) for q in MULTI]
    d.dist_many_cached(state, items)
    t0 = time.perf_counter()
    res = d.dist_many_cached(state, items)
    dt = time.perf_counter() - t0
    t0 = time.perf_counter()
    full = [d.dist(state, q, o)[0] for q, o in items]
    dt_full = time.perf_counter() - t0
    diff = max(abs(full[i][k] - res[i][0][k]) for i in range(len(items)) for k in full[i])
    print(f"[{len(items)} questions, same state] with cache+batch {dt * 1000:.0f} ms | no cache {dt_full * 1000:.0f} ms"
          f" | max difference {diff:.4f}")
    for (q, o), (p, _) in zip(items, res):
        top = max(p, key=p.get)
        lab = q["criteria"][int(top)] if q["type"] == "score" and isinstance(q.get("criteria"), list) else top
        print(f"  {q['instructions'][:60]!r} → {lab} ({p[top]:.0%})")
    # larger fan-out (agent pattern): 24 questions about the same state
    big = [(q, options_of(q)) for q in (MULTI * 8)]
    d.dist_many_cached(state, big)
    t0 = time.perf_counter()
    d.dist_many_cached(state, big)
    dt = time.perf_counter() - t0
    print(f"[{len(big)} questions, same state] {dt * 1000:.0f} ms total = {dt * 1000 / len(big):.0f} ms per decision")
    if os.environ.get("COMPARE_TORCH") == "1":
        from letter_adapter import LetterAdapter
        del d
        mx.clear_cache()
        ad = LetterAdapter(M, device="mps", calib=os.path.join(M, "calib.json"))
        ad.load()
        mxd = 0.0
        for (name, st, q), pm in zip(CASES, res_single):
            pt, _ = ad.dist(st, q, options_of(q))
            mxd = max(mxd, max(abs(pt[k] - pm[k]) for k in pt))
        print(f"MLX × PyTorch (MPS): max probability difference {mxd:.4f}")


if __name__ == "__main__":
    _demo(sys.argv[1])
