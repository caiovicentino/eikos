"""Decision readout from letter logits (one forward pass per decision), used in evaluation (the JevBench harness and
the evaluation suites) and by the server. Format/prompt come from decision_core (same as in training). Options are
labelled A..Z, AA, AB, ... (decision_core.LABELS, one token each): up to 588 options are read in one pass.

Modes:
- local (transformers): model_path = model path/repo (+ optional LoRA adapter);
- server: model_path = "sglang:<url>|<tokenizer>" (reads the letter logprobs from a running SGLang server);
- vLLM: model_path = "vllm:<url>|<tokenizer>" (reads the letter logprobs from a vLLM server, see serve_vllm.sh);
- verify: verify_budget > 0 → short reasoning (thinking) with a token budget, and only then the letter.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

import decision_core
from decision_core import (LABELS, LETTERS, PROMPT_VERSION, STYLE, SYSTEM, messages, options_of,  # noqa: F401
                           render_user, state_text, temp_for, tournament)

import os
try:
    from jevbench.adapters.base import DecisionResult
except ImportError:  # used outside the harness (e.g. the release server)
    @dataclass
    class DecisionResult:  # type: ignore[no-redef]
        adapter: str
        ok: bool
        probs: Optional[dict] = None
        probs_source: str = "unknown"
        model: str = ""
        status: Optional[int] = None
        error: Optional[str] = None
        latency_s: float = 0.0
        usage: dict = field(default_factory=dict)
        raw: Optional[Any] = None
        request_body: Optional[Any] = None
        label: Optional[str] = None


class LetterAdapter:
    name = "ours_letter"
    cost_basis = "self_hosted_gpu"

    def __init__(self, model_path: str, device: str = "cuda", sym: bool = False, max_tokens: int = 12000,
                 adapter_path: str | None = None, temp: float = 1.0, calib: str | None = None,
                 verify_budget: int = 0, think_base: bool = False, readout: str | None = None,
                 jepa_path: str | None = None, mix_w: float | None = None):
        self.model_path = model_path
        self.endpoint = model_path
        self.model = model_path
        self.device = device
        self.sym = sym
        self.max_tokens = max_tokens
        self.adapter_path = adapter_path
        self.temp = temp
        self.calib = json.load(open(calib)) if calib else None
        self.verify_budget = verify_budget
        # hybrid: the base model thinks (reasoning intact) and the trained adapter only reads the letter at the end
        self.think_base = bool(think_base and adapter_path and verify_budget > 0)
        # readout: "letter" (default), "energy" (JEPA: -cos between the predicted answer and the option spans) or "mix"
        import os
        self.readout = readout or os.environ.get("READOUT", "letter")
        self.jepa_path = jepa_path or os.environ.get("JEPA_HEAD")
        self.mix_w = mix_w if mix_w is not None else float(os.environ.get("MIX_W", "0.5"))
        self._jhead = None
        self._loaded = None
        self._lp_cap = None  # --max-logprobs of the vLLM server, learned from its first refusal (None: not limited)
        self._inflight, self._rlock = {}, threading.Lock()  # requests in flight per vLLM server

    # ---------------- loading ----------------
    def _sglang(self):  # "sglang:<url>|<tokenizer>" or "vllm:<url>[,<url>...]|<tokenizer>"
        url, tok_path = self.model_path.split(":", 1)[1].split("|", 1)
        return url.split(",")[0].strip().rstrip("/"), tok_path

    @contextlib.contextmanager
    def _replica(self):
        """Several vLLM servers (e.g. one per GPU, comma-separated URLs): each request goes to the one with the fewest
        requests in flight, and all of its calls stay there (its prefix cache holds the state)."""
        urls = [u.strip().rstrip("/") for u in self.model_path.split(":", 1)[1].split("|", 1)[0].split(",")]
        with self._rlock:
            url = min(urls, key=lambda u: self._inflight.get(u, 0))
            self._inflight[url] = self._inflight.get(url, 0) + 1
        try:
            yield url
        finally:
            with self._rlock:
                self._inflight[url] -= 1

    @property
    def _is_vllm(self):
        return self.model_path.startswith("vllm:")

    def _vllm_letters(self, texts, n_max, url=None):
        """One /v1/completions call with all the prompts; allowed labels = the first n_max; logprobs already
        normalized over the allowed labels (server run with --logprobs-mode processed_logprobs). A server started with
        a lower --max-logprobs (v1.0 and v1.1 used 32) only returns its top labels: the labels it leaves out share the
        remaining probability evenly. Returns (list of label_logprobs, n_tok)."""
        import urllib.error
        import urllib.request
        url = url or self._sglang()[0]
        want = self._loaded[2][:n_max]
        while True:
            n_lp = n_max if self._lp_cap is None else min(n_max, self._lp_cap)
            body = {"model": "decider", "prompt": texts, "max_tokens": 1, "temperature": 0.0, "logprobs": n_lp,
                    "allowed_token_ids": want, "return_tokens_as_token_ids": True}
            req = urllib.request.Request(f"{url}/v1/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    d = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code != 400 or n_lp <= 32 or "logprobs" not in e.read().decode(errors="replace"):
                    raise
                self._lp_cap = 32  # older server: read the top 32 labels and spread the rest
        out = [None] * len(texts)
        for ch in d["choices"]:
            top = ch["logprobs"]["top_logprobs"][0]
            lp = {int(k.split(":", 1)[1]) if k.startswith("token_id:") else k: v for k, v in top.items()}
            rest = [i for i in want if i not in lp]
            if rest:
                tail = max(1.0 - sum(math.exp(v) for v in lp.values()), 1e-12)
                lp.update({i: math.log(tail / len(rest)) for i in rest})
            out[ch["index"]] = lp
        n_tok = d.get("usage", {}).get("prompt_tokens", 0) // max(1, len(texts))
        return out, n_tok

    def _warm_prefix(self, texts, url):
        """vLLM does not share a prefix between the prompts of one call while it is being computed, so N questions about
        one state would compute that state N times. One short call with their common prefix first puts its full cache
        blocks in the prefix cache, and the N questions then read them. EIKOS_SHARED_PREFIX=0 turns this off."""
        if len(texts) < 2 or os.environ.get("EIKOS_SHARED_PREFIX", "1") != "1":
            return
        pre = os.path.commonprefix(texts)
        if len(pre) < int(os.environ.get("EIKOS_SHARED_MIN_CHARS", "3000")):  # less than ~1 cache block: nothing to gain
            return
        import urllib.request
        body = {"model": "decider", "prompt": pre, "max_tokens": 1, "temperature": 0.0}
        req = urllib.request.Request(f"{url}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            r.read()

    @staticmethod
    def _letter_ids(tok):
        ids = []
        for L in LABELS:
            t = tok.encode(L, add_special_tokens=False)
            assert len(t) == 1, f"label {L} is not a single token: {t}"
            ids.append(t[0])
        assert len(set(ids)) == len(ids)
        return ids

    def load(self):
        if self._loaded is not None:
            return self._loaded
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self.model_path.startswith(("sglang:", "vllm:")):
            tok = AutoTokenizer.from_pretrained(self._sglang()[1])
            self._loaded = (None, tok, self._letter_ids(tok))
            return self._loaded
        tok = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=torch.bfloat16,
                                                     device_map={"": self.device})
        if self.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)
            if not self.think_base:
                model = model.merge_and_unload()
        model.eval()
        if self.readout != "letter":
            self._jhead = self._load_jhead(model)
        self._loaded = (model, tok, self._letter_ids(tok))
        return self._loaded

    def _load_jhead(self, model):
        import torch.nn as nn
        ck = torch.load(self.jepa_path, map_location=self.device, weights_only=False)
        d, k = ck["d"], ck["dim"]

        class _H(nn.Module):
            def __init__(self):
                super().__init__()
                mlp = lambda i, o: nn.Sequential(nn.Linear(i, 2 * k), nn.GELU(), nn.Linear(2 * k, o))  # noqa: E731
                self.pred_ans, self.opt = mlp(d, k), nn.Linear(d, k)
                self.pred_view, self.pred_rat = mlp(d, d), mlp(d, d)
        h = _H()
        h.load_state_dict(ck["state"])
        h.tau = ck["tau"]
        return h.to(self.device).float().eval()

    def prompt_ids(self, tok, user: str):  # compat (default prompt)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]

    # ---------------- readout ----------------
    def _probs(self, logits_letters, n_tok, opts, question):
        T = temp_for(self.calib, self.temp, n_tok, len(opts), question.get("type", "choice"))
        p = torch.softmax(logits_letters.float() / T, -1).tolist()
        return {lab: p[i] for i, (lab, _) in enumerate(opts)}

    def _post(self, body):
        import urllib.request
        url, _ = self._sglang()
        req = urllib.request.Request(f"{url}/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())

    def _letters_sglang(self, text, want):
        d = self._post({"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                        "return_logprob": True, "token_ids_logprob": want})
        lp = {int(x[1]): float(x[0]) for x in d["meta_info"]["output_token_ids_logprobs"][0]}
        return torch.tensor([lp[i] for i in want]), int(d["meta_info"].get("prompt_tokens", 0))

    @torch.no_grad()
    def dist(self, state, question, opts):
        model, tok, let_ids = self.load()
        want = let_ids[:len(opts)]
        verify = self.verify_budget > 0
        text = tok.apply_chat_template(messages(state, question, opts, verify=verify), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=verify)
        if verify:  # short reasoning with a budget; closes </think> if the budget runs out
            if model is None:
                d = self._post({"text": text, "sampling_params": {"max_new_tokens": self.verify_budget,
                                                                  "temperature": 0.0, "stop": ["</think>"]}})
                thought = d["text"]
            else:
                ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
                ctx = model.disable_adapter() if self.think_base else contextlib.nullcontext()
                with ctx:
                    g = model.generate(ids, max_new_tokens=self.verify_budget, do_sample=False,
                                       stop_strings=["</think>"], tokenizer=tok)
                thought = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
            if self.think_base:  # readout in the training format (fast prompt) with the thought in the <think> block
                text = tok.apply_chat_template(messages(state, question, opts, verify=False), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
            text = text + thought.split("</think>")[0].rstrip() + "\n</think>\n\n"
        if model is None and self._is_vllm:
            with self._replica() as url:
                (lp,), n_tok = self._vllm_letters([text], len(opts), url)
            lg = torch.tensor([lp.get(i, -1e9) for i in want])
            return self._probs(lg, n_tok, opts, question), n_tok
        if model is None:
            lg, n_tok = self._letters_sglang(text, want)
            return self._probs(lg, n_tok, opts, question), n_tok
        if self._jhead is not None and not verify:
            return self._dist_energy(model, tok, text, state, question, opts, want)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if ids.shape[1] > self.max_tokens:
            raise ValueError(f"prompt with {ids.shape[1]} tokens > {self.max_tokens}")
        out = model(input_ids=ids.to(self.device), logits_to_keep=1)
        return self._probs(out.logits[0, -1, want], int(ids.shape[1]), opts, question), int(ids.shape[1])

    def _dist_energy(self, model, tok, text, state, question, opts, want):
        """JEPA readout: energy between the predicted answer (last hidden state) and each option (its span in the
        context)."""
        from decision_core import option_spans, render_user
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor([enc["input_ids"]], device=self.device)
        if ids.shape[1] > self.max_tokens:
            raise ValueError(f"prompt with {ids.shape[1]} tokens > {self.max_tokens}")
        spans = option_spans(text, render_user(state, question, opts), opts, question, enc["offset_mapping"])
        H = model.model(input_ids=ids).last_hidden_state[0]
        h = H[-1].float()
        p_let = self._probs(model.lm_head(H[-1:])[0, want], int(ids.shape[1]), opts, question)
        if not spans:
            return p_let, int(ids.shape[1])
        pooled = torch.stack([H[a:b].float().mean(0) for a, b in spans])
        z = torch.nn.functional.normalize(self._jhead.pred_ans(h), dim=-1)
        c = torch.nn.functional.normalize(self._jhead.opt(pooled), dim=-1)
        p_en = torch.softmax((c @ z) / self._jhead.tau, -1).tolist()
        p_en = {lab: p_en[i] for i, (lab, _) in enumerate(opts)}
        if self.readout == "energy":
            return p_en, int(ids.shape[1])
        w = self.mix_w  # product of experts: letters^(1-w) · energy^w
        mix = {k: math.exp((1 - w) * math.log(max(p_let[k], 1e-12)) + w * math.log(max(p_en[k], 1e-12))) for k in p_let}
        s = sum(mix.values())
        return {k: v / s for k, v in mix.items()}, int(ids.shape[1])

    @torch.no_grad()
    def dist_many(self, state, items):
        """Parallel: several questions about the same state in a single GPU pass (batch, right-padding).
        items = [(question, opts), ...]. With vLLM or SGLang, all prompts go in one call and the server's prefix cache
        reuses the state. Falls back to the sequential path for the energy readout, verify mode, a single question or
        more options than one pass reads (decision_core.MAX_ONE_PASS)."""
        model, tok, let_ids = self.load()
        if (self._jhead is not None or self.verify_budget > 0 or len(items) == 1
                or any(len(o) > decision_core.MAX_ONE_PASS for _, o in items)):
            return [self.dist_any(state, q, o) for q, o in items]
        if model is None and self._is_vllm:  # vLLM: one call; the server's prefix cache reuses the state
            texts = [tok.apply_chat_template(messages(state, q, o), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for q, o in items]
            with self._replica() as url:
                self._warm_prefix(texts, url)
                lps, n_tok = self._vllm_letters(texts, max(len(o) for _, o in items), url)
            return [(self._probs(torch.tensor([lp.get(i, -1e9) for i in let_ids[:len(o)]]), n_tok, o, q), n_tok)
                    for (q, o), lp in zip(items, lps)]
        if model is None:  # SGLang: one call with the list of prompts; the prefix cache processes the state once
            texts = [tok.apply_chat_template(messages(state, q, o), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for q, o in items]
            wants = [let_ids[:len(o)] for _, o in items]
            d = self._post({"text": texts, "sampling_params": [{"max_new_tokens": 1, "temperature": 0.0}] * len(texts),
                            "return_logprob": [True] * len(texts), "token_ids_logprob": wants})
            out = []
            for (q, o), w, r in zip(items, wants, d):
                lp = {int(x[1]): float(x[0]) for x in r["meta_info"]["output_token_ids_logprobs"][0]}
                n_tok = int(r["meta_info"].get("prompt_tokens", 0))
                out.append((self._probs(torch.tensor([lp[i] for i in w]), n_tok, o, q), n_tok))
            return out
        seqs = []
        for q, o in items:
            text = tok.apply_chat_template(messages(state, q, o), tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            seqs.append(tok(text, add_special_tokens=False)["input_ids"])
        L = max(len(x) for x in seqs)
        if L > self.max_tokens:
            raise ValueError(f"prompt with {L} tokens > {self.max_tokens}")
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
        H = model.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device)).last_hidden_state
        last = torch.tensor([len(x) - 1 for x in seqs], device=self.device)
        logits = model.lm_head(H[torch.arange(len(seqs), device=self.device), last])
        return [(self._probs(logits[i, let_ids[:len(o)]], len(seqs[i]), o, q), len(seqs[i]))
                for i, (q, o) in enumerate(items)]

    @staticmethod
    def _expand_cache(cache, n):
        """Replicates the cache (attention: keys/values; linear: conv/recurrent) for a batch of n questions.
        Accepts both transformers formats: a plain tensor (5.12) or a dict keyed by state_idx (5.17+)."""
        def rep(t):
            if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] == 1:
                return t.repeat_interleave(n, dim=0)
            return t
        for layer in getattr(cache, "layers", []):
            for attr in ("keys", "values", "conv_states", "recurrent_states"):
                t = getattr(layer, attr, None)
                if isinstance(t, dict):
                    for k in t:
                        t[k] = rep(t[k])
                elif isinstance(t, list):
                    t[:] = [rep(x) for x in t]
                else:
                    setattr(layer, attr, rep(t))
        return cache

    @torch.no_grad()
    def dist_many_cached(self, state, items, return_timing=False):
        """Parallel, à la Jev: the common prefix (system + state) goes through the GPU once; the cache is replicated
        and the suffixes of the N questions run together in one batch. Same result as the full prompt."""
        import time as _t
        model, tok, let_ids = self.load()
        if (model is None or self._jhead is not None or self.verify_budget > 0
                or any(len(o) > decision_core.MAX_ONE_PASS for _, o in items)):
            return self.dist_many(state, items)
        texts = [tok.apply_chat_template(messages(state, q, o), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for q, o in items]
        seqs = tok(texts, add_special_tokens=False)["input_ids"]  # batched tokenization (Rust)
        if max(len(x) for x in seqs) > self.max_tokens:
            raise ValueError(f"prompt with {max(len(x) for x in seqs)} tokens > {self.max_tokens}")
        P = 0  # longest common prefix in tokens (tokenization identical to that of the full prompt)
        while all(len(x) > P + 1 for x in seqs) and len({x[P] for x in seqs}) == 1:
            P += 1
        t0 = _t.perf_counter()
        pre = torch.tensor([seqs[0][:P]], device=self.device)
        base_cache = model.model(input_ids=pre, use_cache=True).past_key_values
        t1 = _t.perf_counter()
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        out = []
        CH = int(__import__("os").environ.get("CACHE_CHUNK", "64"))  # questions per batch (replicated-cache memory)
        for c0 in range(0, len(seqs), CH):
            part = seqs[c0:c0 + CH]
            n = len(part)
            cache = self._expand_cache(copy.deepcopy(base_cache), n)
            suf = [x[P:] for x in part]
            S = max(len(x) for x in suf)
            ids = torch.full((n, S), pad, dtype=torch.long)
            att = torch.zeros((n, P + S), dtype=torch.long)
            att[:, :P] = 1
            for i, x in enumerate(suf):
                ids[i, :len(x)] = torch.tensor(x)
                att[i, P:P + len(x)] = 1
            H = model.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device), past_key_values=cache,
                            use_cache=True).last_hidden_state
            last = torch.tensor([len(x) - 1 for x in suf], device=self.device)
            logits = model.lm_head(H[torch.arange(n, device=self.device), last])
            for i in range(n):
                q, o = items[c0 + i]
                out.append((self._probs(logits[i, let_ids[:len(o)]], len(part[i]), o, q), len(part[i])))
            del cache, H
        if return_timing:
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            return out, {"prefix_tokens": P, "max_suffix_tokens": S, "prefix_ms": round((t1 - t0) * 1000, 1),
                         "total_ms": round((_t.perf_counter() - t0) * 1000, 1)}
        return out

    def dist_any(self, state, question, opts, chunk=None, keep=None):
        return tournament(lambda o: self.dist(state, question, o), opts, chunk, keep)

    # ---------------- harness ----------------
    def run(self, task) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        try:
            self.load()
        except Exception as e:  # noqa: BLE001
            res.error = f"load failed: {type(e).__name__}: {str(e)[:250]}"
            return res
        opts = options_of(task.question, list(task.labels))
        t0 = time.perf_counter()
        try:
            probs, n_in = self.dist_any(task.state, task.question, opts)
            if self.sym and len(opts) > 1:
                probs2, _ = self.dist_any(task.state, task.question, list(reversed(opts)))
                probs = {k: 0.5 * (probs[k] + probs2[k]) for k in probs}
        except Exception as e:  # noqa: BLE001
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        if torch.cuda.is_available() and not self.model_path.startswith("sglang:"):
            torch.cuda.synchronize()
        res.latency_s = time.perf_counter() - t0
        res.probs = probs
        res.usage = {"input_tokens": n_in * (2 if self.sym else 1), "output_tokens": 0}
        res.raw = {"prompt_version": PROMPT_VERSION, "sym": self.sym, "verify_budget": self.verify_budget}
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
