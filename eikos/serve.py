"""Typed-decision server compatible with the TypeSafe API (POST /v1/systemone and /v1/evaluate).

Uses the same core as training and evaluation (decision_core + letter_adapter): same prompt, same
letter readout, same calibration. All questions in a request go in one pass (batch); >26 options via a tournament;
System One: by default every question is answered in one pass, without generating text. Optional (off by
default, not used for the reported results): "mode": "verify" per question turns on a short reasoning step before
the letter (--verify-budget N).

usage: python serve.py --model <model_dir> [--port 8000] [--sym] [--device cuda]
     Production (parallel + cache): start vLLM with serve_vllm.sh and use --vllm-url http://127.0.0.1:8001 —
     continuous batching across users and the hybrid model's prefix cache (the state is reused across questions and
     calls).
     Agent sessions: POST /v1/sessions {"state"} → {"session_id"}; POST /v1/sessions/<id>/append {"text"};
     POST /v1/sessions/<id>/systemone {"questions"}; DELETE /v1/sessions/<id>.
     If the folder has a decision_config.json (model exported by merge_export.py), the prompt style and
     calibration are read from it: the model is standalone, with no LoRA, no external LLM and no API.
     (development: --adapter <lora> --calib <calib.json>, and PROMPT_STYLE in the environment)
License: MIT.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

class Decider:
    def __init__(self, a):
        self.remote = bool(a.vllm_url or a.sglang_url)
        if a.device == "mlx":  # Apple Silicon: mlx_decide.MLXDecider has the same dist interface
            from mlx_decide import MLXDecider
            self.fast = MLXDecider(a.model)
            self.verify = None
        else:
            from letter_adapter import LetterAdapter
            common = dict(device=a.device, adapter_path=a.adapter, temp=a.temp, calib=a.calib, max_tokens=a.max_tokens)
            src = (f"vllm:{a.vllm_url}|{a.model}" if a.vllm_url else
                   f"sglang:{a.sglang_url}|{a.model}" if a.sglang_url else a.model)
            self.fast = LetterAdapter(src, **common)
            self.fast.load()
            self.verify = LetterAdapter(src, verify_budget=a.verify_budget, **common) if a.verify_budget else None
            if self.verify:
                self.verify._loaded = self.fast._loaded  # same weights
        self.sym = a.sym
        # Local PyTorch: one pass at a time on the GPU. vLLM/SGLang: no lock (the server batches concurrent requests).
        self.lock = Lock() if not self.remote else contextlib.nullcontext()
        self.sessions = {}  # id -> {"state": str, "t": last activity}
        self.slock = Lock()
        self.name = a.model

    def decide_all(self, state, questions):
        """All questions in the request in a single GPU pass (parallel, like Jev). --sym goes into the same batch."""
        from decision_core import options_of
        names = list(questions)
        if self.verify and any((questions[nm] or {}).get("mode") == "verify" for nm in names):
            return {nm: self.decide(state, questions[nm]) for nm in names}  # verify: sequential path
        items, idx = [], []
        for nm in names:
            q = questions[nm]
            opts = options_of(q)
            if len(opts) < 2:
                raise ValueError(f"{nm}: at least 2 options are required")
            idx.append((nm, len(items), opts))
            items.append((q, opts))
            if self.sym:
                items.append((q, list(reversed(opts))))
        with self.lock:
            if self.remote or len(items) < 2:
                res = self.fast.dist_many(state, items)
            else:  # local PyTorch: state processed once (prefix cache) and questions in a batch
                res = self.fast.dist_many_cached(state, items)
        out = {}
        for nm, k, opts in idx:
            probs, n = res[k]
            if self.sym:
                p2, n2 = res[k + 1]
                probs = {x: 0.5 * (probs[x] + p2[x]) for x in probs}
                n += n2
            out[nm] = (self._format(questions[nm], probs), n)
        return out

    def _format(self, q, probs):
        t = q.get("type")
        top = max(probs, key=probs.get)
        if t in ("noul", "boolean"):
            return {"type": t, "noul": probs["yes"], "probability": probs["yes"], "value": probs["yes"] >= 0.5,
                    "confidence": max(probs.values())}
        if t == "score":
            return {"type": "score", "probabilities": probs, "score": int(top),
                    "expected": sum(float(k) * v for k, v in probs.items()), "confidence": probs[top]}
        return {"type": "choice", "choice": top, "probabilities": probs, "confidence": probs[top]}

    def decide(self, state, q):
        from decision_core import options_of
        opts = options_of(q)
        if len(opts) < 2:
            raise ValueError("at least 2 options are required")
        ad = self.verify if (q.get("mode") == "verify" and self.verify) else self.fast
        with self.lock:
            probs, n = ad.dist_any(state, q, opts)
            if self.sym:
                p2, n2 = ad.dist_any(state, q, list(reversed(opts)))
                probs = {k: 0.5 * (probs[k] + p2[k]) for k in probs}
                n += n2
        t = q.get("type")
        top = max(probs, key=probs.get)
        if t in ("noul", "boolean"):
            ans = {"type": t, "noul": probs["yes"], "probability": probs["yes"], "value": probs["yes"] >= 0.5,
                   "confidence": max(probs.values())}
        elif t == "score":
            ans = {"type": "score", "probabilities": probs, "score": int(top),
                   "expected": sum(float(k) * v for k, v in probs.items()), "confidence": probs[top]}
        else:
            ans = {"type": "choice", "choice": top, "probabilities": probs, "confidence": probs[top]}
        return ans, n


def make_handler(dec: Decider):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: clients reuse the connection (every response sets Content-Length)

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                self._send(200, {"ok": True, "model": dec.name})
            else:
                self._send(404, {"error": "not found"})

        def _body(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def do_DELETE(self):
            parts = self.path.strip("/").split("/")
            if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
                with dec.slock:
                    ok = dec.sessions.pop(parts[2], None) is not None
                return self._send(200 if ok else 404, {"ok": ok})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            parts = self.path.strip("/").split("/")
            t0 = time.perf_counter()
            if parts[:2] == ["v1", "sessions"]:  # agent sessions: the state grows by append; the cache reuses it
                try:
                    body = self._body()
                    if len(parts) == 2:
                        st = body.get("state", "")
                        if not isinstance(st, str):
                            st = json.dumps(st, ensure_ascii=False)
                        sid = os.urandom(8).hex()
                        with dec.slock:
                            dec.sessions[sid] = {"state": st, "t": time.time()}
                        return self._send(200, {"session_id": sid, "chars": len(st)})
                    sid = parts[2]
                    with dec.slock:
                        sess = dec.sessions.get(sid)
                    if sess is None:
                        return self._send(404, {"error": "unknown session"})
                    if len(parts) == 4 and parts[3] == "append":
                        txt = body.get("text", "")
                        with dec.slock:
                            sess["state"] += txt if isinstance(txt, str) else json.dumps(txt, ensure_ascii=False)
                            sess["t"] = time.time()
                        return self._send(200, {"ok": True, "chars": len(sess["state"])})
                    if len(parts) == 4 and parts[3] in ("systemone", "evaluate"):
                        res = dec.decide_all(sess["state"], body.get("questions") or {})
                        sess["t"] = time.time()
                        return self._send(200, {"model": dec.name, "session_id": sid,
                                                "answers": {k: v[0] for k, v in res.items()},
                                                "usage": {"input_tokens": sum(v[1] for v in res.values()), "output_tokens": 0},
                                                "latency_s": time.perf_counter() - t0})
                    return self._send(404, {"error": "not found"})
                except ValueError as e:
                    return self._send(422, {"error": str(e)})
                except Exception as e:  # noqa: BLE001
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            if self.path not in ("/v1/systemone", "/v1/evaluate"):
                return self._send(404, {"error": "not found"})
            try:
                body = self._body()
                res = dec.decide_all(body.get("state", ""), body.get("questions") or {})
                answers = {k: v[0] for k, v in res.items()}
                n_in = sum(v[1] for v in res.values())
                self._send(200, {"model": dec.name, "answers": answers,
                                 "usage": {"input_tokens": n_in, "output_tokens": 0},
                                 "latency_s": time.perf_counter() - t0})
            except ValueError as e:
                self._send(422, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--verify-budget", type=int, default=0, help="optional: enables 'mode': 'verify' (short reasoning)")
    ap.add_argument("--vllm-url", default=None, help="production backend: vLLM URL (serve_vllm.sh)")
    ap.add_argument("--sglang-url", default=None, help="production backend (parallel): URL of the SGLang server")
    a = ap.parse_args()
    cfg_path = os.path.join(a.model, "decision_config.json")
    if os.path.exists(cfg_path):  # exported model: the configuration ships with the weights
        cfg = json.load(open(cfg_path))
        os.environ.setdefault("PROMPT_STYLE", cfg.get("prompt_version", "letter-v1-semif").rsplit("-", 1)[-1])
        if a.calib is None and cfg.get("calib"):
            a.calib = os.path.join(a.model, cfg["calib"])
    dec = Decider(a)
    dec.decide("warm-up", {"type": "noul", "instructions": "Is this a warm-up?", "criteria": {"true": "yes", "false": "no"}})
    print(f"ready at http://{a.host}:{a.port} (calib={bool(a.calib)}, sym={a.sym}, backend={'vllm' if a.vllm_url else 'sglang' if a.sglang_url else 'pytorch'}, verify={a.verify_budget})", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(dec)).serve_forever()


if __name__ == "__main__":
    main()
