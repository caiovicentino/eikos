"""Typed-decision server compatible with the TypeSafe API (POST /v1/systemone and /v1/evaluate).

Uses the same core as training and evaluation (decision_core + letter_adapter): same prompt, same
letter readout, same calibration. All questions in a request go in one pass (batch), each with up to 588 options
(labels A..Z, AA, AB, ...; start vLLM with serve_vllm.sh, --max-logprobs 600); more options than that via a tournament;
System One: by default every question is answered in one pass, without generating text. Optional (off by
default, not used for the reported results): "mode": "verify" per question turns on a short reasoning step before
the letter (--verify-budget N).
Images (v1.3, vLLM backend): add "images" to the request (a list of data URIs or base64 strings; URLs only with
--allow-image-urls), or send multipart/form-data with the JSON in a "request" field and the images as "image" files.
Image data URIs inside the state (a string, dict or list values, chat-style content parts) are taken out in order and
replaced with "[image N]"; an "image_data" field is read too. Every question in the request reads the same images,
attached before the decision text.

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
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

VERSION = "1.3"
_MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF8", "image/gif"))


def _mime(data: bytes):
    for sig, mime in _MAGIC:
        if data.startswith(sig):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


_IMG_URI = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/]+=*")


def lift_images(state, start=0):
    """Image data URIs inside the state (a string, dict or list values, chat-style content parts) are taken out in order
    and replaced with "[image N]", numbered after the request's own images. Returns (state, data URIs)."""
    found = []

    def take(m):
        found.append(m.group(0))
        return f"[image {start + len(found)}]"

    def walk(x):
        if isinstance(x, str):
            return _IMG_URI.sub(take, x) if "data:image/" in x else x
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(state), found


def has_images(body) -> bool:
    """Whether a request carries images in any of the forms /v1/systemone reads."""
    if body.get("images") or body.get("image_data"):
        return True
    return any("data:image/" in (x if isinstance(x, str) else json.dumps(x)) for x in (body.get("state"), body.get("text")))


def load_images(items, allow_urls=False, max_images=4, max_pixels=3840 * 2160, max_bytes=48_000_000):
    """Validates the request's images and returns them as data URIs (URLs pass through only with --allow-image-urls).
    Images larger than max_pixels are scaled down, keeping their aspect ratio. Any problem is a ValueError (422)."""
    import base64
    import io
    if not items:
        return None
    if not isinstance(items, list):
        raise ValueError('"images" must be a list')
    if len(items) > max_images:
        raise ValueError(f"at most {max_images} images per request (got {len(items)})")
    out = []
    for i, it in enumerate(items):
        if isinstance(it, dict):  # {"url": ...} or {"data": <base64>}
            it = it.get("url") or it.get("data") or ""
        if isinstance(it, str) and it.startswith(("http://", "https://")):
            if not allow_urls:
                raise ValueError("image URLs are off: send the image as base64, a data URI or a multipart file "
                                 "(or start serve.py with --allow-image-urls)")
            out.append(it)
            continue
        if isinstance(it, str):
            try:
                data = base64.b64decode(it.split(",", 1)[1] if it.startswith("data:") else it)
            except Exception:  # noqa: BLE001
                raise ValueError(f"image {i}: not valid base64") from None
        else:
            data = bytes(it)
        if len(data) > max_bytes:
            raise ValueError(f"image {i}: larger than {max_bytes // 1_000_000} MB")
        mime = _mime(data)
        if mime is None:
            raise ValueError(f"image {i}: not a PNG, JPEG, GIF or WebP image")
        try:
            from PIL import Image
        except ImportError:  # no Pillow: pass the image as it is
            out.append(f"data:{mime};base64," + base64.b64encode(data).decode())
            continue
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception:  # noqa: BLE001
            raise ValueError(f"image {i}: unreadable") from None
        w, h = im.size
        if w * h > max_pixels:
            k = (max_pixels / (w * h)) ** 0.5
            im = im.convert("RGB").resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "PNG")
            data, mime = buf.getvalue(), "image/png"
        out.append(f"data:{mime};base64," + base64.b64encode(data).decode())
    return out


def parse_multipart(ctype: str, raw: bytes) -> dict:
    """multipart/form-data as imajev's clients send it: a "request" field with the JSON body, and the images as file
    fields named "image" (or "images"), in order."""
    from email.parser import BytesParser
    from email.policy import HTTP
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + raw)
    if not msg.is_multipart():
        raise ValueError("malformed multipart body")
    body, files = None, []
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        if name == "request":
            body = json.loads(data.decode("utf-8"))
        elif name in ("image", "images"):
            files.append(data)
    if not isinstance(body, dict):
        raise ValueError('a multipart request needs a "request" field holding the JSON body')
    body["images"] = list(body.get("images") or []) + files
    return body


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

    def decide_all(self, state, questions, images=None):
        """All questions in the request in a single GPU pass (parallel, like Jev). --sym goes into the same batch.
        With images, every question reads the same images (vLLM backend)."""
        from decision_core import options_of
        names = list(questions)
        if self.verify and any((questions[nm] or {}).get("mode") == "verify" for nm in names):
            return {nm: self.decide(state, questions[nm], images=images) for nm in names}  # verify: sequential path
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
            if images:
                res = self.fast.dist_many(state, items, images=images)
            elif self.remote or len(items) < 2:
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

    def decide(self, state, q, images=None):
        from decision_core import options_of
        opts = options_of(q)
        if len(opts) < 2:
            raise ValueError("at least 2 options are required")
        ad = self.verify if (q.get("mode") == "verify" and self.verify) else self.fast
        with self.lock:
            probs, n = ad.dist_any(state, q, opts, images=images)
            if self.sym:
                p2, n2 = ad.dist_any(state, q, list(reversed(opts)), images=images)
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


class BodyTooLarge(Exception):
    pass


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
                self._send(200, {"ok": True, "model": dec.name, "version": VERSION})
            else:
                self._send(404, {"error": "not found"})

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            if n > dec.max_body:
                self.close_connection = True  # the unread body cannot stay on a keep-alive connection
                raise BodyTooLarge(f"request body over {dec.max_body // 1_000_000} MB")
            raw = self.rfile.read(n)
            ctype = self.headers.get("Content-Type", "")
            if ctype.lower().startswith("multipart/form-data"):
                return parse_multipart(ctype, raw)
            return json.loads(raw or b"{}")

        def _images(self, body):
            """The request's images: "images", then "image_data", then the data URIs taken out of the state (body["state"]
            keeps "[image N]" in their place)."""
            items = body.get("images") or []
            if not isinstance(items, list):
                raise ValueError('"images" must be a list')
            extra = body.get("image_data") or []
            items = items + (extra if isinstance(extra, list) else [extra])
            body["state"], lifted = lift_images(body.get("state", ""), start=len(items))
            return load_images(items + lifted, allow_urls=dec.allow_image_urls, max_images=dec.max_images,
                               max_pixels=dec.max_image_pixels, max_bytes=dec.max_image_bytes)

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
                    if has_images(body):
                        return self._send(422, {"error": "sessions take text only; send images to /v1/systemone"})
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
                except BodyTooLarge as e:
                    return self._send(413, {"error": str(e)})
                except ValueError as e:
                    return self._send(422, {"error": str(e)})
                except Exception as e:  # noqa: BLE001
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            if self.path not in ("/v1/systemone", "/v1/evaluate"):
                return self._send(404, {"error": "not found"})
            try:
                body = self._body()
                images = self._images(body)
                res = dec.decide_all(body.get("state", ""), body.get("questions") or {}, images=images)
                answers = {k: v[0] for k, v in res.items()}
                n_in = sum(v[1] for v in res.values())
                self._send(200, {"model": dec.name, "answers": answers,
                                 "usage": {"input_tokens": n_in, "output_tokens": 0},
                                 "latency_s": time.perf_counter() - t0})
            except BodyTooLarge as e:
                self._send(413, {"error": str(e)})
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
    ap.add_argument("--vllm-url", default=None, help="production backend: vLLM URL (serve_vllm.sh); several "
                    "comma-separated URLs (e.g. one server per GPU) spread the requests, least busy first")
    ap.add_argument("--sglang-url", default=None, help="production backend (parallel): URL of the SGLang server")
    ap.add_argument("--max-one-pass", type=int, default=None,
                    help="most options read in one pass (default: the model's decision_config.json, else 588)")
    ap.add_argument("--max-images", type=int, default=4, help="most images per request (vLLM backend)")
    ap.add_argument("--max-image-pixels", type=int, default=3840 * 2160,
                    help="larger images are scaled down to this many pixels, keeping their aspect ratio")
    ap.add_argument("--allow-image-urls", action="store_true",
                    help="let requests give images as http(s) URLs, which the vLLM server then fetches (off by default: "
                         "a server open to others would fetch any address it is given)")
    ap.add_argument("--max-image-mb", type=int, default=48,
                    help="largest image file accepted, before scaling (a 48 MB image is 64 MB in base64)")
    ap.add_argument("--max-body-mb", type=int, default=64, help="largest request body accepted")
    a = ap.parse_args()
    cfg_path = os.path.join(a.model, "decision_config.json")
    cfg = {}
    if os.path.exists(cfg_path):  # exported model: the configuration ships with the weights
        cfg = json.load(open(cfg_path))
        os.environ.setdefault("PROMPT_STYLE", cfg.get("prompt_version", "letter-v1-semif").rsplit("-", 1)[-1])
        if a.calib is None and cfg.get("calib"):
            a.calib = os.path.join(a.model, cfg["calib"])
    import decision_core  # after PROMPT_STYLE is set
    decision_core.set_max_one_pass(a.max_one_pass or cfg.get("max_one_pass"))
    dec = Decider(a)
    dec.max_images, dec.max_image_pixels, dec.allow_image_urls = a.max_images, a.max_image_pixels, a.allow_image_urls
    dec.max_body = a.max_body_mb * 1_000_000
    dec.max_image_bytes = a.max_image_mb * 1_000_000
    dec.decide("warm-up", {"type": "noul", "instructions": "Is this a warm-up?", "criteria": {"true": "yes", "false": "no"}})
    print(f"serve.py {VERSION} ready at http://{a.host}:{a.port} (calib={bool(a.calib)}, sym={a.sym}, backend={'vllm' if a.vllm_url else 'sglang' if a.sglang_url else 'pytorch'}, verify={a.verify_budget}, one pass up to {decision_core.MAX_ONE_PASS} options, images={'on' if a.vllm_url else 'off'})", flush=True)
    ThreadingHTTPServer.request_queue_size = 1024  # the default (5) drops connections when many clients arrive at once
    ThreadingHTTPServer((a.host, a.port), make_handler(dec)).serve_forever()


if __name__ == "__main__":
    main()
