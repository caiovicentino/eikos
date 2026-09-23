"""Typed-decision distillation: the student (Qwen3.5) learns, in a single forward pass, the teacher's distribution
over the option letters (same prompt as the LetterAdapter used in evaluation).

- Loss: soft cross-entropy -Σ target·log p (≡ KL up to a constant) on the first token of the answer.
- Target: the teacher's probabilities (or exact gold_probs on the programmatic items).
- Augmentation: shuffled option order (choice), yes/no in random order (noul); score in natural order.
- LoRA (default) or full fine-tune (FULL=1). Internal dev set (never trained on) for selection.

usage: python train_dec.py  (config via env vars, see below)
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, evaluation suites and checkpoints
JEVBENCH_DIR = os.environ.get("JEVBENCH_DIR", "jevbench")  # JevBench clone
sys.path.insert(0, JEVBENCH_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from letter_adapter import LETTERS, SYSTEM, render_user  # noqa: E402

E = os.environ.get
BASE = E("BASE")
DATA = E("DATA", f"{EIKOS_HOME}/data/labeled.jsonl,{EIKOS_HOME}/data/prog_prob.jsonl")
OUT = E("OUT", f"{EIKOS_HOME}/ckpt/run")
FULL = E("FULL", "0") == "1"
LR = float(E("LR", "1e-4" if not FULL else "1e-5"))
EPOCHS = float(E("EPOCHS", "2"))
BS = int(E("BS", "8"))
GRAD = int(E("GRAD", "4"))
MAXLEN = int(E("MAXLEN", "6144"))
RANK = int(E("RANK_LORA", "64"))
SEED = int(E("SEED", "7"))
PERM_P = float(E("PERM_P", "1.0"))
ONEHOT_MIX = float(E("ONEHOT_MIX", "0.0"))
DEV_FRAC = float(E("DEV_FRAC", "0.04"))
DEV_FRAC_GEN = float(E("DEV_FRAC_GEN", str(DEV_FRAC)))  # generated items: larger dev fraction = more calibration data
TOK_BUDGET = int(E("TOK_BUDGET", "0"))  # >0: token-budget batches (long items go in smaller batches)
EVAL_EVERY = int(E("EVAL_EVERY", "200"))
MAX_PER_SOURCE = json.loads(E("MAX_PER_SOURCE", "{}"))  # e.g. {"prog_prob": 2000}
DEVICE = E("DEVICE", "cuda:0")
# generalization: whole families/topics/languages held out of training (evaluated separately at the end)
HOLD_FAM = set(filter(None, E("HOLDOUT_FAMILIES", "").split(",")))
HOLD_TOPIC = set(filter(None, E("HOLDOUT_TOPICS", "").split(",")))
HOLD_LANG = set(filter(None, E("HOLDOUT_LANGS", "").split(",")))
EXCLUDE = set(json.load(open(E("EXCLUDE")))) if E("EXCLUDE") else set()  # contaminated ids (dedup_check)
RAT_W = float(E("RATIONALE_W", "0"))  # >0: auxiliary LM loss on the rationale (distilling step-by-step)
RAT_MAX = int(E("RATIONALE_MAX_TOK", "192"))
# JEPA (training only; inference is still a single pass): prediction in representation space
JE_W = float(E("JEPA_E_W", "0"))    # energy readout: E(s,c) = -cos(pred(h_s), proj(option span)) / tau
JV_W = float(E("JEPA_V_W", "0"))    # cross-view invariance: the same decision in another language predicts
                                    # the same representation
JR_W = float(E("JEPA_R_W", "0"))    # latent reasoning: the decision predicts the embedding of the teacher's rationale
JV_P = float(E("JEPA_V_P", "0.5"))  # fraction of the items with a view that enter with their view in each batch
JDIM = int(E("JEPA_DIM", "1024"))
JTAU = float(E("JEPA_TAU", "0.05"))
VIEWS = E("VIEWS", "")               # jsonl {id, view: {state, instructions, criteria, lang}} (make_views.py)
VIEW_AUG = E("VIEW_AUG", "0") == "1"   # views are added as training examples (augmentation), even without JEPA
JVKL_W = float(E("JEPA_VKL_W", "0"))  # output consistency: symmetric KL between the two views' distributions
JLAYER = int(E("JEPA_LAYER", "-1"))   # layer of the V/R latent losses (-1 = 3/4 of the depth); never the answer vector
JR_TARGET = E("JEPA_R_TARGET", "self")  # rationale target: "self" (same model, no gradient) or "base" (LoRA off)
JDROP = float(E("JEPA_DROP", "0.5"))  # "loss dropout": skips the latent losses in a fraction of the micro-batches
JEPA_ON = JE_W > 0 or JV_W > 0 or JR_W > 0 or VIEW_AUG or JVKL_W > 0
os.makedirs(OUT, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)


def log(*a):
    print(*a, flush=True)
    with open(f"{OUT}/train.log", "a") as f:
        print(*a, file=f)


# ---------------- data ----------------
def load_rows():
    rows = []
    for path in DATA.split(","):
        if not path:
            continue
        n_src = {}
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # file still being written
                continue
            if not r.get("agree", True) or r.get("id") in EXCLUDE:
                continue
            src = (r.get("source") or "gen").split(":")[0]
            if src in MAX_PER_SOURCE and n_src.get(src, 0) >= MAX_PER_SOURCE[src]:
                continue
            n_src[src] = n_src.get(src, 0) + 1
            tgt = r.get("gold_probs") or r.get("teacher_probs")
            if not tgt:
                continue
            tgt = {str(k): max(0.0, float(v)) for k, v in tgt.items()}
            s = sum(tgt.values())
            if s <= 0:
                continue
            tgt = {k: v / s for k, v in tgt.items()}
            if ONEHOT_MIX > 0:
                exp = str(r["expected"])
                tgt = {k: (1 - ONEHOT_MIX) * v + ONEHOT_MIX * (1.0 if k == exp else 0.0) for k, v in tgt.items()}
            r["_tgt"] = tgt
            r["_src"] = src
            rows.append(r)
        log(f"[data] {path}: {n_src}")
    return rows


def is_dev(r):
    h = int(hashlib.sha1(str(r.get("id")).encode()).hexdigest()[:8], 16)
    frac = DEV_FRAC_GEN if (r.get("_src") or "gen") == "gen" else DEV_FRAC
    return (h % 10000) < frac * 10000


def canonical_opts(r):
    """(label, description) in the canonical presentation order (same as the LetterAdapter)."""
    q = r["question"]
    crit = q.get("criteria") or {}
    if q["type"] == "noul":
        return [("yes", crit.get("true", "yes")), ("no", crit.get("false", "no"))]
    if q["type"] == "score":
        if isinstance(crit, dict):
            return [(str(l), str(crit.get(str(l), l))) for l in r["labels"]]
        return [(str(i), str(c)) for i, c in enumerate(crit)]
    return [(l, str(crit.get(l, l)) if isinstance(crit, dict) else l) for l in r["labels"]]


def present(r, rng, train=True):
    opts = canonical_opts(r)
    t = r["question"]["type"]
    if train and t in ("choice", "noul") and rng.random() < PERM_P:
        opts = opts[:]
        rng.shuffle(opts)
    tgt = [r["_tgt"].get(lab, 0.0) for lab, _ in opts]
    s = sum(tgt) or 1.0
    return opts, [x / s for x in tgt]


# ---------------- model ----------------
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": DEVICE})
model.config.use_cache = False
LET_IDS = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
if FULL:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
else:
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, bias="none",
                     target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
_tc = getattr(model.config, "text_config", None) or model.config
D_MODEL = _tc.hidden_size


class JEPAHead(torch.nn.Module):
    """JEPA predictors (small, fp32). Can be discarded for letter-readout inference; the energy readout is optional."""

    def __init__(self, d, k):
        super().__init__()

        def mlp(i, o):
            return torch.nn.Sequential(torch.nn.Linear(i, 2 * k), torch.nn.GELU(), torch.nn.Linear(2 * k, o))
        self.pred_ans = mlp(d, k)          # decision state -> predicted answer embedding
        self.opt = torch.nn.Linear(d, k)   # option span in context -> joint space
        self.pred_view = mlp(d, d)         # view A -> view B (model space)
        self.pred_rat = mlp(d, d)          # decision -> rationale (space of the frozen base model)

    def energy_logits(self, h, pooled):
        z = F.normalize(self.pred_ans(h), dim=-1)
        c = F.normalize(self.opt(pooled), dim=-1)
        return (c @ z) / JTAU


jhead = None
if JEPA_ON:
    assert not FULL or JR_W == 0, "JEPA_R_W uses the frozen base model (disable_adapter): LoRA only"
    assert os.environ.get("PROMPT_STYLE", "ours") == "semif" or JE_W == 0, "span energy requires PROMPT_STYLE=semif"
    jhead = JEPAHead(D_MODEL, JDIM).to(DEVICE).float()
    params += list(jhead.parameters())
_JL = {}
if JV_W > 0 or JR_W > 0:
    _base = model.get_base_model() if hasattr(model, "get_base_model") else model
    _layers = _base.model.layers
    if JLAYER < 0:
        JLAYER = int(0.75 * len(_layers))

    def _jl_hook(mod, inp, out):
        _JL["h"] = out[0] if isinstance(out, tuple) else out
    _layers[JLAYER].register_forward_hook(_jl_hook)
VIEW_MAP = {}
for _vp in filter(None, VIEWS.split(",")):
    for _l in open(_vp):
        try:
            _v = json.loads(_l)
            VIEW_MAP[_v["id"]] = _v["view"]
        except Exception:  # noqa: BLE001
            pass


def encode(r, rng, train=True):
    opts, tgt = present(r, rng, train)
    state = r["state"]
    for _ in range(4):
        user = render_user(state, r["question"], opts)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= MAXLEN:
            return ids, tgt, len(opts)
        s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        keep = int(len(s) * MAXLEN / len(ids) * 0.9)
        state = s[: int(keep * 0.7)] + "\n[...]\n" + s[-int(keep * 0.3):]
    return None


def rationale_ids(r, enc):
    """Gold letter + rationale + EOS, for the auxiliary loss (empty if the item has no rationale)."""
    if RAT_W <= 0 or not isinstance(r.get("rationale"), str) or len(r["rationale"]) < 20:
        return []
    _, tgt, n = enc
    g = max(range(n), key=lambda k: tgt[k])
    tail = tok(LETTERS[g] + "\nRationale: " + r["rationale"].strip(), add_special_tokens=False)["input_ids"]
    return tail[:RAT_MAX] + [tok.eos_token_id]


def _parts():
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model, m.lm_head


def forward_with_rationale(encs, rats):
    """Right-padding; logits only at the positions needed: the answer (letters) + the rationale tokens."""
    seqs = [e[0] + rt for e, rt in zip(encs, rats)]
    L = max(len(x) for x in seqs)
    pad = tok.pad_token_id
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    dec, head = _parts()
    H = dec(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE)).last_hidden_state
    ans_pos = torch.tensor([len(e[0]) - 1 for e in encs], device=DEVICE)
    ans_logits = head(H[torch.arange(len(encs), device=DEVICE), ans_pos]).float()
    rows, cols, tgts = [], [], []
    for i, (e, rt) in enumerate(zip(encs, rats)):
        p0 = len(e[0])
        for t in range(len(rt) - 1):  # position p0+t predicts rt[t+1] (rt[0] is the letter, already covered above)
            rows.append(i); cols.append(p0 + t); tgts.append(rt[t + 1])
    rat_loss = torch.zeros((), device=DEVICE)
    if rows:
        hs = H[torch.tensor(rows, device=DEVICE), torch.tensor(cols, device=DEVICE)]
        rat_loss = F.cross_entropy(head(hs).float(), torch.tensor(tgts, device=DEVICE))
    return ans_logits, rat_loss


# ---------------- JEPA ----------------
def option_spans(text, user, opts, q, offsets):
    """Token spans of each option's description inside the semif (JSON) prompt."""
    noul = q.get("type") in ("noul", "boolean")
    shown = (lambda lab: {"yes": "true", "no": "false"}.get(lab, lab)) if noul else (lambda lab: lab)  # noqa: E731
    base = text.find(user)
    k = user.find('"options"')
    if base < 0 or k < 0:
        return None
    spans = []
    for lab, desc in opts:
        needle = json.dumps(f"{shown(lab)}: {desc}", ensure_ascii=False)
        pos = user.find(needle, k)
        if pos < 0:
            return None
        a, b = base + pos + 1, base + pos + len(needle) - 1
        toks = [t for t, (s0, e0) in enumerate(offsets) if e0 > a and s0 < b]
        if not toks:
            return None
        spans.append((toks[0], toks[-1] + 1))
        k = pos + len(needle)
    return spans


def encode_ex(r, opts, tgt, view=None):
    """Like encode(), with the option spans; view = the same decision rewritten (another language), same order."""
    q = r["question"]
    state = r["state"]
    if view is not None:
        q = dict(q, instructions=view["instructions"], criteria=view["criteria"])
        vd = dict(canonical_opts(dict(r, question=q)))
        opts = [(lab, vd.get(lab, desc)) for lab, desc in opts]
        state = view["state"]
    for _ in range(4):
        user = render_user(state, q, opts)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=JE_W > 0)
        ids = enc["input_ids"]
        if len(ids) <= MAXLEN:
            spans = option_spans(text, user, opts, q, enc["offset_mapping"]) if JE_W > 0 else None
            return {"ids": ids, "tgt": tgt, "n": len(opts), "spans": spans}
        st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        keep = int(len(st) * MAXLEN / len(ids) * 0.9)
        state = st[: int(keep * 0.7)] + "\n[...]\n" + st[-int(keep * 0.3):]
    return None


def forward_jepa(exs, rats):
    """Right-padding. Returns the letter logits, the rationale LM loss, h at the answer position, and H."""
    seqs = [e["ids"] + (rt or []) for e, rt in zip(exs, rats)]
    L = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    dec, head = _parts()
    H = dec(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE)).last_hidden_state
    ar = torch.arange(len(exs), device=DEVICE)
    h_ans = H[ar, torch.tensor([len(e["ids"]) - 1 for e in exs], device=DEVICE)]
    ans_logits = head(h_ans).float()
    h_mid = None
    if "h" in _JL and (JV_W > 0 or JR_W > 0):
        h_mid = _JL["h"][ar, torch.tensor([len(e["ids"]) - 1 for e in exs], device=DEVICE)].float()
    rows, cols, tgts = [], [], []
    for i, (e, rt) in enumerate(zip(exs, rats)):
        p0 = len(e["ids"])
        for t in range(len(rt or []) - 1):
            rows.append(i); cols.append(p0 + t); tgts.append(rt[t + 1])
    rat_loss = torch.zeros((), device=DEVICE)
    if rows:
        hs = H[torch.tensor(rows, device=DEVICE), torch.tensor(cols, device=DEVICE)]
        rat_loss = F.cross_entropy(head(hs).float(), torch.tensor(tgts, device=DEVICE))
    return ans_logits, rat_loss, h_ans.float(), H, h_mid


def energy_logits_of(e, i, h_ans, H):
    pooled = torch.stack([H[i, a:b].float().mean(0) for a, b in e["spans"]])
    return jhead.energy_logits(h_ans[i], pooled)


def energy_loss(exs, h_ans, H):
    tot, n = torch.zeros((), device=DEVICE), 0
    for i, e in enumerate(exs):
        if e.get("spans"):
            lp = F.log_softmax(energy_logits_of(e, i, h_ans, H), -1)
            tot = tot - (torch.tensor(e["tgt"], device=DEVICE) * lp).sum()
            n += 1
    return tot / max(n, 1)


def view_kl(ans_logits, exs, pairs):
    """Symmetric KL between the answer distributions of the two views (same option order = same letters)."""
    if not pairs:
        return torch.zeros((), device=DEVICE)
    tot = torch.zeros((), device=DEVICE)
    for i, j in pairs:
        n = exs[i]["n"]
        li = F.log_softmax(ans_logits[i, LET_IDS[:n]], -1)
        lj = F.log_softmax(ans_logits[j, LET_IDS[:n]], -1)
        tot = tot + 0.5 * (F.kl_div(lj, li, log_target=True, reduction="sum") + F.kl_div(li, lj, log_target=True, reduction="sum"))
    return tot / len(pairs)


def view_loss(h_ans, pairs):
    if not pairs:
        return torch.zeros((), device=DEVICE)
    a = torch.stack([h_ans[i] for i, _ in pairs])
    b = torch.stack([h_ans[j] for _, j in pairs])
    la = 1 - F.cosine_similarity(jhead.pred_view(a), b.detach(), dim=-1)
    lb = 1 - F.cosine_similarity(jhead.pred_view(b), a.detach(), dim=-1)
    return 0.5 * (la + lb).mean()


@torch.no_grad()
def rationale_targets(rows):
    """Embedding of the rationale + correct option, at the JEPA layer, at the last token (as in LLM-JEPA).
    JR_TARGET=self: the model itself without gradient (CODI: better than a frozen teacher); base: LoRA disabled."""
    texts = []
    for r in rows:
        gold = str(r.get("expected"))
        crit = (r.get("question") or {}).get("criteria") or {}
        gtxt = crit.get({"yes": "true", "no": "false"}.get(gold, gold), gold) if isinstance(crit, dict) else gold
        texts.append(f"Rationale: {r['rationale'].strip()}\nAnswer: {gold}: {gtxt}")
    toks = [tok(t, add_special_tokens=False)["input_ids"][:RAT_MAX] for t in texts]
    L = max(len(x) for x in toks)
    ids = torch.full((len(toks), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(toks), L), dtype=torch.long)
    for i, x in enumerate(toks):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    dec, _ = _parts()
    ctx = model.disable_adapter() if JR_TARGET == "base" else contextlib.nullcontext()
    with ctx:
        dec(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE))
    last = torch.tensor([len(x) - 1 for x in toks], device=DEVICE)
    return _JL["h"][torch.arange(len(toks), device=DEVICE), last].float()


def rationale_latent_loss(rrows, h_mid):
    idx = [i for i, r in enumerate(rrows) if r is not None and isinstance(r.get("rationale"), str) and len(r["rationale"]) >= 20]
    if not idx or h_mid is None:
        return torch.zeros((), device=DEVICE)
    pred = jhead.pred_rat(h_mid[torch.tensor(idx, device=DEVICE)])  # before the target: the hook gets overwritten
    t = rationale_targets([rrows[i] for i in idx])
    return (1 - F.cosine_similarity(pred, t, dim=-1)).mean()


def jepa_step(chunk, rng):
    """One micro-batch with the JEPA losses. Returns (total loss, dict of components)."""
    exs, rats, pairs, rrows = [], [], [], []
    for r in chunk:
        opts, tgt = present(r, rng)
        e = encode_ex(r, opts, tgt)
        if not e:
            continue
        i = len(exs)
        exs.append(e)
        rats.append(rationale_ids(r, (e["ids"], e["tgt"], e["n"])) if RAT_W > 0 else [])
        rrows.append(r)
        v = VIEW_MAP.get(r.get("id"))
        if v is not None and (JV_W > 0 or VIEW_AUG or JVKL_W > 0) and rng.random() < JV_P:
            ev = encode_ex(r, opts, tgt, view=v)
            if ev:
                exs.append(ev); rats.append([]); rrows.append(None); pairs.append((i, len(exs) - 1))
    if not exs:
        return None, {}
    ans_logits, rat_loss, h_ans, H, h_mid = forward_jepa(exs, rats)
    parts = {"ce": soft_ce(ans_logits, [(e["ids"], e["tgt"], e["n"]) for e in exs])}
    latent_on = rng.random() >= JDROP
    if RAT_W > 0:
        parts["rat_lm"] = RAT_W * rat_loss
    if JE_W > 0:
        parts["energy"] = JE_W * energy_loss(exs, h_ans, H)
    if JVKL_W > 0:
        parts["view_kl"] = JVKL_W * view_kl(ans_logits, exs, pairs)
    if JV_W > 0 and latent_on and h_mid is not None:
        parts["view_latent"] = JV_W * view_loss(h_mid, pairs)
    if JR_W > 0 and latent_on:
        parts["rat_lat"] = JR_W * rationale_latent_loss(rrows, h_mid)
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}


@torch.no_grad()
def evaluate_energy(dev_rows, n_max=600):
    """Accuracy of the energy readout (JEPA) on the dev set — comparable to that of the letter readout."""
    model.eval()
    rng = random.Random(123)
    hits, n = 0, 0
    for b in range(0, min(len(dev_rows), n_max), BS):
        exs = []
        for r in dev_rows[b:b + BS]:
            opts, tgt = present(r, rng, train=False)
            e = encode_ex(r, opts, tgt)
            if e and e.get("spans"):
                exs.append(e)
        if not exs:
            continue
        _, _, h_ans, H, _ = forward_jepa(exs, [[] for _ in exs])
        for i, e in enumerate(exs):
            k = int(energy_logits_of(e, i, h_ans, H).argmax())
            hits += int(k == max(range(e["n"]), key=lambda j: e["tgt"][j]))
            n += 1
    model.train()
    return {"energy_acc": round(hits / max(n, 1), 4), "n": n}


def batch_forward(encs):
    L = max(len(e[0]) for e in encs)
    ids = torch.full((len(encs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(encs), L), dtype=torch.long)
    for i, (x, _, _) in enumerate(encs):
        ids[i, L - len(x):] = torch.tensor(x)
        att[i, L - len(x):] = 1
    out = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), logits_to_keep=1)
    return out.logits[:, -1, :].float()


def soft_ce(logits, encs):
    loss = 0.0
    for i, (_, tgt, n) in enumerate(encs):
        lp = F.log_softmax(logits[i, LET_IDS[:n]], -1)
        loss = loss - (torch.tensor(tgt, device=lp.device) * lp).sum()
    return loss / len(encs)


@torch.no_grad()
def evaluate(dev_rows, n_max=600):
    model.eval()
    rng = random.Random(123)
    hits, nll, confs, corr = 0, 0.0, [], []
    rows = dev_rows[:n_max]
    for b in range(0, len(rows), BS):
        chunk = rows[b:b + BS]
        encs = [e for e in (encode(r, rng, train=False) for r in chunk) if e]
        if not encs:
            continue
        lg = batch_forward(encs)
        for i, (_, tgt, n) in enumerate(encs):
            p = torch.softmax(lg[i, LET_IDS[:n]], -1)
            g = max(range(n), key=lambda k: tgt[k])
            k = int(p.argmax())
            hits += int(k == g)
            nll -= math.log(max(float(p[g]), 1e-9))
            confs.append(float(p[k]))
            corr.append(int(k == g))
    model.train()
    n = max(len(confs), 1)
    ece = 0.0
    for b in range(10):
        idx = [i for i, c in enumerate(confs) if b / 10 < c <= (b + 1) / 10]
        if idx:
            ece += len(idx) / n * abs(sum(corr[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx))
    return {"dev_acc": round(hits / n, 4), "dev_nll": round(nll / n, 4), "dev_ece": round(ece, 4), "n": n}


@torch.no_grad()
def dump_dev(dev_rows):
    """Saves the letter logits + target of every dev item (to fit the temperature outside the public set)."""
    model.eval()
    rng = random.Random(123)
    with open(f"{OUT}/dev_preds.jsonl", "w") as f:
        for b in range(0, len(dev_rows), BS):
            chunk = dev_rows[b:b + BS]
            encs = [(r, encode(r, rng, train=False)) for r in chunk]
            encs = [(r, e) for r, e in encs if e]
            if not encs:
                continue
            lg = batch_forward([e for _, e in encs])
            for i, (r, (ids_, tgt, n)) in enumerate(encs):
                f.write(json.dumps({"id": r.get("id"), "family": r.get("family"), "src": r.get("_src"),
                                    "qtype": r["question"]["type"], "n_tok": len(ids_), "n_opts": n,
                                    "difficulty": r.get("difficulty"),
                                    "logits": lg[i, LET_IDS[:n]].tolist(), "target": tgt}) + "\n")
    model.train()


def main():
    rows = load_rows()
    held = lambda r: (r.get("family") in HOLD_FAM or r.get("topic") in HOLD_TOPIC  # noqa: E731
                      or r.get("lang") in HOLD_LANG)
    gen_eval = [r for r in rows if held(r)]
    rows = [r for r in rows if not held(r)]
    dev = [r for r in rows if is_dev(r)]
    train = [r for r in rows if not is_dev(r)]
    log(f"[generalization] held-out items: {len(gen_eval)} "
        f"(families={sorted(HOLD_FAM)} topics={sorted(HOLD_TOPIC)} languages={sorted(HOLD_LANG)})")
    log(f"[data] train={len(train)} dev={len(dev)} base={BASE} full={FULL} lr={LR} bs={BS}x{GRAD}")
    json.dump(sorted(str(r.get("id")) for r in dev), open(f"{OUT}/dev_ids.json", "w"))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0 if not FULL else 0.01, betas=(0.9, 0.99))
    steps_total = int(math.ceil(len(train) / BS) * EPOCHS / GRAD)
    warm = max(10, int(0.03 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(steps_total, 1)))))
    log(f"[train] optimizer steps={steps_total} warmup={warm} | initial dev: {evaluate(dev, 300)}")
    rng = random.Random(SEED)
    step, micro, t0, hist = 0, 0, time.time(), []
    comp_hist = {}
    if JEPA_ON:
        log(f"[jepa] E={JE_W} V={JV_W} R={JR_W} VKL={JVKL_W} aug={VIEW_AUG} layer={JLAYER} R_target={JR_TARGET} "
            f"drop={JDROP} dim={JDIM} tau={JTAU} views={len(VIEW_MAP)} "
            f"(training items with a view: {sum(1 for r in train if r.get('id') in VIEW_MAP)})")
    ep = 0.0
    model.train()
    while ep < EPOCHS and step < steps_total:
        order = train[:]
        rng.shuffle(order)
        # bucket by length to reduce padding
        blocks = [order[i:i + BS * 32] for i in range(0, len(order), BS * 32)]
        batches = []
        for bl in blocks:
            clen = lambda r: len(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"]))  # noqa: E731
            bl.sort(key=clen)
            if TOK_BUDGET <= 0:
                batches += [bl[i:i + BS] for i in range(0, len(bl), BS)]
                continue
            cur = []  # budget: n_items x largest_item (approx. 1 token ~ 3.5 characters) <= TOK_BUDGET
            for r in bl:
                if cur and (len(cur) + 1) * max(clen(x) for x in cur + [r]) / 3.5 > TOK_BUDGET or len(cur) >= BS:
                    batches.append(cur)
                    cur = []
                cur.append(r)
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        for chunk in batches:
            if JEPA_ON:
                loss, comp = jepa_step(chunk, rng)
                if loss is None:
                    continue
                loss = loss / GRAD
                for k_, v_ in comp.items():
                    comp_hist.setdefault(k_, []).append(v_)
                loss.backward()
                hist.append(float(loss.detach()) * GRAD)
                micro += 1
                if micro % GRAD == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 20 == 0:
                        cs = " ".join(f"{k_}={sum(v_[-80:]) / len(v_[-80:]):.4f}" for k_, v_ in comp_hist.items())
                        log(f"st{step}/{steps_total} loss={sum(hist[-80:]) / len(hist[-80:]):.4f} [{cs}] "
                            f"lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
                    if step % EVAL_EVERY == 0:
                        log(f"[dev] st{step} {evaluate(dev)} {evaluate_energy(dev) if JE_W > 0 else ''}")
                    if step >= steps_total:
                        break
                continue
            pairs = [(r, e) for r, e in ((r, encode(r, rng)) for r in chunk) if e]
            if not pairs:
                continue
            encs = [e for _, e in pairs]
            if RAT_W > 0:
                rats = [rationale_ids(r, e) for r, e in pairs]
                ans_logits, rat_loss = forward_with_rationale(encs, rats)
                loss = (soft_ce(ans_logits, encs) + RAT_W * rat_loss) / GRAD
            else:
                loss = soft_ce(batch_forward(encs), encs) / GRAD
            loss.backward()
            hist.append(float(loss.detach()) * GRAD)
            micro += 1
            if micro % GRAD == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    log(f"st{step}/{steps_total} loss={sum(hist[-80:]) / len(hist[-80:]):.4f} "
                        f"lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
                if step % EVAL_EVERY == 0:
                    log(f"[dev] st{step} {evaluate(dev)}")
                if step >= steps_total:
                    break
        ep += 1
    log(f"[final] {evaluate(dev, 2000)} time={time.time() - t0:.0f}s")
    if JE_W > 0:
        log(f"[final-energy] {evaluate_energy(dev, 2000)}")
    dump_dev(dev)
    if gen_eval:
        for key, vals in (("family", HOLD_FAM), ("topic", HOLD_TOPIC), ("lang", HOLD_LANG)):
            for v in sorted(vals):
                sub = [r for r in gen_eval if r.get(key) == v]
                if sub:
                    log(f"[generalization] {key}={v}: {evaluate(sub, 1000)}")
    if FULL:
        model.save_pretrained(OUT)
    else:
        model.save_pretrained(f"{OUT}/adapter")
    if jhead is not None:
        torch.save({"state": jhead.state_dict(), "d": D_MODEL, "dim": JDIM, "tau": JTAU,
                    "weights": {"E": JE_W, "V": JV_W, "R": JR_W}}, f"{OUT}/jepa_head.pt")
    tok.save_pretrained(OUT)
    json.dump({k: v for k, v in os.environ.items() if k in (
        "BASE", "DATA", "FULL", "LR", "EPOCHS", "BS", "GRAD", "MAXLEN", "RANK_LORA", "SEED", "PERM_P",
        "ONEHOT_MIX", "DEV_FRAC", "MAX_PER_SOURCE", "RATIONALE_W", "JEPA_E_W", "JEPA_V_W", "JEPA_R_W",
        "JEPA_V_P", "JEPA_DIM", "JEPA_TAU", "VIEWS", "PROMPT_STYLE", "VIEW_AUG", "JEPA_VKL_W", "JEPA_LAYER",
        "JEPA_R_TARGET", "JEPA_DROP", "DEV_FRAC_GEN", "TOK_BUDGET", "HOLDOUT_FAMILIES", "HOLDOUT_TOPICS",
        "HOLDOUT_LANGS", "EXCLUDE", "RATIONALE_MAX_TOK", "EVAL_EVERY")}, open(f"{OUT}/run_config.json", "w"), indent=1)
    log("saved", OUT)


if __name__ == "__main__":
    main()
