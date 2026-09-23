"""Builds quantized GPU checkpoints (vLLM/SGLang) with llm-compressor, in the official Qwen3.5 format
(Qwen3_5ForConditionalGeneration): only the linear layers of the language model are quantized; vision, lm_head,
embeddings, conv1d and the Gated DeltaNet parameters stay in bf16.
  fp8  : FP8_DYNAMIC (per-channel FP8 weights + dynamic per-token FP8 activations), no calibration.
  int4 : GPTQ W4A16 (groups of 128) calibrated on training items (snap_final1), NEVER on evaluation suites.
usage: python quantize_llmc.py <release> <output> <fp8|int4> [n_calib]"""
import glob
import json
import os
import random
import shutil
import sys

import torch
EIKOS_HOME = os.environ.get("EIKOS_HOME", ".")  # data, snapshots, evaluation suites and checkpoints
os.environ.setdefault("PROMPT_STYLE", "semif")

SRC, OUT, SCHEME = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 256
IGNORE = ["re:.*lm_head", "re:.*visual.*", "re:.*vision.*", "re:.*mtp.*", "re:.*embed.*"]

from llmcompressor import oneshot  # noqa: E402
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForImageTextToText.from_pretrained(SRC, dtype=torch.bfloat16, device_map="auto")
print("class:", type(model).__name__, flush=True)

if SCHEME == "fp8":
    oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE))
else:
    from datasets import Dataset

    from decision_core import messages, options_of
    rng = random.Random(7)
    pool = []
    S = os.environ.get("CALIB_SNAPSHOT", f"{EIKOS_HOME}/snap_final1")  # training snapshot (never evaluation data)
    for f in sorted(glob.glob(f"{S}/*.jsonl")):
        if os.path.basename(f).startswith("long"):
            continue  # calibration with short items (the long ones repeat the same pattern and cost memory)
        for line in open(f):
            r = json.loads(line)
            if r.get("agree", True):
                pool.append(r)
    rng.shuffle(pool)
    texts = []
    for r in pool:
        o = options_of(r["question"], list(r["labels"]))
        if len(o) > 26:
            continue
        texts.append(tok.apply_chat_template(messages(r["state"], r["question"], o), tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False))
        if len(texts) >= N:
            break
    assert texts, f"no calibration items found in {S} (set CALIB_SNAPSHOT to a training snapshot)"
    enc = tok(texts, add_special_tokens=False, truncation=True, max_length=2048)
    ds = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    print(f"calibration: {len(texts)} training items, {sum(map(len, enc['input_ids'])) // len(texts)} tokens avg",
          flush=True)
    oneshot(model=model, dataset=ds, processor=tok, recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE),
            max_seq_length=2048, num_calibration_samples=len(texts))

model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
for f in ("calib.json", "decision_config.json", "chat_template.jinja", "preprocessor_config.json",
          "video_preprocessor_config.json", "LICENSE"):
    if os.path.exists(os.path.join(SRC, f)):
        shutil.copy(os.path.join(SRC, f), os.path.join(OUT, f))
tot = sum(os.path.getsize(p) for p in glob.glob(f"{OUT}/*") if os.path.isfile(p))
print(f"saved {OUT}: {tot / 1e9:.2f} GB ({SCHEME})", flush=True)
