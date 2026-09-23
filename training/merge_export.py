"""Merges the LoRA into the weights and exports a single standalone model (safetensors + tokenizer + calibration).
Whoever downloads it uses only this folder: no separate LoRA, no external LLM, no API.
usage: python merge_export.py <base> <adapter> <out_dir> [calib.json]
"""
import json
import os
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, adapter, out = sys.argv[1:4]
calib = sys.argv[4] if len(sys.argv) > 4 else None
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"})
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(base).save_pretrained(out)
if calib:
    shutil.copy(calib, os.path.join(out, "calib.json"))
json.dump({"prompt_version": f"letter-v1-{os.environ.get('PROMPT_STYLE', 'ours')}", "readout": "letter-logit",
           "calib": "calib.json" if calib else None}, open(os.path.join(out, "decision_config.json"), "w"), indent=1)
tot = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
print(f"exported {out}: {tot / 1e9:.2f} GB, params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
# Official checkpoint layout (the Qwen3_5ForCausalLM saved above does not load in vLLM/SGLang): swaps the language
# weights of the base checkpoint for ours and keeps the original config, vision and MTP weights
if os.environ.get("OFFICIAL_LAYOUT", "1") == "1":
    import subprocess
    rel = out.rstrip("/") + "_release"
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fix_export.py"),
                    base, out, rel], check=True)
    print(f"official layout in {rel}")
