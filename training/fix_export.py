"""Repackages the merged model in the same layout as the official Qwen3.5 checkpoint (original config, original names,
vision/MTP weights from the base), swapping only the language weights for ours. This way it runs in transformers,
SGLang and vLLM without adaptation. usage: python fix_export.py <base_snapshot> <merged_dir> <out_dir>
"""
import json
import os
import shutil
import sys

from safetensors.torch import load_file, save_file

base, merged, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for f in os.listdir(base):
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
        shutil.copy(os.path.join(base, f), os.path.join(out, f))
bi = json.load(open(f"{base}/model.safetensors.index.json"))["weight_map"]
mi = json.load(open(f"{merged}/model.safetensors.index.json"))["weight_map"]
ours = {}
for shard in sorted(set(mi.values())):
    ours.update(load_file(f"{merged}/{shard}"))
n_ours = len(ours)
index, trocados = {}, 0
for shard in sorted(set(bi.values())):
    t = load_file(f"{base}/{shard}")
    for k in list(t):
        if k in ours:
            assert ours[k].shape == t[k].shape, (k, ours[k].shape, t[k].shape)
            t[k] = ours.pop(k).contiguous()
            trocados += 1
    save_file(t, f"{out}/{shard}", metadata={"format": "pt"})
    index.update({k: shard for k in t})
if ours:
    print("our keys with no match in the base:", list(ours)[:5])
    save_file({k: v.contiguous() for k, v in ours.items()}, f"{out}/model-extra.safetensors", metadata={"format": "pt"})
    index.update({k: "model-extra.safetensors" for k in ours})
json.dump({"metadata": {}, "weight_map": index}, open(f"{out}/model.safetensors.index.json", "w"), indent=1)
for f in ("calib.json", "decision_config.json"):
    if os.path.exists(f"{merged}/{f}"):
        shutil.copy(f"{merged}/{f}", f"{out}/{f}")
print(f"our weights: {n_ours} | swapped into the original checkpoint: {trocados} | total tensors: {len(index)}")
