"""Model soup: weight average of merged models from the same base, trained on the same data (Wortsman et al., 2022).
Identical tensors (vision, MTP, whatever the LoRA did not touch) are copied; differing ones become their fp32 mean.
usage: python make_soup.py <out_dir> <release1> <release2> [...]   (folders in the official layout, same index)"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

out, srcs = sys.argv[1], sys.argv[2:]
idx = [json.load(open(os.path.join(s, "model.safetensors.index.json")))["weight_map"] for s in srcs]
assert all(i == idx[0] for i in idx[1:]), "different indexes: the models do not have the same layout"
os.makedirs(out, exist_ok=True)
n_avg = n_same = 0
for sh in sorted(set(idx[0].values())):
    ts = [load_file(os.path.join(s, sh)) for s in srcs]
    res = {}
    for k in ts[0]:
        if all(torch.equal(ts[0][k], t[k]) for t in ts[1:]):
            res[k] = ts[0][k]
            n_same += 1
        else:
            res[k] = (sum(t[k].float() for t in ts) / len(ts)).to(ts[0][k].dtype)
            n_avg += 1
    save_file(res, os.path.join(out, sh), metadata={"format": "pt"})
for f in os.listdir(srcs[0]):
    p = os.path.join(srcs[0], f)
    if os.path.isfile(p) and not f.endswith(".safetensors"):
        shutil.copy(p, os.path.join(out, f))
print(f"soup in {out}: {n_avg} tensors averaged, {n_same} identical ones copied, from {len(srcs)} models")
