#!/bin/bash
# Final training recipes of the released Eikos models. Shared settings:
#   LoRA rank 64 on the Qwen base, 1 epoch, lr 1e-4, seed 7;
#   soft cross-entropy on the option-letter logits, with option-order permutation;
#   auxiliary rationale loss 0.3.
# Recipes:
#   4b-B : Qwen3.5-4B + PT<->EN views as augmentation + symmetric KL between views (0.5)
#   4b-E : 4b-B + light JEPA losses (state-option energy, view invariance, latent rationale: 0.2 each; dropout 0.5)
#   27b  : Qwen3.8-27B, no views, inputs up to 14k tokens (long dossiers <= 12k)
# The released Eikos-4B is the weight average of 4b-B and 4b-E (training/make_soup.py).
# The snapshot comes from data_pipeline/snapshot_final.sh or from release_tools/release_to_train.py.
# usage: scripts/train_final.sh <4b-B|4b-E|27b> <gpu> <out_dir> <snapshot_dir> [views.jsonl[,more]]
set -eu
RECIPE=$1; G=$2; CK=$3; S=$4; VIEWS=${5:-}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
FILES=$(ls "$S"/*.jsonl | grep -v excluir | tr "\n" "," | sed "s/,$//")
case $RECIPE in
  4b-B) BASE=${BASE_MODEL:-Qwen/Qwen3.5-4B}; X="BS=8 GRAD=4 MAXLEN=36000 VIEW_AUG=1 JEPA_VKL_W=0.5 JEPA_V_P=1.0" ;;
  4b-E) BASE=${BASE_MODEL:-Qwen/Qwen3.5-4B}
        X="BS=8 GRAD=4 MAXLEN=36000 VIEW_AUG=1 JEPA_VKL_W=0.5 JEPA_V_P=1.0 JEPA_E_W=0.2 JEPA_V_W=0.2 JEPA_R_W=0.2 JEPA_DROP=0.5" ;;
  27b)  BASE=${BASE_MODEL:-Qwen/Qwen3.8-27B}; X="BS=2 GRAD=16 MAXLEN=14000" ;;
  *) echo "recipe must be 4b-B, 4b-E or 27b"; exit 1 ;;
esac
case $RECIPE in 4b-*) [ -n "$VIEWS" ] || { echo "the 4B recipes need the PT<->EN views file(s) as 5th argument"; exit 1; } ;; esac
mkdir -p "$CK"
# shellcheck disable=SC2086
env DATA="$FILES" EXCLUDE="$S/excluir.json" MAX_PER_SOURCE="$(cat "$S/caps.json")" \
  "HOLDOUT_TOPICS=healthcare administration" HOLDOUT_FAMILIES=tradeoff HOLDOUT_LANGS=Spanish \
  PROMPT_STYLE=semif EPOCHS=1 FULL=0 LR=1e-4 RANK_LORA=64 SEED=7 EVAL_EVERY=400 TOK_BUDGET=40000 \
  DEV_FRAC_GEN=0.08 RATIONALE_W=0.3 VIEWS="$VIEWS" BASE="$BASE" DEVICE="cuda:$G" OUT="$CK" $X \
  "$PY" "$HERE/training/train_dec.py"
# released calibration: T = 1 (see the model card, "Calibration")
"$PY" -c "import json, sys; json.dump({'b': 0.0, 'w': [0, 0, 0, 0], 'features': ['log_ntok', 'log_nopts', 'noul', 'score'], 't_global': 1.0, 'mode': 'T1'}, open(sys.argv[1], 'w'))" "$CK/calib.json"
echo "done: $CK/adapter + $CK/calib.json"
echo "next: python training/merge_export.py <base_dir> $CK/adapter $CK/standalone $CK/calib.json"
