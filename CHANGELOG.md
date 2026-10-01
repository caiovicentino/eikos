# Changelog

## serve v1.3 (2026-10-01)

### What changes

**Images.** Eikos reads images with the vision encoder of its base model, which our training left untouched (the
fine-tuning was text only). `serve.py` v1.3 passes them through vLLM's chat endpoint with the same prompt and the same
one-pass letter readout as text, so every option still gets a calibrated probability and no text is generated.

- **JSON:** add `"images"` to the request, a list of data URIs or base64 strings (or `{"url": ...}` / `{"data": ...}`
  objects). An `image_data` field is read too.
- **multipart/form-data:** the JSON in a `request` field and the images as `image` files. This is the form imajev's
  clients use for TypeSafe's wire format with images.
- **Images inside the state:** image data URIs anywhere in the `state` (a string, a dict or list value, chat-style
  `image_url` parts) are taken out in order and replaced with `[image N]`, the way imajev's server reads them.
- Every question in the request reads the same images, attached before the decision text; with several images, each
  is labelled `[image N]`. With images and no text state, the evidence is "the attached image(s)".
- **Limits:** up to 4 images per request (`--max-images`); images above 3840×2160 pixels are scaled down, keeping
  their aspect ratio (`--max-image-pixels`); files up to 48 MB (`--max-image-mb`) and request bodies up to 64 MB
  (`--max-body-mb`). Image URLs are refused unless the server starts with `--allow-image-urls`: a server open to others
  would otherwise fetch any address it is given.
- **Builds:** the vLLM path of all six GPU builds (27B and 4B, in bf16, FP8 and INT4) has the vision weights; the FP8
  and INT4 builds keep the vision tower in bf16. The three MLX builds have no vision weights (their conversion kept the
  text model only), and the local PyTorch path is text only. Sessions stay text only.
- **vLLM 0.30 quirks handled:** vLLM can answer 500 once to a request whose image it saw in a request it refused (its
  two image caches drift apart); `serve.py` retries that request once. A vLLM server started with the older scripts
  (`--max-logprobs 32`, v1.0 and v1.1) reads images like text: the top 32 labels, the rest spread evenly.

**Other changes**

- An option whose description is `null` now shows its own name. v1.2 showed the text "None".
- When vLLM refuses a request (a prompt longer than the context, an unreadable image), `serve.py` answers 422 with
  vLLM's own message instead of a bare 500.
- `serve_vllm.sh` takes `MAX_MODEL_LEN` (default 16384). A 4K screenshot is about 8,300 tokens.

### Unchanged

- **Text prompts:** byte for byte the same as v1.2.1 for every question without a `null` description: 984 of the
  988 questions of our regression set (the Decision Index compatibility and stress requests); the 4 that differ are
  the ones with `null` descriptions.
- **Text answers:** across those 952 regular questions, no answer changed in two runs against v1.2.1. With the same
  prompts, the probabilities differ only by vLLM's run-to-run noise (mean 0.0001).
- **No retraining:** the images are read zero-shot.

### Measured (zero-shot, one RTX PRO 6000 per model, vLLM 0.30)

Public sets, used for evaluation only. GUI grounding is asked as "where is the element for this action?": a 3×3 grid
described in words (no marks), or a labelled 4×4 grid drawn on the screenshot; the answer is the cell holding the
centre of the element.

| Set | Items | Chance | Eikos-27B-FP8 | Eikos-4B |
|---|---|---|---|---|
| MME (yes/no) | 2,374 | 50% | 89.8% · score 2,495.8 | 87.6% · score 2,397.2 |
| MMStar | 1,493 | 25% | 72.3% | 66.9% |
| SEED-Bench-2-Plus (charts, maps, web pages) | 2,277 | 25% | 74.5% | 72.5% |
| ScreenSpot-v2, no marks (3×3) | 1,272 | 11% | 73.7% | 57.5% |
| ScreenSpot-v2, 4×4 grid drawn | 1,272 | 6% | 60.9% | 53.9% |
| ScreenSpot-Pro, no marks (3×3), 500 of 1,581 | 500 | 11% | 65.0% | 51.8% |

Calibration (ECE of the top answer) is 0.028–0.095 for the 27B and 0.035–0.099 for the 4B. Both are too confident on
SEED-Bench-2-Plus (27B: 84% mean confidence for 74.5% accuracy).

**Against the base model.** Same server, prompt and readout, the base model Eikos-27B was trained from
(Qwen/Qwen3.8-27B, bf16), on 300 items per set. Eikos-27B is better on all six sets and better calibrated on all six:

| Set (300 items each) | Eikos-27B-FP8 | Qwen3.8-27B | only Eikos right / only base right |
|---|---|---|---|
| MME | 90.0% | 73.7% | 50 / 1 |
| MMStar | 70.7% | 65.3% | 33 / 17 |
| ScreenSpot-Pro, no marks | 65.7% | 59.0% | 35 / 15 |
| ScreenSpot-v2, no marks | 71.3% | 68.7% | 25 / 17 |
| SEED-Bench-2-Plus | 74.3% | 73.3% | 12 / 9 |
| ScreenSpot-v2, 4×4 grid drawn | 60.0% | 59.0% | 26 / 23 |

The text-only decision training carried over to images. This compares both models in this one-pass typed format,
not the base model at its best with generated answers.

**Every GPU build** on the same 300 items per set (vLLM 0.30, `serve.py` v1.3):

| 300 items per set | Eikos-27B | Eikos-27B-FP8 | Eikos-27B-INT4 | Eikos-4B | Eikos-4B-FP8 | Eikos-4B-INT4 |
|---|---|---|---|---|---|---|
| MME (yes/no) | 89.0% | 90.0% | 89.7% | 89.7% | 89.3% | 87.3% |
| MMStar | 70.7% | 70.7% | 70.7% | 67.0% | 67.3% | 66.7% |
| SEED-Bench-2-Plus | 74.0% | 74.3% | 73.3% | 73.7% | 74.0% | 71.7% |
| ScreenSpot-v2, no marks | 70.7% | 71.3% | 69.0% | 55.7% | 56.3% | 55.0% |
| ScreenSpot-v2, 4×4 grid drawn | 61.0% | 60.0% | 58.7% | 53.3% | 54.3% | 54.0% |
| ScreenSpot-Pro, no marks | 66.0% | 65.7% | 61.7% | 52.7% | 51.7% | 52.0% |
| All 1,800 items | 71.9% | 72.0% | 70.5% | 65.3% | 65.5% | 64.4% |
| Same answer as bf16: all / confident (≥0.9) | — | 97.6% / 100.0% | 94.8% / 100.0% | — | 95.2% / 100.0% | 88.9% / 99.2% |

Answers that differ from bf16 are almost all on items where bf16 itself was below 0.7 confidence (96–100% of them). The largest gap is Eikos-27B-INT4 on ScreenSpot-Pro: 61.7% against 66.0% for bf16.

## serve v1.2.1 (2026-09-26)

**`serve_vllm.sh` now sets `--max-num-seqs 64`** (change it with the `MAX_NUM_SEQS` environment variable). Without
it, vLLM 0.30 uses its default of 1024 on large GPUs, and every request in flight needs one block of the hybrid
(Mamba) cache. With the script's default memory setting (0.85), one RTX PRO 6000 (96 GB) has 530 such blocks for
Eikos-27B and 1,003 for Eikos-27B-FP8, so the server refused to start ("max_num_seqs (1024) exceeds available Mamba
cache blocks"). The 27B-INT4 and the 4B builds started, with less room on smaller GPUs.

Launched with the published script and with the fixed one on one RTX PRO 6000 (vLLM 0.30), each followed by a test
request:

| build | v1.2 script | v1.2.1 script |
|---|---|---|
| Eikos-27B | does not start (530 blocks) | starts |
| Eikos-27B-FP8 | does not start (1,003 blocks) | starts |
| Eikos-27B-INT4 | starts | starts |
| Eikos-4B | starts | starts |
| Eikos-4B-FP8, Eikos-4B-INT4 | not run | start |

The flag only caps how many requests are processed at once; prompts and the readout are unchanged. Our own servers
have run with 64 since launch.

## serve v1.2 (2026-09-26)

### What changes

**Up to 588 options read in one forward pass.** Options were labelled A–Z, so a question with more than 26 options
went through a tournament: blocks of 20, the top 2 of each, then a final round. That meant two model calls, and an
option that came third in its block could never win. Labels now continue after Z with the two-letter codes AA, AB, …
that the Qwen3.5 tokenizer keeps as one token (562 of the 676 codes; the 114 that split are skipped). A–Z plus these
codes give 588 labels, all read in the same pass.

**How many options are read in one pass is set per model** (`max_one_pass` in `decision_config.json`, or
`serve.py --max-one-pass`), from measurements against the tournament:

- **27B:** 160. One pass is better up to 151 options, the largest lists in our suites.
- **4B:** 100. One pass is better up to about 100 options and worse from about 120. On clinc150 subsets, one pass
  vs tournament: 78.7 vs 75.3 at 60 options, 70.3 vs 69.7 at 100, 64.0 vs 67.3 at 151.

Above the limit, a tournament runs with blocks of that size, and the best of each block go to one final pass.

Model folders downloaded before v1.2 have no `max_one_pass`. With the v1.2 code, add it to their
`decision_config.json` (160 for the 27B builds, 100 for the 4B builds) or pass `serve.py --max-one-pass`; without
it, questions of up to 588 options are read in one pass. For the two MLX 4-bit builds, use the files in their
repositories (v1.1).

- **Unchanged below 27 options:** all 9,301 evaluation items with up to 26 options produce byte-identical prompts to
  v1.0/v1.1, in both prompt styles, so every published result for those items stands.
- **No retraining:** the model reads the new labels zero-shot.

### Measured (Eikos-27B-FP8, one RTX PRO 6000, same vLLM 0.30 engine, v1.1 vs v1.2)

| Items with more than 26 options (7 release suites) | v1.1 tournament | v1.2 one pass |
|---|---|---|
| db_medium (16 items, 70 options) | 100% · 967 ms | 100% · 405 ms |
| db_hard (16 items, 70 options) | 93.8% · 686 ms | 100% · 251 ms |
| banking77 (300 items, 77 options) | 77.3% · 287 ms | 80.0% · 204 ms |
| clinc150 (300 items, 151 options) | 81.3% · 403 ms | 84.7% · 282 ms |

Calibration (ECE, 10 bins) stays in the same range (about 0.12 on banking77 and clinc150).

**Full evaluation of the exact release files** (the published weights with the v1.2 files, offline vLLM, the same
settings as the release evaluation; macro accuracy, release → v1.2):

| build | JevBench | DecisionBench | general | fin | fin2 | trade | rules |
|---|---|---|---|---|---|---|---|
| 27B | 93.99 → 93.99 | 83.62 → 84.30 | 84.90 → 84.94 | 85.37 → 85.37 | 58.04 → 58.29 | 87.16 → 87.17 | 95.09 → 95.27 |
| 27B-FP8 | 94.59 → 93.99 | 83.79 → 83.79 | 85.13 → 85.13 | 85.26 → 85.48 | 56.92 → 57.13 | 87.16 → 87.04 | 95.09 → 95.28 |
| 27B-INT4 | 94.29 → 94.29 | 83.45 → 83.79 | 84.54 → 84.72 | 84.92 → 84.92 | 57.13 → 57.00 | 87.38 → 87.48 | 95.14 → 95.14 |
| 4B | 88.51 → 88.51 | 71.84 → 72.69 | 78.34 → 78.12 | 74.66 → 74.66 | 56.37 → 56.21 | 77.11 → 77.11 | 93.06 → 92.94 |
| 4B-FP8 | 87.75 → 87.75 | 72.52 → 73.21 | 78.25 → 78.27 | 75.23 → 75.23 | 56.75 → 56.75 | 77.31 → 77.31 | 92.92 → 92.92 |
| 4B-INT4 | 87.91 → 87.91 | 71.67 → 71.33 | 78.46 → 78.82 | 75.66 → 75.44 | 55.21 → 55.34 | 77.41 → 77.62 | 92.71 → 92.71 |

The two tasks where every item has more than 26 options:

| build | banking77 (77 options) | clinc150 (151 options) |
|---|---|---|
| 27B | 78.0 → 80.0 | 81.0 → 82.0 |
| 27B-FP8 | 77.0 → 80.0 | 81.7 → 83.0 |
| 27B-INT4 | 77.7 → 79.7 | 82.7 → 86.7 |
| 4B | 70.7 → 72.7 | 67.3 → 67.3 (above the limit: tournament) |
| 4B-FP8 | 71.7 → 73.7 | 68.3 → 66.7 (above the limit: tournament) |
| 4B-INT4 | 71.7 → 74.3 | 66.7 → 68.0 (above the limit: tournament) |

DecisionBench has 32 items with 70 options; the rest of its items, and every other suite, have byte-identical
prompts. Their small differences are items where the top choice is a near-tie (0.2–0.6): they flip when vLLM
re-tunes its kernels for a new model path, and they net out to within about ±0.3 points.

**MLX builds** (MLX on CUDA, the items with more than 26 options; release → v1.2):

| build | DecisionBench (32 items, 70 options) | banking77 (77 options) | clinc150 (151 options) |
|---|---|---|---|
| 4B-MLX-8bit | 84.4 → 96.9 | 71.3 → 72.7 | 68.0 → 68.0 (above the limit: tournament) |
| 27B-MLX-4bit | 96.9 → 100 | 78.0 → 80.0 | 77.7 → 73.0 |
| 4B-MLX-4bit | 84.4 → 87.5 | 72.0 → 69.7 | 67.0 → 65.0 (above the limit: tournament) |

The two MLX 4-bit builds lose accuracy when a long list is read in one pass (clinc150 on the 27B, banking77 on the
4B), unlike the GPU INT4 builds. So they keep their v1.1 files (a tournament above 26 options) until a limit is
measured for them. The 4B-MLX-8bit moves to v1.2.

**Browser agent (browser-use/jev-ultrafast):** on the requests with more than 26 options (Wikipedia and Google Flights
pages, 35–52 elements), the whole request drops from 361 ms to 144 ms, with the same choice in 8 of 10 pages. The
other 2 were equivalent: the same article reached as `option` or as `link`, and a click target on a page where the
chosen operation was DONE. Requests with up to 26 options are unchanged: 228/228 same choices, 120 → 118 ms.

20 recorded runs per task against Jev (both started together, results checked by code):

- **Hotel:** 20/20 vs Jev 19/20.
- **Wikipedia:** 18/18 each, Eikos faster in 13 of 16 runs (2 of 17 with v1.1).
- **Google Flights:** 14/20 vs Jev 19/19, with 6 premature DONEs.

The flights drop is a timing effect, not a different choice. On the captured requests, v1.1 and v1.2 choose the same
action for the same page. The agent only waits 50 ms after a click and re-reads the page only if it changed before
acting. With decisions 2.5× faster, it acts on the date picker before it closes and declares DONE before the search
results are in. v1.1 escaped this because its decisions were slower.

The model knew it was unsure: the 6 wrong DONEs had confidence 0.41–0.50, while all 165 right DONEs across three
sessions had 0.90 or more. The benchmark agent was left unchanged. The fix belongs in the model, which should wait
while a page is still changing.

### Other changes

- **`serve.py`**
  - The connection backlog goes from 5 to 1024. With the default, 32 simultaneous clients got connection resets.
  - Reports `version` in `/health`.
- **`--vllm-url`** takes several comma-separated URLs (e.g. one vLLM server per GPU). Each request goes to the least
  busy server, and all of its calls stay on that server.
- **`letter_adapter`**
  - Works against older vLLM servers started with `--max-logprobs 32`: it reads their top 32 labels and spreads the
    remaining probability.
  - Optional shared-prefix warm-up (`EIKOS_SHARED_PREFIX=1`, off by default): one short call computes the common
    state first, so the N questions of a request read it from the prefix cache instead of computing it N times.
- **`serve_vllm.sh`**
  - `--max-logprobs 600`.
  - `--mamba-cache-mode align`: vLLM 0.30 already falls back to it from `all`, with a deprecation warning.
- **`eval_vllm_suite.py`** and **`mlx_decide.py`** use the same labels. `TOURNAMENT_26=1` reproduces the v1.0/v1.1
  evaluation.

### Throughput notes (one RTX PRO 6000, 27B-FP8, agent requests of ~3k tokens, no cache shared between agents)

- **Default settings:** about 3 decisions/s with the GPU saturated in prefill (about 8.8k tokens/s); p50 5.5 s with
  16 agents at once.
- **Optional high-throughput mode:** `--mamba-ssm-cache-dtype bfloat16` plus `EIKOS_SHARED_PREFIX=1` gives 3.4
  decisions/s (+17%). The Gated DeltaNet state is stored in bf16, which halves the prefix-cache block from 784 to 400
  tokens. Choices on the agent replay were unchanged (86/86), with probabilities within 0.15. Still to pass the full
  evaluation before it can be recommended.
- **No gain:** larger prefill steps (16k tokens) and async scheduling.
- **More throughput:** more GPUs, one vLLM per GPU behind one `serve.py`.

## v1.1 (2026-09-24)

- `serve.py`: HTTP/1.1 keep-alive (clients reuse the connection).
- MLX: questions with more than 26 options use the tournament.

## v1.0 (2026-09-23)

- First release: Eikos-27B and Eikos-4B (bf16, FP8, INT4, MLX), training data and code.
