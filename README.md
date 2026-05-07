# GeoDrive-Bench

A multi-country driving-scene benchmark for evaluating geo- and culture-aware reasoning in vision-language models, plus the **DriveOPD** Self-Distillation training pipeline used to internalise traffic knowledge into a base VLM.

```
GeoDrive-Bench (this repo)
├── eval/                   evaluation harness (vLLM, HuggingFace, Bedrock, OpenAI)
├── driveopd/               SDFT training pipeline (student / teacher VLM)
├── traffic_handbook/       per-country handbooks used at training/eval
└── README.md (this file)
```

The benchmark JSON + image files are hosted on HuggingFace:
**[`GeoDriveBench/GeoDrive-Bench`](https://huggingface.co/datasets/GeoDriveBench/GeoDrive-Bench)**

- 5,053 multiple-choice questions
- 6 countries: China (cn), USA (us), UK (uk), Japan (jp), Singapore (sg), India (ind)
- 4 task categories: perception, prediction, planning, region
- 7,160 unique driving images from 6 source datasets (nuScenes, ONCE, IDD, CoVLA, LingoQA, Waymo)

---

## 1. Environment

```bash
# 1. Create a conda env (Python 3.10+ recommended)
conda create -n driveopd python=3.10 -y
conda activate driveopd

# 2. Install PyTorch with CUDA 12.1 (or whatever matches your driver)
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest
pip install -U \
  "transformers>=4.46,<5.0" \
  "vllm>=0.10" \
  "accelerate>=0.34" \
  "datasets>=3.0" \
  "trl>=0.10" \
  "deepspeed>=0.15" \
  "lmdeploy>=0.6" \
  pillow tqdm pandas pyarrow boto3 openai mlcroissant

# 4. (HuggingFace cache) — point to a fast local disk if available
export HF_HOME=$HOME/.cache/huggingface
```

`git-lfs` is required if you plan to download the dataset via Croissant or `git clone`:

```bash
conda install -c conda-forge git-lfs && git lfs install
```

---

## 2. Get the Benchmark Data

Three equivalent options.

**Option A — `datasets` library (recommended for inference)**
```python
from datasets import load_dataset
ds = load_dataset("GeoDriveBench/GeoDrive-Bench", split="train")
print(len(ds), ds[0].keys())
```

**Option B — Plain JSON + image files (recommended for evaluation scripts)**
```bash
# Clone the dataset repo (≈ 3 GB, lots of images)
git lfs install
git clone https://huggingface.co/datasets/GeoDriveBench/GeoDrive-Bench data
ls data/                # culturebenchmark_eval_v2.json, images/, README.md
```
The image paths in `culturebenchmark_eval_v2.json` are **relative to the repo root**, so the eval scripts in this repo will work out-of-the-box if you point them at `data/`:

```bash
cp data/culturebenchmark_eval_v2.json .
ln -s data/images images
```

**Option C — Croissant**
```python
from mlcroissant import Dataset
ds = Dataset(jsonld="https://huggingface.co/api/datasets/GeoDriveBench/GeoDrive-Bench/croissant")
records = list(ds.records("default"))
```

### Schema

| Field | Type | Description |
|---|---|---|
| `id` | int | Unique question id |
| `country` | str | `cn`, `us`, `uk`, `jp`, `sg`, `ind` |
| `image_path` | list[str] | Driving frames (relative paths, 1 / 3 / 6 frames per question) |
| `question` | str | Anonymised question (no country names that would leak the answer) |
| `options` | list[str] | Four answer options A–D |
| `answer` | str | Ground-truth letter `A`/`B`/`C`/`D` |
| `question_type` | str | Always `multiple_choice` |
| `question_category` | str | `perception` / `prediction` / `planning` / `region` |
| `rule_reference` | list[str] | Relevant traffic-rule IDs (e.g. `S3`, `S8`) keyed into `traffic_handbook/` |
| `explanation` | str | Human-written explanation |

---

## 3. Run VLM inference

The unified evaluator is **`eval/infer_vllm.py`**. It supports three backends — vLLM (default), HuggingFace transformers, and lmdeploy — and seven prompt settings.

### 3.1 Quick example

```bash
# Single command: evaluate Qwen2.5-VL-7B on the v2 benchmark, "reasoning" setting
python eval/infer_vllm.py \
    --model qwen25vl \
    --setting reasoning \
    --benchmark ./culturebenchmark_eval_v2.json \
    --output  ./eval/results/qwen25vl_reasoning_v2_results.json
```

### 3.2 Built-in models (see `eval/infer_vllm.py::MODEL_REGISTRY`)

| `--model` | HF id |
|---|---|
| `llava` | llava-hf/llava-v1.6-mistral-7b-hf |
| `qwen25vl` | Qwen/Qwen2.5-VL-7B-Instruct |
| `qwen3vl` | Qwen/Qwen3-VL-8B-Instruct |
| `internvl3` | OpenGVLab/InternVL3-8B |
| `internvl35` | OpenGVLab/InternVL3_5-8B |
| `gemma3` | google/gemma-3-12b-it |
| `llama32` | meta-llama/Llama-3.2-11B-Vision-Instruct |
| `driveopd_qwen25vl` | **your** SDFT-tuned Qwen2.5-VL-7B (set `DRIVEOPD_QWEN25VL_PATH`) |
| `driveopd_internvl3` | **your** SDFT-tuned InternVL3-8B (set `DRIVEOPD_INTERNVL3_PATH`) |

### 3.3 Settings (`--setting`)

| Setting | Prompt to the student |
|---|---|
| `direct` | Image + Q + options. Output one letter. |
| `reasoning` | 4-step CoT (geographic → cultural rule → visual → answer). |
| `rule_given` | Provide the *relevant rule excerpt* from the handbook, then 4-step CoT. |
| `full_handbook` | Provide the *entire* country handbook, then 4-step CoT. |
| `wrong_rule` | Provide a *random unrelated rule*, then 4-step CoT (robustness probe). |
| `no_image{,_reasoning,_rule_given}` | Same prompts but no image (knowledge-only baseline). |
| `image_corruption` | Direct prompt over a Gaussian-blurred image (vision-only probe). |

### 3.4 Closed-source / API models

`eval/infer_api.py` mirrors the same interface for OpenAI (e.g. GPT-5) and AWS Bedrock-hosted Qwen3-VL. Set the relevant env vars before running:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# AWS Bedrock
export AWS_BEARER_TOKEN_BEDROCK='...'
export AWS_REGION=us-east-2

python eval/infer_api.py --backend openai --model gpt-5 --setting reasoning \
    --benchmark ./culturebenchmark_eval_v2.json \
    --output ./eval/results/gpt5_reasoning_v2_results.json

# Bedrock-hosted Qwen3-VL-235B
python eval/infer_api.py --backend bedrock --model qwen.qwen3-vl-235b-a22b --setting reasoning \
    --benchmark ./culturebenchmark_eval_v2.json \
    --output ./eval/results/qwen3vl235b_reasoning_v2_results.json
```

### 3.5 Batch / robustness sweeps

```bash
# Run the full multi-model × multi-setting sweep in sequence (queue script)
bash eval/run_v2_queue.sh 0 qwen25vl direct  qwen25vl reasoning  qwen25vl rule_given
bash eval/run_v2_robustness.sh    # 4 base VLMs × 6 perturbation settings
bash eval/run_v2_sdft.sh          # 2 DriveOPD checkpoints × 5 settings
```

### 3.6 Resume / atomic writes

`infer_vllm.py` saves results every 50 rows with **atomic rename** (`*.tmp` → final). Re-running the same `--output` file resumes seamlessly: it loads completed rows, drops `ERROR` rows for retry, and starts where it left off. Pass `--overwrite` to force a fresh run.

---

## 4. Train DriveOPD (Self-Distillation Fine-Tuning)

**Idea.** The student VLM sees the `reasoning` prompt (no rule). The teacher (the same backbone, in-context) sees the `full_handbook` prompt. The student is on-policy distilled from the teacher's chain-of-thought, so the cultural traffic knowledge is internalised into the student's weights — at deploy time no rule needs to be supplied.

### 4.1 Single-node, multi-GPU run

```bash
cd driveopd

# Defaults: Qwen2.5-VL-7B, 4 GPUs, ZeRO-3, 1 epoch, lr 1e-5
MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
DATASET=$(realpath ../culturebenchmark_eval_v2.json) \
OUT_ROOT=./runs \
bash run_v2.sh

# Or InternVL3
MODEL_NAME=OpenGVLab/InternVL3-8B-hf \
DATASET=$(realpath ../culturebenchmark_eval_v2.json) \
bash run_v2.sh
```

Outputs (`runs/<model>_sdft_v2_<timestamp>/checkpoint-*`):
- `checkpoint-100`, `-200`, ... — periodic snapshots
- `checkpoint-XXX` (last) — what you'll plug into `eval/infer_vllm.py`

### 4.2 Slurm

`driveopd/sbatch_train.sh` is a template. Edit `--account`, `--partition`, `CONDA_ENV`, then `sbatch sbatch_train.sh`. It runs a 16-sample smoke test first; full training only starts if the smoke passes.

### 4.3 Plug your DriveOPD checkpoint into the evaluator

```bash
export DRIVEOPD_QWEN25VL_PATH=/abs/path/to/runs/qwen25vl_sdft_v2_*/checkpoint-631
python eval/infer_vllm.py --model driveopd_qwen25vl --setting reasoning \
    --benchmark ./culturebenchmark_eval_v2.json \
    --output  ./eval/results/driveopd_qwen25vl_reasoning_v2_results.json
```

### 4.4 Hyper-parameter knobs (`driveopd/main_culturaldrive.py`)

| Flag | Default | Notes |
|---|---|---|
| `--learning_rate` | 1e-5 | LR for the student |
| `--num_train_epochs` | 1 | one pass over the dataset is usually enough |
| `--num_prompts_per_batch` | 2 | == gradient_accumulation_steps |
| `--max_images` | 1 | training uses 1 image per item to keep prompt < 6 k tokens |
| `--max_prompt_length` | 6144 | needs to fit the full handbook + question + options |
| `--max_completion_length` | 512 | CoT teacher / student output |
| `--ref_model_mixup_alpha` | 0.01 | small EMA mix of student into teacher each step |
| `--smoke` | off | 16-sample sanity check |

---

## 5. Repo layout (complete)

```
eval/
  infer_vllm.py        unified MM evaluator (vLLM / HF / lmdeploy)
  infer_api.py         OpenAI + Bedrock evaluator (same flags)
  _utils.py            answer extraction, atomic resume, etc.
  watchdog.sh          background GPU-usage logger / zombie killer
  run_v2_queue.sh      sequential queue runner (model × setting list)
  run_v2_sdft.sh       SDFT-checkpoint sweep (5 settings × 2 models)
  run_v2_robustness.sh perturbation sweep (no-image / image-corruption)
  analysis/            error analysis + region rule mappings (uses OpenAI)
driveopd/
  main_culturaldrive.py SDFT trainer entry-point (student + teacher)
  distil_trainer.py     KL distillation loop (TRL-based)
  distil_config.py      DistilConfig
  ds_z3.json            DeepSpeed ZeRO-3 config
  ds_zero3.yaml         accelerate config (4 GPUs by default)
  run_local.sh          single-node single-model launcher
  run_v2.sh             v2-benchmark launcher
  run_chain.sh          launch a 2nd run after the 1st finishes
  sbatch_train.sh       Slurm template
traffic_handbook/
  cn.py jp.py uk.py us.py sg.py ind.py   # the rule corpora
  __init__.py            # exports `<country>_traffic_handbook` strings
```

---

## 6. Reproducing the paper

The exact commands used for the paper are listed in `eval/run_v2_*.sh`. The result files end with `_v2_results.json` and are loaded by `eval/_utils.load_with_resume`.

| Component | Script |
|---|---|
| Eval all base VLMs × 3 settings | `run_v2_queue.sh` |
| Robustness (no-image, image-corruption) | `run_v2_robustness.sh` |
| DriveOPD checkpoints × 5 settings | `run_v2_sdft.sh` |
| Aggregation tables | `eval/analysis/` |

---

## License

The dataset and code are released for research use under the CC-BY-4.0 license. The third-party driving image sources (nuScenes, ONCE, IDD, CoVLA, LingoQA, Waymo) are subject to their own licenses; please consult each one before redistribution.
