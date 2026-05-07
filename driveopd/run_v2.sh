#!/bin/bash
# SDFT training on the GeoDrive-Bench v2 evaluation set.
# Usage:
#   MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct bash run_v2.sh
#   MODEL_NAME=OpenGVLab/InternVL3-8B-hf  bash run_v2.sh
#
# Optional env vars:
#   CONDA_ENV   — conda env (default: driveopd)
#   HF_HOME     — HuggingFace cache (default: ~/.cache/huggingface)
#   OUT_ROOT    — checkpoint output root (default: ./runs)
#   DATASET     — path to culturebenchmark_eval_v2.json (default: ../culturebenchmark_eval_v2.json)
set -o pipefail

CONDA_ENV=${CONDA_ENV:-driveopd}
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "$CONDA_ENV"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS=ignore
export WANDB_PROJECT=${WANDB_PROJECT:-geodrive-sdft-v2}
export WANDB_MODE=${WANDB_MODE:-offline}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}
SHORT=$(basename "$MODEL_NAME" | tr '/.' '_')

SRC=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT=${OUT_ROOT:-"$SRC/runs"}
DATASET=${DATASET:-"$SRC/../culturebenchmark_eval_v2.json"}
mkdir -p "$OUT_ROOT" "$SRC/logs"
cd "$SRC"
TAG=$(date +%Y%m%d_%H%M%S)
REAL_DIR="$OUT_ROOT/${SHORT}_sdft_v2_$TAG"

echo "=== TRAINING $MODEL_NAME on $(basename $DATASET) (4 GPU) ==="
echo "  output: $REAL_DIR"
echo "  start:  $(date)"

accelerate launch --config_file ds_zero3.yaml main_culturaldrive.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET" \
    --output_dir "$REAL_DIR" \
    --num_train_epochs 1 --num_prompts_per_batch 2 --learning_rate 1e-5 \
    --max_images 1 --max_prompt_length 6144 --max_completion_length 512 \
    > "$SRC/logs/train_${SHORT}_v2_$TAG.log" 2>&1
TRAIN_RC=$?
echo "TRAIN exit=$TRAIN_RC  $(date)"
ls -la "$REAL_DIR" 2>/dev/null | head -10
