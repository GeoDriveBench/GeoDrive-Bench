#!/bin/bash
# Full SDFT training. MODEL_NAME can be overridden (env var) to chain runs.
#
# Required env vars (or override at the top here):
#   CONDA_ENV        — conda env that has the deps installed (default: driveopd)
#   HF_HOME          — HuggingFace cache root (default: ~/.cache/huggingface)
#   OUT_ROOT         — where checkpoints will be written (default: ./runs)
#   MODEL_NAME       — student model id (default: Qwen/Qwen2.5-VL-7B-Instruct)
#   WANDB_PROJECT    — wandb project (default: geodrive-sdft)
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
export WANDB_PROJECT=${WANDB_PROJECT:-geodrive-sdft}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}
SHORT=$(basename "$MODEL_NAME" | tr '/.' '_')

SRC=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT=${OUT_ROOT:-"$SRC/runs"}
mkdir -p "$OUT_ROOT" "$SRC/logs"
cd "$SRC"
TAG=$(date +%Y%m%d_%H%M%S)
REAL_DIR="$OUT_ROOT/${SHORT}_sdft_$TAG"

echo "=== FULL TRAINING  model=$MODEL_NAME  $(date)  ->  $REAL_DIR ==="
accelerate launch --config_file ds_zero3.yaml main_culturaldrive.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$REAL_DIR" \
    --num_train_epochs 1 --num_prompts_per_batch 2 --learning_rate 1e-5 \
    --max_images 1 --max_prompt_length 2560 --max_completion_length 512 \
    > "$SRC/logs/train_${SHORT}_$TAG.log" 2>&1 || true
TRAIN_RC=$?
echo "TRAIN exit=$TRAIN_RC  $(date)"
ls -la "$REAL_DIR" 2>/dev/null | head -20 || true
