#!/bin/bash
# Slurm template for SDFT training. Edit the placeholders below for your cluster.
# (Account, partition, output paths, conda env.)
#
#SBATCH --job-name=sdft_train
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sdft_%j.out
#SBATCH --error=logs/sdft_%j.err

set -eo pipefail

CONDA_ENV=${CONDA_ENV:-driveopd}
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "$CONDA_ENV"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS=ignore
export WANDB_MODE=${WANDB_MODE:-offline}    # no login; upload later if desired

SRC=$(cd "$(dirname "$0")" && pwd)
cd "$SRC"
mkdir -p logs runs

JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_DIR="$SRC/runs/smoke_$JOB_TAG"
REAL_DIR="$SRC/runs/sdft_$JOB_TAG"

echo "=== Node: $(hostname)  Job: $JOB_TAG ==="
nvidia-smi -L
echo

# ------------------------------------------------------------
# Phase 1: smoke test (16 samples, 1 accumulation step) to validate
#          model load + generation + loss + backward end-to-end.
# ------------------------------------------------------------
echo "=== PHASE 1: smoke  $(date) ==="
accelerate launch --config_file ds_zero3.yaml main_culturaldrive.py \
    --smoke --output_dir "$SMOKE_DIR" \
    --num_prompts_per_batch 4 \
    --max_train_samples 16 \
    2>&1 | tee logs/smoke_${JOB_TAG}.log
SMOKE_RC=${PIPESTATUS[0]}
echo "SMOKE exit=$SMOKE_RC  $(date)"
if [ "$SMOKE_RC" -ne 0 ]; then
    echo "=== SMOKE FAILED. Halting before full training. ==="
    exit 1
fi

# ------------------------------------------------------------
# Phase 2: full SDFT training.
# ------------------------------------------------------------
echo "=== PHASE 2: full training  $(date) ==="
accelerate launch --config_file ds_zero3.yaml main_culturaldrive.py \
    --output_dir "$REAL_DIR" \
    --num_train_epochs 1 \
    --num_prompts_per_batch 16 \
    --learning_rate 1e-5 \
    --max_prompt_length 2048 \
    --max_completion_length 512 \
    2>&1 | tee logs/train_${JOB_TAG}.log
echo "=== DONE  $(date) ==="
