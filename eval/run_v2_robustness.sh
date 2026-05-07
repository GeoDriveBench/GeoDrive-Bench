#!/bin/bash
# Robustness ablation on v2 benchmark (5053 items).
# 4 base VLMs × 6 perturbed settings (no_image/image_corruption × direct/reasoning/rule_given).
# Image corruption = Gaussian blur, severity 3 (radius 6 px).
#
# 2-GPU layout (sequential per GPU):
#   GPU 0: llava   → gemma3
#   GPU 1: internvl3 → qwen3vl
#
# Each python is launched in its own process group via setsid so timeout
# kills the whole group (vLLM EngineCore included). Each (model, setting)
# gets up to 3 retries; ABORT after 3 failures and continue.
#
# Usage: nohup bash eval/run_v2_robustness.sh > eval/logs/v2_robust_master.log 2>&1 &
set -o pipefail
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate dllm
export TRANSFORMERS_VERBOSITY=error VLLM_LOGGING_LEVEL=WARNING \
       TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore

EVAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$EVAL_DIR")
BENCHMARK="$REPO_ROOT/culturebenchmark_eval_v2.json"
OUTDIR="$EVAL_DIR/results"; LOGDIR="$EVAL_DIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

GPU0_MODELS=("llava" "gemma3")
GPU1_MODELS=("internvl3" "qwen3vl")
SETTINGS=(
    "no_image"
    "image_corruption"
    "no_image_reasoning"
    "image_corruption_reasoning"
    "no_image_rule_given"
    "image_corruption_rule_given"
)

# direct prompts: 1.5h timeout, batch 16; long-output prompts: 4h, batch 8
get_timeout_batch() {
    case "$1" in
        no_image|image_corruption) echo "5400 16" ;;
        *)                         echo "14400 8" ;;
    esac
}

run_queue() {
    local gpu=$1; shift
    local models=("$@")
    export CUDA_VISIBLE_DEVICES=$gpu
    for model in "${models[@]}"; do
        for s in "${SETTINGS[@]}"; do
            extra=""
            if [[ "$s" == image_corruption* ]]; then
                extra="--corruption_type blur --corruption_severity 3"
            fi
            read -r tlimit batch < <(get_timeout_batch "$s")
            out="$OUTDIR/${model}_${s}_v2_results.json"
            for try in 1 2 3; do
                echo "[GPU$gpu] FILL $model/$s try=$try START $(date +%H:%M:%S) tlimit=${tlimit}s"
                setsid timeout --kill-after=10 ${tlimit} python "$EVAL_DIR/infer_vllm.py" \
                    --model "$model" --setting "$s" \
                    --output "$out" \
                    --max_new_tokens_long 512 --batch_size $batch \
                    --benchmark "$BENCHMARK" $extra \
                    >> "$LOGDIR/v2_robust_${model}_${s}.log" 2>&1
                rc=$?
                python3 -c "
import json, os
p='$out'
if os.path.exists(p):
    d=json.load(open(p))
    c=[r for r in d if not str(r.get('pred','')).startswith('ERROR')]
    json.dump(c,open(p,'w'),ensure_ascii=False,indent=2)" 2>/dev/null
                n=$(python3 -c "import json,os; p='$out'; print(len(json.load(open(p))) if os.path.exists(p) else 0)")
                echo "[GPU$gpu] $model/$s try=$try END rc=$rc total=$n / 5053"
                sleep 5
                [ "$n" -ge 5000 ] && break
            done
            if [ "${n:-0}" -lt 5000 ]; then
                echo "[GPU$gpu] $model/$s ABORTED after 3 tries (only $n / 5053). Continuing."
            fi
        done
        echo "[GPU$gpu] === $model DONE $(date) ==="
    done
    echo "[GPU$gpu] QUEUE DONE $(date)"
}

run_queue 0 "${GPU0_MODELS[@]}" > "$LOGDIR/v2_robust_gpu0.log" 2>&1 &
run_queue 1 "${GPU1_MODELS[@]}" > "$LOGDIR/v2_robust_gpu1.log" 2>&1 &
wait
echo "ALL DONE $(date)"
