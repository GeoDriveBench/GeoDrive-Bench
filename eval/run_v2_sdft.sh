#!/bin/bash
# v2 SDFT evaluation: 2 SDFT-v2 checkpoints × 5 settings on v2 benchmark.
#   Models   : qwen25vl_sdft_v2, internvl3_sdft_v2
#   Settings : direct, reasoning, rule_given, full_handbook, wrong_rule
#
# 2-GPU layout (sequential per GPU):
#   GPU 0 : qwen25vl_sdft_v2
#   GPU 1 : internvl3_sdft_v2
#
# Each python is launched in its own process group via setsid so timeout
# kills the whole group (vLLM EngineCore included). Up to 3 retries per
# (model, setting); ABORT after 3 failures and continue.
#
# Usage: nohup bash eval/run_v2_sdft.sh > eval/logs/v2_sdft_master.log 2>&1 &
set -o pipefail
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate dllm
export TRANSFORMERS_VERBOSITY=error VLLM_LOGGING_LEVEL=WARNING \
       TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore

EVAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$EVAL_DIR")
BENCHMARK="$REPO_ROOT/culturebenchmark_eval_v2.json"
OUTDIR="$EVAL_DIR/results"; LOGDIR="$EVAL_DIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

SETTINGS=("direct" "reasoning" "rule_given" "full_handbook" "wrong_rule")

# direct: 1.5h, batch 16; long-output: 4h, batch 8
get_timeout_batch() {
    case "$1" in
        direct) echo "5400 16" ;;
        *)      echo "14400 8" ;;
    esac
}

run_queue() {
    local gpu=$1
    local model=$2
    export CUDA_VISIBLE_DEVICES=$gpu
    for s in "${SETTINGS[@]}"; do
        read -r tlimit batch < <(get_timeout_batch "$s")
        out="$OUTDIR/${model}_${s}_v2_results.json"
        for try in 1 2 3; do
            echo "[GPU$gpu] FILL $model/$s try=$try START $(date +%H:%M:%S) tlimit=${tlimit}s"
            setsid timeout --kill-after=10 ${tlimit} python "$EVAL_DIR/infer_vllm.py" \
                --model "$model" --setting "$s" \
                --output "$out" \
                --max_new_tokens_long 512 --batch_size $batch \
                --benchmark "$BENCHMARK" \
                >> "$LOGDIR/v2_sdft_${model}_${s}.log" 2>&1
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
    echo "[GPU$gpu] === $model QUEUE DONE $(date) ==="
}

run_queue 0 qwen25vl_sdft_v2  > "$LOGDIR/v2_sdft_gpu0.log" 2>&1 &
run_queue 1 internvl3_sdft_v2 > "$LOGDIR/v2_sdft_gpu1.log" 2>&1 &
wait
echo "ALL DONE $(date)"
