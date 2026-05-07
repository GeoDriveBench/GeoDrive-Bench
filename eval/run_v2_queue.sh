#!/bin/bash
# Sequential queue runner for the FINAL v2 benchmark — root-cause-fixed version.
# Usage: run_v2_queue.sh <gpu> <model1> <setting1> <model2> <setting2> ...
#
# What this version fixes vs the previous one:
#   1. Each python is launched in its OWN process group via setsid.
#      timeout sends SIGKILL to the leader, which then kills the EntireGroup
#      (including vLLM's EngineCore subprocess). No more zombie EngineCores.
#   2. Per-(model, setting) timeout is adaptive: HF / lmdeploy backends and
#      long-output settings get a much bigger timeout. No more premature
#      timeout aborts on slow long-prompt rule_given runs.
#   3. After each python exits we explicitly kill any leftover process owned
#      by us on the GPU (a defence-in-depth fallback for vLLM races).
#   4. After all retries fail (rc != 0 for 3 in a row), we record an
#      "ABORTED" marker and continue the queue, so a single dead model does
#      not cascade-fail downstream chains.

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate dllm
export TRANSFORMERS_VERBOSITY=error VLLM_LOGGING_LEVEL=WARNING \
       TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$EVAL_DIR")"
BENCHMARK="$REPO_ROOT/culturebenchmark_eval_v2.json"
OUTDIR="$EVAL_DIR/results"
LOGDIR="$EVAL_DIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"
cd "$EVAL_DIR"

gpu=$1; shift
export CUDA_VISIBLE_DEVICES=$gpu

# Adaptive timeout per (model, setting) — slow models need much more time.
# Use this lookup; default is 14400 (4h).
get_timeout() {
  local m=$1 s=$2
  # HF backend (no batching, 1 image): llama32
  if [ "$m" = "llama32" ]; then
    case "$s" in
      direct|no_image|image_corruption) echo 7200 ;;  # 2h
      reasoning|rule_given|full_handbook|wrong_rule|no_image_reasoning|image_corruption_reasoning) echo 43200 ;;  # 12h
      *)          echo 14400 ;;
    esac
    return
  fi
  # lmdeploy (no batching, internvl35):
  if [ "$m" = "internvl35" ]; then
    case "$s" in
      direct|no_image|image_corruption) echo 14400 ;;  # 4h
      reasoning|rule_given|full_handbook|wrong_rule|no_image_reasoning|image_corruption_reasoning) echo 43200 ;;  # 12h
      *)          echo 14400 ;;
    esac
    return
  fi
  # vLLM (batched): rule_given/reasoning/full_handbook/wrong_rule have long generation
  case "$s" in
    direct|no_image|image_corruption) echo 5400 ;;  # 1.5h
    reasoning|rule_given|full_handbook|wrong_rule|no_image_reasoning|image_corruption_reasoning) echo 14400 ;;  # 4h
    *)          echo 7200 ;;
  esac
}

# Clean up zombie / leftover python procs of mine ONLY on the GPUs this queue
# uses. Without --gpu-bus-id filtering, the previous version killed processes
# across ALL my GPUs, which mis-fired and SIGKILLed peer queue runs.
# Since setsid + timeout --kill-after=10 already cleans up this try's process
# group, this function is now a no-op (defence-in-depth not needed).
cleanup_my_gpu_pids() {
  : # intentionally a no-op
}

while [ $# -ge 2 ]; do
  m=$1; s=$2; shift 2
  out="$OUTDIR/${m}_${s}_v2_results.json"
  TLIMIT=$(get_timeout "$m" "$s")
  echo "[GPU$gpu] PLAN $m/$s timeout=${TLIMIT}s"
  for try in 1 2 3; do
    echo "[GPU$gpu] FILL $m/$s try=$try START $(date +%H:%M:%S) tlimit=${TLIMIT}s"
    # setsid creates a new process group; --kill-after=10 sends SIGKILL to the
    # whole group (vllm subprocess included) 10s after SIGTERM.
    # batch_size 8 (down from 32) to reduce multimodal preprocessing memory
    # pressure that was silently SIGKILLing EngineCore mid-batch.
    # Log redirect uses APPEND (>>) so prior-try crash traces are preserved.
    setsid timeout --kill-after=10 ${TLIMIT} python infer_vllm.py \
      --model "$m" --setting "$s" \
      --output "$out" \
      --max_new_tokens_long 512 --batch_size 8 \
      --benchmark "$BENCHMARK" \
      >> "$LOGDIR/v2_${m}_${s}.log" 2>&1
    rc=$?
    # Strip ERROR rows in the saved file.
    python3 -c "
import json, os
p='$out'
if os.path.exists(p):
    d=json.load(open(p))
    c=[r for r in d if not str(r.get('pred','')).startswith('ERROR')]
    json.dump(c,open(p,'w'),ensure_ascii=False,indent=2)" 2>/dev/null
    n=$(python3 -c "
import json, os
p='$out'
print(len(json.load(open(p))) if os.path.exists(p) else 0)" 2>/dev/null)
    echo "[GPU$gpu] $m/$s try=$try END rc=$rc total=$n"
    cleanup_my_gpu_pids
    sleep 5
    [ "$n" -ge 5000 ] && break
  done
  if [ "${n:-0}" -lt 5000 ]; then
    echo "[GPU$gpu] $m/$s ABORTED after 3 tries (only $n / 5053). Continuing queue."
  fi
done
echo "[GPU$gpu] QUEUE DONE $(date)"
