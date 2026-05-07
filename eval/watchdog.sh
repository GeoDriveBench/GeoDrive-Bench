#!/bin/bash
# Background watchdog: every 5 min, dump GPU + result-file progress to a log,
# and auto-kill orphaned vLLM EngineCore processes (parent already dead but
# GPU memory still held).
#
# Configure with env vars:
#   RESULTS_DIR — where evaluator writes <model>_<setting>_v2_results.json
#                 (default: ./results)
#   LOG         — output log path (default: ./logs/watchdog.log)
SRC=$(cd "$(dirname "$0")" && pwd)
RESULTS_DIR=${RESULTS_DIR:-"$SRC/results"}
LOG=${LOG:-"$SRC/logs/watchdog.log"}
mkdir -p "$(dirname "$LOG")"

echo "watchdog started $(date)" >> $LOG
while true; do
  ts=$(date +"%H:%M:%S")
  echo "--- $ts ---" >> $LOG
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $LOG
  for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null); do
    user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    parent=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ "$user" = "$USER" ] && [ "$parent" = "1" ]; then
      runtime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
      echo "  KILLING zombie EngineCore pid=$pid runtime=$runtime" >> $LOG
      kill -9 "$pid" 2>/dev/null
    fi
  done
  for f in "$RESULTS_DIR"/*_v2_results.json; do
    [ -f "$f" ] || continue
    fname=$(basename "$f" _v2_results.json)
    cnt=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null)
    printf "  %-30s %4s\n" "$fname" "${cnt:-0}" >> $LOG
  done
  sleep 300
done
