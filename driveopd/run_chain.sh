#!/bin/bash
# Wait for the current SDFT run to finish, then launch a follow-up
# (e.g. InternVL3-8B) with the same SDFT config.
#
# Usage:
#     nohup bash run_chain.sh > logs/chain.log 2>&1 &
set -o pipefail

SRC=$(cd "$(dirname "$0")" && pwd)
cd "$SRC"

echo "=== CHAIN: waiting for running SDFT to finish ($(date)) ==="
while pgrep -f "main_culturaldrive" > /dev/null 2>&1; do
    sleep 120
done
echo "=== CHAIN: previous run ended at $(date). Launching follow-up ==="

# Give GPUs a moment to fully release
sleep 30

export MODEL_NAME=${NEXT_MODEL:-"OpenGVLab/InternVL3-8B-hf"}
bash "$SRC/run_local.sh"
echo "=== CHAIN: follow-up run finished at $(date) ==="
