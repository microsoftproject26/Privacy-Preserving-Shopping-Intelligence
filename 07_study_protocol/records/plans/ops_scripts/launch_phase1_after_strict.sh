#!/bin/bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
test ! -e "$V/runs_ds3/phase1/phase1.log"
nohup setsid -f bash "$V/runs_ds3/phase1_after_strict.sh" > "$V/runs_ds3/phase1_wrapper.log" 2>&1 < /dev/null
echo PHASE1_QUEUE_LAUNCHED
