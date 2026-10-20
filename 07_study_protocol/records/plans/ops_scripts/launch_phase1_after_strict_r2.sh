#!/bin/bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
test ! -e "$V/runs_ds3/phase1_r2/phase1_r2.log"
test ! -e "$V/contracts/eval_device/EVAL_DEVICE_PIN_A40.json"
nohup setsid -f bash "$V/runs_ds3/phase1_after_strict_r2.sh" > "$V/runs_ds3/phase1_r2_wrapper.log" 2>&1 < /dev/null
echo PHASE1_R2_QUEUE_LAUNCHED
