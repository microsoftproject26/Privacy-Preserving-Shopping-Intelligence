#!/bin/bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
test ! -e "$V/runs_ds3/strict_a40/strict.log"
nohup setsid -f bash "$V/runs_ds3/strict_a40_run_r2.sh" > "$V/runs_ds3/strict_a40_r2_wrapper.log" 2>&1 < /dev/null
echo STRICT_R2_LAUNCHED
