#!/bin/bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
test ! -e "$V/runs_ds3/pre/gate_rest.log"
test ! -e "$V/runs_ds3/pre/smoke/PF_r10"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nohup setsid -f bash "$V/runs_ds3/pre/gate_rest_pre.sh" > "$V/runs_ds3/pre/gate_rest_wrapper.log" 2>&1 < /dev/null
echo GATE_REST_LAUNCHED
