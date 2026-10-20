#!/usr/bin/env bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
RUN=$V/data_cfull/_run
test ! -e "$RUN/final_pre.status"
nohup setsid -f bash "$V/runs_ds3/cfull/final_cfull_pre.sh" > "$RUN/final_pre_wrapper.log" 2>&1 < /dev/null
echo FINAL_LAUNCHED
