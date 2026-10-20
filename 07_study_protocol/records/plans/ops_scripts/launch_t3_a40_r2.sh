#!/usr/bin/env bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
E=$V/runs_ds3/T3_A40_r2
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
test ! -e "$E/out/launch.status"
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
echo "$(date -u +%FT%TZ) T3_A40_R2_START gpu0" > "$E/out/launch.status"
nohup setsid -f bash -c '"$1" -B "$2" --gpu 0 --families GRU,SASREC --n-threads 4 > "$3" 2>&1; rc=$?; echo "$(date -u +%FT%TZ) T3_A40_R2_DONE rc=$rc" >> "$4"' _ "$PY" "$E/code/gpu_t3_receipt.py" "$E/out/stdout.txt" "$E/out/launch.status" < /dev/null
echo T3_A40_R2_LAUNCHED
