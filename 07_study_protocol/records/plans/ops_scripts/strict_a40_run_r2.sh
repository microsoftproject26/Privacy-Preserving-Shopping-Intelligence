#!/bin/bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
CR=$V/code_ds3strict
R=$V/runs_ds3/strict_a40
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
test -f "$R/strict.status"
grep -q 'STRICT_QUEUED gpu0' "$R/strict.status"
test ! -e "$R/strict.log"
test ! -e "$R/SYS04C_VARIANT_BENCH.json"
grep -q 'GATE_REST_DONE rc=0' "$V/runs_ds3/pre/gate_rest.log"
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$CR"
echo "$(date -u +%FT%TZ) STRICT_R1_QUEUE_FAILED unbound_G_before_compute" >> "$R/strict.status"
echo "$(date -u +%FT%TZ) STRICT_START gpu0 code_ds3strict" >> "$R/strict.status"
if "$PY" -B tests_train_gpu/sys04c_variant_bench.py all --out-dir "$R" --variants SASREC_D64_B2 SASREC_D256_B3 --reps 2 --updates 300 > "$R/strict.log" 2>&1; then rc=0; else rc=$?; fi
echo "$(date -u +%FT%TZ) STRICT_DONE rc=$rc" >> "$R/strict.status"
exit "$rc"
