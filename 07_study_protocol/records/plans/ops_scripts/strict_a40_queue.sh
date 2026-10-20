#!/bin/bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
CR=$V/code_ds3strict
R=$V/runs_ds3/strict_a40
GATE=$V/runs_ds3/pre/gate_rest.log
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
mkdir -p "$R"
test ! -e "$R/strict.status"
echo "$(date -u +%FT%TZ) STRICT_QUEUED gpu0" > "$R/strict.status"
for _ in $(seq 1 1440); do
  if grep -q 'GATE_REST_DONE rc=' "$G" 2>/dev/null; then break; fi
  sleep 10
done
if ! grep -q 'GATE_REST_DONE rc=' "$G" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) STRICT_REFUSED gate-timeout" >> "$R/strict.status"
  exit 4
fi
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$CR"
echo "$(date -u +%FT%TZ) STRICT_START gpu0" >> "$R/strict.status"
if "$PY" -B tests_train_gpu/sys04c_variant_bench.py all --out-dir "$R" --variants SASREC_D64_B2 SASREC_D256_B3 --reps 2 --updates 300 > "$R/strict.log" 2>&1; then rc=0; else rc=$?; fi
echo "$(date -u +%FT%TZ) STRICT_DONE rc=$rc" >> "$R/strict.status"
exit "$rc"
