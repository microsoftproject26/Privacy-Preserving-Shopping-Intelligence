#!/usr/bin/env bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$V/data_cfull
RUN=$OUT/_run
CR=$V/code_ds3pre
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
test -f "$RUN/build_pre.status"
grep -q 'DATA08_BUILD_DONE rc=0' "$RUN/build_pre.status"
test ! -e "$OUT/receipts/DATA08_SERVER_RECEIPT.json"
mkdir -p "$RUN/tmp"
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1 TMPDIR=$RUN/tmp
cd "$CR"
echo "$(date -u +%FT%TZ) DATA08_FINAL_START code_ds3pre" > "$RUN/final_pre.status"
if "$PY" -B -m v4data08.d8_finalize registry > "$RUN/registry_pre.log" 2>&1 && \
   (cd tests_data08 && D8_PYTEST_BASETEMP="$RUN/pytest_final_pre" "$PY" -B run_all.py "$OUT/receipts/TESTS_server.json" > "$RUN/tests_final_pre.log" 2>&1) && \
   "$PY" -B -m v4data08.d8_finalize final --tests-receipt "$OUT/receipts/TESTS_server.json" --rss-log "$RUN/rss_build_pre.jsonl" --run-dir "$RUN" > "$RUN/final_pre.log" 2>&1; then rc=0; else rc=$?; fi
echo "$(date -u +%FT%TZ) DATA08_FINAL_DONE rc=$rc" >> "$RUN/final_pre.status"
exit "$rc"
