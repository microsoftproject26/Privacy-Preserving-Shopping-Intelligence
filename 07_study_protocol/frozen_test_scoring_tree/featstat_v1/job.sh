#!/usr/bin/env bash
# FEATSTAT-01 server job: unit tests, then fs_run under the 16 GiB RSS guard. CPU only. Outputs JSON/md only.
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
OUT=$W/featstat_v5_100
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
# return freed memory promptly: explicit mmap threshold (disables the dynamic raise) and a small arena count
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=33554432 MALLOC_ARENA_MAX=4
cd "$W/code" || exit 2
echo "job start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
df -B1 /home/jovyan | tail -1
FREE=$(df -B1 /home/jovyan | tail -1 | awk '{print $4}')
if [ "$FREE" -lt 10737418240 ]; then echo "ABORT: free disk below 10 GiB ($FREE)"; exit 4; fi
"$PY" -B featstat_v1/test_fs_core.py > "$OUT/unit_tests.log" 2>&1
TRC=$?
tail -1 "$OUT/unit_tests.log" > "$OUT/FEATSTAT_UNIT_TESTS.json"
if [ "$TRC" -ne 0 ]; then echo "ABORT: unit tests failed rc=$TRC"; exit 5; fi
echo "unit tests ok $(cat "$OUT/FEATSTAT_UNIT_TESTS.json")"
RC=0
"$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$OUT/rss.jsonl" -- \
  "$PY" -B featstat_v1/fs_run.py --out "$OUT" --buckets 0-63 \
  --state "$W/data_v2/TRANSFORM_STATE_CLEAN_V2.json" --fd-workers 3 || RC=$?
echo "job end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$RC"
exit $RC
