#!/usr/bin/env bash
# tests_hist/server/job5.sh (owner: Data/integration engineer, DATA-05) - the final pass after the DECISION_LOG 47/48
# keys, the registry verification path and the manifest binding: the test suites, the session-feature measurement
# (every pool VALIDATION decision of buckets 0-7 plus the pool TRAIN decisions of buckets 0-3), the provider
# throughput with the new keys (pool, USER_HISTORY_L50), then the receipt re-assembly. Same rules as job.sh.
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$W/runs_g1/data05
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
M=tests_hist/server/data05_measure.py
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$OUT/parts" "$OUT/logs" "$OUT/tmp5"
export TMPDIR="$OUT/tmp5"
cd "$W/code" || exit 6
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/job5_start_utc.txt"

step() {  # step <name> <cmd...>
  local name=$1; shift
  local FREE
  FREE=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
  if [ "$FREE" -lt 10737418240 ]; then
    echo "{\"step\": \"$name\", \"skipped\": \"free disk below 10 GiB\"}" >> "$OUT/steps.jsonl"; echo "STOP: free $FREE"; return 5
  fi
  local t0; t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[job5] $name start $t0 free=$FREE"
  CUDA_VISIBLE_DEVICES=-1 "$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$OUT/rss.jsonl" -- \
    "$PY" -B "$@" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  local t1; t1=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local f1; f1=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
  echo "{\"step\": \"$name\", \"rc\": $rc, \"utc_start\": \"$t0\", \"utc_end\": \"$t1\", \"free_bytes_before\": $FREE, \"free_bytes_after\": $f1, \"rss_guard\": $(tail -1 "$OUT/rss.jsonl")}" >> "$OUT/steps.jsonl"
  echo "[job5] $name rc=$rc end $t1"
  return $rc
}

{ date -u; sha256sum v4hist/*.py v4int/*.py tests_hist/*.py tests_hist/server/* tests_int/*.py; } > "$OUT/preflight5.txt" 2>&1
step tests_final3 tests_hist/run_all.py --receipts-dir "$OUT/tests_final3"
step session_features "$M" session-features --buckets 0-7
for K in train eval; do
  step "provider_pool_${K}_USER_HISTORY_L50_with_session_keys" "$M" provider --source pool --kind "$K" \
    --policy USER_HISTORY_L50 --buckets 0-7 --tag with_session_keys
done
step assemble5 "$M" assemble
rm -rf "$OUT/tmp5"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/job5_end_utc.txt"
echo "JOB5_DONE $(cat "$OUT/job5_end_utc.txt")"
