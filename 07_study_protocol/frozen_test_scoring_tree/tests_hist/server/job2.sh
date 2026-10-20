#!/usr/bin/env bash
# tests_hist/server/job2.sh (owner: Data/integration engineer, DATA-05) - the final server pass after the
# transform_state addition (DECISION_LOG 42): the final test suites, the provider throughput with an explicit
# transform state (the locked Clean V2 state file, so the values are the default's), then the receipt re-assembly.
# Same rules as job.sh: 16 GiB RSS guard, CPU only, free-disk check before each step, temporary state deleted.
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$W/runs_g1/data05
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
M=tests_hist/server/data05_measure.py
TS=$W/data_v2/TRANSFORM_STATE_CLEAN_V2.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$OUT/parts" "$OUT/logs" "$OUT/tmp"
export TMPDIR="$OUT/tmp"
cd "$W/code" || exit 6
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/job2_start_utc.txt"

step() {  # step <name> <cmd...>
  local name=$1; shift
  local FREE
  FREE=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
  if [ "$FREE" -lt 10737418240 ]; then
    echo "{\"step\": \"$name\", \"skipped\": \"free disk below 10 GiB\"}" >> "$OUT/steps.jsonl"; echo "STOP: free $FREE"; return 5
  fi
  local t0; t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[job2] $name start $t0 free=$FREE"
  CUDA_VISIBLE_DEVICES=-1 "$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$OUT/rss.jsonl" -- \
    "$PY" -B "$@" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  local t1; t1=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local f1; f1=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
  echo "{\"step\": \"$name\", \"rc\": $rc, \"utc_start\": \"$t0\", \"utc_end\": \"$t1\", \"free_bytes_before\": $FREE, \"free_bytes_after\": $f1, \"rss_guard\": $(tail -1 "$OUT/rss.jsonl")}" >> "$OUT/steps.jsonl"
  echo "[job2] $name rc=$rc end $t1"
  return $rc
}

{ date -u; sha256sum v4hist/*.py v4int/*.py tests_hist/*.py tests_hist/server/* tests_int/*.py "$TS"; } > "$OUT/preflight2.txt" 2>&1
step tests_final tests_hist/run_all.py --receipts-dir "$OUT/tests_final"
step provider_manifest_train_USER_HISTORY_L50_with_transform_state "$M" provider --source manifest --kind train \
  --policy USER_HISTORY_L50 --manifest M05_CALIB_TRAIN --transform-state "$TS"
step provider_manifest_eval_USER_HISTORY_L50_with_transform_state "$M" provider --source manifest --kind eval \
  --policy USER_HISTORY_L50 --manifest M06_CALIB_VALIDATION_WARM --transform-state "$TS"
step assemble2 "$M" assemble
rm -rf "$OUT/tmp"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/job2_end_utc.txt"
echo "JOB2_DONE $(cat "$OUT/job2_end_utc.txt")"
