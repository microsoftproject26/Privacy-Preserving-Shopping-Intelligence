#!/usr/bin/env bash
# tests_hist/server/job.sh (owner: Data/integration engineer, DATA-05) - the DATA-05 server job. Launch detached:
#   (nohup setsid bash job.sh > runs_g1/data05/job.log 2>&1 < /dev/null &)
# Writes only under ppsi_v4/runs_g1/data05/ (receipts); the v1 dataset is read-only; no cache or weight is written.
# Every CPU step runs under the v4data.run_bounded 16 GiB RSS guard with CUDA_VISIBLE_DEVICES=-1; the bench step
# exports physical GPU 0 itself (v4int.launch_central.export_environment) after checking that GPU 0 is free.
# Free disk is checked before every step: below 10 GiB the job stops.
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$W/runs_g1/data05
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
M=tests_hist/server/data05_measure.py
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$OUT/parts" "$OUT/logs" "$OUT/tmp"
export TMPDIR="$OUT/tmp"
cd "$W/code" || exit 6
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/job_start_utc.txt"

free_ok() {
  FREE=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
  if [ "$FREE" -lt 10737418240 ]; then
    echo "STOP: free space $FREE bytes is below 10 GiB"; return 1
  fi
  return 0
}

step() {  # step <name> <cuda_visible_devices> <cmd...>
  local name=$1 cvd=$2; shift 2
  if ! free_ok; then
    echo "{\"step\": \"$name\", \"skipped\": \"free disk below 10 GiB\", \"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$OUT/steps.jsonl"
    return 5
  fi
  local t0 f0
  t0=$(date -u +%Y-%m-%dT%H:%M:%SZ); f0=$FREE
  echo "[job] $name start $t0 free=$f0"
  CUDA_VISIBLE_DEVICES=$cvd "$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$OUT/rss.jsonl" -- \
    "$PY" -B "$@" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  local t1 f1
  t1=$(date -u +%Y-%m-%dT%H:%M:%SZ); f1=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
  local g
  g=$(tail -1 "$OUT/rss.jsonl")
  echo "{\"step\": \"$name\", \"rc\": $rc, \"utc_start\": \"$t0\", \"utc_end\": \"$t1\", \"free_bytes_before\": $f0, \"free_bytes_after\": $f1, \"rss_guard\": $g}" >> "$OUT/steps.jsonl"
  echo "[job] $name rc=$rc end $t1"
  return $rc
}

{ date -u; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader;
  nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader; free -g;
  cat /sys/fs/cgroup/memory.current; df -B1 /home/jovyan | tail -1;
  sha256sum v4hist/*.py v4int/*.py tests_hist/*.py tests_hist/server/* tests_int/*.py; } > "$OUT/preflight.txt" 2>&1

step tests -1 tests_hist/run_all.py --receipts-dir "$OUT/tests"
step census -1 "$M" census
for K in train eval; do
  for P in USER_HISTORY_L50 SESSION_L50; do
    step "provider_pool_${K}_${P}" -1 "$M" provider --source pool --kind "$K" --policy "$P" --buckets 0-7
  done
done
for P in USER_HISTORY_L50 SESSION_L50; do
  step "provider_manifest_train_${P}" -1 "$M" provider --source manifest --kind train --policy "$P" --manifest M05_CALIB_TRAIN
  step "provider_manifest_eval_${P}" -1 "$M" provider --source manifest --kind eval --policy "$P" --manifest M06_CALIB_VALIDATION_WARM
done
step bench 0 "$M" bench --buckets 0-3 --max-wait-min 60
step assemble -1 "$M" assemble
rm -rf "$OUT/tmp"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/job_end_utc.txt"
df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ' | sed 's/^/free_bytes_after=/' > "$OUT/df_after.txt"
echo "JOB_DONE $(cat "$OUT/job_end_utc.txt")"
