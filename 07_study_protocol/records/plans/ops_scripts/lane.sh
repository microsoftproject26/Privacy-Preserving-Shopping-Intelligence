#!/bin/bash
# MASTER_PLAN_V5 lane runner: runs the job lines of <lane>.jobs sequentially on OUR A40, one at a time, resumable.
# Usage (server, detached): (nohup setsid bash lane.sh <lane_name> > <lane_name>.nohup 2>&1 < /dev/null &)
# <lane_name>.jobs: one job per line = "<run_dir> <full command>"; a line is skipped when <run_dir>/RUN_SUMMARY.json exists
# (DONE) and re-run with --resume when <run_dir>/ckpt/latest.pt or <run_dir>/shards exists. Log: <lane_name>.log; end: QUEUE_DONE rc=<n>.
set -uo pipefail
LANE=${1:?lane name}
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
CR=$V/code_ds3
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
Q=$V/runs_ds3/queues
LOG=$Q/$LANE.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "$CR" || { echo "$(ts) REFUSED: no $CR" >> "$LOG"; echo "QUEUE_DONE rc=90" >> "$LOG"; exit 90; }
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
RC_ALL=0
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue; case "$line" in \#*) continue;; esac
  RUN_DIR=${line%% *}; CMD=${line#* }
  if [ -f "$RUN_DIR/RUN_SUMMARY.json" ] || [ -f "$RUN_DIR/RESULT_ROWS.json" ]; then echo "$(ts) SKIP done $RUN_DIR" >> "$LOG"; continue; fi
  if [ -f "$RUN_DIR/ckpt/latest.pt" ] || [ -d "$RUN_DIR/shards" ]; then CMD="$CMD --resume"; fi
  mkdir -p "$RUN_DIR"
  echo "$(ts) START $RUN_DIR :: $CMD" >> "$LOG"
  $PY -B $CMD > "$RUN_DIR/launch.log" 2>&1; rc=$?
  echo "$(ts) JOB $RUN_DIR rc=$rc" >> "$LOG"
  [ $rc -ne 0 ] && RC_ALL=1
done < "$Q/$LANE.jobs"
echo "QUEUE_DONE rc=$RC_ALL" >> "$LOG"
