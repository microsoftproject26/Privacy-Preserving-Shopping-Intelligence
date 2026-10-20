#!/usr/bin/env bash
# v4data08/job_cfull_dryrun.sh (owner: Data engineer, DATA-08) - server job 3 (after job_final.sh overall_pass): the
# C_FULL launcher dry run through the shim (v4int.c_full --dry-run-out): extension + view + BOTH real providers (the
# 26.7M-row TRAIN cache: the training-data-loading RSS measurement) + comparison key + RunConfig.validate() + the
# trainer's own registry binding checks. No trainer, no GPU kernel, nothing written under the run dir. Guard 16 GiB
# (DATA-08 packet); if the guard trips, the handoff's fallback applies. Usage: bash job_cfull_dryrun.sh <seed> <gpu>
set -uo pipefail
SEED=${1:-2026}; GPU=${2:-0}
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
RUN=$W/data_cfull/_run
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
cd "$W/code" || exit 6
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/dryrun_s${SEED}_start_utc.txt"
"$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$RUN/rss_dryrun_s${SEED}.jsonl" -- \
  "$PY" -B "$W/code/v4int/c_full.py" --dry-run-out "$RUN/C_FULL_DRYRUN_s${SEED}.json" -- \
  --family SASREC --seed "$SEED" --phase MAIN --recipe-id 02B_SASREC --gpu "$GPU" \
  --run-dir "$W/runs_cfull/C_FULL_02B_SASREC_s${SEED}" --history-policy USER_HISTORY_L50 > "$RUN/dryrun_s${SEED}.log" 2>&1
RC=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/dryrun_s${SEED}_end_utc.txt"
tail -3 "$RUN/dryrun_s${SEED}.log"
echo "JOB_DRYRUN_DONE rc=$RC"
