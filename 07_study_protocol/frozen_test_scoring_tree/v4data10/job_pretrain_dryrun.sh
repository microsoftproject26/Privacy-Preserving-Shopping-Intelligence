#!/usr/bin/env bash
# v4data10/job_pretrain_dryrun.sh (owner: Data engineer, DATA-10) - server job 3 (after job_final.sh overall_pass): the
# PRETRAIN launcher dry run through the shim (v4int.pretrain_small --dry-run-out): extension + view + the SMALL recipe +
# BOTH real providers (the 24.1M-row TRAIN cache: the training-data-loading RSS measurement) + comparison key +
# RunConfig.validate() (run_label PRETRAIN_FOR_FL, effective_auto_continuation) + the trainer's own registry binding
# checks. No trainer, no GPU kernel, nothing written under the run dir. Guard 24 GiB (DATA-10 packet item 4).
# Usage: bash job_pretrain_dryrun.sh <gpu>
set -uo pipefail
GPU=${1:?usage: job_pretrain_dryrun.sh <gpu>}
SEED=2026
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
RUN=$W/data_pretrain/_run
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
mkdir -p "$RUN"
cd "$W/code" || exit 6
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/dryrun_s${SEED}_start_utc.txt"
"$PY" -B -m v4data.run_bounded --max-rss-mib 24576 --log "$RUN/rss_dryrun_s${SEED}.jsonl" -- \
  "$PY" -B "$W/code/v4int/pretrain_small.py" --dry-run-out "$RUN/PRETRAIN_DRYRUN_s${SEED}.json" -- \
  --family SASREC --seed "$SEED" --phase MAIN --recipe-id SMALL_SASREC_D64_B2 --gpu "$GPU" \
  --run-dir "$W/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s${SEED}" --history-policy USER_HISTORY_L50 \
  > "$RUN/dryrun_s${SEED}.log" 2>&1
RC=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/dryrun_s${SEED}_end_utc.txt"
tail -3 "$RUN/dryrun_s${SEED}.log"
echo "JOB_DRYRUN_DONE rc=$RC"
