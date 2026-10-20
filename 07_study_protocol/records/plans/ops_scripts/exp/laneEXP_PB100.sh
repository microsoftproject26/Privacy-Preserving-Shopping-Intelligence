#!/usr/bin/env bash
# laneEXP_PB100.sh — EXP-PB100 (exploratory, DL 486): theta_pre(100k) on PRE ranks [131072, 231072), then warm FL on all
# MAIN users from it with the two registered Phase-1 warm recipes (S8 FA_WARM_FROZEN, S7 FA_WARM), concurrently.
# Sequential, stops at the first failure (marker FAIL), resumable by markers in $X/_state. Value-free log.
set -u
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; CODE=$W/code_ds3r7; X=$W/exp_pb100; R=$W/runs_exp_pb100; ST=$R/_state; G=${G:-1}
mkdir -p $R $ST; LOG=$R/lane.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
step() { local n=$1; shift; [ -f $ST/$n.ok ] && return 0; say "$n: start"; if "$@" >> $R/$n.out 2>&1; then touch $ST/$n.ok; say "$n: ok"; else touch $ST/FAIL; say "$n: FAIL rc=$?"; exit 1; fi; }
h_running() { ps -eo args | grep -q -E '^bash .*[H]_test\.sh'; }
h_running && { say "H step running: refuse"; exit 2; }
export PB100_CODE=$CODE PYTHONDONTWRITEBYTECODE=1 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
cd $CODE
step band env CUDA_VISIBLE_DEVICES="" $PY -B $X/pb100_build.py band
RC=$W/data_pretrain_pb100/receipts/DATA10_BAND_RECEIPT.json
export PB100_NDEC=$($PY -c "import json;print(json.load(open('$RC'))['n_eligible_examples'])")
export PB100_SHA=$($PY -c "import json;print(json.load(open('$RC'))['population']['population_sha256_selection_key_v2'])")
step build env CUDA_VISIBLE_DEVICES="" $PY -B -m v4data.run_bounded --max-rss-mib 16384 --log $R/rss.jsonl -- $PY -B $X/pb100_build.py build
step registry env CUDA_VISIBLE_DEVICES="" $PY -B $X/pb100_build.py registry
unset CUDA_VISIBLE_DEVICES; export CUBLAS_WORKSPACE_CONFIG=:4096:8
RD=$R/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_PB100K_s2026; EXT=$W/contracts/manifest_registry/REGISTRY_V2_PRETRAIN_EXT_PB100.json
A=(-- --family SASREC --seed 2026 --phase MAIN --recipe-id SMALL_SASREC_D64_B2 --history-policy USER_HISTORY_L50 --gpu $G --allow-shared-gpu --coschedule-06g --run-dir $RD)
step pretrain_dry $PY -B $X/pretrain_pb100.py --ext-registry $EXT --dry-run-out $R/pretrain_dry.json "${A[@]}"
step pretrain $PY -B $X/pretrain_pb100.py --ext-registry $EXT "${A[@]}"
step flinit $PY -B $X/pretrain_pb100.py --ext-registry $EXT --fl-init-only "${A[@]}"
I=(--init-from $RD/PRETRAIN_FOR_FL_INIT.json --fl-tuning $W/contracts/fl/FL_TUNING_SELECTION.json)
FLC=(--gpu $G --history-policy USER_HISTORY_L50 --allow-shared-gpu)
S8RD=$R/FA_WARM_FROZEN_PB100K_s2026; S7RD=$R/FA_WARM_PB100K_s2026
step fl_dry_s8 $PY -B $X/launch_fl_pb100.py "${FLC[@]}" --arm FA_WARM_FROZEN --seed 2026 "${I[@]}" --freeze-item-tables --warm-selection $W/contracts/fl/FL_WARM_SELECTION_S8.json --run-dir $S8RD --dry-run
step fl_dry_s7 $PY -B $X/launch_fl_pb100.py "${FLC[@]}" --arm FA_WARM --seed 2026 "${I[@]}" --warm-selection $W/contracts/fl/FL_WARM_SELECTION_S7.json --run-dir $S7RD --dry-run
say "fl: start S8 + S7 concurrently (gpu $G)"
( step fl_s8 $PY -B $X/launch_fl_pb100.py "${FLC[@]}" --arm FA_WARM_FROZEN --seed 2026 "${I[@]}" --freeze-item-tables --warm-selection $W/contracts/fl/FL_WARM_SELECTION_S8.json --run-dir $S8RD ) &
( step fl_s7 $PY -B $X/launch_fl_pb100.py "${FLC[@]}" --arm FA_WARM --seed 2026 "${I[@]}" --warm-selection $W/contracts/fl/FL_WARM_SELECTION_S7.json --run-dir $S7RD ) &
wait; say "lane done"
