#!/bin/bash
# D3 (DL 317 / 319): K9b: FT_C and LO on UNSEEN_HOLDOUT s2026; code_ds3r6
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneD3_PANEL.log; G=1
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r6.sh
SFA=$V/contracts/device/SELECTION_FT_FA.json
wait_file $R/FL_T/FA_WARM_s2026/RUN_SUMMARY.json
wait_file $R/FL_T/FA_WARM_s2027/RUN_SUMMARY.json
wait_file $R/FL_T/FA_WARM_FROZEN_s2026/RUN_SUMMARY.json
wait_file $R/FL_T/FA_WARM_FROZEN_s2027/RUN_SUMMARY.json
job FT_C_UNSEEN_HOLDOUT_s2026 dev --arm FT_C --seed 2026 --role UNSEEN_HOLDOUT --base $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --base-kind central --selection $V/contracts/device/SELECTION_FT_C.json --run-dir $R/DEV/FT_C_UNSEEN_HOLDOUT_s2026 --n-shards 64
job LO_UNSEEN_HOLDOUT_s2026 dev --arm LO --seed 2026 --role UNSEEN_HOLDOUT --selection $V/contracts/device/SELECTION_LO.json --run-dir $R/DEV/LO_UNSEEN_HOLDOUT_s2026 --n-shards 64
step "QUEUE_DONE rc=0"
