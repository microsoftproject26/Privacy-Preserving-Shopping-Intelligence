#!/bin/bash
# 06N N3.3 cold (random-start) re-run from scratch, after the warm line frees the cards (DL 306); code_ds3r6 (cold digests r5 == r6)
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneN33_DP8_2026.log; G=1
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r6.sh
T="--fl-tuning $V/contracts/fl/FL_TUNING_SELECTION.json"; SC="--clip-record $V/contracts/dp/S_RECORD.json"
wait_file $R/FL_T/FA_1024_WARM_FROZEN_s2026/RUN_SUMMARY.json
wait_file $R/FL_T/FA_1024_WARM_FROZEN_s2027/RUN_SUMMARY.json
wait_file $R/FL_T/DP8_FROZEN_s2026/RUN_SUMMARY.json
wait_file $R/FL_T/DP8_FROZEN_s2027/RUN_SUMMARY.json
wait_file $R/FL_T/DP1_WARM_FROZEN_s2026/RUN_SUMMARY.json
job DRY_N33_DP8_s2026 fl --arm DP --epsilon 8 --seed 2026 $T $SC --run-dir $R/FL_T/N33_DP8_s2026 --dry-run
job N33_DP8_s2026 fl --arm DP --epsilon 8 --seed 2026 $T $SC --run-dir $R/FL_T/N33_DP8_s2026
step "QUEUE_DONE rc=0"
