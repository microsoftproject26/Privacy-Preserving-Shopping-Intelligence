#!/bin/bash
# D3 (DL 317 / 319): FT_FA MAIN s2027 (base cold FA s2027 best) after SELECTION_FT_FA.json; code_ds3r6
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneD3_FTFA_2027.log; G=1
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r6.sh
SFA=$V/contracts/device/SELECTION_FT_FA.json
wait_file $SFA
job FT_FA_s2027 dev --arm FT_FA --seed 2027 --role MAIN --run-dir $R/DEV/FT_FA_s2027 --base $R/FL_T/FA_s2027/ckpt/FL_FA_SMALL_s2027/best.pt --base-kind fl --selection $SFA --n-shards 64
step "QUEUE_DONE rc=0"
