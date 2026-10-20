#!/bin/bash
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneFTW_W_2027.log; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r6.sh
B=$R/FL_T/FA_WARM_s2027; BP=$B/ckpt/FL_FA_WARM_SMALL_s2027/best.pt
wait_file $B/RUN_SUMMARY.json
[ -f $BP ] || { step "LANE_STOP no base $BP"; exit 8; }
job SMK_FT_FA_WARM_s2027 dev --arm FT_FA_WARM --seed 2027 --role MAIN --run-dir $R/DEV/FT_FA_WARM_SMOKE_s2027 --base $BP --base-kind fa_warm --n-shards 64 --smoke --smoke-clients 8
grep -rq e47965d2 $R/DEV/FT_FA_WARM_SMOKE_s2027/LAUNCH*.json && grep -rq fa_warm $R/DEV/FT_FA_WARM_SMOKE_s2027/LAUNCH*.json || { step "LANE_STOP smoke check (selection sha / start kind)"; exit 9; }
step "SMOKE_CHECK_OK"
job FT_FA_WARM_s2027 dev --arm FT_FA_WARM --seed 2027 --role MAIN --run-dir $R/DEV/FT_FA_WARM_s2027 --base $BP --base-kind fa_warm --n-shards 64
step "QUEUE_DONE rc=0"
