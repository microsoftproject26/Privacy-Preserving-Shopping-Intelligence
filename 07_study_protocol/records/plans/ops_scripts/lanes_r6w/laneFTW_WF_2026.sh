#!/bin/bash
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneFTW_WF_2026.log; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r6.sh
B=$R/FL_T/FA_WARM_FROZEN_s2026; BP=$B/ckpt/FL_FA_WARM_FROZEN_SMALL_s2026/best.pt
wait_file $B/RUN_SUMMARY.json
[ -f $BP ] || { step "LANE_STOP no base $BP"; exit 8; }
job SMK_FT_FA_WARM_FROZEN_s2026 dev --arm FT_FA_WARM_FROZEN --seed 2026 --role MAIN --run-dir $R/DEV/FT_FA_WARM_FROZEN_SMOKE_s2026 --base $BP --base-kind fa_warm --n-shards 64 --smoke --smoke-clients 8
grep -rq e47965d2 $R/DEV/FT_FA_WARM_FROZEN_SMOKE_s2026/LAUNCH*.json && grep -rq fa_warm $R/DEV/FT_FA_WARM_FROZEN_SMOKE_s2026/LAUNCH*.json || { step "LANE_STOP smoke check (selection sha / start kind)"; exit 9; }
step "SMOKE_CHECK_OK"
job FT_FA_WARM_FROZEN_s2026 dev --arm FT_FA_WARM_FROZEN --seed 2026 --role MAIN --run-dir $R/DEV/FT_FA_WARM_FROZEN_s2026 --base $BP --base-kind fa_warm --n-shards 64
step "QUEUE_DONE rc=0"
