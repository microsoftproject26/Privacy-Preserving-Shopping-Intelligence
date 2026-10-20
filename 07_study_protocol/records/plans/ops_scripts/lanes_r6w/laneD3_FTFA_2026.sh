#!/bin/bash
# D3 (DL 317 / 319; O_EXCL write-once per Fable audit M1, DL 331): SEL_FT_FA on UNSEEN_VAL (base cold FA s2026 best) -> SELECTION_FT_FA.json write-once -> FT_FA MAIN s2026; code_ds3r6
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneD3_FTFA_2026.log; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r6.sh
SFA=$V/contracts/device/SELECTION_FT_FA.json
wait_file $R/FL_T/FA_WARM_s2026/RUN_SUMMARY.json
wait_file $R/FL_T/FA_WARM_s2027/RUN_SUMMARY.json
wait_file $R/FL_T/FA_WARM_FROZEN_s2026/RUN_SUMMARY.json
wait_file $R/FL_T/FA_WARM_FROZEN_s2027/RUN_SUMMARY.json
job SEL_FT_FA dev --arm FT_FA --seed 2026 --role UNSEEN_VAL --select --base $R/FL_T/FA_s2026/ckpt/FL_FA_SMALL_s2026/best.pt --base-kind fl --run-dir $R/DEV/SEL_FT_FA
$PY -B -c "import sys,os,hashlib; s=open(sys.argv[1],'rb').read(); f=open(sys.argv[2],'xb'); f.write(s); f.close(); os.chmod(sys.argv[2],0o444); assert hashlib.sha256(open(sys.argv[2],'rb').read()).digest()==hashlib.sha256(s).digest()" $R/DEV/SEL_FT_FA/SELECTION.json $SFA || { step "LANE_STOP selection write-once (O_EXCL) failed"; exit 7; }
step "REGISTERED $SFA $(sha256sum $SFA | cut -c1-16) (O_EXCL, sha = source)"
job FT_FA_s2026 dev --arm FT_FA --seed 2026 --role MAIN --run-dir $R/DEV/FT_FA_s2026 --base $R/FL_T/FA_s2026/ckpt/FL_FA_SMALL_s2026/best.pt --base-kind fl --selection $SFA --n-shards 64
step "QUEUE_DONE rc=0"
