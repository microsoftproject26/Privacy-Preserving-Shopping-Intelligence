#!/bin/bash
# usage: reeval_06o_r6.sh DRY|REAL <gpu> — 06O DESCRIPTIVE previews on UNSEEN_VAL: C_FULL s2026/s2027 PB+CB, PRE s2026 CB; code_ds3r6
# expected sha256 comes from each run's sealed/CHECKPOINTS.json (PRE: the registered fl_init sha), never from the file itself
MODE=$1; G=$2
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneRV06O_${MODE}_shared.log
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r6.sh
EXTRA=""; SUF=""; [ "$MODE" = DRY ] && { EXTRA="--dry-run"; SUF="_DRY"; }
RV="$PY -B v4int/reeval.py --ckpt-kind central --history-policy USER_HISTORY_L50 --gpu $G --allow-shared-gpu --role UNSEEN_VAL $EXTRA"
ck_sha() { $PY -B -c "import json,sys; d=json.load(open(sys.argv[1]))[sys.argv[2]]; print(d['sha256'] if isinstance(d,dict) else d)" "$1" "$2"; }
for SD in 2026 2027; do CF=$R/CFULL/C_FULL_02B_SASREC_s$SD
 for T in PB:best:PRACTICAL_BEST CB:endpoint_6.0:CONTROLLED_BUDGET; do IFS=: read TAG CK RULE <<< "$T"
  H=$(ck_sha $CF/sealed/CHECKPOINTS.json $CK); [ ${#H} -eq 64 ] || { step "LANE_STOP no sha $SD $CK"; exit 6; }
  job RV06O_CF_s${SD}_$TAG$SUF $RV --ckpt $CF/sealed/checkpoints/$CK.pt --expect-sha256 $H --variant SASREC_D256_B3 --recipe-id 02B_SASREC --seed $SD --arm-label C_FULL --selection-rule $RULE --checkpoint-id C_FULL_02B_SASREC_s$SD:$CK --out-dir $R/REEVAL_A40/C_FULL_UNSEEN_VAL_s${SD}_$TAG$SUF
 done; done
PRE=$V/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026; H=cf08588c11f019cc760f129ba7ba6b23806297660e0170f9938436a5cb88cf78
job RV06O_PRE_s2026_CB$SUF $RV --ckpt $PRE/sealed/checkpoints/endpoint_6.0.pt --expect-sha256 $H --variant SASREC_D64_B2 --recipe-id SMALL_SASREC_D64_B2 --seed 2026 --arm-label PRE --selection-rule CONTROLLED_BUDGET --checkpoint-id PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026:endpoint_6.0 --out-dir $R/REEVAL_A40/PRE_UNSEEN_VAL_s2026_CB$SUF
step "QUEUE_DONE rc=0"
