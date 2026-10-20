#!/bin/bash
# B2_reeval_cfull_s2027.sh DRY|REAL <gpu> — review §1 B2 / ANA-05 step 2: C_FULL s2027 × {MAIN, UNSEEN_HOLDOUT,
# SPARSE_PANEL, HEAVY_PANEL} × {PB, CB} = 8 DESCRIPTIVE reevals (refused on r5, admitted since r6; run on r7).
# Not in the D3 list but on the review checklist and in §11.2 ("C_FULL s2027 DESCRIPTIVE"); no DL row shows it ran.
# FINAL r7 form (OPS-ENDGAME-2), NOT RUN for real. Expected sha from the s2027 sealed/CHECKPOINTS.json (keys best /
# endpoint_6.0, MEASURED). Model: reeval_cmds_r5.sh / reeval_06o_r6.sh. All 8 dry runs passed on r7 (README §6).
# Predecessors: DRY needs $R/deploy_r7/R7_VERIFIED.ok; REAL needs $Q/endgame/ok/B2_DRY.ok. Output: ok/B2_REAL.ok.
MODE=$1; G=$2
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/endgame_B.log
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r7.sh
source $Q/endgame/_endgame_lib.sh
case "$MODE" in DRY|REAL) ;; *) eg_refuse "usage: B2_reeval_cfull_s2027.sh DRY|REAL <gpu 0|1>";; esac
[[ "$G" =~ ^[01]$ ]] || eg_refuse "gpu must be 0 or 1"
need $R/deploy_r7/R7_VERIFIED.ok; [ "$MODE" = REAL ] && need $OK/B2_DRY.ok; once $OK/B2_$MODE.ok
step "PHASE_START B2 $MODE gpu=$G tree=$CR"
EXTRA=""; SUF=""; [ "$MODE" = DRY ] && { EXTRA="--dry-run"; SUF="_DRY"; }
RV="$PY -B v4int/reeval.py --ckpt-kind central --history-policy USER_HISTORY_L50 --gpu $G --allow-shared-gpu $EXTRA"
CF=$R/CFULL/C_FULL_02B_SASREC_s2027; need $CF/sealed/CHECKPOINTS.json
for ROLE in MAIN UNSEEN_HOLDOUT SPARSE_PANEL HEAVY_PANEL; do RN=$([ $ROLE = MAIN ] && echo "" || echo "${ROLE}_")
 for T in PB:best:PRACTICAL_BEST CB:endpoint_6.0:CONTROLLED_BUDGET; do IFS=: read TAG CK RULE <<< "$T"
  H=$(central_sha $CF/sealed/CHECKPOINTS.json $CK); [[ "$H" =~ ^[0-9a-f]{64}$ ]] || { step "LANE_STOP no sha s2027 $CK"; exit 6; }
  OUT=$R/REEVAL_A40/C_FULL_${RN}s2027_$TAG$SUF; [ "$MODE" = REAL ] && [ -e $OUT ] && { step "LANE_STOP $OUT exists"; exit 7; }
  ejob EGB2_CF_${ROLE}_s2027_$TAG$SUF $RV --ckpt $CF/sealed/checkpoints/$CK.pt --expect-sha256 $H --variant SASREC_D256_B3 \
    --recipe-id 02B_SASREC --seed 2027 --role $ROLE --arm-label C_FULL --selection-rule $RULE \
    --checkpoint-id C_FULL_02B_SASREC_s2027:$CK --out-dir $OUT
  # ANA-05 expectation: central_source.seed_status DESCRIPTIVE — DRY: the one-line dry-run JSON (reeval main prints it);
  # REAL: the REEVAL_RECORD.json central_source (identity fields only, never a value)
  if [ "$MODE" = DRY ]; then grep -q '"seed_status": "DESCRIPTIVE"' "$Q/EGB2_CF_${ROLE}_s2027_$TAG$SUF.log"
  else $PY -B -c "import json,sys; sys.exit(0 if (json.load(open(sys.argv[1])).get('central_source') or {}).get('seed_status') == 'DESCRIPTIVE' else 1)" "$OUT/REEVAL_RECORD.json"; fi \
    || { step "LANE_STOP no DESCRIPTIVE seed_status ($ROLE $TAG)"; exit 8; }
 done
done
date -u +%FT%TZ > $OK/B2_$MODE.ok; step "QUEUE_DONE B2 $MODE rc=0"
