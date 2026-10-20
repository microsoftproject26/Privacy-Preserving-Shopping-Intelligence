#!/bin/bash
# B_reeval_panels.sh DRY|REAL <gpu> — D3(b) panel reevals (DL 317; review §1 B3, §3 D3), seed 2026, on code_ds3r7.
# FINAL r7 form (OPS-ENDGAME-2), NOT RUN for real. Roles UNSEEN_HOLDOUT / SPARSE_PANEL / HEAVY_PANEL (+ TRUE_COLD where
# reeval admits the source: reeval refuses TRUE_COLD only for the registered central sources C_FULL / PRE (SOURCE_ROLES,
# reeval.py L420); none of the six sources below is one, so all six are expected ADMITTED (README Q2)).
# Sources: C_BIG best, C_SMALL best, FA best (cold FL_T/FA_s2026), N33_DP8 endpoint (cold N33 re-run),
#          FA_WARM_FROZEN best, DP8_FROZEN endpoint. DESCRIPTIVE panel rows (different populations).
# --expect-sha256: central = the run's sealed/CHECKPOINTS.json; FL = the sealed FL checkpoint record (seal_fl_ckpts:
#          ckpt/<run_id> + payload run_id, events.jsonl bytes + content digest, DEV start.file_sha256 cross-check;
#          README Q1) — never sha256sum of the file alone.
# --allow-shared-gpu: owner-approved while FL jobs run (DL 292). DRY writes nothing under REEVAL_A40.
# Predecessors: DRY needs $R/deploy_r7/R7_VERIFIED.ok (r7 = the reeval that knows the warm labels; review G-11);
#               REAL needs $Q/endgame/ok/B_DRY.ok. Output: $Q/endgame/ok/B_REAL.ok (the D_freeze predecessor).
MODE=$1; G=$2
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/endgame_B.log
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r7.sh
source $Q/endgame/_endgame_lib.sh
case "$MODE" in DRY|REAL) ;; *) eg_refuse "usage: B_reeval_panels.sh DRY|REAL <gpu 0|1>";; esac
[[ "$G" =~ ^[01]$ ]] || eg_refuse "gpu must be 0 or 1"
need $R/deploy_r7/R7_VERIFIED.ok
[ "$MODE" = REAL ] && need $OK/B_DRY.ok
once $OK/B_$MODE.ok
step "PHASE_START B $MODE gpu=$G tree=$CR"

ROLES="UNSEEN_HOLDOUT SPARSE_PANEL HEAVY_PANEL TRUE_COLD"
CBIG=$V/runs_p3/SASREC/02B_SASREC_s2026; CSM=$V/runs_ds/C_SMALL_s2026
FA=$R/FL_T/FA_s2026; N8=$R/FL_T/N33_DP8_s2026; WF=$R/FL_T/FA_WARM_FROZEN_s2026; D8F=$R/FL_T/DP8_FROZEN_s2026
for d in $FA $N8 $WF $D8F; do need_done $d; done
FA_CK=$(fl_ckdir $FA)/best.pt; N8_CK=$(fl_ckdir $N8)/endpoint_6.0.pt
WF_CK=$(fl_ckdir $WF)/best.pt; D8F_CK=$(fl_ckdir $D8F)/endpoint_6.0.pt
for f in $CBIG/sealed/checkpoints/best.pt $CSM/sealed/checkpoints/best.pt $FA_CK $N8_CK $WF_CK $D8F_CK; do need $f; done
FLREC=$EG/FL_CKPT_RECORD_B.json
seal_fl_ckpts $FLREC $FA_CK $N8_CK $WF_CK $D8F_CK >> "$LOG" 2>&1 || { step "LANE_STOP FL checkpoint record REFUSED"; exit 96; }

# ARM | kind | ckpt | expected sha | variant | recipe | rule | tag | checkpoint-id. README Q3 (RESOLVED): central ids =
# the VALIDATION REEVAL_A40 records (C_BIG_s2026_PB '02B_SASREC_MAIN_s2026:best', C_SMALL '..._MAIN_s<S>:best', MEASURED);
# FL ids = '<run_id>:best' / '<run_id>:endpoint_6.0' with <run_id> = the ckpt/<run_id> dir = launch_fl's run id (its own
# CONTROLLED_BUDGET rows use '<run_id>:endpoint_6.0', launch_fl L1641). A-F10 reads the row efe for PRACTICAL_BEST.
H() { local h; h=$("$@") && [ ${#h} -eq 64 ] && echo $h || { step "LANE_STOP no recorded sha ($*)"; exit 6; }; }
D2=SASREC_D256_B3:02B_SASREC; D6=SASREC_D64_B2:SMALL_SASREC_D64_B2
SRC=(
 "C_BIG|central|$CBIG/sealed/checkpoints/best.pt|$(H central_sha $CBIG/sealed/CHECKPOINTS.json best)|$D2|PRACTICAL_BEST|PB|02B_SASREC_MAIN_s2026:best"
 "C_SMALL|central|$CSM/sealed/checkpoints/best.pt|$(H central_sha $CSM/sealed/CHECKPOINTS.json best)|$D6|PRACTICAL_BEST|PB|SMALL_SASREC_D64_B2_MAIN_s2026:best"
 "FA|fl|$FA_CK|$(H fl_sha $FLREC $(readlink -f $FA_CK))|$D6|PRACTICAL_BEST|PB|$(basename $(dirname $FA_CK)):best"
 "DP8|fl|$N8_CK|$(H fl_sha $FLREC $(readlink -f $N8_CK))|$D6|CONTROLLED_BUDGET|CB|$(basename $(dirname $N8_CK)):endpoint_6.0"
 "FA_WARM_FROZEN|fl|$WF_CK|$(H fl_sha $FLREC $(readlink -f $WF_CK))|$D6|PRACTICAL_BEST|PB|$(basename $(dirname $WF_CK)):best"
 "DP8_FROZEN|fl|$D8F_CK|$(H fl_sha $FLREC $(readlink -f $D8F_CK))|$D6|CONTROLLED_BUDGET|CB|$(basename $(dirname $D8F_CK)):endpoint_6.0"
)
for L in "${SRC[@]}"; do IFS='|' read -r ARM _ _ HX _ <<< "$L"; [[ "$HX" =~ ^[0-9a-f]{64}$ ]] || { step "LANE_STOP no recorded sha for $ARM"; exit 6; }; done
EXTRA=""; SUF=""; [ "$MODE" = DRY ] && { EXTRA="--dry-run"; SUF="_DRY"; }
RV="$PY -B v4int/reeval.py --history-policy USER_HISTORY_L50 --gpu $G --allow-shared-gpu $EXTRA"
TCA=$EG/B_TRUE_COLD_ADMISSION.tsv          # written by DRY (one line per source), read by REAL
if [ "$MODE" = DRY ]; then : > $TCA; else need $TCA; chmod a-w $TCA; fi   # DRY may be repeated until B_DRY.ok
for L in "${SRC[@]}"; do IFS='|' read ARM KIND CK HX VR RULE TAG CKID <<< "$L"; IFS=: read VAR REC <<< "$VR"
 for ROLE in $ROLES; do
  N=EGB_${ARM}_${ROLE}_s2026_$TAG$SUF; OUT=$R/REEVAL_A40/${ARM}_${ROLE}_s2026_$TAG$SUF   # Q4 RESOLVED: index L359-363
  # (<head>_s<S>_<PB|CB>, head == arm or arm + '_' + ...; the cold N33 source is the block arm DP8, launch arm DP eps 8)
  ARGS=(--ckpt $CK --ckpt-kind $KIND --expect-sha256 $HX --variant $VAR --recipe-id $REC --seed 2026 --role $ROLE
        --arm-label $ARM --selection-rule $RULE --checkpoint-id $CKID --out-dir $OUT)
  if [ "$ROLE" = TRUE_COLD ] && [ "$MODE" = DRY ]; then        # the ONE tolerated refusal: TRUE_COLD not admitted for a source
   step "JOB_START $N gpu=$G"; $RV "${ARGS[@]}" > "$Q/$N.log" 2>&1; rc=$?; step "JOB_DONE $N rc=$rc"
   if [ $rc -eq 0 ] && ! grep -q 'REFUSED' "$Q/$N.log"; then echo -e "$ARM\tADMITTED" >> $TCA
   elif grep -E 'REEVAL REFUSED' "$Q/$N.log" | grep -q 'TRUE_COLD'; then echo -e "$ARM\tNOT_ADMITTED" >> $TCA; step "TRUE_COLD_NOT_ADMITTED $ARM"
   else step "LANE_STOP $N rc=$rc (refusal is not the TRUE_COLD admission rule)"; exit 97; fi
   continue
  fi
  if [ "$ROLE" = TRUE_COLD ]; then grep -qP "^$ARM\tADMITTED$" $TCA || { step "SKIP_TRUE_COLD_NOT_ADMITTED $ARM"; continue; }; fi
  [ "$MODE" = REAL ] && [ -e "$OUT" ] && { step "LANE_STOP $OUT exists"; exit 7; }
  ejob $N $RV "${ARGS[@]}"
 done
done
if [ "$MODE" = REAL ]; then   # B5: record fields only (role, arm, checkpoint sha, pin, status) — never a metric
 $PY -B - "$R/REEVAL_A40" "$PIN" <<'PY' >> "$LOG" 2>&1 || { step "LANE_STOP B5 post-check failed"; exit 98; }
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); pin = json.load(open(sys.argv[2])); bad = 0
for d in sorted(root.glob("*_s2026_[PC]B")):
    parts = d.name.split("_")
    if not any(r in d.name for r in ("_UNSEEN_HOLDOUT_", "_SPARSE_PANEL_", "_HEAVY_PANEL_", "_TRUE_COLD_")): continue
    r = json.load(open(d / "REEVAL_RECORD.json"))
    ok = r.get("status") == "COMPLETE" and d.name.startswith(r.get("arm_label", "?") + "_" + r.get("role", "?") + "_") \
         and len(r.get("checkpoint_sha256", "")) == 64 and r.get("device_pin_sha256", "").startswith("db744ca3")
    bad += not ok; print("B5", "OK" if ok else "BAD", d.name, r.get("role"), r.get("arm_label"), r.get("checkpoint_sha256", "")[:12])
sys.exit(1 if bad else 0)
PY
fi
date -u +%FT%TZ > $OK/B_$MODE.ok; step "QUEUE_DONE B $MODE rc=0"
