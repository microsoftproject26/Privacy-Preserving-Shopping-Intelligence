#!/usr/bin/env bash
# laneX_C2_proof.sh <lock> - VALIDATION proof of the EXP_PB100_TEST_LOCK_v2 candidate (proofC re-run after the
# x2_admissions fix; POST_HOC_EXPLORATORY; no TEST role resolved): every entry TEST-role plan must refuse (rc 2);
# --proof-role MAIN --dry-run; MAIN scoring (GPU 0, the registered A40 class); x_analysis on role MAIN on CPU with
# --code-root code_x1 (the tree laneX_TEST.sh binds; Fable T2 condition 1).
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; R=$V/runs_x_rees46; L=$1; LS=$(sha256sum $L | cut -c1-64); LOG=$R/proofC2.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
cd $V/x_rees46; say "lock $LS"
KEYS=$($PY -c "import json;d=json.load(open('$L'));print(' '.join(e['key'] for e in d['sealed_arms']))")
for k in $KEYS; do
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --plan > $R/logs/proofC2_$k.test_refusal.out 2>&1; say "$k TEST-role plan rc=$? (expected 2)"
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --proof-role MAIN --dry-run > $R/logs/proofC2_$k.dry.out 2>&1; say "$k MAIN dry-run rc=$?"
done
for k in $KEYS; do
  [ -f $R/PROOF/${k}_MAIN/X_RECEIPT.json ] && continue
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --proof-role MAIN > $R/logs/proofC2_$k.main.out 2>&1; say "$k MAIN (VALIDATION) scoring rc=$?"
done
W6=$V/runs_p2/MAIN/P2_WSTAR_SMALL_s2026; W7=$V/runs_p2/MAIN/P2_WSTAR_SMALL_s2027
$PY -B x_proof_spec.py $L $R/locks/proofC2_spec.json "P2_WSTAR|2026|PB=$W6" "P2_WSTAR|2026|CB=$W6" "P2_WSTAR|2027|PB=$W7" "P2_WSTAR|2027|CB=$W7" >> $LOG 2>&1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY -B x_analysis.py --spec $R/locks/proofC2_spec.json --role MAIN --out $R/PROOF/ANALYSIS_C2_MAIN --code-root $V/code_x1 >> $R/logs/proofC2_analysis.out 2>&1
say "x_analysis MAIN rc=$?"
say "proofC2 done"
