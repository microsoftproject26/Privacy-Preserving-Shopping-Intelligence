#!/usr/bin/env bash
# laneX_T2_proof.sh <lock> - VALIDATION proof of EXP_XREES46_TEST_LOCK (POST_HOC_EXPLORATORY; no TEST role resolved):
#   every entry: TEST-role plan must refuse (rc 2, no BIND entry); --proof-role MAIN --dry-run; MAIN scoring (GPU 0);
#   then x_analysis on role MAIN (proof ranks + confirmatory VALIDATION ranks + in-run marks of FA / P2_WSTAR).
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; R=$V/runs_x_rees46; L=$1; LS=$(sha256sum $L | cut -c1-64); LOG=$R/proofT2.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
cd $V/x_rees46; say "lock $LS"
KEYS=$($PY -c "import json;d=json.load(open('$L'));print(' '.join(e['key'] for e in d['sealed_arms']))")
for k in $KEYS; do
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --plan > $R/logs/proofT2_$k.test_refusal.out 2>&1; say "$k TEST-role plan rc=$? (expected 2)"
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --proof-role MAIN --dry-run > $R/logs/proofT2_$k.dry.out 2>&1; say "$k MAIN dry-run rc=$?"
done
for k in $KEYS; do
  [ -f $R/PROOF/${k}_MAIN/X_RECEIPT.json ] && continue
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --proof-role MAIN > $R/logs/proofT2_$k.main.out 2>&1; say "$k MAIN (VALIDATION) scoring rc=$?"
done
F6=$V/runs_ds3/FL_T/FA_s2026; F7=$V/runs_ds3/FL_T/FA_s2027; W6=$V/runs_p2/MAIN/P2_WSTAR_SMALL_s2026; W7=$V/runs_p2/MAIN/P2_WSTAR_SMALL_s2027
$PY -B x_proof_spec.py $L $R/locks/proofT2_spec.json "FA|2026|PB=$F6" "FA|2026|CB=$F6" "FA|2027|PB=$F7" "FA|2027|CB=$F7" \
  "P2_WSTAR|2026|PB=$W6" "P2_WSTAR|2026|CB=$W6" "P2_WSTAR|2027|PB=$W7" "P2_WSTAR|2027|CB=$W7" >> $LOG 2>&1
$PY -B x_analysis.py --spec $R/locks/proofT2_spec.json --role MAIN --out $R/PROOF/ANALYSIS_T2_MAIN --code-root $V/code_x1a >> $R/logs/proofT2_analysis.out 2>&1
say "x_analysis MAIN rc=$?"
say "proofT2 done"
