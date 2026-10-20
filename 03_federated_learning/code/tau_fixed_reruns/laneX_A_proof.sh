#!/usr/bin/env bash
# laneX_A_proof.sh <lock> <lock-sha> - part A VALIDATION proof (POST_HOC_EXPLORATORY; no TEST role is resolved):
#   1 every entry: reeval_x on MAIN_TEST must REFUSE (rc 2: no register entry binds the EXP lock yet)
#   2 every entry: --proof-role MAIN --dry-run, then the real MAIN (VALIDATION) scoring for the reeval entries
#   3 x_analysis on role MAIN over the proof rank files + the confirmatory VALIDATION rank files
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; L=$1; LS=$2; R=$V/runs_x_rees46; LOG=$R/proofA.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
cd $V/x_rees46
KEYS=$($PY -c "import json;d=json.load(open('$L'));print(' '.join(e['key'] for e in d['sealed_arms']+d['device_optional']))")
for k in $KEYS; do
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --plan >> $R/logs/proofA_$k.test_refusal.out 2>&1; rc=$?
  say "$k TEST-role plan rc=$rc (expected 2 = refused, no register entry)"
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --proof-role MAIN --dry-run >> $R/logs/proofA_$k.dry.out 2>&1; rc=$?
  say "$k MAIN dry-run rc=$rc"
done
for k in $KEYS; do
  case $k in FT_*) continue;; esac
  [ -f $R/PROOF/${k}_MAIN/X_RECEIPT.json ] && { say "$k MAIN proof already done"; continue; }
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --proof-role MAIN >> $R/logs/proofA_$k.main.out 2>&1; rc=$?
  say "$k MAIN (VALIDATION) scoring rc=$rc"
done
W2=$V/runs_p2/MAIN/P2_WSTAR_SMALL_s2026
$PY -B x_proof_spec.py $L $R/locks/proofA_spec.json "P2_WSTAR|PB=$W2" "P2_WSTAR|CB=$W2" >> $LOG 2>&1
rm -rf $R/PROOF/ANALYSIS_A_MAIN.tmp
$PY -B x_analysis.py --spec $R/locks/proofA_spec.json --role MAIN --out $R/PROOF/ANALYSIS_A_MAIN --code-root $V/code_x1a >> $R/logs/proofA_analysis.out 2>&1
say "x_analysis MAIN rc=$?"
say "proofA done"
