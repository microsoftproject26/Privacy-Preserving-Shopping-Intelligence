#!/usr/bin/env bash
# laneX_PB2.sh - after laneX_PB.sh ("lane done"): build the X2 (EXP lock v2) CANDIDATE exploratory lock and prove it on VALIDATION
# (POST_HOC_EXPLORATORY; no TEST role is ever resolved here). GPU 0 only. Steps:
#   1 wait for "lane done" in runs_x_rees46/lane.log (stop on FAIL)
#   2 x_lock_build.py B -> runs_x_rees46/locks/EXP_PB100_TEST_LOCK_v2.candidate.json (the final lock is built by Ops
#     into company/gates/EXP_PB100_TEST_LOCK_v2.json after reading this proof)
#   3 every entry: TEST-role plan must refuse (rc 2); --proof-role MAIN --dry-run; then MAIN (VALIDATION) scoring
#   4 x_analysis on role MAIN (proof ranks + confirmatory VALIDATION ranks + W* in-run marks)
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; R=$V/runs_x_rees46; LOG=$R/proofC.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
until grep -q "lane done" $R/PB100/lane.log; do [ -f $R/PB100/_state/FAIL ] && { say "lane B FAIL: stop"; exit 1; }; sleep 300; done
cd $V/x_rees46
L=$R/locks/EXP_PB100_TEST_LOCK_v2.candidate.json
[ -e $L ] || $PY -B x_lock_build.py C --out $L --x-main-dirs $R/PB100/MAIN/X2_*_s2026 $R/PB100/MAIN/X2_*_s2027 >> $LOG 2>&1 || { say "lock build failed"; exit 1; }
LS=$(sha256sum $L | cut -c1-64); say "candidate lock $LS"
KEYS=$($PY -c "import json;d=json.load(open('$L'));print(' '.join(e['key'] for e in d['sealed_arms']))")
for k in $KEYS; do
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --plan >> $R/logs/proofC_$k.test_refusal.out 2>&1; say "$k TEST-role plan rc=$? (expected 2)"
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --proof-role MAIN --dry-run >> $R/logs/proofC_$k.dry.out 2>&1; say "$k MAIN dry-run rc=$?"
done
for k in $KEYS; do
  [ -f $R/PROOF/${k}_MAIN/X_RECEIPT.json ] && continue
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k --proof-role MAIN >> $R/logs/proofC_$k.main.out 2>&1; say "$k MAIN (VALIDATION) scoring rc=$?"
done
W6=$V/runs_p2/MAIN/P2_WSTAR_SMALL_s2026; W7=$V/runs_p2/MAIN/P2_WSTAR_SMALL_s2027
$PY -B x_proof_spec.py $L $R/locks/proofC_spec.json  >> $LOG 2>&1
$PY -B x_analysis.py --spec $R/locks/proofC_spec.json --role MAIN --out $R/PROOF/ANALYSIS_C_MAIN --code-root $V/code_x1a >> $R/logs/proofC_analysis.out 2>&1
say "x_analysis MAIN rc=$?"
say "proofB done"
