#!/bin/bash
# F_validation_read.sh <freeze_sha256> --i-confirm — review §1 Phase F / MASTER_PLAN §9: THE SINGLE VALIDATION READ.
# This is the FIRST read of any MAIN value. FINAL r7 form (OPS-ENDGAME-2), NOT RUN. It does NOT rebuild the index: it re-runs the E report line
# on the SAME block (tree digest must equal E_CHECK_PASS.json) without --check-only, into $R/ANA/out_val_<tag> (O_EXCL).
# D4 corrections live in _endgame_ana_args.sh (--fl $R/FL_T, --pin, --contrasts v4, records incl. S_RECORD_WARM_FROZEN,
# FL_WARM_SELECTION_S7 / S8, SELECTION_FT_FA; no --allow-single-seed); the arm set is the §11.2 / §11.3 list of the
# index (N33 dirs as the cold arms, FT_FA ×2, the warm line) — the command text is checked against the freeze.
# Predecessor: $R/ANA/E_CHECK_PASS_<tag>.json + ok/E_<tag>.ok. Output: $R/ANA/out_val_<tag>/RETENTION_RESULT.json + ok/F_<tag>.ok.
# After this: anything new is POST_VALIDATION_EXPLORATORY (06L-C C5); no re-run, no selection, no recipe change.
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/endgame_F.log
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r7.sh
source $Q/endgame/_endgame_lib.sh
confirm "$2"
source $Q/endgame/_endgame_ana_args.sh
source $Q/endgame/_endgame_freeze_check.sh
need $ANA/E_CHECK_PASS_$ANA_TAG.json; need $OK/E_$ANA_TAG.ok; once $OK/F_$ANA_TAG.ok; once $ANA_OUT_VAL
freeze_check "$1"
PASS=$ANA/E_CHECK_PASS_$ANA_TAG.json
pf() { $PY -B -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" $PASS "$1"; }
[ "$(pf freeze_sha256)" = "$1" ] || eg_refuse "E_CHECK_PASS was made under another freeze"
[ "$(tree_digest $ANA_BLOCK_VAL)" = "$(pf block_tree_digest)" ] || eg_refuse "block_val changed since the check-only pass"
step "PHASE_START F VALIDATION READ freeze=${1:0:16} (first MAIN read)"
ejob EGF_VALIDATION_READ $(ana_report_val "$RSHA" "$ANA_OUT_VAL")
need $ANA_OUT_VAL/RETENTION_RESULT.json
chmod -R a-w $ANA_OUT_VAL; date -u +%FT%TZ > $OK/F_$ANA_TAG.ok
# FIX-5: the registered DESCRIPTIVE full metric table (frozen metric_table.py, sha in the freeze `scripts`, re-checked by
# freeze_check above) from the SAME block (index sha = E_CHECK_PASS). Values are copied from the sealed rows, never computed.
# It runs AFTER the F marker: the single read is done and is never repeated; a refusal here only stops this table
# (Ops re-runs exactly this line into a new --out after the cause is recorded in the DL).
MTV=$ANA/metric_table_val_$ANA_TAG
if [ ! -e $MTV ]; then step "JOB_START EGF_METRIC_TABLE"
  $PY -B $EG/metric_table.py --block $ANA_BLOCK_VAL --split VALIDATION --expect-block-index-sha256 "$(pf block_index_sha256)" --out $MTV > $Q/EGF_METRIC_TABLE.log 2>&1
  rc=$?; step "JOB_DONE EGF_METRIC_TABLE rc=$rc"; [ $rc -eq 0 ] && chmod -R a-w $MTV || step "METRIC_TABLE_NOT_WRITTEN rc=$rc (see the job log; DL row)"
fi
step "QUEUE_DONE F rc=0 RETENTION_RESULT $(sha $ANA_OUT_VAL/RETENTION_RESULT.json | cut -c1-16) — fetch RETENTION_RESULT.json + REPORT.md + curves; DL row"
