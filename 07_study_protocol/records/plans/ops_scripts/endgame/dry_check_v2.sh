#!/bin/bash
# dry_check_v2.sh <candidate_retention_report_sha256> — Fable §6 / DL 395: a PRE-FREEZE value-free dry check.
# Runs the exact §9 CHECK command (retention_report --check-only) with a CANDIDATE tool (never in $V/tools) on the
# read-only block_val_v2 (index unchanged). Output: $R/ANA/_dry_check_v2/<utc>_<sha12>/ (+ .log), read-only. Prints
# only PASS / REFUSED lines. Never F, never TEST, never out_val*.
set -uo pipefail
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/endgame_DRY.log
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r7.sh
export ANA_TAG=v2
source $Q/endgame/_endgame_lib.sh
source $Q/endgame/_endgame_ana_args.sh
CS=${1:?candidate sha}; S12=${CS:0:12}
DRY=$ANA/_dry_check_v2; IN=$DRY/incoming/retention_report.$S12.py; TC=$DRY/tools_cand/$S12
[ -d "$ANA_BLOCK_VAL" ] && [ "$ANA_BLOCK_VAL" = "$ANA/block_val_v2" ] || { echo "REFUSED_DRY: block_val_v2 missing"; exit 2; }
[ "$(sha $IN)" = "$CS" ] || { echo "REFUSED_DRY: incoming candidate sha != $CS"; exit 2; }
if [ ! -d $TC ]; then
  mkdir -p $TC
  for f in $V/tools/*.py; do [ "$(basename $f)" = retention_report.py ] || cp -p $f $TC/; done
  cp $IN $TC/retention_report.py; chmod -R a-w $TC
fi
[ "$(sha $TC/retention_report.py)" = "$CS" ] || { echo "REFUSED_DRY: tools_cand sha mismatch"; exit 2; }
for f in g2_run.py ana_block_index.py test_exposure_register.py; do cmp -s $TC/$f $V/tools/$f || { echo "REFUSED_DRY: $f differs from \$V/tools"; exit 2; }; done
U=$(date -u +%Y%m%dT%H%M%SZ); OUT=$DRY/${U}_$S12; DL=$DRY/${U}_$S12.log
TOOLS=$TC
CMD=$(ana_report_val "$CS" "$OUT" --check-only)
case "$CMD" in *--check-only*) ;; *) echo "REFUSED_DRY: not a check-only command"; exit 2;; esac
case "$CMD" in *out_val*|*TEST*) echo "REFUSED_DRY: forbidden output target"; exit 2;; esac
echo "$(date -u +%FT%TZ) DRY_START cand=$S12" >> $LOG
$CMD > $DL 2>&1; rc=$?
chmod a-w $DL; [ -e $OUT ] && chmod -R a-w $OUT
if [ $rc -eq 0 ] && ! grep -qE '(^|[^A-Z_])REFUSED([: ]|$)' $DL; then R=PASS; else R=REFUSED; fi
echo "$(date -u +%FT%TZ) DRY_$R rc=$rc cand=$S12 log=$DL" >> $LOG
echo "DRY_$R rc=$rc cand=$S12"; grep -hE 'REFUSED|Traceback|Error' $DL | tail -3
