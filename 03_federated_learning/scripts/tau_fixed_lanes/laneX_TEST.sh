#!/usr/bin/env bash
# laneX_TEST.sh <EXP lock> <lock sha> [--with-ft] - the exploratory TEST look (POST_HOC_EXPLORATORY) of one EXP lock.
# RUN ONLY after: Fable gate PASS + Ops appended the LOCKED_EVALUATION / RETRO_TEST register entry that binds <lock sha>
# (reeval_x refuses every entry otherwise). GPU 0 only. Steps: every sealed arm on MAIN_TEST (reeval_x), optional FT
# (device_optional), then x_analysis on MAIN_TEST -> runs_x_rees46/TEST/ANALYSIS_<document>/X_RESULT.json.
# The analysis log holds no values; read X_RESULT.json only after it exists (S8 style).
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; L=$1; LS=$2; FT=${3:-}; R=$V/runs_x_rees46
DOC=$($PY -c "import json;print(json.load(open('$L'))['document'])"); LOG=$R/TEST_$DOC.log; mkdir -p $R/logs $R/TEST
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
cd $V/x_rees46
KEYS=$($PY -c "import json;d=json.load(open('$L'));print(' '.join(e['key'] for e in d['sealed_arms']))")
for k in $KEYS; do
  $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k >> $R/logs/TEST_$k.out 2>&1; rc=$?
  say "$k MAIN_TEST rc=$rc"; [ $rc -eq 0 ] || { say "STOP"; exit $rc; }
done
if [ "$FT" = --with-ft ]; then
  for k in $($PY -c "import json;d=json.load(open('$L'));print(' '.join(e['key'] for e in d.get('device_optional',[])))"); do
    $PY -B reeval_x.py --lock $L --lock-sha $LS --entry $k >> $R/logs/TEST_$k.out 2>&1; say "$k MAIN_TEST FT rc=$?"
  done
fi
$PY -c "
import json,hashlib;d=json.load(open('$L'));a=d['analysis'];e=a['entries']
for k,v in e.items():
    if v['sha256'] is None: v['sha256']=hashlib.sha256(open(v['ranks'],'rb').read()).hexdigest()
json.dump({'label':'POST_HOC_EXPLORATORY','entries':e,'contrasts':a['contrasts']},open('$R/TEST/spec_$DOC.json','w'),indent=1)"
$PY -B x_analysis.py --spec $R/TEST/spec_$DOC.json --role MAIN_TEST --out $R/TEST/ANALYSIS_$DOC --code-root $V/$($PY -c "import json;print(json.load(open('$L'))['code']['scoring_tree']['name'])") --lock $L --lock-sha $LS >> $R/logs/TEST_analysis_$DOC.out 2>&1
say "x_analysis MAIN_TEST rc=$? (X_RESULT sha $(sha256sum $R/TEST/ANALYSIS_$DOC/X_RESULT.json 2>/dev/null | cut -c1-16))"
