#!/bin/bash
# T1: build the ONE final exploratory lock (final_x.py build-lock --final) as soon as all 12 C_SMALL runs are DONE_MAIN
# (Fable T1 condition 3). Polls every 5 min; stops if a C_SMALL run ends in another status. Never scores anything.
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
P="nice -n 19 /home/jovyan/venvs/ppsi-gpu/bin/python -B"
LOG=$V/runs_bench_x/x2_state/logs/lock_when_ready.log
cd $V/code_bench_x2
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
say "waiter start (final_x $(sha256sum tools_x2/final_x.py | cut -c1-16))"
while true; do
  n=0; bad=0
  for d in $V/runs_bench_x/*/C_SMALL__B24_L50_D0.1_s202[67] $V/runs_bench_x/ml1m/ml_1m/C_SMALL__B24_L50_D0.1_s202[67]; do
    st=$(/home/jovyan/venvs/ppsi-gpu/bin/python -c "import json,sys;print(json.load(open(sys.argv[1])).get('status',''))" $d/RESULT.json 2>/dev/null || echo NONE)
    [ "$st" = DONE_MAIN ] && n=$((n+1))
    case "$st" in INCOMPLETE*) bad=1;; esac
  done
  if [ $bad -eq 1 ]; then say "a C_SMALL run is INCOMPLETE: stop, no lock"; exit 1; fi
  if [ $n -eq 12 ]; then
    if pgrep -f '^/home/jovyan/venvs/ppsi-gpu/bin/python -B -m v4bench.run run .*--arm C_SMALL --seed 202[67] --budget-efe 24 --dropout 0.1 --threads' >/dev/null; then sleep 60; continue; fi
    say "12 / 12 C_SMALL DONE_MAIN: build-lock --final"
    $P tools_x2/final_x.py build-lock --final >> $LOG 2>&1; rc=$?
    say "build-lock --final rc=$rc"
    sha256sum $V/company/gates/EXP_BENCH_X_TEST_LOCK.json >> $LOG 2>&1
    say "WAITER_END"; exit $rc
  fi
  sleep 300
done
