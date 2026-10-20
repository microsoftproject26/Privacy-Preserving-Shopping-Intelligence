#!/bin/bash
# D3 K16 INT8 PASS 2 = latency only (DL 368; B14 precedence (ii); OPS-B13-P2). NOT LAUNCHED by its author; Ops arms it.
# 1. WAIT until every Wave-1 driver has exited (code_bench3 / r4 run_wave1*.sh, laneBENCH_RESUME.sh and the non-Wave-2 v4bench jobs
#    they start), 2 consecutive quiet polls. Wave 2 does NOT block pass 2.
# 2. DRAIN Wave 2: SIGSTOP the run_wave2.sh driver(s) and every running v4bench job with --dataset amazon2023_* (pids logged);
#    a trap SIGCONTs exactly those pids on any exit (success, failure, INT / TERM). Wall time only, never a value.
# 3. Write the pass-2 host load beside the result (<dir>/PASS2_HOST_LOAD.json), run `v4dev.bench --latency-only` on the pinned
#    pass-1 dir ONNX_R7_U16384 (shas asserted; no re-export / accuracy), then SIGCONT.
# Env: CODE (default code_ds3r7; must contain the P2 bench.py), DRYWAIT=1 (print blockers + Wave-2 pids, exit 0),
#      TESTHOLD=<sec> (test only: stop, sleep, continue; skips the wait and the bench), W1_BUSY_RE / W2_DRV_RE / W2_JOB_RE (test overrides).
LOG=${LOG_OVERRIDE:-/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneD3_INT8_pass2.log}; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r7.sh
[ -n "${CODE:-}" ] && { CR=$CODE; export PYTHONPATH=$CR; cd "$CR" || exit 90; }
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# patterns are bracket-quoted so this script's own command line can never match them
W1_BUSY_RE=${W1_BUSY_RE:-'[r]un_wave1\.sh|[r]un_wave1_r4\.sh|[l]aneBENCH_RESUME\.sh|-m v4bench[.]run run .*--dataset (ml_1m|s3_)'}
W2_DRV_RE=${W2_DRV_RE:-'[r]un_wave2\.sh'}
W2_JOB_RE=${W2_JOB_RE:-'-m v4bench[.]run run .*--dataset amazon2023_'}
w1busy() { ps -eo pid,args | grep -E -e "$W1_BUSY_RE" | grep -v grep; }
w2pids() { { ps -eo pid,args | grep -E -e "$W2_DRV_RE" | grep -v grep; ps -eo pid,args | grep -E -e "$W2_JOB_RE" | grep -v grep; } | awk '{print $1}' | sort -un; }
STOPPED=""
cont_all() {
  local p
  for p in $STOPPED; do [ -d /proc/$p ] && kill -CONT "$p" 2>/dev/null; done
  [ -n "$STOPPED" ] && step "SIGCONT sent to: $STOPPED"
  STOPPED=""
}
if [ "${DRYWAIT:-0}" = 1 ]; then
  echo "DRYWAIT: Wave-1 blockers right now:"; w1busy | cut -c1-200
  echo "DRYWAIT: n_wave1_blockers=$(w1busy | wc -l); would SIGSTOP these Wave-2 pids: $(w2pids | tr '\n' ' '); code=$CR"; exit 0
fi
trap 'cont_all' EXIT
trap 'exit 143' INT TERM
if [ -z "${TESTHOLD:-}" ]; then
  step "WAIT every Wave-1 driver / job to exit (code=$CR); Wave 2 does not block"
  quiet=0
  while [ $quiet -lt 2 ]; do
    if [ -n "$(w1busy)" ]; then quiet=0; else quiet=$((quiet + 1)); fi
    [ $quiet -lt 2 ] && sleep 120
  done
  step "WAVE1_DONE"
  grep -q -- '--latency-only' "$CR/v4dev/bench.py" || { step "LANE_STOP $CR/v4dev/bench.py has no --latency-only (deploy the P2 candidate first)"; exit 7; }
  CK=$V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt
  O=$R/ONNX_R7_U16384
  [ -e $O/BENCH_RESULT.json ] || { step "LANE_STOP no pass-1 record in $O"; exit 2; }
  [ -e $O/LATENCY_PASS2.json ] || [ -e $O/PASS2_HOST_LOAD.json ] && { step "LANE_STOP pass-2 output already exists in $O (write-once)"; exit 2; }
fi
# drain Wave 2: driver(s) first (no new job launches), then the jobs; repeat to catch a job launched in between
for pass in 1 2 3; do
  for p in $(w2pids); do
    case " $STOPPED " in *" $p "*) continue;; esac
    kill -STOP "$p" 2>/dev/null && { STOPPED="$STOPPED $p"; step "SIGSTOP pid=$p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-160)"; }
  done
  sleep 2
done
LOAD0=$(cut -d' ' -f1-3 /proc/loadavg)
step "WAVE2_DRAINED stopped_pids=[$STOPPED ] loadavg=$LOAD0"
if [ -n "${TESTHOLD:-}" ]; then sleep "$TESTHOLD"; [ "${FAILTEST:-0}" = 1 ] && exit 9; step "TESTHOLD done"; exit 0; fi
sleep 20   # let the stopped jobs' caches / threads go quiet before timing
step "JOB_START INT8R7_PASS2 gpu=cpu loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
timeout 3600 $PY -B -m v4dev.bench --latency-only --ckpt $CK --family SASREC --variant SASREC_D64_B2 --v1-root /home/jovyan/shared_data/rees46_nextitem_v1 --out $O > $Q/INT8R7_PASS2.log 2>&1; rc=$?
LOAD1=$(cut -d' ' -f1-3 /proc/loadavg)
step "JOB_DONE INT8R7_PASS2 rc=$rc loadavg_end=$LOAD1"
set -- $LOAD0; a1=$1; a5=$2; a15=$3; set -- $LOAD1
printf '{"label":"PASS2_HOST_LOAD","loadavg_before_1_5_15":[%s,%s,%s],"loadavg_after_1_5_15":[%s,%s,%s],"nproc":%s,"wave2_stopped_pids":[%s],"wave1_drivers_running":false,"utc":"%s","rc":%s}\n' \
  "$a1" "$a5" "$a15" "$1" "$2" "$3" "$(nproc)" "$(echo $STOPPED | tr ' ' ',')" "$(ts)" "$rc" > $O/PASS2_HOST_LOAD.json
cont_all
[ $rc -eq 0 ] || { step "LANE_STOP rc=$rc"; exit $rc; }
step "QUEUE_DONE rc=0"
