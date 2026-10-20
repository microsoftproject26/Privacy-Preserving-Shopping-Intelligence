#!/bin/bash
# X2 driver (DL 553 A1 disjoint PRE + A2(b) FL at D256; POST_HOC_EXPLORATORY; inner validation only; never a TEST look).
# CPU by default; owner 2026-10-09: an opt-in GPU lane (--gpu-jobs <json run dirs> --gpu N --gpu-max K) for listed jobs.
#
#   bash run_bench_x2.sh --slots 24 --threads 4 [--only A1|A2B] [--dry-run]
#   relaunching the same command resumes (DONE_MAIN skipped, partial runs --resume, live runs not started twice).
# Refused: slots x threads > 96 (owner cap; every live runs_bench_x job holds a slot); a running "v4bench.run final";
# a code_bench_x2 file whose sha256 differs from X2_SHA256SUMS; no PLAN_X2.json; a second driver (flock).
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
X2=$V/code_bench_x2
ST=$V/runs_bench_x/x2_state
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
SLOTS=24; THREADS=4; DRY=0; ONLY=""; GPUJOBS=""; GPU=""; GPUMAX=2
while [ $# -gt 0 ]; do
  case "$1" in
    --slots) SLOTS="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --gpu-jobs) GPUJOBS="$2"; shift 2;;     # owner 2026-10-09: listed run dirs run on --device cuda --gpu N
    --gpu) GPU="$2"; shift 2;;
    --gpu-max) GPUMAX="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
mkdir -p "$ST/logs"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$ST/driver.log"; }
refuse() { say "REFUSED: $*"; exit 3; }
[ $((SLOTS * THREADS)) -le 112 ] || refuse "at most 112 CPU threads (slots x threads; owner 2026-10-09)"
if pgrep -f '^/home/jovyan/venvs/ppsi-gpu/bin/python[0-9.]*( -[A-Za-z]+)* -m v4bench\.run final' >/dev/null; then
  refuse "a 'v4bench.run final' process is running"
fi
(cd "$X2" && sha256sum -c --quiet X2_SHA256SUMS) || refuse "code_bench_x2 sha256 differs from X2_SHA256SUMS"
[ -f "$ST/PLAN_X2.json" ] || refuse "no $ST/PLAN_X2.json (run tools_x2/plan_x2.py --write)"
exec 9>"$ST/.driver.lock"
flock -n 9 || refuse "another driver holds $ST/.driver.lock"
ARGS=(--slots "$SLOTS" --threads "$THREADS"); [ -n "$ONLY" ] && ARGS+=(--only "$ONLY"); [ $DRY -eq 1 ] && ARGS+=(--dry-run)
if [ -n "$GPUJOBS" ]; then
  case "$GPU" in 0|1) ;; *) refuse "--gpu-jobs needs --gpu 0|1";; esac
  ARGS+=(--gpu-jobs "$GPUJOBS" --gpu "$GPU" --gpu-max "$GPUMAX")
fi
say "x2 driver: slots=$SLOTS threads=$THREADS only=${ONLY:-all} gpu_jobs=${GPUJOBS:-none} gpu=${GPU:-none} dry=$DRY plan_sha=$(sha256sum "$ST/PLAN_X2.json" | cut -c1-16)"
cd "$X2"
export CUDA_VISIBLE_DEVICES=""
nice -n 19 "$PY" -B tools_x2/sched_x2.py "${ARGS[@]}"
