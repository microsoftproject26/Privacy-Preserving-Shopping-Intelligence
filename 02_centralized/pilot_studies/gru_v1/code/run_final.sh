#!/usr/bin/env bash
# run_final.sh (owner: Opus) -- autonomous post-training pipeline, FULL_RUN_PROTOCOL.md sec. 7-8.
# 1. wait until BOTH supervisors wrote SUPERVISOR_DONE (abort on SUPERVISOR_STOPPED)
# 2. both summary.json: final_status COMPLETE (the lock re-checks this, strictly)
# 3. frozen-dataset integrity re-verification (sec. 1: before TEST)
# 4. lock -> confirm (VALIDATION_CONFIRM once) -> test (TEST once)
# Runs detached on the server so it does not depend on any client session. Each step's
# output goes to $W/final_eval/run_final.log; a failure stops the chain (set -e).
set -euo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1
R=$W/experiments/full
F0=$R/F0_item_only_h256
F4=$R/F4_rich_side_information_h256
P=~/venvs/ppsi-gpu/bin/python
mkdir -p $W/final_eval
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$1"; }

log "waiting for both arms"
while true; do
  for d in $F0 $F4; do
    if [ -f "$d/SUPERVISOR_STOPPED" ]; then log "ABORT: $d stopped: $(cat $d/SUPERVISOR_STOPPED)"; exit 3; fi
  done
  if [ -f "$F0/SUPERVISOR_DONE" ] && [ -f "$F4/SUPERVISOR_DONE" ]; then break; fi
  sleep 60
done
log "both arms DONE"
for d in $F0 $F4; do
  $P -c "import json,sys; s=json.load(open('$d/summary.json')); print('$d', s['final_status'], 'best_epoch', s['best_epoch'], 'best', s['best_metric'], 'resumes', s['resume_count']); sys.exit(0 if s['final_status']=='COMPLETE' else 1)"
done

# both GPUs are free once the trainers exit; use GPU 0, refuse if a foreign process holds it
n=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$n" != "0" ]; then log "GPU0 busy, trying GPU1"; GPU=1; else GPU=0; fi
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=$GPU
cd $W/code

log "integrity check of the frozen dataset"
$P verify_frozen.py --out $W/final_eval/verify_frozen_before_test.json | tail -3
$P -c "import json,sys; d=json.load(open('$W/final_eval/verify_frozen_before_test.json')); print('all_ok', d['all_ok'], d['verified']); sys.exit(0 if d['all_ok'] else 1)"

log "lock"
$P lock_and_eval.py lock --f0-run $F0 --f4-run $F4
log "confirm (VALIDATION_CONFIRM, once)"
$P lock_and_eval.py confirm
log "test (TEST, once)"
$P lock_and_eval.py test
log "FINAL_PIPELINE_DONE"
