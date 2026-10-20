#!/usr/bin/env bash
# final_stage.sh (owner: Opus): registry v3 amendment 2 final stage, run once by tools/campaign_v3b.py.
# select (full VS only) -> lock (O_EXCL, before any VC/TEST row) -> ONE retrospective VC/TEST evaluation
# on both GPUs (disjoint shards, resumable) -> finalize (archived fingerprint gates, bootstraps) -> report.
set -u
S=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1
P=~/venvs/ppsi-gpu/bin/python
cd $S/code
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log "select"; $P lock_retro_v3b.py select || { log "select failed"; exit 2; }
log "lock";   $P lock_retro_v3b.py lock   || { log "lock failed"; exit 2; }
for attempt in 1 2; do
  log "retrospective eval, attempt $attempt"
  CUDA_VISIBLE_DEVICES=0 $P lock_retro_v3b.py eval --worker 0 --workers 2 > $S/logs/retro_w0.log 2>&1 &
  p0=$!
  CUDA_VISIBLE_DEVICES=1 $P lock_retro_v3b.py eval --worker 1 --workers 2 > $S/logs/retro_w1.log 2>&1 &
  p1=$!
  wait $p0; r0=$?; wait $p1; r1=$?
  log "workers rc $r0 $r1"
  [ "$r0" = 0 ] && [ "$r1" = 0 ] && break
done
log "finalize"; $P lock_retro_v3b.py finalize || { log "finalize failed"; exit 3; }
log "report (automatic rendering; the lead writes the final narrative)"
$P render_report.py > $S/logs/render_report.log 2>&1 || log "render_report returned non-zero (see log)"
date -u +%FT%TZ > $S/state/FINAL_STAGE_DONE
log "FINAL STAGE DONE"
