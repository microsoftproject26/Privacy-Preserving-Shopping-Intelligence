#!/usr/bin/env bash
# FS Stage B training queue (FS-02). Mirrors runs_g1/ops07_sync3/fs_stageA.sh exactly (same skip-if-done rule, same
# log shape, same env), generalized to the schema id(s) v4fs.stage_b names: one sequential queue per physical GPU.
# Usage: fs_stageB.sh <gpu> <seed> <schema_id> [<schema_id> ...]   -- SASREC then GRU, per schema, in the given order.
set -u
V4=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
GPU=${1:?gpu}; SEED=${2:?seed}; shift 2
if [ "$#" -eq 0 ]; then echo "usage: fs_stageB.sh <gpu> <seed> <schema_id> [<schema_id> ...]"; exit 2; fi
SCHEMAS=("$@")
OUT=$V4/runs_fs/stageB
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
cd "$V4/code"
for SCH in "${SCHEMAS[@]}"; do
  for FAM in SASREC GRU; do
    RUN="$OUT/${SCH}__${FAM}__${SEED}"
    if [ -f "$RUN/FS_SUMMARY.json" ]; then echo "$(date -u +%FT%TZ) skip done $RUN" >> "$OUT/logs/queue_gpu${GPU}.log"; continue; fi
    echo "$(date -u +%FT%TZ) start $SCH $FAM $SEED gpu$GPU free=$(df -BG /home/jovyan | tail -1 | awk '{print $4}')" >> "$OUT/logs/queue_gpu${GPU}.log"
    "$PY" -B v4fs/run_fs.py --family "$FAM" --schema "$SCH" --seed "$SEED" --gpu "$GPU" --out "$RUN" \
        > "$OUT/logs/${SCH}__${FAM}__${SEED}.log" 2>&1
    echo "$(date -u +%FT%TZ) end $SCH $FAM $SEED exit=$?" >> "$OUT/logs/queue_gpu${GPU}.log"
  done
done
echo "$(date -u +%FT%TZ) QUEUE_DONE gpu$GPU" >> "$OUT/logs/queue_gpu${GPU}.log"
