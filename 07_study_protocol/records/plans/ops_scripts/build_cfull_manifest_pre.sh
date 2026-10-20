#!/usr/bin/env bash
# DATA-08 CPU manifest build from the already deployed interim code tree.
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$V/data_cfull
RUN=$OUT/_run
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
CR=$V/code_ds3pre
test -d "$CR/v4data08"
test ! -e "$OUT/manifests/FULL_TRAIN_5_100_TIEBREAK.OK"
mkdir -p "$RUN/tmp"
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1 TMPDIR=$RUN/tmp
free=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
test "$free" -ge 10737418240
echo "$(date -u +%FT%TZ) DATA08_BUILD_START code_ds3pre" > "$RUN/build_pre.status"
cd "$CR"
if "$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$RUN/rss_build_pre.jsonl" -- \
  "$PY" -B -m v4data08.d8_build --out "$OUT" > "$RUN/build_pre.log" 2>&1; then rc=0; else rc=$?; fi
echo "$(date -u +%FT%TZ) DATA08_BUILD_DONE rc=$rc" >> "$RUN/build_pre.status"
exit "$rc"
