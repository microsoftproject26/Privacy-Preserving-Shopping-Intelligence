#!/usr/bin/env bash
# v4data08/job_build.sh (owner: Data engineer, DATA-08) - server job 1: disk check, then the C_FULL training manifest
# FULL_TRAIN_5_100_TIEBREAK under the 16 GiB RSS guard. CPU only; writes only under ppsi_v4/data_cfull/; the v1 dataset
# is read-only and every read is read_non_test. Launch detached:
#   mkdir -p <W>/data_cfull/_run && (nohup setsid bash <W>/code/v4data08/job_build.sh > <W>/data_cfull/_run/job_build.log 2>&1 < /dev/null &)
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$W/data_cfull
RUN=$OUT/_run
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/build_job_start_utc.txt"
FREE=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
echo "free_bytes_before_build=$FREE" > "$RUN/df_before_build.txt"
if [ "$FREE" -lt 10737418240 ]; then
  echo "STOP: free space $FREE bytes is below 10 GiB"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/build_job_end_utc.txt"; exit 5
fi
if [ -e "$OUT/manifests/FULL_TRAIN_5_100_TIEBREAK.OK" ]; then
  echo "STOP: $OUT/manifests/FULL_TRAIN_5_100_TIEBREAK.OK exists (already built; move it aside explicitly to rebuild)"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/build_job_end_utc.txt"; exit 7
fi
cd "$W/code" || exit 6
echo "[job] DATA-08 build start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$RUN/rss_build.jsonl" -- \
  "$PY" -B -m v4data08.d8_build --out "$OUT" > "$RUN/build.log" 2>&1
BRC=$?
tail -5 "$RUN/build.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/build_job_end_utc.txt"
df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ' | sed 's/^/free_bytes_after_build=/' > "$RUN/df_after_build.txt"
rmdir "$RUN/tmp" 2>/dev/null
echo "JOB_BUILD_DONE rc=$BRC end $(cat "$RUN/build_job_end_utc.txt")"
