#!/usr/bin/env bash
# v4data09/job_test_build.sh (owner: Data engineer, DATA-09) - server job 1 of the single TEST look: the TEST gate check
# (ALL_METHODS_LOCK.json + its MANIFEST_BUILD TEST_EXPOSURE_REGISTER entry; refuses otherwise), the disk check, then the
# five RETRO_TEST manifests under the 16 GiB RSS guard. CPU only; writes only under ppsi_v4/data_test/; the v1 dataset is
# read-only and TEST rows are read ONLY through v4data09.d9_read.read_with_test under the GateRecord. Run it ONLY after
# Ops has written company/gates/ALL_METHODS_LOCK.json and appended the MANIFEST_BUILD register entry. Launch detached:
#   mkdir -p <W>/data_test/_run && (nohup setsid bash <W>/code/v4data09/job_test_build.sh > <W>/data_test/_run/job_test_build.log 2>&1 < /dev/null &)
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$W/data_test
RUN=$OUT/_run
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/build_job_start_utc.txt"
cd "$W/code" || exit 6
"$PY" -B -m v4data09.d9_gate > "$RUN/gate_before_build.json" 2> "$RUN/gate_before_build.err"
GRC=$?
if [ "$GRC" -ne 0 ]; then
  echo "STOP: TEST gate closed (rc=$GRC): $(cat "$RUN/gate_before_build.err")"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/build_job_end_utc.txt"; exit 8
fi
FREE=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
echo "free_bytes_before_build=$FREE" > "$RUN/df_before_build.txt"
if [ "$FREE" -lt 10737418240 ]; then
  echo "STOP: free space $FREE bytes is below 10 GiB"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/build_job_end_utc.txt"; exit 5
fi
for M in MAIN_RETRO_TEST_5_100_131072 UNEW_HOLDOUT_RETRO_TEST_32768 TRUE_COLD_RETRO_TEST SPARSE_PANEL_16384_RETRO_TEST HEAVY_PANEL_16384_RETRO_TEST; do
  if [ -e "$OUT/manifests/$M.OK" ]; then
    echo "STOP: $OUT/manifests/$M.OK exists (the look is built once; E10.3 allows only an interrupted build to restart)"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/build_job_end_utc.txt"; exit 7
  fi
done
echo "[job] DATA-09 gated RETRO_TEST build start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$RUN/rss_build.jsonl" -- \
  "$PY" -B -m v4data09.d9_build --out "$OUT" > "$RUN/build.log" 2>&1
BRC=$?
tail -5 "$RUN/build.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/build_job_end_utc.txt"
df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ' | sed 's/^/free_bytes_after_build=/' > "$RUN/df_after_build.txt"
rmdir "$RUN/tmp" 2>/dev/null
echo "JOB_TEST_BUILD_DONE rc=$BRC end $(cat "$RUN/build_job_end_utc.txt")"
