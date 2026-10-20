#!/usr/bin/env bash
# v4fs_data/job_manifests.sh (owner: Data engineer, FS-01) - server job 1: disk check, the synthetic pre-build test
# gate, then the FS build and the M05/M06 equivalence run in parallel under ONE 16 GiB RSS guard. CPU only; writes only
# under ppsi_v4/data_fs/; the v1 dataset is read-only. Launch detached:
#   (nohup setsid bash <W>/code/v4fs_data/job_manifests.sh > <W>/data_fs/_run/job_manifests.log 2>&1 < /dev/null &)
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$W/data_fs
RUN=$OUT/_run
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/manifests_job_start_utc.txt"
FREE=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
echo "free_bytes_before_manifests=$FREE" > "$RUN/df_before_manifests.txt"
if [ "$FREE" -lt 10737418240 ]; then
  echo "STOP: free space $FREE bytes is below 10 GiB"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/manifests_job_end_utc.txt"; exit 5
fi
cd "$W/code/tests_fs_data" || exit 6
echo "[job] pre-build tests start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
FS_TESTS_SYNTHETIC_ONLY=1 FS_PYTEST_BASETEMP="$RUN/pytest_pre" "$PY" -B run_all.py "$RUN/TESTS_pre_build.json"
TRC=$?
if [ "$TRC" -ne 0 ]; then
  echo "STOP: pre-build tests failed rc=$TRC"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/manifests_job_end_utc.txt"; exit 7
fi
cd "$W/code" || exit 6
echo "[job] builds start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$RUN/rss_manifests.jsonl" -- \
  bash "$W/code/v4fs_data/builds.sh" "$PY" "$OUT" "$RUN"
BRC=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/manifests_job_end_utc.txt"
df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ' | sed 's/^/free_bytes_after_manifests=/' > "$RUN/df_after_manifests.txt"
rmdir "$RUN/tmp" 2>/dev/null
echo "JOB_MANIFESTS_DONE rc=$BRC $(cat "$RUN/build_rc.txt" 2>/dev/null) end $(cat "$RUN/manifests_job_end_utc.txt")"
