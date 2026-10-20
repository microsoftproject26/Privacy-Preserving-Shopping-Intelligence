#!/usr/bin/env bash
# census_v2/job.sh (owner: Data Lead, CENSUS-02) - server job: unit tests, then the RSS-guarded census, then the
# receipt finalisation. Launch detached:
#   (nohup setsid bash job.sh > census_v5_100/_run/job.log 2>&1 < /dev/null &)
# CPU only; writes only under ppsi_v4/census_v5_100/; the v1 dataset is read-only.
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
LA=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai
DS=/home/jovyan/shared_data/rees46_nextitem_v1
OUT=$W/census_v5_100
RUN=$OUT/_run
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/job_start_utc.txt"
FREE=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
echo "free_bytes_before=$FREE" > "$RUN/df_before.txt"
if [ "$FREE" -lt 10737418240 ]; then
  echo "STOP: free space $FREE bytes is below 10 GiB"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/job_end_utc.txt"; exit 5
fi
cd "$W/code" || exit 6
echo "[job] tests start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m census_v2.test_census02 --receipt "$RUN/TESTS_server.json"
TRC=$?
if [ "$TRC" -ne 0 ]; then
  echo "STOP: unit tests failed rc=$TRC"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/job_end_utc.txt"; exit 7
fi
echo "[job] census start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$RUN/rss.jsonl" -- \
  "$PY" -B -m census_v2.census02 --dataset "$DS" --out "$OUT" --prior-root "$LA" \
  --tests-receipt "$RUN/TESTS_server.json"
CRC=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/job_end_utc.txt"
echo "[job] census rc=$CRC end $(cat "$RUN/job_end_utc.txt")"
"$PY" -B -m census_v2.finalize --out "$OUT" --rss-log "$RUN/rss.jsonl" --job-start "$RUN/job_start_utc.txt" \
  --job-end "$RUN/job_end_utc.txt"
FRC=$?
df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ' | sed 's/^/free_bytes_after=/' > "$RUN/df_after.txt"
rmdir "$RUN/tmp" 2>/dev/null
echo "JOB_DONE census_rc=$CRC finalize_rc=$FRC"
