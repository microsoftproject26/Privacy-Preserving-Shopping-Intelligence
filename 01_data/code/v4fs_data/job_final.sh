#!/usr/bin/env bash
# v4fs_data/job_final.sh (owner: Data engineer, FS-01) - server job 2 (after job_manifests.sh): disk check, the optional
# TRANSFORM_STATE_FS fit under the 16 GiB RSS guard, the registry, the full test suite (synthetic + real) and the final
# receipt. CPU only; writes only under ppsi_v4/data_fs/ and contracts/manifest_registry/. Launch detached:
#   (nohup setsid bash <W>/code/v4fs_data/job_final.sh > <W>/data_fs/_run/job_final.log 2>&1 < /dev/null &)
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$W/data_fs
RUN=$OUT/_run
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_start_utc.txt"
FREE=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
echo "free_bytes_before_final=$FREE" > "$RUN/df_before_final.txt"
if [ "$FREE" -lt 10737418240 ]; then
  echo "STOP: free space $FREE bytes is below 10 GiB"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_end_utc.txt"; exit 5
fi
cd "$W/code" || exit 6
echo "[job] transform state start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$RUN/rss_transform.jsonl" -- \
  "$PY" -B -m v4fs_data.fs_transform --workers 6 > "$RUN/transform.log" 2>&1
SRC=$?
tail -2 "$RUN/transform.log"
if [ "$SRC" -ne 0 ]; then
  echo "STOP: transform state rc=$SRC"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_end_utc.txt"; exit 8
fi
echo "[job] registry $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m v4fs_data.fs_finalize registry > "$RUN/registry.log" 2>&1
RRC=$?
cat "$RUN/registry.log"
if [ "$RRC" -ne 0 ]; then
  echo "STOP: registry rc=$RRC"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_end_utc.txt"; exit 9
fi
echo "[job] full tests $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$W/code/tests_fs_data" || exit 6
FS_PYTEST_BASETEMP="$RUN/pytest_final" "$PY" -B run_all.py "$OUT/receipts/TESTS_server.json"
TRC=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_end_utc.txt"
df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ' | sed 's/^/free_bytes_after_final=/' > "$RUN/df_after_final.txt"
cd "$W/code" || exit 6
"$PY" -B -m v4fs_data.fs_finalize final --tests-receipt "$OUT/receipts/TESTS_server.json" \
  --rss-log "$RUN/rss_manifests.jsonl" --rss-log "$RUN/rss_transform.jsonl" --run-dir "$RUN"
FRC=$?
rmdir "$RUN/tmp" 2>/dev/null
echo "JOB_FINAL_DONE transform_rc=$SRC registry_rc=$RRC tests_rc=$TRC final_rc=$FRC end $(cat "$RUN/final_job_end_utc.txt")"
