#!/usr/bin/env bash
# v4data10/job_final.sh (owner: Data engineer, DATA-10) - server job 2 (after job_build.sh rc 0): disk check, the
# registry extension contracts/manifest_registry/REGISTRY_V2_PRETRAIN_EXT.json (a NEW file; never overwrites different
# bytes; REGISTRY_V2 is only read), the full test suite (synthetic + real) and the final receipt. CPU only. Launch:
#   (nohup setsid bash <W>/code/v4data10/job_final.sh > <W>/data_pretrain/_run/job_final.log 2>&1 < /dev/null &)
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$W/data_pretrain
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
echo "[job] registry extension $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m v4data10.d10_finalize registry > "$RUN/registry.log" 2>&1
RRC=$?
cat "$RUN/registry.log"
if [ "$RRC" -ne 0 ]; then
  echo "STOP: registry rc=$RRC"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_end_utc.txt"; exit 9
fi
echo "[job] full tests $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$W/code/tests_data10" || exit 6
D10_PYTEST_BASETEMP="$RUN/pytest_final" "$PY" -B run_all.py "$OUT/receipts/TESTS_server.json"
TRC=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_end_utc.txt"
df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ' | sed 's/^/free_bytes_after_final=/' > "$RUN/df_after_final.txt"
cd "$W/code" || exit 6
"$PY" -B -m v4data10.d10_finalize final --tests-receipt "$OUT/receipts/TESTS_server.json" \
  --rss-log "$RUN/rss_build.jsonl" --run-dir "$RUN"
FRC=$?
rmdir "$RUN/tmp" 2>/dev/null
echo "JOB_FINAL_DONE registry_rc=$RRC tests_rc=$TRC final_rc=$FRC end $(cat "$RUN/final_job_end_utc.txt")"
