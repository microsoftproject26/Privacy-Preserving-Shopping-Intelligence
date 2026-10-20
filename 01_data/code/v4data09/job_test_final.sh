#!/usr/bin/env bash
# v4data09/job_test_final.sh (owner: Data engineer, DATA-09) - server job 2 of the TEST look (after job_test_build.sh
# rc 0): the TEST gate check, the disk check, the registry extension contracts/manifest_registry/REGISTRY_V2_TEST_EXT.json
# (O_EXCL; never overwrites different bytes; REGISTRY_V2 is only read), the loader-path verification of every TEST role
# (v4int.test_roles), the post-build G0 integrity check (counts only), the DATA-09 test suite and the final receipt.
# CPU only. Launch:
#   (nohup setsid bash <W>/code/v4data09/job_test_final.sh > <W>/data_test/_run/job_test_final.log 2>&1 < /dev/null &)
set -uo pipefail
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$W/data_test
RUN=$OUT/_run
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
mkdir -p "$RUN/tmp"
export TMPDIR="$RUN/tmp"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_start_utc.txt"
cd "$W/code" || exit 6
"$PY" -B -m v4data09.d9_gate > "$RUN/gate_before_final.json" 2> "$RUN/gate_before_final.err"
GRC=$?
if [ "$GRC" -ne 0 ]; then
  echo "STOP: TEST gate closed (rc=$GRC): $(cat "$RUN/gate_before_final.err")"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_end_utc.txt"; exit 8
fi
FREE=$(df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ')
echo "free_bytes_before_final=$FREE" > "$RUN/df_before_final.txt"
if [ "$FREE" -lt 10737418240 ]; then
  echo "STOP: free space $FREE bytes is below 10 GiB"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_end_utc.txt"; exit 5
fi
echo "[job] registry extension $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m v4data09.d9_finalize registry > "$RUN/registry.log" 2>&1
RRC=$?
cat "$RUN/registry.log"
if [ "$RRC" -ne 0 ]; then
  echo "STOP: registry rc=$RRC"; date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_end_utc.txt"; exit 9
fi
echo "[job] loader-path verification $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m v4data.run_bounded --max-rss-mib 16384 --log "$RUN/rss_verify.jsonl" -- \
  "$PY" -B -m v4data09.d9_finalize verify > "$RUN/verify.log" 2>&1
VRC=$?
tail -3 "$RUN/verify.log"
echo "[job] G0 integrity (counts only) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -B -m v4data09.d9_finalize g0 > "$RUN/g0.log" 2>&1
GORC=$?
cat "$RUN/g0.log"
echo "[job] tests $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$W/code/tests_data09" || exit 6
D9_PYTEST_BASETEMP="$RUN/pytest_final" "$PY" -B run_all.py "$OUT/receipts/TESTS_server.json"
TRC=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/final_job_end_utc.txt"
df -B1 --output=avail /home/jovyan | tail -1 | tr -d ' ' | sed 's/^/free_bytes_after_final=/' > "$RUN/df_after_final.txt"
cd "$W/code" || exit 6
"$PY" -B -m v4data09.d9_finalize final --tests-receipt "$OUT/receipts/TESTS_server.json" \
  --rss-log "$RUN/rss_build.jsonl" --rss-log "$RUN/rss_verify.jsonl" --run-dir "$RUN"
FRC=$?
rmdir "$RUN/tmp" 2>/dev/null
echo "JOB_TEST_FINAL_DONE registry_rc=$RRC verify_rc=$VRC g0_rc=$GORC tests_rc=$TRC final_rc=$FRC end $(cat "$RUN/final_job_end_utc.txt")"
