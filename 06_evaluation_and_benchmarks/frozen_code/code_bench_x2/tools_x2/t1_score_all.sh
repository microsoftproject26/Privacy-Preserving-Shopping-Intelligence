#!/bin/bash
# T1 scoring lane (POST_HOC_EXPLORATORY; run ONLY after the BIND entry is in the register, local + server).
# Each dataset: final_x.py score (gate -> fingerprint -> holdout -> TEST_LOOK -> results), then its shas are printed
# (TEST_LOOK / TEST_RESULTS / TEST_USERS / TEST_ATTEMPTS lines; values are never printed). Then analyze -> X_RESULT,
# and the "after" inventory of runs_bench / runs_bench_gpu compared with "before". Stops at the first failure.
#   setsid nohup bash tools_x2/t1_score_all.sh > $V/runs_bench_x/x2_state/logs/t1_score_all.log 2>&1 &
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
P="nice -n 19 /home/jovyan/venvs/ppsi-gpu/bin/python -B"
T=$V/runs_bench_x/TEST
THREADS=${THREADS:-16}
cd $V/code_bench_x2
export CUDA_VISIBLE_DEVICES=""
echo "[$(date -u +%FT%TZ)] T1 start; lock $(sha256sum $V/company/gates/EXP_BENCH_X_TEST_LOCK.json | cut -c1-64); final_x $(sha256sum tools_x2/final_x.py | cut -c1-16); register $(sha256sum $V/company/registers/TEST_EXPOSURE_REGISTER.json | cut -c1-16)"
for ds in s3_toys s3_beauty s3_sports ml_1m amazon2023_video_games amazon2023_baby_products; do
  echo "[$(date -u +%FT%TZ)] score $ds"
  $P tools_x2/final_x.py score --dataset $ds --threads $THREADS
  echo "SHAS $ds"; (cd $T/$ds && sha256sum TEST_LOOK.json TEST_RESULTS.json TEST_USERS.int64.npy && wc -l TEST_ATTEMPTS.jsonl)
  chmod -R a-w $T/$ds
done
echo "[$(date -u +%FT%TZ)] analyze"
$P tools_x2/final_x.py analyze
sha256sum $T/ANALYSIS_EXP_BENCH_X/X_RESULT.json
$P tools_x2/inventory_x.py snap --label after
$P tools_x2/inventory_x.py compare before after
echo "[$(date -u +%FT%TZ)] T1_SCORE_ALL_END"
