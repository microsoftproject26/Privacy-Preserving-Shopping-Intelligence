#!/bin/bash
# D3 K16 INT8 on code_ds3r7 (DEV-04 Ops command; DL 343). Waits until the Wave-1 benchmark driver has exited, so the
# single-thread latency / RSS numbers are measured on a quiet CPU. MASTER_PLAN 8.2 fallback: bench > 2 h -> --first-users 16384.
# Outputs are sealed MAIN values; Ops never opens BENCH_RESULT / ranks before the §9 read.
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneD3_INT8_r7.log; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r7.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
step "WAIT wave1 driver exit"
while pgrep -f "runs_bench/run_wave1.sh|bash run_wave1.sh" > /dev/null; do sleep 300; done
step "WAVE1_DONE_CPU_QUIET"
CK=$V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt
O=$R/ONNX_R7; [ -e $O/BENCH_RESULT.json ] && { step "LANE_STOP $O already has a result (write-once)"; exit 2; }
job INT8R7_DUMP nice -n 15 $PY -B -m v4dev.dump_rows --role MAIN --history-policy USER_HISTORY_L50 --out $O/rows.npz --ckpt $CK --family SASREC
MN=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['manifest_n'])" $O/rows.meta.json) || { step "LANE_STOP no manifest_n"; exit 6; }
step "MANIFEST_N $MN"
step "JOB_START INT8R7_BENCH gpu=cpu"
timeout 7200 nice -n 15 $PY -B -m v4dev.bench --ckpt $CK --family SASREC --variant SASREC_D64_B2 --v1-root /home/jovyan/shared_data/rees46_nextitem_v1 --out $O --rows $O/rows.npz --manifest-n $MN --tag onnx_int8 --quiet > $Q/INT8R7_BENCH.log 2>&1; rc=$?
step "JOB_DONE INT8R7_BENCH rc=$rc"
if [ $rc -eq 124 ]; then
  step "FALLBACK_8.2 bench > 2 h -> --first-users 16384 (disclosed)"
  O=$R/ONNX_R7_U16384
  job INT8R7U_DUMP nice -n 15 $PY -B -m v4dev.dump_rows --role MAIN --history-policy USER_HISTORY_L50 --out $O/rows.npz --ckpt $CK --family SASREC --first-users 16384
  MN=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['manifest_n'])" $O/rows.meta.json)
  job INT8R7U_BENCH nice -n 15 $PY -B -m v4dev.bench --ckpt $CK --family SASREC --variant SASREC_D64_B2 --v1-root /home/jovyan/shared_data/rees46_nextitem_v1 --out $O --rows $O/rows.npz --manifest-n $MN --tag onnx_int8 --quiet
elif [ $rc -ne 0 ]; then step "LANE_STOP rc=$rc"; exit $rc; fi
step "QUEUE_DONE rc=0"
