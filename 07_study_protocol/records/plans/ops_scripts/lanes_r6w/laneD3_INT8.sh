#!/bin/bash
# D3 (DL 317 / 319): C_SMALL INT8 (K16) on CPU, MASTER_PLAN 8.2; outputs sealed (MAIN), never read before the §9 read
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneD3_INT8.log; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r6.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
O=$R/ONNX; mkdir -p $O
job INT8_DUMP_ROWS nice -n 15 $PY -B -m v4dev.dump_rows --role MAIN --history-policy USER_HISTORY_L50 --out $O/rows.npz
MN=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['manifest_n'])" $O/rows.meta.json) || { step "LANE_STOP no manifest_n"; exit 6; }
step "MANIFEST_N $MN"
job INT8_BENCH nice -n 15 $PY -B -m v4dev.bench --ckpt $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --family SASREC --variant SASREC_D64_B2 --v1-root /home/jovyan/shared_data/rees46_nextitem_v1 --out $O --rows $O/rows.npz --manifest-n $MN --tag onnx_int8
step "QUEUE_DONE rc=0"
