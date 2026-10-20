#!/bin/bash
# X2 GPU smoke (SMOKE_NOT_EVIDENCE; runs_bench_x/x2_smoke_gpu): ML-1M D256 FA rounds and PRE steps on --device cuda --gpu 1
# (NVIDIA_A40-FP32-DET class), plus a D256 FA_WARM on cuda from a CPU smoke PRE (cross-class warm start, recorded).
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
R=$V/runs_bench_x/x2_smoke_gpu; G=${GPU:-1}
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
export BM_ALLOW_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $V/code_bench_x2
r() { echo "=== $*"; nice -n 19 "$PY" -B -m v4bench.run run --threads 4 --runs-root "$R" --smoke --device cuda --gpu $G "$@"; echo "rc=$?"; }
r --dataset ml_1m --arm FA --seed 2027 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.1 --local-passes 1 --smoke-rounds 8 --model-variant SASREC_D256_B3
r --dataset ml_1m --arm PRE --seed 2027 --budget-efe 96 --dropout 0.2 --smoke-epochs 0.02 --model-variant SASREC_D256_B3
r --dataset s3_toys --arm FA_WARM --seed 2026 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.3 --local-passes 1 --pre-run $V/runs_bench_x/x2_smoke/s3_toys/SMOKE/PRE__B96_L50_D0.2__V256_s2026 --pre-ckpt endpoint --smoke-rounds 3 --model-variant SASREC_D256_B3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo GPU_SMOKE_END
