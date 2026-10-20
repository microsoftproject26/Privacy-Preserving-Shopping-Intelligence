#!/bin/bash
# X2 CPU smoke (SMOKE_NOT_EVIDENCE; writes only runs_bench_x/x2_smoke). Proves: PRE on S, FA / FA_WARM / C_SMALL on F,
# the refusals (wrong role, PRE of another split, D256 C_SMALL), PRE D256 -> FA_WARM D256, and times D64 vs D256 rounds.
# A smoke PRE writes only its endpoint checkpoint, so the warm smokes use --pre-ckpt endpoint (real runs: best).
# SMOKE_PART=warm runs only the warm cases.
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
X2=$V/code_bench_x2; R=$V/runs_bench_x/x2_smoke; SP=$V/runs_bench_x/x2_A1/splits
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
export CUDA_VISIBLE_DEVICES="" BM_ALLOW_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$X2"
r() { echo "=== $*"; nice -n 19 "$PY" -B -m v4bench.run run --threads 4 --runs-root "$R" --smoke "$@"; echo "rc=$?"; }
D=s3_toys
PRE50=$R/$D/SMOKE/PRE__B96_L50_D0.2__UDJ50-S_s2026
PRE256=$R/$D/SMOKE/PRE__B96_L50_D0.2__V256_s2026
if [ "${SMOKE_PART:-all}" = warm ]; then
  r --dataset $D --arm FA_WARM --seed 2026 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.3 --local-passes 1 --pre-run $PRE50 --pre-ckpt endpoint --smoke-rounds 3 --user-subset $SP/${D}__DJ50.json --subset-role F
  r --dataset $D --arm FA_WARM --seed 2026 --budget-efe 24 --server-tau 1e-6 --pre-run $PRE50 --pre-ckpt endpoint --smoke-rounds 2 --user-subset $SP/${D}__DJ15.json --subset-role F
  r --dataset $D --arm FA_WARM --seed 2026 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.3 --local-passes 1 --pre-run $PRE256 --pre-ckpt endpoint --smoke-rounds 3 --model-variant SASREC_D256_B3
  r --dataset $D --arm FA_WARM --seed 2026 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.3 --local-passes 1 --pre-run $PRE256 --pre-ckpt endpoint --smoke-rounds 2
  r --dataset $D --arm FA_WARM --seed 2026 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.3 --local-passes 1 --pre-run $PRE50 --pre-ckpt endpoint --smoke-rounds 2
  echo SMOKE_SCRIPT_END; exit 0
fi
r --dataset $D --arm PRE --seed 2026 --budget-efe 96 --dropout 0.2 --smoke-epochs 0.05 --user-subset $SP/${D}__DJ50.json --subset-role S
PRE50=$R/$D/SMOKE/PRE__B96_L50_D0.2__UDJ50-S_s2026
r --dataset $D --arm FA_WARM --seed 2026 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.3 --local-passes 1 --pre-run $PRE50 --pre-ckpt endpoint --smoke-rounds 3 --user-subset $SP/${D}__DJ50.json --subset-role F
r --dataset $D --arm FA --seed 2026 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.1 --local-passes 1 --smoke-rounds 3 --user-subset $SP/${D}__DJ50.json --subset-role F
r --dataset $D --arm C_SMALL --seed 2026 --budget-efe 24 --dropout 0.1 --smoke-epochs 0.05 --user-subset $SP/${D}__DJ50.json --subset-role F
echo "--- expected refusals (rc=2):"
r --dataset $D --arm FA --seed 2026 --budget-efe 24 --server-tau 1e-6 --smoke-rounds 2 --user-subset $SP/${D}__DJ50.json --subset-role S
r --dataset $D --arm FA_WARM --seed 2026 --budget-efe 24 --server-tau 1e-6 --pre-run $PRE50 --pre-ckpt endpoint --smoke-rounds 2 --user-subset $SP/${D}__DJ15.json --subset-role F
r --dataset $D --arm C_SMALL --seed 2026 --budget-efe 24 --smoke-epochs 0.01 --model-variant SASREC_D256_B3
r --dataset $D --arm FA --seed 2026 --budget-efe 24 --smoke-rounds 2 --runs-root $V/runs_bench_x/lean_fake
echo "--- A2(b) D256"
r --dataset $D --arm PRE --seed 2026 --budget-efe 96 --dropout 0.2 --smoke-epochs 0.05 --model-variant SASREC_D256_B3
PRE256=$R/$D/SMOKE/PRE__B96_L50_D0.2__V256_s2026
r --dataset $D --arm FA_WARM --seed 2026 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.3 --local-passes 1 --pre-run $PRE256 --pre-ckpt endpoint --smoke-rounds 3 --model-variant SASREC_D256_B3
echo "--- refusal: D64 FA_WARM from the D256 PRE (rc=2)"
r --dataset $D --arm FA_WARM --seed 2026 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.3 --local-passes 1 --pre-run $PRE256 --pre-ckpt endpoint --smoke-rounds 2
echo "--- timing D64 vs D256 (8 rounds each)"
for DS in s3_toys ml_1m; do
  r --dataset $DS --arm FA --seed 2027 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.1 --local-passes 1 --smoke-rounds 8
  r --dataset $DS --arm FA --seed 2027 --budget-efe 24 --dropout 0.1 --server-tau 1e-6 --server-lr 0.001 --client-lr 0.1 --local-passes 1 --smoke-rounds 8 --model-variant SASREC_D256_B3
done
echo SMOKE_SCRIPT_END
