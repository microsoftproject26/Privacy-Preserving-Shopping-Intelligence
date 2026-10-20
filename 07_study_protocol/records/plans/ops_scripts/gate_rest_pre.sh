#!/bin/bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
CR=$V/code_ds3pre
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
R=$V/runs_ds3/pre
LOG=$R/gate_rest.log
test ! -e "$LOG"
test ! -e "$R/smoke/PF_r10"
test -f "$V/runs_ds3/T3_A40_r2/out/T3_GPU_RECEIPT.json"
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
unset CUDA_VISIBLE_DEVICES
cd "$CR"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
step() { echo "$(ts) $*" >> "$LOG"; }
run() {
  name=$1; shift; t0=$(date +%s); step "START $name"
  if "$@" > "$R/smoke/$name.log" 2>&1; then rc=0; else rc=$?; fi
  step "END $name rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  if [ "$rc" -ne 0 ]; then echo "GATE_REST_DONE rc=$rc" >> "$LOG"; exit "$rc"; fi
}
FL="$PY -B v4int/launch_fl.py --gpu 0 --history-policy USER_HISTORY_L50 --smoke"
DEV="$PY -B v4int/launch_device.py --gpu 0 --history-policy USER_HISTORY_L50 --smoke"
run PF_r10        $FL --arm PF --seed 2026 --run-dir $R/smoke/PF_r10 --smoke-rounds 10 --lr-mult 1
run FA_Q8_r10     $FL --arm FA_Q8 --seed 2026 --run-dir $R/smoke/FA_Q8_r10 --smoke-rounds 10 --lr-mult 1
run S_CAL_r3      $FL --arm S_CAL --seed 2026 --run-dir $R/smoke/S_CAL_r3 --smoke-rounds 3 --lr-mult 1
run DP8_r3        $FL --arm DP --epsilon 8 --seed 2026 --run-dir $R/smoke/DP8_r3 --smoke-rounds 3 --lr-mult 1 --clip-record $R/smoke/S_CAL_r3/S_RECORD.json
run LO_c64        $DEV --arm LO --seed 2026 --role MAIN --run-dir $R/smoke/LO_c64 --smoke-clients 64
run SEL_FT_C_c64  $DEV --arm FT_C --seed 2026 --role UNSEEN_VAL --select --base $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --base-kind central --run-dir $R/smoke/SEL_FT_C_c64 --smoke-clients 64
mkdir -p "$R/sys04c"
step "START sys04c D256_B3"
if CUDA_VISIBLE_DEVICES=0 $PY -B tests_train_gpu/sys04c_variant_bench.py all --out-dir $R/sys04c --variants SASREC_D256_B3 --reps 2 --updates 300 > "$R/sys04c/bench.log" 2>&1; then rc=0; else rc=$?; fi
step "END sys04c rc=$rc"
echo "GATE_REST_DONE rc=$rc" >> "$LOG"
exit "$rc"
