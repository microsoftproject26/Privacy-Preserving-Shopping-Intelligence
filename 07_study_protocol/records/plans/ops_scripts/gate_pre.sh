#!/bin/bash
# MASTER_PLAN_V5 §3 measurement gate (interim tree code_ds3pre; GPU 0 = ours). Smokes only (SMOKE_NOT_EVIDENCE, weights deleted),
# the SYS-04c STRICT bench of SASREC_D256_B3 (2 reps) and an nvidia-smi sampler. Nothing here is a production run; no value is read.
# Usage (server, detached): (nohup setsid bash gate_pre.sh 0 > gate_pre.nohup 2>&1 < /dev/null &)
set -euo pipefail
G=${1:?gpu index}
[ "$G" = 0 ] || { echo "REFUSED: this project owns GPU 0 only" >&2; exit 90; }
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
CR=$V/code_ds3pre
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
R=$V/runs_ds3/pre
LOG=$R/gate.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
step() { echo "$(ts) $*" >> "$LOG"; }
mkdir -p "$R/smoke" "$R/sys04c"
cd "$CR" || { step "REFUSED no $CR"; echo "GATE_DONE rc=90" >> "$LOG"; exit 90; }
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
# nvidia-smi sampler (our card only), 10 s
( while true; do echo "$(ts),$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i $G | tr -d ' ')"; sleep 10; done ) >> "$R/nvsmi.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT
run() {
  name=$1; shift; t0=$(date +%s); step "START $name"
  if "$@" > "$R/smoke/$name.log" 2>&1; then rc=0; else rc=$?; fi
  step "END $name rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  if [ "$rc" -ne 0 ]; then echo "GATE_DONE rc=$rc" >> "$LOG"; exit "$rc"; fi
}
FL="$PY -B v4int/launch_fl.py --gpu $G --history-policy USER_HISTORY_L50 --smoke"
DEV="$PY -B v4int/launch_device.py --gpu $G --history-policy USER_HISTORY_L50 --smoke"
run FA_r20        $FL --arm FA --seed 2026 --run-dir $R/smoke/FA_r20 --smoke-rounds 20 --lr-mult 1
run FA_CALIB_r20  $FL --arm FA --role CALIBRATION --lr-mult 1 --seed 2026 --run-dir $R/smoke/FA_CALIB_r20 --smoke-rounds 20
run PF_r10        $FL --arm PF --seed 2026 --run-dir $R/smoke/PF_r10 --smoke-rounds 10 --lr-mult 1
run FA_Q8_r10     $FL --arm FA_Q8 --seed 2026 --run-dir $R/smoke/FA_Q8_r10 --smoke-rounds 10 --lr-mult 1
run S_CAL_r3      $FL --arm S_CAL --seed 2026 --run-dir $R/smoke/S_CAL_r3 --smoke-rounds 3 --lr-mult 1
run DP8_r3        $FL --arm DP --epsilon 8 --seed 2026 --run-dir $R/smoke/DP8_r3 --smoke-rounds 3 --lr-mult 1 --clip-record $R/smoke/S_CAL_r3/S_RECORD.json
run LO_c64        $DEV --arm LO --seed 2026 --role MAIN --run-dir $R/smoke/LO_c64 --smoke-clients 64
run SEL_FT_C_c64  $DEV --arm FT_C --seed 2026 --role UNSEEN_VAL --select --base $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --base-kind central --run-dir $R/smoke/SEL_FT_C_c64 --smoke-clients 64
# STRICT bench (synthetic data; TUNING role; 2 reps; SASREC_D256_B3 only: D64_B2 needs INT-06's ladder option)
t0=$(date +%s); step "START sys04c D256_B3"
if CUDA_VISIBLE_DEVICES=$G $PY -B tests_train_gpu/sys04c_variant_bench.py all --out-dir $R/sys04c --variants SASREC_D256_B3 --reps 2 --updates 300 > $R/sys04c/bench.log 2>&1; then rc=0; else rc=$?; fi
step "END sys04c rc=$rc wall_s=$(( $(date +%s) - t0 ))"
if [ "$rc" -ne 0 ]; then echo "GATE_DONE rc=$rc" >> "$LOG"; exit "$rc"; fi
echo "GATE_DONE rc=0" >> "$LOG"
