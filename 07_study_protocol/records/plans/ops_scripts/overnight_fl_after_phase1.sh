#!/bin/bash
# Registered MAIN FL arms, serialized on GPU 0 after the A40 gate and LR selection.
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
CR=$V/code_ds3r2
R=$V/runs_ds3
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
LOG=$R/overnight_fl/queue.log
P1=$R/phase1_r2/phase1_r2.log
mkdir -p "$R/overnight_fl" "$R/FL" "$V/contracts/fl"
test ! -e "$LOG"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
step() { echo "$(ts) $*" >> "$LOG"; }
step 'WAIT_PHASE1 gpu0 code_ds3r2'
for _ in $(seq 1 2880); do
  if grep -q 'PHASE1_DONE' "$P1" 2>/dev/null; then break; fi
  if grep -qE '(STOP_|rc=[1-9])' "$P1" 2>/dev/null; then step 'STOP_PHASE1_FAILED'; exit 4; fi
  sleep 30
done
if ! grep -q 'PHASE1_DONE' "$P1" 2>/dev/null; then step 'STOP_PHASE1_TIMEOUT'; exit 5; fi
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$CR"
SEL=$V/contracts/fl/LR_SELECTION.json
test ! -e "$SEL"
step 'LR_SELECT_START'
if "$PY" -B v4int/lr_screen.py select --runs "$R/LRSCREEN/x0.5" "$R/LRSCREEN/x1" "$R/LRSCREEN/x2" --out "$SEL" > "$R/overnight_fl/lr_select.log" 2>&1; then rc=0; else rc=$?; fi
step "LR_SELECT_DONE rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
run_fl() {
  arm=$1; seed=$2
  out=$R/FL/${arm}_s${seed}
  test ! -e "$out"
  step "RUN_START ${arm}_s${seed}"
  if "$PY" -B v4int/launch_fl.py --arm "$arm" --seed "$seed" --gpu 0 --run-dir "$out" \
      --history-policy USER_HISTORY_L50 --lr-selection "$SEL" > "$R/overnight_fl/${arm}_s${seed}.log" 2>&1; then rc=0; else rc=$?; fi
  step "RUN_DONE ${arm}_s${seed} rc=$rc"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
}
for seed in 2026 2027; do
  run_fl FA "$seed"
  run_fl PF "$seed"
  run_fl FA_Q8 "$seed"
done
step 'OVERNIGHT_FL_DONE rc=0'
