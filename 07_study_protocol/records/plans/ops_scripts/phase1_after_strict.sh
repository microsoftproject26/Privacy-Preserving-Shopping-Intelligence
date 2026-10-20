#!/bin/bash
# Serialized A40 gate -> central re-evaluation -> registered CALIB LR training. TEST remains sealed.
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
CR=$V/code_ds3r1
R=$V/runs_ds3
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
SLOG=$R/strict_a40/strict.status
LOG=$R/phase1/phase1.log
mkdir -p "$R/phase1" "$R/REEVAL_A40" "$R/LRSCREEN"
test ! -e "$LOG"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
step() { echo "$(ts) $*" >> "$LOG"; }
step "WAIT_STRICT gpu0 code_ds3r1"
for _ in $(seq 1 1440); do
  if grep -q 'STRICT_DONE rc=' "$SLOG" 2>/dev/null; then break; fi
  sleep 10
done
if ! grep -q 'STRICT_DONE rc=0' "$SLOG" 2>/dev/null; then step "STOP_STRICT_NOT_PASS"; exit 4; fi
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$CR"
PIN=$V/contracts/eval_device/EVAL_DEVICE_PIN_A40.json
test ! -e "$PIN"
step "PIN_START"
if "$PY" -B -m v4eval.device_pin write --device-index 0 \
  --t3-receipt "$R/T3_A40_r2/out/T3_GPU_RECEIPT.json" \
  --strict-receipt "$R/strict_a40/SYS04C_VARIANT_BENCH.json" --out "$PIN" \
  > "$R/phase1/pin.log" 2>&1; then step "PIN_DONE rc=0"; else rc=$?; step "PIN_DONE rc=$rc"; exit "$rc"; fi
run_reeval() {
  arm=$1; seed=$2; rule=$3; sha=$4
  if [ "$arm" = C_BIG ]; then
    base=$V/runs_p3/SASREC/02B_SASREC_s$seed
    variant=SASREC_D256_B3; recipe=02B_SASREC; source=02B_SASREC_MAIN_s$seed
  else
    base=$V/runs_ds/C_SMALL_s$seed
    variant=SASREC_D64_B2; recipe=SMALL_SASREC_D64_B2; source=SMALL_SASREC_D64_B2_MAIN_s$seed
  fi
  if [ "$rule" = PB ]; then selection=PRACTICAL_BEST; ckpt=best; ckptid=$source:best; else selection=CONTROLLED_BUDGET; ckpt=endpoint_6.0; ckptid=$source:endpoint_6.0; fi
  out=$R/REEVAL_A40/${arm}_s${seed}_${rule}
  test ! -e "$out"
  step "REEVAL_START ${arm}_s${seed}_${rule}"
  if "$PY" -B v4int/reeval.py --ckpt "$base/sealed/checkpoints/$ckpt.pt" --expect-sha256 "$sha" \
    --ckpt-kind central --variant "$variant" --recipe-id "$recipe" --seed "$seed" --role MAIN \
    --arm-label "$arm" --selection-rule "$selection" --checkpoint-id "$ckptid" \
    --gpu 0 --out-dir "$out" --history-policy USER_HISTORY_L50 \
    > "$R/phase1/reeval_${arm}_s${seed}_${rule}.log" 2>&1; then rc=0; else rc=$?; fi
  step "REEVAL_DONE ${arm}_s${seed}_${rule} rc=$rc"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
}
run_reeval C_BIG   2026 PB df35157f8ea93762ffed75048a43217d8ce1406de208b8429fd3f04af226c749
run_reeval C_BIG   2026 CB 7fb323b282539b788c7b1b4f590a8a8f033a7fd68b9f7ca3adb39d1233f723ed
run_reeval C_SMALL 2026 PB e2db4376797756ba06e9b42ea3ef26a4b9b77a01366938a0cda18f37cd1b710d
run_reeval C_SMALL 2026 CB 80921fc87a6163a0cef404c72f1d0656343f28408e153c704cc498b70ac34073
run_reeval C_BIG   2027 PB a8bbfff90bbe35ca6de239503e6ec9225e16fcb68a9eb87e64d6bf4e8aa84b75
run_reeval C_BIG   2027 CB 4b9a7d566ad6bd720a57f51e57dcafa70355624b4e5b8160eae43e0077559492
run_reeval C_SMALL 2027 PB ae9d840d91eea72e778d6a5c9804e241638cd2ff8c1bba5af3cdac8deff6d3f7
run_reeval C_SMALL 2027 CB 312cdf38d47a3f842bcffe2ffee7a32555c01dd257af8c2c78c3da0339c50967
step "REEVAL_ALL_DONE"
for M in 0.5 1 2; do
  out=$R/LRSCREEN/x$M
  test ! -e "$out"
  step "LRSCREEN_START x$M"
  if "$PY" -B v4int/launch_fl.py --arm FA --role CALIBRATION --lr-mult "$M" --seed 2026 \
    --gpu 0 --run-dir "$out" --history-policy USER_HISTORY_L50 > "$R/LRSCREEN/x$M.log" 2>&1; then rc=0; else rc=$?; fi
  step "LRSCREEN_DONE x$M rc=$rc"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
step "PHASE1_DONE"
