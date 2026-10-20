#!/usr/bin/env bash
# laneX_COLD.sh - C1: REES46 COLD FA with the FedAdam tau fix (POST_HOC_EXPLORATORY), GPU 0 beside Part B, nice 10,
# 4 threads per job. Original confirmatory recipe: FedAdam eta 0.01, tau 1e-3, client SGD-m 0.1, 1 pass, batch 16,
# fl_const, 6 EFE (13,654 MAIN rounds). Selection on CALIB_VALIDATION by the registered P2 rule (x_select2.py).
#   M  evidence: 65-round CALIB smokes of the original recipe at tau 1e-3 and at tau 1e-6 (X_TAU_DIAG.jsonl)
#   G  CALIB grid tau 1e-6, eta {1e-4, 3e-4, 1e-3, 1e-2 (original)}, 6 EFE (concurrent with M)
#   E  if the selected cell's best mark is its last mark (still rising): the same cell at 12 EFE; re-select
#   R  MAIN seeds 2026 + 2027 of the selected cell (its budget)
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; CODE=$V/code_x1; X=$V/x_rees46; R=$V/runs_x_rees46/COLD; ST=$R/_state
mkdir -p $ST $R/logs; LOG=$R/lane.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
fail() { touch $ST/FAIL; say "FAIL: $*"; exit 1; }
[ -f $ST/FAIL ] && { say "FAIL marker present: refuse"; exit 2; }
export X_CODE=$CODE PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES
T="--arm FA --gpu 0 --allow-shared-gpu --history-policy USER_HISTORY_L50 --server-opt fedadam --client-opt sgd_m --client-lr 0.1 --lr-shape fl_const --local-passes 1 --local-batch 16 --lr-mult 1"
cd $CODE
# job <name> <args...>
job() { local n=$1; shift; [ -f $ST/$n.ok ] && return 0; say "$n: start"
  if nice -n 10 $PY -B $X/launch_fl_x3.py "$@" >> $R/logs/$n.out 2>&1; then touch $ST/$n.ok; say "$n: ok"; else say "$n: rc=$?"; return 1; fi; }
P=()
job M_tau1e-3 --x-diag-first 100 $T --server-lr 0.01 --server-tau 1e-3 --role CALIBRATION --seed 2026 --smoke --smoke-rounds 50 --run-dir $R/M/FA_eta0.01_tau1e-3 & P+=($!)
job M_tau1e-6 --x-diag-first 100 $T --server-lr 0.01 --server-tau 1e-6 --role CALIBRATION --seed 2026 --smoke --smoke-rounds 50 --run-dir $R/M/FA_eta0.01_tau1e-6 & P+=($!)
G=()
for e in 0.0001 0.0003 0.001 0.01; do
  job G_eta${e} $T --server-lr $e --server-tau 1e-6 --role CALIBRATION --seed 2026 --run-dir $R/G/FA_eta${e}_tau1e-6_E6 & P+=($!); G+=($R/G/FA_eta${e}_tau1e-6_E6)
done
rc=0; for p in "${P[@]}"; do wait $p || rc=$((rc+1)); done; say "M + G done (failed jobs: $rc)"
$PY -B $X/x_select2.py $R/X_COLD_SELECTION_G.json 4 "${G[@]}" > $R/logs/select_G.out 2>&1 || fail "select G"
W=$(grep '^WINNER' $R/logs/select_G.out | cut -d' ' -f2)
RISE=$($PY -c "import json;print(int(json.load(open('$R/X_COLD_SELECTION_G.json'))['winner']['rising_at_end']))")
ETA=$($PY -c "import json;print(format(json.load(open('$R/X_COLD_SELECTION_G.json'))['winner']['eta'],'g'))")
say "G winner eta $ETA (rising at end: $RISE)"
SEL=$R/X_COLD_SELECTION_G.json
if [ "$RISE" = 1 ]; then
  job E_eta${ETA}_E12 --x-efe 12 $T --server-lr $ETA --server-tau 1e-6 --role CALIBRATION --seed 2026 --run-dir $R/G/FA_eta${ETA}_tau1e-6_E12 || fail "E12"
  $PY -B $X/x_select2.py $R/X_COLD_SELECTION_FINAL.json 4 "${G[@]}" $R/G/FA_eta${ETA}_tau1e-6_E12 > $R/logs/select_FINAL.out 2>&1 || fail "select final"
  SEL=$R/X_COLD_SELECTION_FINAL.json
else cp $SEL $R/X_COLD_SELECTION_FINAL.json; SEL=$R/X_COLD_SELECTION_FINAL.json; fi
read ETA E < <($PY -c "import json;w=json.load(open('$SEL'))['winner'];print(format(w['eta'],'g'),format(w['efe'],'g'))")
say "FINAL eta $ETA E $E (X_COLD_SELECTION_FINAL sha $(sha256sum $SEL | cut -c1-16))"
P=()
for s in 2026 2027; do job R_s$s --x-efe $E $T --server-lr $ETA --server-tau 1e-6 --role MAIN --seed $s --run-dir $R/MAIN/X_FA_COLD_eta${ETA}_tau1e-6_E${E}_s$s & P+=($!); done
rc=0; for p in "${P[@]}"; do wait $p || rc=$((rc+1)); done; [ $rc -eq 0 ] || fail "MAIN"
say "lane done"
