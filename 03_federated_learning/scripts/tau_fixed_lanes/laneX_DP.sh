#!/usr/bin/env bash
# laneX_DP.sh cold|pb100 - C2 + C3 (POST_HOC_EXPLORATORY). Waits for the winner lane, then:
#   C2 DP8 redo with tau 1e-6 on the winner recipe, the registered mechanism unchanged (Poisson m 1,024 of N 131,072
#      = q 2^-7, T 384, z 0.5642 from the FL-04 receipt, epsilon 8, delta 5e-6), S re-calibrated by the registered S_CAL
#      rule (20 rounds, seed 2026, median client-delta norm) on the NEW recipe; DP8 MAIN seeds 2026 + 2027 (endpoint).
#        cold : launch_fl_x3.py --arm S_CAL -> runs_x_rees46/DP/COLD/S_CAL/S_RECORD.json; --arm DP --epsilon 8 (GPU 0)
#        pb100: launch_fl_x2.py --arm P2_WSTAR_S_CAL -> S_RECORD_P2.json; --arm P2_WSTAR_DP8 (GPU 1)
#   C3 E1 MIA audit (audit_e1_x.py; TRAIN-window rows only) of the winner (s2026 best.pt) + its DP8 (s2026 endpoint),
#      beside the four mandatory confirmatory targets (re-scored read-only), into runs_x_rees46/E1/<mode>_v1.
set -uo pipefail
MODE=$1
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; X=$V/x_rees46; XR=$V/runs_x_rees46; LOG=$XR/DP_$MODE.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
export X_CODE=$V/code_x1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES
cd $V/code_x1
if [ $MODE = cold ]; then W=$XR/COLD; else W=$XR/PB100; fi
until grep -q "lane done" $W/lane.log 2>/dev/null; do [ -f $W/_state/FAIL ] && { say "winner lane FAIL: stop"; exit 1; }; sleep 600; done
run() { local n=$1; shift; [ -f $XR/DP/_state_$MODE/$n.ok ] && return 0; mkdir -p $XR/DP/_state_$MODE; say "$n: start"
  if nice -n 10 "$@" >> $XR/DP/logs_$MODE.$n.out 2>&1; then touch $XR/DP/_state_$MODE/$n.ok; say "$n: ok"; else say "$n: rc=$? STOP"; exit 1; fi; }
if [ $MODE = cold ]; then
  ETA=$($PY -c "import json;print(format(json.load(open('$W/X_COLD_SELECTION_FINAL.json'))['winner']['eta'],'g'))")
  T="--gpu 0 --allow-shared-gpu --history-policy USER_HISTORY_L50 --server-opt fedadam --server-tau 1e-6 --server-lr $ETA --client-opt sgd_m --client-lr 0.1 --lr-shape fl_const --lr-mult 1"
  D=$XR/DP/COLD
  run S_CAL $PY -B $X/launch_fl_x3.py --arm S_CAL --role MAIN --seed 2026 $T --run-dir $D/S_CAL
  for s in 2026 2027; do run DP8_s$s $PY -B $X/launch_fl_x3.py --arm DP --epsilon 8 --clip-record $D/S_CAL/S_RECORD.json --role MAIN --seed $s $T --run-dir $D/DP8_s$s & done; wait
  M=$(ls -d $W/MAIN/*_s2026); TGT="X_FA_COLD=$(ls $M/ckpt/*/best.pt) X_FA_COLD_DP8=$(ls $D/DP8_s2026/ckpt/*/endpoint_6.0.pt)"
else
  WR=$($PY -c "import json;print(json.load(open('$W/X2_SELECTION_FINAL.json'))['winner']['run_dir'])")
  RF=$($PY -c "import json;print(json.load(open('$WR/LAUNCH_0.json'))['p2_recipe']['path'])")
  INIT=$V/runs_exp_pb100/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_PB100K_s2026/PRETRAIN_FOR_FL_INIT.json
  C="--x-tau 1e-6 --gpu 1 --allow-shared-gpu --history-policy USER_HISTORY_L50 --init-from $INIT --p2-recipe $RF"
  D=$XR/PB100/DP
  run S_CAL $PY -B $X/launch_fl_x2.py --arm P2_WSTAR_S_CAL --seed 2026 $C --run-dir $D/S_CAL
  for s in 2026 2027; do run DP8_s$s $PY -B $X/launch_fl_x2.py --arm P2_WSTAR_DP8 --seed $s $C --clip-record $D/S_CAL/S_RECORD_P2.json --run-dir $D/DP8_s$s & done; wait
  M=$(ls -d $W/MAIN/*_s2026)
  export X2_RECIPE_SHA=$($PY -c "import json;print(json.load(open('$M/LAUNCH_0.json'))['p2_recipe']['sha256'])")
  export X2_RUN_ID_FMT=$($PY -c "import json;print(json.load(open('$M/LAUNCH_0.json'))['run_id'].replace('_s2026','_s{seed}'))")
  export X2_INIT_SHA=$($PY -c "import json;print(json.load(open('$M/LAUNCH_0.json'))['arm_spec']['init_from']['checkpoint_sha256'])")
  export X2_INIT_DIGEST=$($PY -c "import json;print(json.load(open('$M/LAUNCH_0.json'))['arm_spec']['init_from']['state_digest'])")
  TGT="X2_PB100=$(ls $M/ckpt/*/best.pt) X2_PB100_DP8=$(ls $D/DP8_s2026/ckpt/*/endpoint_6.0.pt)"
fi
for f in DP8_s2026 DP8_s2027; do [ -f $XR/DP/_state_$MODE/$f.ok ] || { say "$f missing: STOP"; exit 1; }; done
BASE=$($PY -c "import json;a=json.load(open('$V/runs_p2/E1/E1_AUDIT_v1/E1_RUN_RECORD.json'))['argv'];i=a.index('--targets')+1
out=[]
while i<len(a) and not a[i].startswith('--'): out.append(a[i]); i+=1
print(' '.join(x for x in out if not x.startswith('P2_WSTAR_DP')))")
run E1 $PY -B $X/audit_e1_x.py --device cuda --targets $BASE $TGT --out $XR/E1/${MODE}_v1
say "lane done"
