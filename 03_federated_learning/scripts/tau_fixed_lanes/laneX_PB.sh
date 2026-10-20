#!/usr/bin/env bash
# laneX_PB.sh - X2: the strongest FL on top of theta_pre(100k) (PB100), POST_HOC_EXPLORATORY, GPU 1 only, nice 10,
# 4 CPU threads per job. Selection on CALIB_VALIDATION by the registered P2 rule (x_select2.py); NO TEST role.
#   S1 CALIB 6 EFE, passes 1, tau 1e-6: eta {3e-6,1e-5,3e-5,1e-4} x item-LR mult {0.1,1} x client LR {0.1,0.3} (16)
#      + 1 control at tau 1e-3 (eta 1e-3, c 0.03, ilr 1 = the P2 rendering of the S7 recipe of FA_WARM_PB100K)
#   S2 CALIB passes 2 for the top-2 S1 cells   ||   S3 CALIB 12 and 24 EFE for the top-3 S1 cells   (concurrent)
#   S4 final selection over S1-S3 (tau 1e-6), then MAIN seeds 2026 + 2027 of the winner (its EFE / passes)
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; CODE=$V/code_x1; X=$V/x_rees46; R=$V/runs_x_rees46/PB100; ST=$R/_state
G=1; mkdir -p $ST $R/logs $R/recipes; LOG=$R/lane.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
fail() { touch $ST/FAIL; say "FAIL: $*"; exit 1; }
[ -f $ST/FAIL ] && { say "FAIL marker present: refuse"; exit 2; }
export X_CODE=$CODE PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SLOTS=${SLOTS:-12}      # GPU 1 holds ~14 runs of ~3.1 GB; 17 at once OOMed twice (MEASURED 19:49Z)
unset CUDA_VISIBLE_DEVICES
INIT=$V/runs_exp_pb100/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_PB100K_s2026/PRETRAIN_FOR_FL_INIT.json
C="--gpu $G --allow-shared-gpu --history-policy USER_HISTORY_L50 --init-from $INIT"
cd $CODE
recipe() {   # recipe <eta> <c> <ilr> -> path
  local f=$R/recipes/eta$1_c$2_ilr$3.json
  [ -f $f ] || $PY -B -c "
import json,sys;sys.path.insert(0,'.');from v4int import launch_fl as F
F.P2_CLIENT_LRS=dict(F.P2_CLIENT_LRS,x2=(0.3,));F.P2_ITEM_LR_MULTS=tuple(F.P2_ITEM_LR_MULTS)+(1.0,)
o=F.p2_recipe_file_obj({'server_opt':'fedadam','server_lr':$1,'client_lr':$2,'items':'unfrozen','item_lr_mult':$3})
open('$f','w').write(json.dumps(o,sort_keys=True,indent=1))" || return 1
  echo $f; }
# job <name> <tau> <efe> <passes> <role> <seed> <recipe> <run_dir>
job() { local n=$1; [ -f $ST/$n.ok ] && return 0; say "$n: start"
  if nice -n 10 $PY -B $X/launch_fl_x2.py --x-tau $2 --x-efe $3 --x-passes $4 --arm P2_W --role $5 --seed $6 $C \
       --p2-recipe $7 --run-dir $8 >> $R/logs/$n.out 2>&1; then touch $ST/$n.ok; say "$n: ok"; else say "$n: rc=$?"; return 1; fi; }
waitall() { local rc=0; for p in "$@"; do wait $p || rc=$((rc+1)); done; return $rc; }
slot() { while [ $(jobs -rp | wc -l) -ge $SLOTS ]; do wait -n; done; }

# ---------------- S1 ----------------
P=()
for e in 3e-06 1e-05 3e-05 0.0001; do for i in 0.1 1; do for c in 0.1 0.3; do
  rf=$(recipe $e $c $i) || fail "recipe $e $c $i"
  t=eta${e}_c${c}_ilr${i}_tau1e-6_E6_P1
  slot; job S1_$t 1e-6 6 1 CALIBRATION 2026 $rf $R/S1/$t & P+=($!)
done; done; done
rf=$(recipe 0.001 0.03 1) || fail "control recipe"
slot; job S1_CONTROL_tau1e-3 1e-3 6 1 CALIBRATION 2026 $rf $R/S1/eta0.001_c0.03_ilr1_tau1e-3_E6_P1 & P+=($!)
waitall "${P[@]}"; say "S1 done (failed jobs: $?)"
$PY -B $X/x_select2.py $R/X2_SELECTION_S1.json 3 $R/S1/* > $R/logs/select_S1.out 2>&1 || fail "select S1"
mapfile -t TOP < <(grep '^TOP' $R/logs/select_S1.out | cut -d' ' -f2)
[ ${#TOP[@]} -ge 3 ] || fail "fewer than 3 ranked S1 cells"
say "S1 top-3 (CALIB): $(for t in "${TOP[@]}"; do basename $t; done | tr '\n' ' ')"
rec_of() { $PY -c "import json;print(json.load(open('$1/LAUNCH_0.json'))['p2_recipe']['path'])"; }
tag_of() { basename $1 | sed 's/_E6_P1$//'; }
# ---------------- S2 || S3 ----------------
P=()
for t in "${TOP[@]:0:2}"; do b=$(tag_of $t); job S2_${b}_P2 1e-6 6 2 CALIBRATION 2026 $(rec_of $t) $R/S2/${b}_E6_P2 & P+=($!); done
for t in "${TOP[@]:0:3}"; do b=$(tag_of $t)
  for E in 12 24; do job S3_${b}_E$E 1e-6 $E 1 CALIBRATION 2026 $(rec_of $t) $R/S3/${b}_E${E}_P1 & P+=($!); done; done
waitall "${P[@]}"; say "S2 + S3 done (failed jobs: $?)"
# ---------------- S4 ----------------
$PY -B $X/x_select2.py $R/X2_SELECTION_FINAL.json 5 $R/S1/* $R/S2/* $R/S3/* > $R/logs/select_FINAL.out 2>&1 || fail "select final"
W=$(grep '^WINNER' $R/logs/select_FINAL.out | cut -d' ' -f2)
read E PS < <($PY -c "import json;w=json.load(open('$R/X2_SELECTION_FINAL.json'))['winner'];print(format(w['efe'],'g'),w['passes'])")
say "FINAL winner $(basename $W) (E $E, passes $PS; X2_SELECTION_FINAL sha $(sha256sum $R/X2_SELECTION_FINAL.json | cut -c1-16))"
b=$(basename $W | sed 's/_E[0-9]*_P[0-9]$//'); RF=$(rec_of $W); P=()
for s in 2026 2027; do job MAIN_s$s 1e-6 $E $PS MAIN $s $RF $R/MAIN/X2_${b}_E${E}_P${PS}_s$s & P+=($!); done
waitall "${P[@]}" || fail "MAIN runs"
say "lane done"
