#!/usr/bin/env bash
# re-run ONLY the E1 step of laneX_DP.sh pb100 (DP8 not re-run), with the same targets / env as the lane; then mark it.
set -u
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4; PY=/home/jovyan/venvs/ppsi-gpu/bin/python
X=$V/x_rees46; XR=$V/runs_x_rees46; W=$XR/PB100; D=$XR/PB100/DP; LOG=$XR/DP_pb100.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
export X_CODE=$V/code_x1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES; cd $V/code_x1
M=$(ls -d $W/MAIN/*_s2026)
export X2_RECIPE_SHA=$($PY -c "import json;print(json.load(open('$M/LAUNCH_0.json'))['p2_recipe']['sha256'])")
export X2_RUN_ID_FMT=$($PY -c "import json;print(json.load(open('$M/LAUNCH_0.json'))['run_id'].replace('_s2026','_s{seed}'))")
export X2_INIT_SHA=$($PY -c "import json;print(json.load(open('$M/LAUNCH_0.json'))['arm_spec']['init_from']['checkpoint_sha256'])")
export X2_INIT_DIGEST=$($PY -c "import json;print(json.load(open('$M/LAUNCH_0.json'))['arm_spec']['init_from']['state_digest'])")
TGT="X2_PB100=$(ls $M/ckpt/*/best.pt) X2_PB100_DP8=$(ls $D/DP8_s2026/ckpt/*/endpoint_6.0.pt)"
BASE=$($PY -c "import json;a=json.load(open('$V/runs_p2/E1/E1_AUDIT_v1/E1_RUN_RECORD.json'))['argv'];i=a.index('--targets')+1
out=[]
while i<len(a) and not a[i].startswith('--'): out.append(a[i]); i+=1
print(' '.join(x for x in out if not x.startswith('P2_WSTAR_DP')))")
say "E1: re-run (audit_e1_x X2_PB100_DP8 own-run binding fix; DP8 not re-run)"
if nice -n 10 $PY -B $X/audit_e1_x.py --device cuda --targets $BASE $TGT --out $XR/E1/pb100_v1 >> $XR/DP/logs_pb100.E1_rerun.out 2>&1; then
  touch $XR/DP/_state_pb100/E1.ok; say "E1: ok"; say "lane done"; else say "E1: rc=$? STOP"; fi
