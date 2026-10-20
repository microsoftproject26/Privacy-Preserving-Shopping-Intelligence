V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4; R=$V/runs_ds3/deploy_r4
for T in code_ds3r4 code_ds3r5; do
 ( export PYTHONPATH=$V/$T:$V/$T/tests_int PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
   cd $V/$T; mkdir -p $R/tmp/traj_$T
   timeout 80 /home/jovyan/venvs/ppsi-gpu/bin/python -B - <<PY 2>&1 | tail -1
import sys, pathlib, tempfile, torch
torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
sys.argv=['x']
import test_fl10_fedadam_launch as t
p=pathlib.Path(tempfile.mkdtemp(dir='$R/tmp/traj_$T'))
got={arm: t.run_tiny(p, t._tiny(arm), arm)["summary"]["server_digest"] for arm in t.F.ARMS}
print('$T', sorted(got.items()))
PY
 )
done
