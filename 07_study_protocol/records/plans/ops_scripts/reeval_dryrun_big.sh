set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
CR=$V/code_ds3
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1
cd "$CR"
"$PY" -B v4int/reeval.py --ckpt "$V/runs_p3/SASREC/02B_SASREC_s2026/sealed/checkpoints/best.pt" \
  --expect-sha256 df35157f8ea93762ffed75048a43217d8ce1406de208b8429fd3f04af226c749 \
  --ckpt-kind central --variant SASREC_D256_B3 --recipe-id 02B_SASREC --seed 2026 --role MAIN \
  --arm-label C_BIG --selection-rule PRACTICAL_BEST --checkpoint-id 02B_SASREC_MAIN_s2026:best@3.0 \
  --gpu 0 --out-dir "$V/runs_ds3/REEVAL_A40/C_BIG_s2026_PB" --history-policy USER_HISTORY_L50 --dry-run
