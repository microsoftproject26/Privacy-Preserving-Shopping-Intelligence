set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
tail -50 "$V/runs_ds3/pre/smoke/PF_r10.log" 2>/dev/null || true
find "$V/runs_ds3/pre/smoke" -maxdepth 1 -type f -printf '%f\n'
