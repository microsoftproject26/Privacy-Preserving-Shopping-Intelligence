set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
tail -12 "$V/runs_ds3/pre/gate_rest.log" 2>/dev/null || true
for f in "$V/runs_ds3/pre/smoke"/*.log; do
  test -f "$f" || continue
  printf '%s ' "$(basename "$f")"
  tail -1 "$f"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
