set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
echo GATE
tail -12 "$V/runs_ds3/pre/gate_rest.log" 2>/dev/null || true
echo STRICT
tail -5 "$V/runs_ds3/strict_a40/strict.status" 2>/dev/null || true
tail -4 "$V/runs_ds3/strict_a40/strict.log" 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
