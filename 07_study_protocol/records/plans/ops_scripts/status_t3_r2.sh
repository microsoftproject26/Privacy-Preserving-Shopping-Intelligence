set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
E=$V/runs_ds3/T3_A40_r2
tail -5 "$E/out/launch.status" 2>/dev/null || true
tail -15 "$E/out/stdout.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
