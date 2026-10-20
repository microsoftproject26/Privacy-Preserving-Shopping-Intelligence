set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
tail -5 "$V/runs_ds3/pre/gate.log" 2>/dev/null || true
tail -5 "$V/data_cfull/_run/final_pre.status" 2>/dev/null || true
tail -8 "$V/data_cfull/_run/tests_final_pre.log" 2>/dev/null || true
tail -4 "$V/runs_ds3/T3_A40/out/launch.status" 2>/dev/null || true
tail -8 "$V/runs_ds3/T3_A40/out/stdout.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
