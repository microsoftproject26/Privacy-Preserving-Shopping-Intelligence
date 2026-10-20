set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
E=$V/runs_g1/r5_t3_receipt
find "$E" -maxdepth 2 -type f -printf '%P\n' | head -40
sha256sum "$E/code/gpu_t3_receipt.py" "$V/code_ds3pre/tests_eval/gpu_t3_receipt.py"
