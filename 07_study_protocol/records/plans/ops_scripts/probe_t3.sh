set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
find "$V/runs_g1" -maxdepth 4 \( -name T3_GPU_RECEIPT.json -o -name BUNDLE_MANIFEST.json -o -name gpu_t3_receipt.py \) -print
find "$V/runs_ds3" -maxdepth 4 -name T3_GPU_RECEIPT.json -print
ls -la "$V/runs_g1/eval_receipt/code" | head
ls -la "$V/code_ds3pre/tests_eval/receipts" | head
