set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
test ! -e "$V/code_ds3strict"
cp -a "$V/code_ds3pre" "$V/code_ds3strict"
test -f "$V/code_ds3strict/tests_train_gpu/sys04c_variant_bench.py"
echo STRICT_CODE_STAGED
