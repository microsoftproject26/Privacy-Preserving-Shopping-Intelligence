set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OLD=$V/runs_g1/r5_t3_receipt
NEW=$V/runs_ds3/T3_A40
test ! -e "$NEW/code"
mkdir -p "$NEW/code" "$NEW/contracts" "$NEW/out"
cp -a "$OLD/code/." "$NEW/code/"
cp -a "$OLD/contracts/PREG1_LOCK_VALUES.json" "$NEW/contracts/"
echo T3_A40_STAGED
