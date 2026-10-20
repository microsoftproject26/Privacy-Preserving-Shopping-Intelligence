set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
test ! -e "$V/code_ds3"
mkdir -p "$V/runs_ds3/deploy"
echo DEPLOY_DIR_READY
