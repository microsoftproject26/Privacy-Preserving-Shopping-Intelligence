set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
CR=$V/code_ds3
R=$V/runs_ds3/deploy
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
mkdir -p "$R/tmp"
export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="" TMPDIR=$R/tmp
cd "$CR"
"$PY" -B -m pytest tests_eval/test_device_pin.py tests_int/test_int06_reeval.py -q -p no:cacheprovider --basetemp "$R/tmp/pytest_int06" > "$R/targeted_tests.log" 2>&1
tail -4 "$R/targeted_tests.log"
