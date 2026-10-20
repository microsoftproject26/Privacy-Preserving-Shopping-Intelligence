#!/usr/bin/env bash
# SCHEMA-01: theta0 hash registration of the derived schemas L0_CLEAN_V2 / L3_NEAR_DUPLICATE on the server build
# (torch 2.13.0+cu130). Physical GPU 1 only (fail-closed ownership check first); no training; no weights written.
set -euo pipefail
V4=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
OUT=$V4/runs_g1/schema01
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
V1=/home/jovyan/shared_data/rees46_nextitem_v1
mkdir -p "$OUT"
echo "utc_start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
BUS1=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' '$1=="1"{print $2}')
if [ -z "$BUS1" ]; then echo "REFUSED: GPU 1 not enumerated"; exit 3; fi
APPS=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | grep -c "$BUS1" || true)
MEM1=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1=="1"{print $2}')
echo "gpu1 bus $BUS1 compute_apps $APPS memory_used_mib $MEM1"
if [ "$APPS" != "0" ] || [ "${MEM1:-9999}" -gt 256 ]; then echo "REFUSED: GPU 1 busy"; exit 3; fi
df -h "$V4" | tail -1
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
cd "$V4/code"
ls -la tests_fl/receipts/gpu/init.json || echo "no FL-02 init.json on the server"
"$PY" -B -m v4fs.theta0 --v1-root "$V1" --out "$OUT/THETA0_SERVER_RECEIPT_R1.json" --device cuda:0 --threads 4
echo "exit $?"
nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader || true
find "$V4/code" -name __pycache__ -path "*v4fs*" -o -name __pycache__ -path "*v4models*" | head
echo "utc_end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
