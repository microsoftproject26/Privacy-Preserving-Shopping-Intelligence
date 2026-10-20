#!/usr/bin/env bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
R=$V/runs_ds3/pre
test -d "$V/code_ds3pre"
test -f "$R/gate_pre.sh"
bash -n "$R/gate_pre.sh"
test ! -e "$R/gate.log"
uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==0 {print $2}')
test -n "$uuid"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
  echo 'REFUSED: GPU 0 already has a compute process' >&2
  exit 4
fi
echo "$(date -u +%FT%TZ) GPU 0 preflight measurement starting"
(cd "$R" && nohup setsid bash gate_pre.sh 0 > gate_pre.nohup 2>&1 < /dev/null &)
sleep 2
tail -n 3 "$R/gate.log"
