#!/bin/bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
test ! -e "$V/runs_ds3/strict_a40/strict.status"
nohup setsid -f bash "$V/runs_ds3/strict_a40_queue.sh" > "$V/runs_ds3/strict_a40_wrapper.log" 2>&1 < /dev/null
echo STRICT_QUEUED
