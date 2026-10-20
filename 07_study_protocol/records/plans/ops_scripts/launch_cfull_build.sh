#!/usr/bin/env bash
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
R=$V/runs_ds3/cfull
bash -n "$R/build_cfull_manifest_pre.sh"
test ! -e "$V/data_cfull/_run/build_pre.status"
(cd "$R" && nohup setsid bash build_cfull_manifest_pre.sh > build_pre.nohup 2>&1 < /dev/null &)
echo "$(date -u +%FT%TZ) DATA08 CPU manifest build launched"
