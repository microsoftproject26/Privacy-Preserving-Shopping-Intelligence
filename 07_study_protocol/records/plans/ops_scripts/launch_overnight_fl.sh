set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
R=$V/runs_ds3
test -f "$R/phase1_r2/phase1_r2.log"
test ! -e "$R/overnight_fl/queue.log"
nohup setsid -f bash "$R/overnight_fl_after_phase1.sh" > "$R/overnight_fl_wrapper.log" 2>&1 < /dev/null &
echo OVERNIGHT_FL_QUEUED
