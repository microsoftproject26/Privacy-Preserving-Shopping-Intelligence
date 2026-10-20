set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
R=$V/runs_ds3
bash -n "$R/overnight_fl_after_phase1.sh"
echo STRICT
tail -4 "$R/strict_a40/strict.status" 2>/dev/null || true
echo PHASE1
tail -4 "$R/phase1_r2/phase1_r2.log" 2>/dev/null || true
echo OVERNIGHT
tail -4 "$R/overnight_fl/queue.log" 2>/dev/null || true
echo PROCESSES
ps -eo pid,etime,args | grep -E 'strict_a40_run_r2|phase1_after_strict_r2|overnight_fl_after_phase1|sys04c_variant_bench' | grep -v grep || true
