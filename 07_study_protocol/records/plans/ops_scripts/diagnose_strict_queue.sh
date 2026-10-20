set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
cat "$V/runs_ds3/strict_a40/strict.status" 2>/dev/null || true
tail -25 "$V/runs_ds3/strict_a40_wrapper.log" 2>/dev/null || true
tail -15 "$V/runs_ds3/strict_a40/strict.log" 2>/dev/null || true
ps -eo pid,etime,pcpu,pmem,args | grep -E 'strict_a40_queue|sys04c_variant_bench|phase1_after_strict' | grep -v grep || true
