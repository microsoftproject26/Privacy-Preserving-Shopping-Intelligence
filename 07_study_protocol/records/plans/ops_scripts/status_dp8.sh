set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
tail -12 "$V/runs_ds3/pre/smoke/DP8_r3.log" 2>/dev/null || true
ps -eo pid,etime,pcpu,pmem,args | grep -E 'launch_fl.py|launch_device.py|sys04c_variant_bench.py' | grep -v grep || true
