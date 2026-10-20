set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
for name in LO_c64 SEL_FT_C_c64; do
  echo "$name"
  tail -15 "$V/runs_ds3/pre/smoke/$name.log" 2>/dev/null || true
done
ps -eo pid,etime,pcpu,pmem,args | grep -E 'launch_device.py' | grep -v grep || true
