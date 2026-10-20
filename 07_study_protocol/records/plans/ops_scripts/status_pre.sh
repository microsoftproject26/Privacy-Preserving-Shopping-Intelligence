#!/usr/bin/env bash
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
R=$V/runs_ds3/pre
date -u
if test -f "$R/gate.log"; then tail -n 12 "$R/gate.log"; else echo NO_GATE_LOG; fi
if test -f "$R/gate_pre.nohup"; then tail -n 8 "$R/gate_pre.nohup"; fi
if test -f "$R/smoke/FA_r20.log"; then tail -n 12 "$R/smoke/FA_r20.log"; fi
if test -f "$V/data_cfull/_run/build_pre.status"; then tail -n 4 "$V/data_cfull/_run/build_pre.status"; fi
if test -f "$V/data_cfull/_run/build_pre.log"; then tail -n 5 "$V/data_cfull/_run/build_pre.log"; fi
ps -eo pid,etime,pcpu,pmem,comm | head -n 18
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
