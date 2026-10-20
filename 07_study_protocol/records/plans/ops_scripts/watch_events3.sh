#!/bin/bash
# exits (and prints) when the set of JOB_DONE / LANE_STOP lines across all lane logs changes vs the baseline
cd "/f/Queens University/Microsoft Project/eda/.local_ai/sasrec_v1/tools"
q='cat /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/lane*.log 2>/dev/null | grep -h "JOB_DONE\|LANE_STOP" | grep -v -e "JOB_DONE BASE4_" -e "JOB_DONE RV_" | sort'
base=$(MSYS_NO_PATHCONV=1 timeout 60 python lobot.py cmd "$q" 2>/dev/null)
[ -z "$base" ] && { echo "BASELINE_FAILED"; exit 1; }
for i in $(seq 1 110); do
  sleep 240
  now=$(MSYS_NO_PATHCONV=1 timeout 60 python lobot.py cmd "$q" 2>/dev/null)
  [ -z "$now" ] && continue
  if [ "$now" != "$base" ]; then date -u; echo "NEW EVENTS:"; comm -13 <(echo "$base") <(echo "$now"); exit 0; fi
done
echo "WATCH_TIMEOUT"
