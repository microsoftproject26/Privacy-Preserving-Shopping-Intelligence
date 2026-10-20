#!/bin/bash
# alerts on any NEW LANE_STOP or QUEUE_DONE line in lane logs vs the baseline (poll 10 min, up to ~18 h)
cd "/f/Queens University/Microsoft Project/eda/.local_ai/sasrec_v1/tools"
q='cat /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/lane*.log 2>/dev/null | grep -h "LANE_STOP\|QUEUE_DONE" | sort'
base=$(MSYS_NO_PATHCONV=1 timeout 60 python lobot.py cmd "$q" 2>/dev/null); [ -z "$base" ] && { echo BASELINE_FAILED; exit 1; }
for i in $(seq 1 110); do sleep 600
  now=$(MSYS_NO_PATHCONV=1 timeout 60 python lobot.py cmd "$q" 2>/dev/null); [ -z "$now" ] && continue
  [ "$now" != "$base" ] && { date -u; echo NEW:; comm -13 <(echo "$base") <(echo "$now"); exit 0; }
done; echo WATCH_TIMEOUT
