#!/bin/bash
# Ops (2026-09-29, owner: "100 % utilisation, finish fast"): replace the WAITING chained lanes 1–3 (FA→FA_Q8, FA→FA_Q8, FP→FP)
# with 6 single-job lanes spread over both A40s so all six FL runs train concurrently. Refuses if any of lanes 1–3 started a job.
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
Q=$V/runs_ds3/queues; C=$Q/_common_code_ds3r2.sh
for k in lane1 lane2 lane3; do
  if grep -q JOB_START "$Q/$k.log" 2>/dev/null; then echo "REFUSED: $k already started a job"; exit 3; fi
done
for k in lane1 lane2 lane3; do
  for p in $(ps -eo pid,args | awk -v s="queues/$k.sh" '$2=="bash" && index($3,s)>0 {print $1}'); do kill "$p"; done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUPERSEDED_BY_SPLIT (was waiting for LR; no job started)" >> "$Q/$k.log"
done
LR='wait_grep $Q/lane0.log "JOB_DONE LR_SELECT rc=0"'
mk() {  # mk <lane> <gpu> <arm> <seed>
  local f=$Q/$1.sh
  [ -e "$Q/$1.log" ] && { echo "REFUSED $1 exists"; exit 4; }
  { echo '#!/bin/bash'; echo "LOG=$Q/$1.log; G=$2"; echo "source $C"; echo "$LR";
    echo "job ${3}_s$4 fl --arm $3 --seed $4 --run-dir \$R/FL/${3}_s$4 --lr-selection \$SEL"; echo 'step "QUEUE_DONE rc=0"'; } > "$f"
  bash -n "$f" || exit 7
}
mk laneF1 0 FA 2026; mk laneF2 0 FA 2027; mk laneF3 0 FP 2026
mk laneF4 1 FA_Q8 2026; mk laneF5 1 FA_Q8 2027; mk laneF6 1 FP 2027
for k in laneF1 laneF2 laneF3 laneF4 laneF5 laneF6; do (nohup setsid bash "$Q/$k.sh" > "$Q/$k.nohup" 2>&1 < /dev/null &); done
sleep 3; date -u; ps -eo args | grep -c "queues/lane"
