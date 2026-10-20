#!/bin/bash
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneWDP_F1024_2026.log; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_warm_r6.sh
wait_file $SREC
dpmain FA_1024_WARM_FROZEN_s2026 --arm FA_1024_WARM_FROZEN --seed 2026
step "QUEUE_DONE rc=0"
