#!/bin/bash
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneWDP_DP1_2026.log; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_warm_r6.sh
wait_file $SREC
dpmain DP1_WARM_FROZEN_s2026 --arm DP_WARM_FROZEN --epsilon 1 --seed 2026
step "QUEUE_DONE rc=0"
