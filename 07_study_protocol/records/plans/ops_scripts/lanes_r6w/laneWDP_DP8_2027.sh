#!/bin/bash
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneWDP_DP8_2027.log; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_warm_r6.sh
wait_file $SREC
dpmain DP8_FROZEN_s2027 --arm DP_WARM_FROZEN --epsilon 8 --seed 2027
step "QUEUE_DONE rc=0"
