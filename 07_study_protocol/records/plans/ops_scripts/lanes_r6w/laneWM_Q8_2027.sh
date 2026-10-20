#!/bin/bash
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneWM_Q8_2027.log; G=1
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_warm_r6.sh
warm3 FA_Q8_WARM_FROZEN_s2027 --arm FA_Q8_WARM_FROZEN --seed 2027
step "QUEUE_DONE rc=0"
