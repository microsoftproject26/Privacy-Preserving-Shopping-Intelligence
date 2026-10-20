#!/bin/bash
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneWM_FP_2027.log; G=1
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_warm_r6.sh
warm3 FP_WARM_FROZEN_s2027 --arm FP_WARM_FROZEN --seed 2027
step "QUEUE_DONE rc=0"
