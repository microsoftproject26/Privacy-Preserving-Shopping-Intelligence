#!/bin/bash
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneWM_PF_2026.log; G=1
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_warm_r6.sh
warm3 PF_WARM_FROZEN_s2026 --arm PF_WARM_FROZEN --seed 2026
step "QUEUE_DONE rc=0"
