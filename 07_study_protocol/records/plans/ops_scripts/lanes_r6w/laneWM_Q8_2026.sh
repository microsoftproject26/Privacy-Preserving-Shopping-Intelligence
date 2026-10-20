#!/bin/bash
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneWM_Q8_2026.log; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_warm_r6.sh
warm3 FA_Q8_WARM_FROZEN_s2026 --arm FA_Q8_WARM_FROZEN --seed 2026
step "QUEUE_DONE rc=0"
