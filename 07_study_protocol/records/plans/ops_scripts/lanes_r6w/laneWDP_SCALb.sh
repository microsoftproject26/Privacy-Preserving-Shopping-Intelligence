#!/bin/bash
# continuation of laneWDP_SCAL after its m1 check looked for LAUNCH_1.json; a fresh run writes LAUNCH_0.json (DL 295)
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/laneWDP_SCALb.log; G=0
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_warm_r6.sh
SS=$SM/S_CAL_WARM_FROZEN_s2026/S_RECORD_WARM_FROZEN.json
$PY -B -c "import json,sys; s=json.load(open(sys.argv[1]))['s_record']; ok=bool(s.get('plan_digest')) and s.get('plan_digest_checked')==s.get('plan_digest'); print('M1', ok, str(s.get('plan_digest'))[:12]); sys.exit(0 if ok else 7)" $SM/FA_1024_WARM_FROZEN_s2026/LAUNCH_0.json >> $LOG 2>&1 || { step "LANE_STOP m1 plan digest check failed"; exit 7; }
step "M1_PLAN_DIGEST_OK (LAUNCH_0)"
job SMK_DP8_FROZEN_s2026 fl --arm DP_WARM_FROZEN --epsilon 8 --seed 2026 $T $I $FZ --clip-record $SS --smoke --smoke-rounds 3 --run-dir $SM/DP8_FROZEN_s2026
job S_CAL_WARM_FROZEN_s2026 fl --arm S_CAL_WARM_FROZEN --seed 2026 $T $I $FZ $W8 --run-dir $R/FL_T/S_CAL_WARM_FROZEN_s2026
cp -n $R/FL_T/S_CAL_WARM_FROZEN_s2026/S_RECORD_WARM_FROZEN.json $SREC && chmod a-w $SREC && step "REGISTERED $SREC $(sha256sum $SREC | cut -c1-16)"
step "QUEUE_DONE rc=0"
