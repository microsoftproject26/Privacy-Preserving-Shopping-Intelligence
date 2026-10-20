#!/usr/bin/env bash
# after the E12 resume (x_execute_fix): mark it ok if RUN_SUMMARY DONE_MAIN, clear the lane FAIL, restart laneX_COLD.sh
# (it skips every .ok job) and laneX_DP.sh cold (it waits for "lane done"). M smokes: done manually at 50 rounds (22:00Z).
set -u
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4; R=$V/runs_x_rees46/COLD; D=$R/G/FA_eta0.001_tau1e-6_E12
say() { echo "[$(date -u +%FT%TZ)] $*" >> $R/lane.log; }
while pgrep -f '^/home/jovyan/venvs/ppsi-gpu/bin/python -B /home/jovyan/.*launch_fl_x3.py .*FA_eta0.001_tau1e-6_E12 --resume' > /dev/null; do sleep 30; done
if grep -q DONE_MAIN $D/STATUS.json 2>/dev/null && [ -f $D/RUN_SUMMARY.json ]; then
  touch $R/_state/E_eta0.001_E12.ok $R/_state/M_tau1e-3.ok $R/_state/M_tau1e-6.ok
  mv $R/_state/FAIL $R/_state/FAIL.E12_6.0_eval_check_20261008 2>/dev/null
  say "E_eta0.001_E12: salvaged by --resume with x_execute_fix (no retraining; post-run step only); lane restarted"
  setsid nohup bash $V/x_rees46/laneX_COLD.sh >> $R/lane_nohup.out 2>&1 < /dev/null &
  sleep 5
  setsid nohup bash $V/x_rees46/laneX_DP.sh cold >> $V/runs_x_rees46/DP/nohup_cold.out 2>&1 < /dev/null &
else say "E_eta0.001_E12 resume did NOT reach DONE_MAIN: manual decision (see logs/E_eta0.001_E12.resume.out)"; fi
