#!/usr/bin/env bash
# server_fill_watchdog.sh — keeps both A40s >= 90 % and the CPU >= 80 % busy while the core Phase-2 lanes (RUN-1/2/3)
# always keep priority (owner instruction, DL 463). Runs unattended every 60 s; value-free (reads only utilisation and
# process lists); acts ONLY by SIGSTOP / SIGCONT of the ML-1M GPU driver group and the Wave-2 CPU driver group, or by
# re-launching the ML-1M GPU driver with --resume when it has exited before finishing. Never touches Phase 2, H or data.
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
LOG=$V/runs_ds3/queues/server_fill_watchdog.log; ST=$V/runs_ds3/queues/.watchdog; mkdir -p "$ST"
exec 9>"$ST/lock.v2"; flock -n 9 || { echo "another watchdog is running"; exit 1; }
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
p2count() { ps -eo args | grep -c -E 'code_p2r[0-9]+/v4int/(launch_fl|central_ft|c_full_small|launch_device_p2|reeval_p2)'; }
gpu_min() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' '; }
cpu_busy() { vmstat 5 2 | tail -1 | awk '{print 100-$15}'; }
grp_state() { local g=$1; [ -n "$g" ] || { echo none; return; }; ps -eo pgid,stat | awk -v g=$g '$1==g{print substr($2,1,1)}' | sort | uniq -c | awk '{if($2=="T")t+=$1; else r+=$1} END{if(t+r==0)print "none"; else if(t>0&&r==0)print "stopped"; else print "running"}'; }
ml1m_pg() { local p; p=$(pgrep -f '^bash company/plans/ops_scripts/bench/[r]un_ml1m_gpu.sh' | head -1); [ -n "$p" ] && ps -o pgid= -p $p | tr -d ' '; }
w2_pg() { local p; p=$(pgrep -f '^bash run_wave2.sh' | head -1); [ -n "$p" ] && ps -o pgid= -p $p | tr -d ' '; }
ml1m_done() { grep -q 'ML-1M GPU jobs finished' $V/runs_bench_gpu_driver.out 2>/dev/null && tail -1 $V/runs_bench_gpu_driver.out | grep -q 'finished'; }
h_running() { ps -eo args | grep -q -E '^bash .*[H]_test\.sh'; }
# per-GPU control (v3): a GPU running a latency-bound Phase-2 job (FL / central FT / device / reeval) is held for it;
# ML-1M jobs on every other GPU run (C2 = c_full_small is near its end and may share)
crit_gpus() { ps -eo args | grep -E '(code_p2r[0-9]+/)?v4int/(launch_fl|central_ft|launch_device_p2|reeval_p2|reeval|launch_device)\.py' | grep -v grep | grep -v 'runs_p2_exp' | grep -oE -- '--gpu [0-9]+' | awk '{print $2}' | sort -u | tr '
' ' '; }
ml1m_jobs_on() { ps -eo pid,args | grep -E '([v]4bench\.run run|[r]unpy\.run_module\("v4bench\.run").*--device cuda' | grep -E -- "--gpu $1( |\$)" | awk '{print $1}'; }
ml1m_shells() { [ -n "$1" ] && ps -eo pid,pgid,args | awk -v g=$1 '$2==g && $3=="bash"{print $1}'; }
held_ref=""
gl=0; cl=0; p2_ref=$(p2count); w2_hold_until=0
log "START p2=$p2_ref"
while true; do
  P2=$(p2count); G=$(gpu_min); C=$(cpu_busy); MG=$(ml1m_pg); WG=$(w2_pg); MS=$(grp_state "$MG"); WS=$(grp_state "$WG"); NOW=$(date +%s)
  # rule 0: H (the TEST sitting) owns everything — pause both fillers
  # rule 0 (v6): while H (the TEST sitting) runs, Wave 2 pauses; ML-1M is held per GPU by rule 1a on H's GPU (H steps run
  # v4int/reeval.py / launch_device.py with --gpu G); the other GPU stays usable (H_test.sh runs --allow-shared-gpu)
  if h_running && [ "$WS" = running ]; then kill -STOP -- -$WG; log "H running: Wave2 paused"; WS=stopped; fi
  # rule 1a (owner 2026-10-04: core speed first, and the server fully used): per GPU, ML-1M jobs pause where a Phase-2
  # job runs and run everywhere else; the ML-1M driver shells always run so the queue keeps moving
  if [ -n "$MG" ]; then
    CG=" $(crit_gpus)"
    for q in $(ml1m_shells "$MG"); do kill -CONT $q 2>/dev/null; done
    for gpu in 0 1; do
      case "$CG" in *" $gpu "*) for q in $(ml1m_jobs_on $gpu); do kill -STOP $q 2>/dev/null; done;;
                    *) for q in $(ml1m_jobs_on $gpu); do kill -CONT $q 2>/dev/null; done;; esac
    done
    [ "$CG" != "$held_ref" ] && { log "per-GPU: core on GPU(s)${CG:- none} -> ML-1M held there, running elsewhere"; held_ref=$CG; }
    MS=pergpu
  fi
  # rule 1: a new core stage started (Phase-2 job count rose by >= 4) -> core gets the GPUs and, during start-up, the CPU
  if [ "$P2" -ge $((p2_ref + 4)) ]; then
    [ "$WS" = running ] && [ "$C" -ge 90 ] && { kill -STOP -- -$WG; w2_hold_until=$((NOW + 1800)); log "core start-up (cpu $C%): Wave2 paused 30 min"; WS=stopped; }
    p2_ref=$P2; gl=0
  fi
  [ "$P2" -lt "$p2_ref" ] && p2_ref=$P2
  # rule 2: a GPU below 90 % for 3 checks -> fill it with ML-1M (resume, or relaunch the driver if it exited unfinished)
  if [ "${G:-100}" -lt 90 ]; then gl=$((gl + 1)); else gl=0; fi
  if [ "$gl" -ge 3 ]; then
    if [ "$MS" = none ] && ! ml1m_done; then rm -f $V/runs_bench_gpu/_gpu_state/YIELD
      (cd $V && PY=$V/company/plans/ops_scripts/bench/py_bm_spawnsafe.sh CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 YIELD_RE='[H]_test\.sh' nohup setsid bash company/plans/ops_scripts/bench/run_ml1m_gpu.sh --resume --gpus "0" --slots-per-gpu 4 >> $V/runs_bench_gpu_driver.out 2>&1 < /dev/null 9>&- &)
      log "gpu min $G% x3: ML-1M driver relaunched (--resume)"; fi
    gl=0; p2_ref=$P2
  fi
  # rule 3: CPU below 80 % for 5 checks -> resume Wave 2 (unless inside a core start-up hold)
  if [ "${C:-100}" -lt 80 ]; then cl=$((cl + 1)); else cl=0; fi
  if [ "$cl" -ge 5 ] && [ "$WS" = stopped ] && [ "$NOW" -ge "$w2_hold_until" ]; then kill -CONT -- -$WG; log "cpu $C% x5: Wave2 resumed"; cl=0; fi
  if [ "$WS" = stopped ] && [ "$w2_hold_until" -gt 0 ] && [ "$NOW" -ge "$w2_hold_until" ] && [ "${C:-100}" -lt 90 ]; then kill -CONT -- -$WG; log "hold over (cpu $C%): Wave2 resumed"; w2_hold_until=0; fi
  echo "$(date -u +%FT%TZ) p2=$P2 gpu_min=$G cpu=$C ml1m=$MS wave2=$WS" > "$ST/status"
  sleep 50
done
