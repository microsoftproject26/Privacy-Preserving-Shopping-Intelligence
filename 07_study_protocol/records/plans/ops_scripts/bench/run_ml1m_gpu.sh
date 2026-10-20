#!/usr/bin/env bash
# run_ml1m_gpu.sh - BM-04: the Wave-1 job graph of run_wave1.sh for ML-1M ONLY on the deterministic A40 device class
# (NVIDIA_A40-FP32-DET; Fable decision FABLE_GATE_ENDGAME_SCRIPTS_20261003 section 8; runner support BM-03).
# GPU VERSION = amendment B16 (TO BE REGISTERED): DO NOT RUN THIS SCRIPT (--launch / --resume) BEFORE B16 IS REGISTERED. Plan mode is safe.
# PLAN MODE BY DEFAULT: nothing is launched unless --launch is given. It stops BEFORE `lock` and `final` (those need Ops).
#
#   run_ml1m_gpu.sh [--plan] [--gpu 0 | --gpus "0 1"] [--max-slots N]   print the jobs + dependency order
#   run_ml1m_gpu.sh --launch [same options]                              run the DAG
#   run_ml1m_gpu.sh --resume [same options]  RESUME: re-runs ONLY failed / skipped / never-run jobs and their dependents
#                                            (finished runs are skipped by their DONE file; interrupted runs continue with `run --resume`).
#
# Differences to run_wave1.sh: CODE default $V/code_bench5b (BM-03 + BM-05; B16, DL 423); RUNS=$V/runs_bench_gpu (a FRESH root; logs in $RUNS/logs, state in
# $RUNS/_gpu_state; refused if it is runs_bench or CODE is code_bench4); dataset fixed to ml_1m; every `run` carries
# `--device cuda --gpu G --runs-root $RUNS` (G alternates over --gpus by job index: deterministic), every other subcommand
# (manifest, decide-warm, trackb-select, seed3) carries only `--runs-root $RUNS`; CUDA is NOT hidden; slots = concurrent
# jobs, default 4 per GPU, refused above 8 in total (small SASRec D64 models on a 48 GB A40; weight 1 per job, FT = 2 workers = 2 slots).
# DAG: see run_wave1.sh (manifest -> C_FULL/PRE/FA -> FA_WARM* -> decide-warm -> FT, phase 2, S_CAL -> FA_1024/DP8
# -> seed3 -> Track B -> trackb-select -> winners at seed 2027).
# YIELD: between jobs the driver polls $RUNS/_gpu_state/YIELD (touch it to stop) and any process matching $YIELD_RE (H_test.sh / Phase-2 lane);
# when set it launches nothing new, lets running jobs finish, exits 0; `--resume` continues. ONE driver only (pid lock + ps checks).
# Slots: --slots-per-gpu (default 4, max 4) per GPU and --max-slots total (default 4 x GPUs, refused above 8). Driver output is value-free.
# Every job: nice 15, write-once (a finished run is skipped, never overwritten), log in $RUNS/logs/;
# a failed job marks its dependents SKIPPED (the branch stops); other branches continue.
set -u
V=${V:-/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4}
CODE=${CODE:-$V/code_bench5b}
PY=${PY:-/home/jovyan/venvs/ppsi-gpu/bin/python}
RUNS=${GPU_RUNS:-$V/runs_bench_gpu}      # GPU_RUNS exists only so the refusal can be tested
LOGS=$RUNS/logs
STATE=$RUNS/_gpu_state
MAXSLOTS=
PERGPU=4                              # explicit slots PER GPU (a slot = one job; FT weighs 2)
GPUS="0"
YIELD_RE=${YIELD_RE:-'[H]_test\.sh|[l]aneP2[A-Za-z0-9_]*\.sh'}   # processes this driver yields to (Ops sets the exact Phase-2 lane pattern)
MODE=plan
RESUME=0
DATASETS="ml_1m"
while [ $# -gt 0 ]; do
  case $1 in
    --plan) MODE=plan;;
    --launch) MODE=launch;;
    --resume) MODE=launch; RESUME=1;;
    --datasets) DATASETS=$2; shift;;
    --gpu) GPUS=$2; shift;;
    --gpus) GPUS=$2; shift;;
    --max-slots) MAXSLOTS=$2; shift;;
    --slots-per-gpu) PERGPU=$2; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
  shift
done
read -r -a GPUARR <<< "$GPUS"
[ ${#GPUARR[@]} -ge 1 ] || { echo "refused: no GPU given" >&2; exit 2; }
for g in "${GPUARR[@]}"; do case $g in ''|*[!0-9]*) echo "refused: bad GPU id '$g'" >&2; exit 2;; esac; done
case $PERGPU in [1-4]) ;; *) echo "refused: --slots-per-gpu must be 1..4" >&2; exit 2;; esac
[ -n "$MAXSLOTS" ] || MAXSLOTS=$(( PERGPU * ${#GPUARR[@]} ))
case $MAXSLOTS in ''|*[!0-9]*) echo "refused: --max-slots must be a number" >&2; exit 2;; esac
if [ "$MAXSLOTS" -gt 8 ]; then echo "refused: at most 8 concurrent GPU job slots in total (4 per GPU)" >&2; exit 2; fi
if [ "$MAXSLOTS" -lt 2 ]; then echo "refused: at least 2 slots (FT needs 2)" >&2; exit 2; fi
[ "$DATASETS" = ml_1m ] || { echo "refused: this GPU driver is ML-1M only (got datasets '$DATASETS')" >&2; exit 2; }
[ "$RUNS" != "$V/runs_bench" ] && [ "$RUNS" != "$V/runs_bench/" ] || { echo "refused: the GPU driver never writes into runs_bench" >&2; exit 2; }
case $CODE in */code_bench5b|*/code_bench5b/) ;; *) echo "refused: CODE ($CODE) must be code_bench5b (B16: one tree for training and final; REV-BM-04)" >&2; exit 2;; esac
case $RUNS in */runs_bench_gpu|*/runs_bench_gpu/) ;; *) echo "refused: the runs root must be runs_bench_gpu (REV-BM-04)" >&2; exit 2;; esac

psl() {                            # "PID COMMAND" lines (procps on the server; the -ef fallback is for Git Bash tests)
  ps -eo pid,args 2>/dev/null || ps -ef 2>/dev/null | awk '{c=$0; sub(/^ *[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +/,"",c); print $2, c}'
}
RUNNER="nice -n 15 $PY -B -m v4bench.run"
J_GPU=(); J_NAME=(); J_W=(); J_DEPS=(); J_DONE=(); J_CMD=()
declare -A J_IDX

# job NAME WEIGHT "DEPS" DONEFILE "COMMAND"   (jobs are registered in topological order)
job() {
  local name=$1 w=$2 deps=$3 dn=$4 cmd=$5 d
  [ -z "${J_IDX[$name]:-}" ] || { echo "duplicate job $name" >&2; exit 1; }
  for d in $deps; do [ -n "${J_IDX[$d]:-}" ] || { echo "job $name: dependency $d is not defined earlier" >&2; exit 1; }; done
  [ "$w" -le "$MAXSLOTS" ] && [ "$w" -le "$PERGPU" ] || { echo "job $name: weight $w exceeds the slots" >&2; exit 1; }
  local idx=${#J_NAME[@]} gpu fl
  gpu=${GPUARR[$(( idx % ${#GPUARR[@]} ))]}                    # deterministic alternation by job index
  fl="--device cuda --gpu $gpu --runs-root $RUNS"
  case $cmd in
    win2027*) cmd="RUNFLAGS=\"$fl\" $cmd";;                     # win2027 inserts $RUNFLAGS after every `run`
    *" run --dataset "*) cmd=${cmd/" run --dataset "/" run $fl --dataset "};;
    *) cmd=${cmd/" --dataset "/" --runs-root $RUNS --dataset "};;
  esac
  J_IDX[$name]=$idx; J_GPU+=("$gpu")
  J_NAME+=("$name"); J_W+=("$w"); J_DEPS+=("$deps"); J_DONE+=("$dn"); J_CMD+=("$cmd")
}

# ---- helpers evaluated when a job RUNS (they read the decision files written earlier in the DAG) -----------------
jget() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }
star()   { jget "$RUNS/$1/FA_WARM_DECISION.json" FA_WARM_STAR; }           # FA_WARM | FA_WARM_FROZEN
suffix() { local s; s=$(star "$1"); echo "${s#FA_}"; }                      # WARM | WARM_FROZEN
cellname() { echo "${1}__B${2}_L${3}_D${4}"; }                              # run.py label of a Track-B cell
pre_dir_for_cell() {                                                        # the 6-EFE PRE of the SAME (max_len, dropout) cell
  local D=$1 ml=$2 dr=$3 seed=$4
  if [ "$ml" = 50 ] && [ "$dr" = 0.1 ]; then echo "$RUNS/$D/PRE_s$seed"; else echo "$RUNS/$D/$(cellname PRE 6 $ml $dr)_s$seed"; fi
}
winner_cell() { $PY -c "import json,sys; c=json.load(open(sys.argv[1]))['winner_cell']; print(c[0], c[1], c[2])" "$RUNS/$1/trackb/WINNER_$2.json"; }
# the seed-2027 run of a Track-B winner (a warm winner also needs the PRE of its cell at seed 2027)
win2027() {
  local D=$1 ARM=$2 realarm b ml dr
  realarm=$ARM; [ "$ARM" = STAR ] && realarm=$(star "$D")
  read -r b ml dr <<< "$(winner_cell "$D" "$realarm")"
  if [ "$b $ml $dr" = "6 50 0.1" ]; then echo "winner = the Track-A cell: its seed-2027 run is part of Track A"; return 0; fi
  local extra="--budget-efe $b --max-len $ml --dropout $dr"
  case $realarm in
    FA_WARM|FA_WARM_FROZEN)
      local pre; pre=$(pre_dir_for_cell "$D" "$ml" "$dr" 2027)
      [ -f "$pre/RESULT.json" ] || $RUNNER run $RUNFLAGS --dataset "$D" --arm PRE --seed 2027 --budget-efe 6 --max-len "$ml" --dropout "$dr" || return 1
      $RUNNER run $RUNFLAGS --dataset "$D" --arm "$realarm" --seed 2027 --max-len "$ml" --dropout "$dr" --pre-run "$pre" ;;
    FA)  $RUNNER run $RUNFLAGS --dataset "$D" --arm FA --seed 2027 --max-len "$ml" --dropout "$dr" ;;
    *)   $RUNNER run $RUNFLAGS --dataset "$D" --arm "$realarm" --seed 2027 $extra ;;
  esac
}
# B12: S_CAL runs once PER PRE SEED; FA_1024 / DP8 seed s bind their own record. The pre-B12 s2026 file is accepted.
srec() {
  local D=$1 s=$2 st f
  st=$(star "$D"); f="$RUNS/$D/dp/S_RECORD_${st}_s$s.json"
  if [ ! -f "$f" ] && [ "$s" = 2026 ] && [ -f "$RUNS/$D/dp/S_RECORD_${st}.json" ]; then f="$RUNS/$D/dp/S_RECORD_${st}.json"; fi
  echo "$f"
}
export -f jget star suffix cellname pre_dir_for_cell winner_cell win2027 srec

# ---- the DAG --------------------------------------------------------------------------------------------------------
build_dataset() {
  local D=$1 s arm x b ml dr cellruns
  local R="$RUNNER"
  local M=manifest.$D
  job $M 1 "" "$RUNS/$D/MANIFEST.json" "$R manifest --dataset $D"
  for s in 2026 2027; do
    job cfull.$D.$s 1 "$M" "$RUNS/$D/C_FULL_s$s/RESULT.json" "$R run --dataset $D --arm C_FULL --seed $s"
    job pre.$D.$s   1 "$M" "$RUNS/$D/PRE_s$s/RESULT.json"    "$R run --dataset $D --arm PRE --seed $s"
    job fa.$D.$s    1 "$M" "$RUNS/$D/FA_s$s/RESULT.json"     "$R run --dataset $D --arm FA --seed $s"
  done
  for s in 2026 2027; do
    job fawarm.$D.$s   1 "pre.$D.$s" "$RUNS/$D/FA_WARM_s$s/RESULT.json" \
        "$R run --dataset $D --arm FA_WARM --seed $s --pre-run $RUNS/$D/PRE_s$s"
    job fafrozen.$D.$s 1 "pre.$D.$s" "$RUNS/$D/FA_WARM_FROZEN_s$s/RESULT.json" \
        "$R run --dataset $D --arm FA_WARM_FROZEN --seed $s --pre-run $RUNS/$D/PRE_s$s"
  done
  local DEC=decide.$D
  job $DEC 1 "fawarm.$D.2026 fawarm.$D.2027 fafrozen.$D.2026 fafrozen.$D.2027" "$RUNS/$D/FA_WARM_DECISION.json" \
      "$R decide-warm --dataset $D --s7-runs FA_WARM_s2026 FA_WARM_s2027 --s8-runs FA_WARM_FROZEN_s2026 FA_WARM_FROZEN_s2027"
  for s in 2026 2027; do                                             # FT on FA_WARM* (2 workers x 16 threads = 2 slots)
    job ft.$D.$s 2 "$DEC" "$RUNS/$D/FT_\$(star $D)_s$s/RESULT.json" \
        "$R run --dataset $D --arm FT_\$(star $D) --seed $s --base-run $RUNS/$D/\$(star $D)_s$s --workers 2"
  done
  for x in FP PF FA_Q8; do                                           # phase 2 (warm on FA_WARM*)
    for s in 2026 2027; do
      job ph2.$x.$D.$s 1 "$DEC pre.$D.$s" "$RUNS/$D/${x}_\$(suffix $D)_s$s/RESULT.json" \
          "$R run --dataset $D --arm ${x}_\$(suffix $D) --seed $s --pre-run $RUNS/$D/PRE_s$s"
    done
  done
  for s in 2026 2027; do                                             # B12: S_CAL per PRE seed
    job scal.$D.$s 1 "$DEC pre.$D.$s" "$RUNS/$D/S_CAL_\$(suffix $D)_s$s/RESULT.json" \
        "$R run --dataset $D --arm S_CAL_\$(suffix $D) --seed $s --pre-run $RUNS/$D/PRE_s$s"
    job fa1024.$D.$s 1 "scal.$D.$s pre.$D.$s" "$RUNS/$D/FA_1024_\$(suffix $D)_s$s/RESULT.json" \
        "$R run --dataset $D --arm FA_1024_\$(suffix $D) --seed $s --pre-run $RUNS/$D/PRE_s$s --s-record \$(srec $D $s)"
    job dp8.$D.$s 1 "scal.$D.$s pre.$D.$s" "$RUNS/$D/DP8_\$(suffix $D)_s$s/RESULT.json" \
        "$R run --dataset $D --arm DP8_\$(suffix $D) --seed $s --pre-run $RUNS/$D/PRE_s$s --s-record \$(srec $D $s)"
  done
  job seed3.$D 1 "cfull.$D.2026 cfull.$D.2027 $DEC" "$RUNS/$D/SEED3_DECISION.json" \
      "$R seed3 --dataset $D --cfull-runs C_FULL_s2026 C_FULL_s2027 --warm-runs \$(star $D)_s2026 \$(star $D)_s2027"

  # ---- Track B (seed 2026): max_len {50; 200 on ML-1M} x dropout {0.1, 0.2, 0.5}; budget axis {6, 24, 96} for the central
  #      arms only (ML-1M C_FULL: {6, 24}, B10 Q4). The Track-A cell is the plain run; it is not repeated.
  local mls="50"; [ "$D" = ml_1m ] && mls="50 200"
  local cfb="6 24 96"; [ "$D" = ml_1m ] && cfb="6 24"
  local cf_runs="C_FULL_s2026" pre_runs="PRE_s2026" fa_runs="FA_s2026" st_runs="\$(star $D)_s2026"
  local cf_deps="cfull.$D.2026" pre_deps="pre.$D.2026" fa_deps="fa.$D.2026" st_deps="$DEC fawarm.$D.2026 fafrozen.$D.2026"
  for ml in $mls; do for dr in 0.1 0.2 0.5; do
    for b in 6 24 96; do
      [ "$b" = 6 ] && [ "$ml" = 50 ] && [ "$dr" = 0.1 ] && continue
      if echo " $cfb " | grep -q " $b "; then
        job tb.cfull.$D.$b.$ml.$dr 1 "$M" "$RUNS/$D/$(cellname C_FULL $b $ml $dr)_s2026/RESULT.json" \
            "$R run --dataset $D --arm C_FULL --seed 2026 --budget-efe $b --max-len $ml --dropout $dr"
        cf_runs="$cf_runs $(cellname C_FULL $b $ml $dr)_s2026"; cf_deps="$cf_deps tb.cfull.$D.$b.$ml.$dr"
      fi
      job tb.pre.$D.$b.$ml.$dr 1 "$M" "$RUNS/$D/$(cellname PRE $b $ml $dr)_s2026/RESULT.json" \
          "$R run --dataset $D --arm PRE --seed 2026 --budget-efe $b --max-len $ml --dropout $dr"
      pre_runs="$pre_runs $(cellname PRE $b $ml $dr)_s2026"; pre_deps="$pre_deps tb.pre.$D.$b.$ml.$dr"
    done
    if [ "$ml" != 50 ] || [ "$dr" != 0.1 ]; then
      job tb.fa.$D.6.$ml.$dr 1 "$M" "$RUNS/$D/$(cellname FA 6 $ml $dr)_s2026/RESULT.json" \
          "$R run --dataset $D --arm FA --seed 2026 --max-len $ml --dropout $dr"
      fa_runs="$fa_runs $(cellname FA 6 $ml $dr)_s2026"; fa_deps="$fa_deps tb.fa.$D.6.$ml.$dr"
      local prejob="tb.pre.$D.6.$ml.$dr"
      job tb.star.$D.6.$ml.$dr 1 "$DEC $prejob" "$RUNS/$D/\$(cellname \$(star $D) 6 $ml $dr)_s2026/RESULT.json" \
          "$R run --dataset $D --arm \$(star $D) --seed 2026 --max-len $ml --dropout $dr --pre-run $(pre_dir_for_cell $D $ml $dr 2026)"
      st_runs="$st_runs \$(cellname \$(star $D) 6 $ml $dr)_s2026"; st_deps="$st_deps tb.star.$D.6.$ml.$dr"
    fi
  done; done
  job tbsel.cfull.$D 1 "$cf_deps"  "$RUNS/$D/trackb/WINNER_C_FULL.json" "$R trackb-select --dataset $D --arm C_FULL --runs $cf_runs"
  job tbsel.pre.$D   1 "$pre_deps" "$RUNS/$D/trackb/WINNER_PRE.json"    "$R trackb-select --dataset $D --arm PRE --runs $pre_runs"
  job tbsel.fa.$D    1 "$fa_deps"  "$RUNS/$D/trackb/WINNER_FA.json"     "$R trackb-select --dataset $D --arm FA --runs $fa_runs"
  job tbsel.star.$D  1 "$st_deps"  "$RUNS/$D/trackb/WINNER_\$(star $D).json" \
      "$R trackb-select --dataset $D --arm \$(star $D) --runs $st_runs"
  for arm in cfull pre fa star; do
    A=$(echo $arm | tr a-z A-Z); [ "$arm" = cfull ] && A=C_FULL
    job win.$arm.$D 1 "tbsel.$arm.$D" "$RUNS/$D/.win2027_$arm.ok" "win2027 $D $A && touch $RUNS/$D/.win2027_$arm.ok"
  done
}

for D in $DATASETS; do build_dataset "$D"; done

# ---- plan mode: print the job list in dependency order ------------------------------------------------------------------
print_plan() {
  local i tw=0
  echo "# ML-1M GPU plan: ${#J_NAME[@]} jobs, datasets: $DATASETS, GPUs: $GPUS, slots: $MAXSLOTS, runs root $RUNS, nice 15"
  echo "# order = registration order (every dependency appears earlier); STOPS before lock / final (Ops)"
  for i in "${!J_NAME[@]}"; do
    printf '%-34s w=%s deps=[%s]\n    %s\n' "${J_NAME[$i]}" "${J_W[$i]}" "${J_DEPS[$i]}" "${J_CMD[$i]}"
    tw=$((tw + J_W[$i]))
  done
  echo "# total weight (job slot units): $tw"
  if [ -d "$STATE" ]; then   # read-only: plan mode works on the real state dir and never writes
    echo "# state (read-only): $(ls "$STATE"/*.ok 2>/dev/null | wc -l) ok, $(ls "$STATE"/*.fail 2>/dev/null | wc -l) fail markers; YIELD file: $([ -e "$STATE/YIELD" ] && echo present || echo absent)"
  fi
}
if [ "$MODE" = plan ]; then print_plan; exit 0; fi
[ "$MODE" = launch ] || exit 0

# ---- launch: a slot scheduler with dependency tracking ------------------------------------------------------------------
mkdir -p "$LOGS" "$STATE"
# exactly ONE ML-1M driver: the atomic pid lock below refuses a second run_ml1m_gpu.sh; ps checks: no run_wave1.sh that covers ml_1m, no CPU ml_1m `run` job outside runs_bench_gpu
OTHER=$(psl | grep -E -e '[r]un_wave1(_r4)?\.sh' | grep -v -e '--datasets' -e 'ml_1m' ; psl | grep -E -e '[r]un_wave1(_r4)?\.sh' | grep -e 'ml_1m' || true)
[ -z "$OTHER" ] || { echo "refused: a run_wave1.sh covering ml_1m is running (the CPU ML-1M branch must be dead)" >&2; exit 2; }
OTHER=$(psl | grep -E -e '-m v4bench[.]run run .*--dataset ml_1m' | grep -v runs_bench_gpu || true)
[ -z "$OTHER" ] || { echo "refused: a CPU ml_1m run job (outside runs_bench_gpu) is active" >&2; exit 2; }
LOCK=$STATE/DRIVER.lock                # pid lock: atomic mkdir, stale locks (dead pid) are taken over
if ! mkdir "$LOCK" 2>/dev/null; then
  if kill -0 "$(cat "$LOCK/pid" 2>/dev/null || echo 0)" 2>/dev/null; then echo "refused: driver lock held by pid $(cat "$LOCK/pid")" >&2; exit 2; fi
  ORPH=$(psl | grep -E -e '-m v4bench[.]run run .*--dataset ml_1m' | grep runs_bench_gpu || true)   # REV-BM-04: no takeover over orphaned GPU jobs
  [ -z "$ORPH" ] || { echo "refused: stale lock but GPU ml_1m run jobs are still active (orphans of the dead driver)" >&2; exit 2; }
  rm -rf "$LOCK"; mkdir "$LOCK" || { echo "refused: cannot take the driver lock" >&2; exit 2; }
fi
echo $$ > "$LOCK/pid"; trap '[ "$BASHPID" = "$$" ] && rm -rf "$LOCK"' EXIT      # background job subshells inherit the trap: only the driver removes the lock
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
declare -A PIDW PIDG
used() { local p t=0; for p in "${!PIDW[@]}"; do if kill -0 "$p" 2>/dev/null; then t=$((t + PIDW[$p])); else unset 'PIDW[$p]'; fi; done; echo $t; }
used_gpu() { local p t=0; for p in "${!PIDW[@]}"; do if kill -0 "$p" 2>/dev/null && [ "${PIDG[$p]}" = "$1" ]; then t=$((t + PIDW[$p])); fi; done; echo $t; }
want_yield() {                     # YIELD file, or an H step / Phase-2 GPU lane is running: launch nothing new
  [ -e "$STATE/YIELD" ] && return 0
  [ -n "$YIELD_RE" ] && psl | grep -E -e "$YIELD_RE" | grep -v grep > /dev/null && return 0
  return 1
}
dep_state() {                       # ok | fail | wait
  local d st=ok
  for d in $1; do
    if [ -f "$STATE/$d.fail" ]; then echo fail; return; fi
    [ -f "$STATE/$d.ok" ] || st=wait
  done
  echo $st
}
run_job() {                         # executed in the background
  local i=$1 name=${J_NAME[$1]}
  local done_eval; done_eval=$(eval "echo \"${J_DONE[$i]}\"")
  if [ -f "$done_eval" ] && ! grep -q '"status": "\(INCOMPLETE\|RUNNING\)' "$done_eval" 2>/dev/null; then
    echo "[$(date -u +%FT%TZ)] $name: already done ($done_eval), skipped" >> "$LOGS/gpu_$name.log"; touch "$STATE/$name.ok"; return 0
  fi
  echo "[$(date -u +%FT%TZ)] $name: start" >> "$LOGS/gpu_$name.log"
  local cmd=${J_CMD[$i]}
  [ "$RESUME" = 1 ] && cmd=${cmd//v4bench.run run /v4bench.run run --resume }      # continue an interrupted run, never restart it
  if ( cd "$CODE" && eval "$cmd" ) >> "$LOGS/gpu_$name.log" 2>&1; then
    touch "$STATE/$name.ok"; echo "[$(date -u +%FT%TZ)] $name: ok" >> "$LOGS/gpu_$name.log"
  else
    touch "$STATE/$name.fail"; echo "[$(date -u +%FT%TZ)] $name: FAILED" >> "$LOGS/gpu_$name.log"
  fi
}
rm -f "$STATE"/*.fail                # a re-launch / resume retries failed + skipped jobs; finished runs are skipped by DONE files
echo "launching ${#J_NAME[@]} jobs (log dir $LOGS)"
declare -a PENDING=("${!J_NAME[@]}")
YIELDED=0
while [ ${#PENDING[@]} -gt 0 ]; do
  if want_yield; then YIELDED=1; echo "[$(date -u +%FT%TZ)] driver: YIELD requested, launching nothing new; waiting for running jobs" >> "$LOGS/gpu_driver.log"; break; fi
  progressed=0; declare -a NEXT=()
  for i in "${PENDING[@]}"; do
    name=${J_NAME[$i]}; st=$(dep_state "${J_DEPS[$i]}")
    if [ "$st" = fail ]; then touch "$STATE/$name.fail"; echo "[$(date -u +%FT%TZ)] $name: SKIPPED (a dependency failed)" >> "$LOGS/gpu_$name.log"; progressed=1; continue; fi
    if [ "$st" = wait ] || [ $(( $(used) + J_W[$i] )) -gt "$MAXSLOTS" ] || [ $(( $(used_gpu "${J_GPU[$i]}") + J_W[$i] )) -gt "$PERGPU" ]; then NEXT+=("$i"); continue; fi
    if want_yield; then NEXT+=("$i"); continue; fi
    run_job "$i" & PIDW[$!]=${J_W[$i]}; PIDG[$!]=${J_GPU[$i]}; progressed=1
  done
  PENDING=("${NEXT[@]}")
  [ ${#PENDING[@]} -gt 0 ] && sleep "${POLL:-20}"
done
wait
if [ "$YIELDED" = 1 ]; then echo "driver YIELDED: running jobs finished, ${#PENDING[@]} jobs not launched; continue later with --resume (identical-hash resume rule)."; exit 0; fi
nfail=$(ls "$STATE"/*.fail 2>/dev/null | wc -l)
echo "ML-1M GPU jobs finished; failed or skipped: $nfail. Next (Ops): lock and final per dataset - NOT run by this script."
