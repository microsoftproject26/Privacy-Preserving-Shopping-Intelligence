#!/usr/bin/env bash
# run_wave1.sh - BM-02 Wave-1 orchestration (PREG1_AMENDMENT_07 B1-B10). PLAN MODE BY DEFAULT: nothing is launched
# unless --launch is given. It stops BEFORE `lock` and `final` (those need Ops).
#
#   run_wave1.sh [--plan] [--datasets "s3_beauty s3_sports s3_toys ml_1m"] [--max-slots 6]   print the jobs + dependency order
#   run_wave1.sh --launch [same options]                                                    run the DAG
#   run_wave1.sh --resume [same options]     RESUME: launch that re-runs ONLY failed / skipped / never-run jobs and their
#                                            dependents (finished runs are skipped by their DONE file and refused by the runner;
#                                            interrupted runs continue with `run --resume`). Never touches a finished run.
#
# DAG per dataset D (seeds 2026 / 2027 unless stated; STAR = the FA_WARM* variant decided by decide-warm):
#   manifest -> C_FULL x2, PRE x2, FA x2 ; PRE s -> FA_WARM s, FA_WARM_FROZEN s
#   -> decide-warm -> FT_STAR x2 ; phase 2: FP / PF / FA_Q8 (STAR) x2
#   -> S_CAL (STAR, seed 2026) -> FA_1024 x2, DP8 x2 (STAR) -> seed3 check
#   -> Track B grid (seed 2026; budget axis for C_FULL / PRE only) -> trackb-select -> winners at seed 2027.
# Resources: one slot = one job at 16 threads (the CPU-FP32 pin); <= 6 slots = <= 96 threads; FT uses 2 workers = 2 slots.
# Every job: nice 15, CUDA hidden, write-once (a finished run is skipped, never overwritten), log in runs_bench/logs/;
# a failed job marks its dependents SKIPPED (the branch stops); other branches continue.
set -u
V=${V:-/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4}
CODE=${CODE:-$V/code_bench4}
PY=${PY:-/home/jovyan/venvs/ppsi-gpu/bin/python}
RUNS=$V/runs_bench
LOGS=$RUNS/logs
STATE=$RUNS/_wave1_state
MAXSLOTS=6
MODE=plan
RESUME=0
DATASETS="s3_beauty s3_sports s3_toys ml_1m"
while [ $# -gt 0 ]; do
  case $1 in
    --plan) MODE=plan;;
    --launch) MODE=launch;;
    --resume) MODE=launch; RESUME=1;;
    --datasets) DATASETS=$2; shift;;
    --max-slots) MAXSLOTS=$2; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
  shift
done
if [ "$MAXSLOTS" -gt 6 ]; then echo "refused: at most 6 slots x 16 threads = 96 threads" >&2; exit 2; fi

RUNNER="nice -n 15 $PY -B -m v4bench.run"
J_NAME=(); J_W=(); J_DEPS=(); J_DONE=(); J_CMD=()
declare -A J_IDX

# job NAME WEIGHT "DEPS" DONEFILE "COMMAND"   (jobs are registered in topological order)
job() {
  local name=$1 w=$2 deps=$3 dn=$4 cmd=$5 d
  [ -z "${J_IDX[$name]:-}" ] || { echo "duplicate job $name" >&2; exit 1; }
  for d in $deps; do [ -n "${J_IDX[$d]:-}" ] || { echo "job $name: dependency $d is not defined earlier" >&2; exit 1; }; done
  [ "$w" -le "$MAXSLOTS" ] || { echo "job $name: weight $w exceeds the slots" >&2; exit 1; }
  J_IDX[$name]=${#J_NAME[@]}
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
      [ -f "$pre/RESULT.json" ] || $RUNNER run --dataset "$D" --arm PRE --seed 2027 --budget-efe 6 --max-len "$ml" --dropout "$dr" || return 1
      $RUNNER run --dataset "$D" --arm "$realarm" --seed 2027 --max-len "$ml" --dropout "$dr" --pre-run "$pre" ;;
    FA)  $RUNNER run --dataset "$D" --arm FA --seed 2027 --max-len "$ml" --dropout "$dr" ;;
    *)   $RUNNER run --dataset "$D" --arm "$realarm" --seed 2027 $extra ;;
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
  echo "# Wave-1 plan: ${#J_NAME[@]} jobs, datasets: $DATASETS, slots: $MAXSLOTS x 16 threads (<= $((MAXSLOTS * 16)) threads), nice 15"
  echo "# order = registration order (every dependency appears earlier); STOPS before lock / final (Ops)"
  for i in "${!J_NAME[@]}"; do
    printf '%-34s w=%s deps=[%s]\n    %s\n' "${J_NAME[$i]}" "${J_W[$i]}" "${J_DEPS[$i]}" "${J_CMD[$i]}"
    tw=$((tw + J_W[$i]))
  done
  echo "# total weight (16-thread job units): $tw"
}
if [ "$MODE" = plan ]; then print_plan; exit 0; fi
[ "$MODE" = launch ] || exit 0

# ---- launch: a slot scheduler with dependency tracking ------------------------------------------------------------------
mkdir -p "$LOGS" "$STATE"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
declare -A PIDW
used() { local p t=0; for p in "${!PIDW[@]}"; do if kill -0 "$p" 2>/dev/null; then t=$((t + PIDW[$p])); else unset 'PIDW[$p]'; fi; done; echo $t; }
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
    echo "[$(date -u +%FT%TZ)] $name: already done ($done_eval), skipped" >> "$LOGS/wave1_$name.log"; touch "$STATE/$name.ok"; return 0
  fi
  echo "[$(date -u +%FT%TZ)] $name: start" >> "$LOGS/wave1_$name.log"
  local cmd=${J_CMD[$i]}
  [ "$RESUME" = 1 ] && cmd=${cmd//v4bench.run run /v4bench.run run --resume }      # continue an interrupted run, never restart it
  if ( cd "$CODE" && eval "$cmd" ) >> "$LOGS/wave1_$name.log" 2>&1; then
    touch "$STATE/$name.ok"; echo "[$(date -u +%FT%TZ)] $name: ok" >> "$LOGS/wave1_$name.log"
  else
    touch "$STATE/$name.fail"; echo "[$(date -u +%FT%TZ)] $name: FAILED" >> "$LOGS/wave1_$name.log"
  fi
}
rm -f "$STATE"/*.fail                # a re-launch / resume retries failed + skipped jobs; finished runs are skipped by DONE files
echo "launching ${#J_NAME[@]} jobs (log dir $LOGS)"
declare -a PENDING=("${!J_NAME[@]}")
while [ ${#PENDING[@]} -gt 0 ]; do
  progressed=0; declare -a NEXT=()
  for i in "${PENDING[@]}"; do
    name=${J_NAME[$i]}; st=$(dep_state "${J_DEPS[$i]}")
    if [ "$st" = fail ]; then touch "$STATE/$name.fail"; echo "[$(date -u +%FT%TZ)] $name: SKIPPED (a dependency failed)" >> "$LOGS/wave1_$name.log"; progressed=1; continue; fi
    if [ "$st" = wait ] || [ $(( $(used) + J_W[$i] )) -gt "$MAXSLOTS" ]; then NEXT+=("$i"); continue; fi
    run_job "$i" & PIDW[$!]=${J_W[$i]}; progressed=1
  done
  PENDING=("${NEXT[@]}")
  [ ${#PENDING[@]} -gt 0 ] && sleep "${POLL:-20}"
done
wait
nfail=$(ls "$STATE"/*.fail 2>/dev/null | wc -l)
echo "wave 1 jobs finished; failed or skipped: $nfail. Next (Ops): lock and final per dataset - NOT run by this script."
