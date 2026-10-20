#!/usr/bin/env bash
# laneP2_RUN1.sh - PPSI V4 Phase 2, RUN-1 driver (P2-RUN1-LANE). Spec: contracts/PREG2.md (REGISTERED) 4.2, 4.3, 5.2, 5.9;
# PHASE2_IMPLEMENTATION_PLAN RUN-1; handoffs P2-01_02 (P2_W, P2-T3), P2-05 (C1 / C2). The file name MUST stay laneP2_RUN1.sh:
# the ML-1M GPU driver (run_ml1m_gpu.sh) yields to any process matching laneP2*.sh. Phase 2 has GPU priority (PREG2 S7).
#
#   laneP2_RUN1.sh [--plan] [options]      PLAN MODE (default): prints every job with its exact command, GPU and output dir; writes nothing
#   laneP2_RUN1.sh --launch [options]      run the jobs (resume semantics below)
# options: --gpus "0 1" (default "0 1")   --slots-per-gpu N (1..16, default 2; DL 459)   --c2-early (C2 right after P2-T3, for makespan)
#          --stages "p2t3 wa c1 c2" (default all four)
#
# Jobs, in order: (0) P2-T3 check (smoke of the incumbent recipe + tools/p2_t3_check.py -> receipt), (1) Stage WA on CALIB: 18 core + 2 ext
# + the incumbent cell (the incumbent runs ONLY if the receipt verdict is not BIT_IDENTICAL: p2_select's reuse rule), (2) the 8 C1 CALIB
# screen runs (4 peak LRs x FULL / FROZEN), (3) C2 on MAIN (1 run, owner O3 = YES, DL 427). It does NOT run: the anchor (P2-07), the
# selections (P2-06, Ops), edge extensions (5.7), WB / WC, any MAIN run other than C2.
# Run dirs live under the FRESH root $V/runs_p2/ (never runs_ds3, never runs_bench*); code tree $V/code_p2r1 (read-only, refused otherwise).
# Semantics: a finished run (RUN_SUMMARY status DONE_MAIN / DONE_CONTINUATION and STATUS not RUNNING / FAILED) is skipped; a started but
# unfinished run is continued with the launcher's own --resume; a failed job never stops the others (isolation); recipe files are generated
# deterministically from the registered grid into $RUNS/recipes (write-once; an existing different file is a refusal); markers in
# $RUNS/_lane_state are write-once (old failure markers are moved aside, never deleted); logs in $RUNS/logs; one lane at a time (pid lock).
# S8: the lane opens no sealed/ file and prints no outcome number: it reports job names, GPU, return codes and seconds only.
# Never launches while an H step (H_test.sh) runs; stops launching new jobs if one appears (exit 3).
set -u
export LC_ALL=C
V=${V:-/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4}
CODE=${CODE:-$V/code_p2r1}
PY=${PY:-/home/jovyan/venvs/ppsi-gpu/bin/python}
DATA=${V4_DATASET_ROOT:-/home/jovyan/shared_data/rees46_nextitem_v1}
RUNS=${P2_RUNS:-$V/runs_p2}             # P2_RUNS exists only so the refusals can be tested
INIT=${P2_INIT:-$V/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026/PRETRAIN_FOR_FL_INIT.json}
S8TC_ROOT=${S8TC_ROOT:-$V/runs_ds3/S7S8_T}   # READ ONLY (the S8Tc run the P2-T3 check compares to)
HP=USER_HISTORY_L50
LAUNCHER_ARGS="--family SASREC --phase MAIN --recipe-id SMALL_SASREC_D64_B2 --history-policy $HP"
RECIPES=$RUNS/recipes; LOGS=$RUNS/logs; STATE=$RUNS/_lane_state; DRY=$RUNS/dry
MODE=plan; GPUS="0 1"; PERGPU=2; C2EARLY=0; STAGES="p2t3 wa c1 c2"
while [ $# -gt 0 ]; do
  case $1 in
    --plan) MODE=plan;;
    --launch) MODE=launch;;
    --gpus) GPUS=$2; shift;;
    --gpu) GPUS=$2; shift;;
    --slots-per-gpu) PERGPU=$2; shift;;
    --c2-early) C2EARLY=1;;
    --stages) STAGES=$2; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
  shift
done
read -r -a GPUARR <<< "$GPUS"
[ ${#GPUARR[@]} -ge 1 ] || { echo "refused: no GPU given" >&2; exit 2; }
for g in "${GPUARR[@]}"; do case $g in ''|*[!0-9]*) echo "refused: bad GPU id '$g'" >&2; exit 2;; esac; done
case $PERGPU in [1-9]|1[0-6]) ;; *) echo "refused: --slots-per-gpu must be 1..16" >&2; exit 2;; esac
for s in $STAGES; do case $s in p2t3|wa|c1|c2) ;; *) echo "refused: unknown stage '$s'" >&2; exit 2;; esac; done
case $RUNS in */runs_ds3|*/runs_ds3/*|*/runs_bench*) echo "refused: the Phase-2 runs root must not be runs_ds3 or runs_bench*" >&2; exit 2;; esac
case $CODE in */code_p2r1|*/code_p2r1/) ;; *) echo "refused: CODE ($CODE) must be the Phase-2 tree code_p2r1" >&2; exit 2;; esac
has_stage() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }

psl() {                            # "PID COMMAND" lines (procps on the server; the -ef fallback is for Git Bash tests)
  ps -eo pid,args 2>/dev/null || ps -ef 2>/dev/null | awk '{c=$0; sub(/^ *[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +/,"",c); print $2, c}'
}
h_step_running() { psl | grep -E -e '[H]_test\.sh' | grep -v grep > /dev/null; }

# ---- job registry ------------------------------------------------------------------------------------------------------
J_NAME=(); J_GPU=(); J_KIND=(); J_RD=(); J_SOFT=(); J_PRE=(); J_CMD=(); J_POST=(); J_REC=(); J_EST=(); J_NOTE=()
declare -A J_IDX
ENVPFX="env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=$DATA"
job() {  # name kind rundir soft pre cmd post recipe_file est note
  local name=$1 idx=${#J_NAME[@]}
  [ -z "${J_IDX[$name]:-}" ] || { echo "duplicate job $name" >&2; exit 1; }
  J_IDX[$name]=$idx; J_GPU+=("${GPUARR[$(( idx % ${#GPUARR[@]} ))]}")
  J_NAME+=("$name"); J_KIND+=("$2"); J_RD+=("$3"); J_SOFT+=("$4"); J_PRE+=("$5"); J_CMD+=("$6"); J_POST+=("$7"); J_REC+=("$8"); J_EST+=("$9"); J_NOTE+=("${10}")
}
tagf() { printf '%g' "$1"; }
ltag() { printf '%.0e' "$1" | sed 's/e-0/e-/'; }      # = central_ft.lr_tag (1e-5, 3e-5, 1e-4, 3e-4)
upper() { echo "$1" | tr a-z A-Z; }
recipe_json() {  # server_opt eta client_lr  (frozen item tables, dense upload: the WA structure; PREG2 5.2)
  local so=$1 eta=$2 c=$3 label
  label="$so|eta=$(tagf "$eta")|c=$(tagf "$c")"
  printf '{"format": "PPSI_V4_P2_RECIPE_1", "label": "%s", "recipe": {"server_opt": "%s", "server_lr": %s, "client_lr": %s, "items": "frozen", "item_lr_mult": null, "upload": "dense", "lora_rank": null, "lora_lr_factor": null, "personal_head": false}}' "$label" "$so" "$eta" "$c"
}
recipe_file() { echo "$RECIPES/P2_W_$(upper "$1")_eta$(tagf "$2")_c$(tagf "$3").json"; }   # server_opt eta client_lr
WA_CORE_SERVER=("fedavg 0.3" "fedavg 1.0" "fedavgm 0.03" "fedavgm 0.1" "fedadam 0.0001" "fedadam 0.0003")   # PREG2 5.2
WA_CORE_CLIENT=(0.003 0.01 0.03)
WA_EXT_SERVER=("fedadam 1e-05" "fedadam 0.0001"); WA_EXT_CLIENT=0.1
WA_INCUMBENT="fedadam 0.001 0.1"                                                                           # S8's f1 / sf0.1
C1_LRS=(1e-05 3e-05 0.0001 0.0003)                                                                         # PREG2 4.2
T3=$RUNS/P2T3; T3_RECEIPT=$T3/P2T3_RECEIPT.json; T3_RECIPE=$RECIPES/P2T3_INCUMBENT_RECIPE.json

add_p2t3() {
  local g=${GPUARR[0]} cmdsmoke cmdcheck
  cmdsmoke="$ENVPFX $PY -B $CODE/v4int/launch_fl.py --arm P2_W --role CALIBRATION --seed 2026 --gpu $g --allow-shared-gpu --history-policy $HP --init-from $INIT --p2-recipe $T3_RECIPE --smoke --smoke-rounds 65 --run-dir $T3/run"
  cmdcheck="$ENVPFX $PY -B $V/tools/p2_t3_check.py --code-root $CODE --p2-run-dir $T3/run --s8tc-search-root $S8TC_ROOT --out $T3_RECEIPT"
  job p2t3 p2t3 "$T3/run" "" "" "$cmdsmoke" "$cmdcheck" "$T3_RECIPE" "0.1 GPU-h" "P2-T3 first: check rc 0 = BIT_IDENTICAL, rc 1 = FAIL (the incumbent is then re-run), rc 2 = refused; the receipt is written once by the tool"
}
add_wa_cell() {  # server_opt eta client_lr [conditional=1]
  local so=$1 eta=$2 c=$3 cond=${4:-} tag rid rd rf i kind=wa
  tag="$(upper "$so")_eta$(tagf "$eta")_c$(tagf "$c")"
  rid="P2_W_${tag}_CALIB_s2026"; rd=$RUNS/WA/$rid; rf=$(recipe_file "$so" "$eta" "$c")
  [ -z "$cond" ] || kind=wacond
  job "wa.$tag" "$kind" "$rd" "$(has_stage p2t3 && echo p2t3)" "" "" "" "$rf" "1.5-2.5 GPU-h" ""
  i=${J_IDX[wa.$tag]}
  J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/launch_fl.py --arm P2_W --role CALIBRATION --seed 2026 --gpu ${J_GPU[$i]} --allow-shared-gpu --history-policy $HP --init-from $INIT --p2-recipe $rf --run-dir $rd"
  [ -z "$cond" ] || J_NOTE[$i]="CONDITIONAL incumbent: runs only if the P2-T3 receipt verdict is not BIT_IDENTICAL (else REUSED: p2_select --incumbent-phase1 + --p2t3-receipt)"
}
add_wa() {
  local p c
  for p in "${WA_CORE_SERVER[@]}"; do for c in "${WA_CORE_CLIENT[@]}"; do add_wa_cell $p "$c"; done; done
  for p in "${WA_EXT_SERVER[@]}"; do add_wa_cell $p "$WA_EXT_CLIENT"; done
  add_wa_cell $WA_INCUMBENT 1
}
add_c1() {
  local v lr rid rd f a g i
  for v in FULL FROZEN; do
    f=""; [ "$v" = FROZEN ] && f="--p2-freeze-item-tables "
    for lr in "${C1_LRS[@]}"; do
      rid="P2_C1_FT_CENTRAL_${v}_CALIB_lr$(ltag "$lr")_s2026"; rd=$RUNS/C1/$rid
      job "c1.${v}.$(ltag "$lr")" c1 "$rd" "$(has_stage p2t3 && echo p2t3)" "" "" "" "" "0.25-0.4 GPU-h (INFERENCE)" ""
      i=${J_IDX[c1.${v}.$(ltag "$lr")]}; g=${J_GPU[$i]}
      a="--role CALIBRATION --peak-lr $lr ${f}--init-from $INIT -- $LAUNCHER_ARGS --seed 2026 --gpu $g --run-dir $rd --allow-shared-gpu --coschedule-06g"
      J_PRE[$i]="$ENVPFX $PY -B $CODE/v4int/central_ft.py --dry-run-out $DRY/$rid.json $a"
      J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/central_ft.py $a"
    done
  done
}
add_c2() {
  local rid=P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026 rd a i g
  rd=$RUNS/C2/$rid
  job c2.main c2 "$rd" "$(has_stage p2t3 && echo p2t3)" "" "" "" "" "12-16 GPU-h (INFERENCE, incl. PRE's automatic C3 continuation)" "C2 on MAIN, seed 2026 only (O3 = YES, DL 427); no smoke (S8); its sealed/ is never opened by this lane"
  i=${J_IDX[c2.main]}; g=${J_GPU[$i]}
  a="-- $LAUNCHER_ARGS --seed 2026 --gpu $g --run-dir $rd --allow-shared-gpu --coschedule-06g"
  J_PRE[$i]="$ENVPFX $PY -B $CODE/v4int/c_full_small.py --dry-run-out $DRY/$rid.json $a"
  J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/c_full_small.py $a"
}
has_stage p2t3 && add_p2t3
[ "$C2EARLY" = 1 ] && has_stage c2 && add_c2
has_stage wa && add_wa
has_stage c1 && add_c1
[ "$C2EARLY" = 0 ] && has_stage c2 && add_c2

# ---- state helpers ------------------------------------------------------------------------------------------------------
run_done() {      # a finished run: public RUN_SUMMARY status only (grep -q: nothing is read out, nothing printed)
  local rd=$1
  [ -f "$rd/RUN_SUMMARY.json" ] || return 1
  grep -q '"status": *"DONE_\(MAIN\|CONTINUATION\)"' "$rd/RUN_SUMMARY.json" || return 1
  if [ -f "$rd/STATUS.json" ] && grep -q '"status": *"\(RUNNING\|FAILED\)"' "$rd/STATUS.json"; then return 1; fi
  return 0
}
run_started() { [ -f "$1/ARM_SPEC.json" ] || [ -f "$1/RUN_START.json" ]; }
receipt_bit_identical() { [ -f "$T3_RECEIPT" ] && grep -q '"verdict": *"BIT_IDENTICAL"' "$T3_RECEIPT"; }
job_state() {     # new | started | done | reused
  local i=$1 rd=${J_RD[$1]}
  case ${J_KIND[$i]} in
    p2t3) if [ -f "$T3_RECEIPT" ]; then echo done; return; fi;;
    wacond) if receipt_bit_identical; then echo reused; return; fi;;
  esac
  if [ "${J_KIND[$i]}" = p2t3 ]; then if run_started "$rd"; then echo started; else echo new; fi; return; fi
  if run_done "$rd"; then echo done; elif run_started "$rd"; then echo started; else echo new; fi
}

# ---- plan -----------------------------------------------------------------------------------------------------------------
print_plan() {
  local i
  echo "# P2 RUN-1 lane plan: ${#J_NAME[@]} jobs, stages: $STAGES, GPUs: $GPUS, slots per GPU: $PERGPU, runs root $RUNS, code tree $CODE"
  echo "# order = launch order; every run goes to the fresh root runs_p2 and the read-only tree code_p2r1; nothing is written in plan mode"
  echo "# S7: Phase 2 never shares a GPU with an H step (refused at start, and no new job after one appears); the ML-1M driver yields to this lane"
  [ -d "$CODE" ] || echo "# NOTE: $CODE does not exist here (plan mode only; --launch refuses)"
  for i in "${!J_NAME[@]}"; do
    printf '%-26s gpu=%s soft-after=[%s] est=%s\n' "${J_NAME[$i]}" "${J_GPU[$i]}" "${J_SOFT[$i]}" "${J_EST[$i]}"
    [ -z "${J_REC[$i]}" ] || printf '    recipe: %s\n' "${J_REC[$i]}"
    [ -z "${J_PRE[$i]}" ] || printf '    dry: %s\n' "${J_PRE[$i]}"
    printf '    cmd: %s\n' "${J_CMD[$i]}"
    [ -z "${J_POST[$i]}" ] || printf '    check: %s\n' "${J_POST[$i]}"
    printf '    out: %s\n' "${J_RD[$i]}"
    printf '    state: %s (resume is automatic: a started run is continued with --resume)\n' "$(job_state "$i")"
    [ -z "${J_NOTE[$i]}" ] || printf '    note: %s\n' "${J_NOTE[$i]}"
  done
  echo "# GPU-h (INFERENCE, PHASE2_IMPLEMENTATION_PLAN section 4 / PREG2 5.9): P2-T3 0.1 + WA 20 x 1.5-2.5 = 30-50 (+1.5-2.5 if the incumbent re-runs) + C1 screen 8 x 0.25-0.4 = 2-3 + C2 12-16 = about 44-69"
  echo "# cap 250 GPU-h for Phase 2 (PREG2 5.9); C2 is the last item in the drop order"
}
if [ "$MODE" = plan ]; then
  h_step_running && echo "# WARNING: an H step (H_test.sh) is running; --launch would refuse"
  print_plan; exit 0
fi

# ---- launch: refusals ---------------------------------------------------------------------------------------------------
if h_step_running; then echo "refused: an H step (H_test.sh) is running; Phase 2 never shares a GPU with an H step (PREG2 S7)" >&2; exit 2; fi
[ -d "$CODE" ] || { echo "refused: the Phase-2 code tree $CODE is missing" >&2; exit 2; }
for f in v4int/launch_fl.py v4int/central_ft.py v4int/c_full_small.py; do
  [ -f "$CODE/$f" ] || { echo "refused: $CODE/$f is missing" >&2; exit 2; }
done
[ -f "$V/tools/p2_t3_check.py" ] || { echo "refused: $V/tools/p2_t3_check.py is missing" >&2; exit 2; }
[ -f "$INIT" ] || { echo "refused: the PRE init record $INIT is missing" >&2; exit 2; }
if [ -n "$(find "$CODE" -type f -perm /222 -print -quit 2>/dev/null)" ]; then echo "refused: $CODE is not read-only (a writable file exists); chmod -R a-w it after the review" >&2; exit 2; fi
case $(uname -s) in MINGW*|MSYS*|CYGWIN*) ;; *)     # directory permission bits are not emulated on Windows hosts
  if [ -n "$(find "$CODE" -type d -perm /222 -print -quit 2>/dev/null)" ]; then echo "refused: $CODE is not read-only (a writable directory exists)" >&2; exit 2; fi;; esac

mkdir -p "$LOGS" "$STATE" "$RECIPES" "$DRY" "$RUNS/WA" "$RUNS/C1" "$RUNS/C2" "$T3"
LOCK=$STATE/DRIVER.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  if kill -0 "$(cat "$LOCK/pid" 2>/dev/null || echo 0)" 2>/dev/null; then echo "refused: lane lock held by pid $(cat "$LOCK/pid")" >&2; exit 2; fi
  rm -rf "$LOCK"; mkdir "$LOCK" || { echo "refused: cannot take the lane lock" >&2; exit 2; }
fi
echo $$ > "$LOCK/pid"; trap '[ "$BASHPID" = "$$" ] && rm -rf "$LOCK"' EXIT

# recipe files: deterministic from the registered grid, write-once (an existing different file is a refusal)
write_recipe() {  # file json
  local f=$1 j=$2
  if [ -e "$f" ]; then printf '%s\n' "$j" | cmp -s - "$f" || { echo "refused: recipe file $f exists with different content" >&2; exit 2; }; return 0; fi
  ( set -C; printf '%s\n' "$j" > "$f" ) || { echo "refused: cannot write recipe file $f" >&2; exit 2; }
  chmod a-w "$f" 2>/dev/null || true
}
write_all_recipes() {
  local p c
  has_stage p2t3 && write_recipe "$T3_RECIPE" "$(recipe_json $WA_INCUMBENT)"
  if has_stage wa; then
    for p in "${WA_CORE_SERVER[@]}"; do for c in "${WA_CORE_CLIENT[@]}"; do write_recipe "$(recipe_file $p "$c")" "$(recipe_json $p "$c")"; done; done
    for p in "${WA_EXT_SERVER[@]}"; do write_recipe "$(recipe_file $p "$WA_EXT_CLIENT")" "$(recipe_json $p "$WA_EXT_CLIENT")"; done
    write_recipe "$(recipe_file $WA_INCUMBENT)" "$(recipe_json $WA_INCUMBENT)"
  fi
}
write_all_recipes

export PYTHONDONTWRITEBYTECODE=1
declare -A PIDG
used_gpu() { local p t=0; for p in "${!PIDG[@]}"; do if kill -0 "$p" 2>/dev/null; then [ "${PIDG[$p]}" = "$1" ] && t=$((t + 1)); else unset 'PIDG[$p]'; fi; done; echo $t; }
used_all() { local p t=0; for p in "${!PIDG[@]}"; do if kill -0 "$p" 2>/dev/null; then t=$((t + 1)); else unset 'PIDG[$p]'; fi; done; echo $t; }
want_yield() { [ -e "$STATE/STOP" ] && return 0; h_step_running && return 0; return 1; }
say() { echo "[$(date -u +%FT%TZ)] $*"; }
ok_marker()   { ( set -C; echo "$2" > "$STATE/$1.ok" ) 2>/dev/null || true; }       # write-once
fail_marker() { echo "$2" > "$STATE/$1.fail"; }
settled() { [ -f "$STATE/$1.ok" ] || [ -f "$STATE/$1.fail" ]; }
for f in "$STATE"/*.fail; do [ -e "$f" ] && mv "$f" "$f.$(date +%s)"; done        # a relaunch retries failed jobs; old failure markers are kept aside

run_job() {       # background subshell; output = names, rc, seconds only
  local i=$1 name=${J_NAME[$1]} log=$LOGS/${J_NAME[$1]}.log g=${J_GPU[$1]} t0 rc cmd
  t0=$(date +%s)
  say "$name: start (gpu $g)"
  echo "[$(date -u +%FT%TZ)] start" >> "$log"
  if [ -n "${J_PRE[$i]}" ]; then
    if ! ( cd "$CODE" && eval "${J_PRE[$i]}" ) >> "$log" 2>&1; then
      fail_marker "$name" "dry run failed"; say "$name: FAILED (dry run) rc=1 sec=$(( $(date +%s) - t0 ))"; return 0
    fi
  fi
  cmd=${J_CMD[$i]}
  if run_started "${J_RD[$i]}" && ! run_done "${J_RD[$i]}"; then cmd="$cmd --resume"; fi
  if [ "${J_KIND[$i]}" = p2t3 ]; then
    if grep -q SMOKE_DONE_NOT_EVIDENCE "$T3/run/STATUS.json" 2>/dev/null; then rc=0
    else ( cd "$CODE" && eval "$cmd" ) >> "$log" 2>&1; rc=$?; fi
    if [ $rc -eq 0 ]; then ( cd "$CODE" && eval "${J_POST[$i]}" ) >> "$log" 2>&1; rc=$?; fi
    # check rc 0 = BIT_IDENTICAL, rc 1 = FAIL verdict (receipt written; the incumbent is then re-run): both settle P2-T3
    if [ $rc -le 1 ] && [ -f "$T3_RECEIPT" ]; then ok_marker "$name" "rc=$rc"; say "$name: ok rc=$rc sec=$(( $(date +%s) - t0 ))"
    else fail_marker "$name" "rc=$rc"; say "$name: FAILED rc=$rc sec=$(( $(date +%s) - t0 ))"; fi
    echo "$name	$g	$rc	$(( $(date +%s) - t0 ))" >> "$LOGS/lane_summary.tsv"
    return 0
  fi
  ( cd "$CODE" && eval "$cmd" ) >> "$log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && run_done "${J_RD[$i]}"; then
    ok_marker "$name" "rc=0"; say "$name: ok rc=0 sec=$(( $(date +%s) - t0 ))"
  else
    [ $rc -ne 0 ] || rc=99                                    # rc 0 but not finished (e.g. INTERRUPTED): a relaunch resumes it
    fail_marker "$name" "rc=$rc"; say "$name: FAILED rc=$rc sec=$(( $(date +%s) - t0 ))"
  fi
  echo "$name	$g	$rc	$(( $(date +%s) - t0 ))" >> "$LOGS/lane_summary.tsv"
  return 0
}

say "launching ${#J_NAME[@]} jobs (stages: $STAGES; GPUs: $GPUS; slots per GPU: $PERGPU; logs $LOGS)"
declare -a PENDING=("${!J_NAME[@]}")
YIELDED=0
while [ ${#PENDING[@]} -gt 0 ]; do
  if want_yield; then YIELDED=1; say "driver: an H step or a STOP file is present; launching nothing new, waiting for running jobs"; break; fi
  declare -a NEXT=()
  for i in "${PENDING[@]}"; do
    name=${J_NAME[$i]}
    if [ -n "${J_SOFT[$i]}" ] && ! settled "${J_SOFT[$i]}"; then NEXT+=("$i"); continue; fi   # P2-T3 settles first (ok or fail; a fail never blocks)
    case $(job_state "$i") in
      done)   say "$name: already finished, skipped"; ok_marker "$name" "already finished"; continue;;
      reused) say "$name: REUSED (P2-T3 receipt BIT_IDENTICAL), not run"; ok_marker "$name" "reused via P2-T3"; continue;;
    esac
    g=${J_GPU[$i]}
    if [ "$(used_all)" -ge $(( PERGPU * ${#GPUARR[@]} )) ] || [ "$(used_gpu "$g")" -ge "$PERGPU" ]; then NEXT+=("$i"); continue; fi
    if want_yield; then NEXT+=("$i"); continue; fi
    run_job "$i" & PIDG[$!]=$g
  done
  PENDING=("${NEXT[@]}")
  [ ${#PENDING[@]} -gt 0 ] && sleep "${POLL:-20}"
done
wait
if [ "$YIELDED" = 1 ]; then say "driver stopped early: ${#PENDING[@]} jobs not launched; relaunch later (finished runs are skipped, started runs resume)."; exit 3; fi
nfail=$(ls "$STATE"/*.fail 2>/dev/null | wc -l)
say "RUN-1 jobs finished; failed jobs: $nfail. Next (Ops): anchor, selections and edge runs - NOT run by this lane."
[ "$nfail" -eq 0 ]
