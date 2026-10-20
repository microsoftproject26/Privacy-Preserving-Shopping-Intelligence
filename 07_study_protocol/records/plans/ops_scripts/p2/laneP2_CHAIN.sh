#!/usr/bin/env bash
# laneP2_CHAIN.sh - PPSI V4 Phase 2 unattended orchestrator (P2-CHAIN). Spec: contracts/PREG2.md (REGISTERED) 4.2, 5.6, 5.7, 5.8, 5.5, 9.1 Q5;
# handoffs P2-06 (p2_select CLI: select --stage C1|WA|FINAL, futility), P2-RUN1, P2-CHAIN (this packet). The name MUST stay laneP2_CHAIN.sh.
#
#   laneP2_CHAIN.sh [--plan] [options]     PLAN MODE (default): the steps, their state, the exact commands; writes nothing
#   laneP2_CHAIN.sh --launch [options]     run the chain (idempotent; a restart resumes at the first step without a marker)
# options: --gpus "0 1" (default)  --slots-per-gpu N (1..16, default 16; passed to RUN-2 / RUN-3)  --wait-c2 (also wait for C2 of RUN-1)
# Steps (markers in $RUNS/_chain_state/<step>.ok, write-once):
#   1 wait_run1    RUN-1 lane finished: no live RUN-1 lock; p2t3 + 21 wa.* + 8 c1.* markers ok, none failed (C2 is not awaited unless --wait-c2)
#   2 wait_anchor  $ANCHOR/REEVAL_RECORD.json status COMPLETE (+ RESULT_ROWS.json)
#   3 c1_select    p2_select select --stage C1 (8 CALIB runs; rc 3 = edge runs missing -> RUN-2 --stages edge -> select again; repeated per unit, <= 3 cells, no cell twice)
#   4 c1_main      laneP2_RUN3.sh --launch --parts c1 in the BACKGROUND (C1 MAIN x4; independent of W*)
#   5 wa_select    p2_select select --stage WA (same edge handling; the reused incumbent via --incumbent-phase1 when the P2-T3 receipt is BIT_IDENTICAL)
#   6 futility     p2_select futility (prints a decision word only): FUTILITY_SKIP_WB_WC -> no RUN-2, else CONTINUE_WB_WC
#   7 run2         laneP2_RUN2.sh --launch --stages "wb wc" (skipped under futility)
#   8 final_select p2_select select --stage FINAL (A* + WB / WC runs; edge handling as above)
#   9 run3         laneP2_RUN3.sh --launch --parts "wstar dp ft" (W* x2, S_CAL -> FA1024 / DP8 / DP1, FT_P2_WSTAR x2), then wait for the c1 lane
# Any refusal, failed job or unexpected rc writes $RUNS/_chain_state/STOP (step, rc, command) and ends with exit 1; nothing is improvised.
# A STOP file (or $RUNS/_lane_state/STOP) present at start: exit 3, nothing run. After review move the STOP file aside and relaunch.
# Output: step names, rc, file sha256 only (selection output, stderr and lane output never reach this log; lane output goes to
# $RUNS/logs/chain_*.log). S8: nothing here opens a sealed/ file or prints a metric. Never launches a lane while H_test.sh runs (it waits).
# Exit: 0 chain complete, 1 STOP, 2 refusal at start, 3 STOP file present / yielded.
set -u
LIBDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$LIBDIR/p2_lane_lib.sh"
LANE_DIR=${LANE_DIR:-$LIBDIR}
WAITC2=0
while [ $# -gt 0 ]; do
  case $1 in
    --plan) MODE=plan;; --launch) MODE=launch;;
    --gpus|--gpu) GPUS=$2; shift;;
    --slots-per-gpu) PERGPU=$2; shift;;
    --wait-c2) WAITC2=1;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
  shift
done
lib_validate
CS=$RUNS/_chain_state
T3_RECEIPT=$RUNS/P2T3/P2T3_RECEIPT.json
SELCMD="$ENVPFX $PY -B $CODE/v4int/p2_select.py"
LANES="--gpus \"$GPUS\" --slots-per-gpu $PERGPU"
POLLS=${POLL:-20}
STEPS="wait_run1 wait_anchor c1_select c1_main wa_select futility run2 final_select run3"

step_ok() { [ -f "$CS/$1.ok" ]; }
state_of() { if step_ok "$1"; then echo ok; else echo pending; fi; }
receipt_bit_identical() { [ -f "$T3_RECEIPT" ] && grep -q '"verdict": *"BIT_IDENTICAL"' "$T3_RECEIPT"; }
dirs_of() { local d; for d in "$@"; do [ -d "$d" ] && echo "$d"; done; }

# ---- plan ---------------------------------------------------------------------------------------------------------------------------
if [ "$MODE" = plan ]; then
  [ -e "$CS/STOP" ] && echo "# WARNING: $CS/STOP exists; --launch would refuse (exit 3)"
  h_step_running && echo "# WARNING: an H step (H_test.sh) is running; lanes are started only after it ends"
  echo "# P2 CHAIN plan: 9 steps, GPUs: $GPUS, slots per GPU: $PERGPU, runs root $RUNS, code tree $CODE, lanes in $LANE_DIR"
  echo "# every step is idempotent; a step with a marker in $CS is skipped; any refusal / failure writes $CS/STOP and ends the chain"
  n=0
  for s in $STEPS; do
    n=$((n + 1)); printf 'step %s %s state=%s\n' "$n" "$s" "$(state_of "$s")"
    case $s in
      wait_run1) echo "    wait: markers in $STATE: p2t3, 21 x wa.*, 8 x c1.* ok (none failed)$([ $WAITC2 = 1 ] && echo ", c2.main and no live $STATE/DRIVER.lock")";;
      wait_anchor) echo "    wait: $ANCHOR/REEVAL_RECORD.json with \"status\": \"COMPLETE\" and RESULT_ROWS.json";;
      c1_select) echo "    cmd: $SELCMD select --stage C1 --runs $RUNS/C1/P2_C1_FT_CENTRAL_*_CALIB_lr*_s2026 --anchor $ANCHOR --out $SEL_C1"
                 echo "    on rc 3 (edge runs missing): bash $LANE_DIR/laneP2_RUN2.sh --launch --stages edge --edge-labels <labels the tool named> $LANES, then the same select once more";;
      c1_main) echo "    cmd: bash $LANE_DIR/laneP2_RUN3.sh --launch --parts c1 $LANES   (background; waits for an H step; its markers c1main.*.ok are checked at step 9)";;
      wa_select) echo "    cmd: $SELCMD select --stage WA --runs $RUNS/WA/P2_W_*_CALIB_s2026 --anchor $ANCHOR [--incumbent-phase1 <s8tc_run_dir of the receipt> --p2t3-receipt $T3_RECEIPT] --out $SEL_A"
                 echo "    on rc 3: RUN-2 --stages edge as above, then the same select once more";;
      futility) echo "    cmd: $SELCMD futility --wa-selection $SEL_A --c1-selection $SEL_C1   (decision word only: FUTILITY_SKIP_WB_WC or CONTINUE_WB_WC)";;
      run2) echo "    cmd: bash $LANE_DIR/laneP2_RUN2.sh --launch --stages \"wb wc\" $LANES   (skipped when futility says FUTILITY_SKIP_WB_WC)";;
      final_select) echo "    cmd: $SELCMD select --stage FINAL --runs <A* run dir> $RUNS/WB/P2_W_*_CALIB_s2026 $RUNS/WC/P2_W_*_CALIB_s2026 --anchor $ANCHOR --wa-selection $SEL_A --c1-selection $SEL_C1 [--incumbent-phase1 ... --p2t3-receipt ...] --out $SEL_FINAL"
                 echo "    on rc 3: RUN-2 --stages edge, then the same select once more (under futility only the A* run is given)";;
      run3) echo "    cmd: bash $LANE_DIR/laneP2_RUN3.sh --launch --parts \"wstar dp ft\" $LANES   (then wait for the c1 lane and check its 4 markers)";;
    esac
  done
  exit 0
fi

# ---- launch: refusals -------------------------------------------------------------------------------------------------------------------
for f in laneP2_RUN2.sh laneP2_RUN3.sh p2_lane_lib.sh; do [ -f "$LANE_DIR/$f" ] || { echo "refused: $LANE_DIR/$f is missing" >&2; exit 2; }; done
[ -d "$CODE" ] || { echo "refused: the Phase-2 code tree $CODE is missing" >&2; exit 2; }
[ -f "$CODE/v4int/p2_select.py" ] || { echo "refused: $CODE/v4int/p2_select.py is missing" >&2; exit 2; }
"$JPY" -c 'import json' 2>/dev/null || { echo "refused: JSON helper interpreter '$JPY' is not usable" >&2; exit 2; }
mkdir -p "$CS" "$LOGS" "$STATE"
if [ -e "$CS/STOP" ] || [ -e "$STATE/STOP" ]; then echo "STOP file present ($CS/STOP or $STATE/STOP); nothing run. Review it, move it aside, relaunch." >&2; exit 3; fi
LOCK=$CS/CHAIN.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  if kill -0 "$(cat "$LOCK/pid" 2>/dev/null || echo 0)" 2>/dev/null; then echo "refused: chain lock held by pid $(cat "$LOCK/pid")" >&2; exit 2; fi
  rm -rf "$LOCK"; mkdir "$LOCK" || { echo "refused: cannot take the chain lock" >&2; exit 2; }
fi
echo $$ > "$LOCK/pid"; trap '[ "$BASHPID" = "$$" ] && rm -rf "$LOCK"' EXIT
export V CODE PY JPY P2_RUNS="$RUNS" P2_INIT="$INIT" P2_SELDIR="$SELDIR" P2_ANCHOR="$ANCHOR" V4_DATASET_ROOT="$DATA" POLL="$POLLS"
ENVA=(env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 "V4_DATASET_ROOT=$DATA")
ERRF=$CS/.stderr.$$; trap '[ "$BASHPID" = "$$" ] && rm -rf "$LOCK" "$ERRF"' EXIT

mark_ok() { ( set -C; echo "$2" > "$CS/$1.ok" ) 2>/dev/null || true; say "$1: ok $2"; }
stop_chain() {   # step rc what : STOP marker (write-once), no values
  ( set -C; printf 'step=%s rc=%s what=%s utc=%s\n' "$1" "$2" "$3" "$(date -u +%FT%TZ)" > "$CS/STOP" ) 2>/dev/null || true
  say "$1: STOP rc=$2 ($3)"; exit 1
}
check_stop() { if [ -e "$STATE/STOP" ] || [ -e "$CS/STOP" ]; then say "STOP file present; chain paused"; exit 3; fi; }
wait_cond() {   # step description function...
  local step=$1 desc=$2; shift 2
  while ! "$@"; do check_stop; sleep "$POLLS"; done
}
wait_no_h() { while h_step_running; do check_stop; sleep "$POLLS"; done; }
lane_alive() { [ -d "$STATE/DRIVER_$1.lock" ] && kill -0 "$(cat "$STATE/DRIVER_$1.lock/pid" 2>/dev/null || echo 0)" 2>/dev/null; }

run1_done() {
  local n
  [ -f "$STATE/p2t3.ok" ] || return 1
  n=$(ls "$STATE"/wa.*.ok 2>/dev/null | wc -l); [ "$n" -ge 21 ] || return 1
  n=$(ls "$STATE"/c1.*.ok 2>/dev/null | wc -l); [ "$n" -ge 8 ] || return 1
  [ "$WAITC2" = 0 ] || [ -f "$STATE/c2.main.ok" ] || return 1
  if [ "$WAITC2" = 1 ]; then [ -d "$STATE/DRIVER.lock" ] && kill -0 "$(cat "$STATE/DRIVER.lock/pid" 2>/dev/null || echo 0)" 2>/dev/null && return 1; fi
  return 0
}
run1_failed() { [ -n "$(ls "$STATE"/p2t3.fail "$STATE"/wa.*.fail "$STATE"/c1.*.fail 2>/dev/null | head -n 1)" ]; }
anchor_done() { [ -f "$ANCHOR/REEVAL_RECORD.json" ] && [ -f "$ANCHOR/RESULT_ROWS.json" ] && grep -q '"status": *"COMPLETE"' "$ANCHOR/REEVAL_RECORD.json"; }

run_lane_fg() {  # name script args... -> rc
  local name=$1 script=$2; shift 2
  wait_no_h; check_stop
  bash "$LANE_DIR/$script" --launch --gpus "$GPUS" --slots-per-gpu "$PERGPU" "$@" >> "$LOGS/chain_$name.log" 2>&1
}

# p2_select: stdout is read only for a decision word / ignored; stderr only for the edge-label line when rc is 3
run_select() {  # args... -> rc ; fills SEL_OUT
  SEL_OUT=$( "${ENVA[@]}" "$PY" -B "$CODE/v4int/p2_select.py" "$@" 2> "$ERRF" ); return $?
}
collect_c1() { SEL_ARGS=(--stage C1 --runs); local d; while IFS= read -r d; do SEL_ARGS+=("$d"); done < <(dirs_of "$RUNS"/C1/P2_C1_FT_CENTRAL_*_CALIB_lr*_s2026); SEL_ARGS+=(--anchor "$ANCHOR" --out "$SEL_C1"); }
incumbent_args() { if receipt_bit_identical; then INC_ARGS=(--incumbent-phase1 "$(jh get "$T3_RECEIPT" s8tc_run_dir)" --p2t3-receipt "$T3_RECEIPT"); else INC_ARGS=(); fi; }
collect_wa() { SEL_ARGS=(--stage WA --runs); local d; while IFS= read -r d; do SEL_ARGS+=("$d"); done < <(dirs_of "$RUNS"/WA/P2_W_*_CALIB_s2026)
               incumbent_args; SEL_ARGS+=(--anchor "$ANCHOR" ${INC_ARGS[@]+"${INC_ARGS[@]}"} --out "$SEL_A"); }
collect_final() {
  SEL_ARGS=(--stage FINAL --runs); local d src
  src=$(jh get "$SEL_A" winner_source)
  if [ "$src" != PHASE1_INCUMBENT ]; then SEL_ARGS+=("$(jh get "$SEL_A" winner_run_dir)"); fi
  if [ ! -f "$CS/futility.skip" ]; then
    while IFS= read -r d; do SEL_ARGS+=("$d"); done < <(dirs_of "$RUNS"/WB/P2_W_*_CALIB_s2026 "$RUNS"/WC/P2_W_*_CALIB_s2026)
    # FINAL edge cells (WB2 -> item-LR 0.001, LoRA rank 64 / 1) live in the same WB / WC dirs
  fi
  INC_ARGS=(); [ "$src" = PHASE1_INCUMBENT ] && incumbent_args
  SEL_ARGS+=(--anchor "$ANCHOR" --wa-selection "$SEL_A" --c1-selection "$SEL_C1" ${INC_ARGS[@]+"${INC_ARGS[@]}"} --out "$SEL_FINAL")
}
do_select() {   # step collect_fn outfile
  # the tool reports missing edge cells one unit at a time (C1: FULL, then FROZEN, each with its own one extension), so
  # rc 3 may repeat; a label requested twice, or more than 3 requested cells in the step (PREG2 5.7 cap), is a STOP
  local step=$1 collect=$2 out=$3 rc labels attempt seen="" l n_req=0
  step_ok "$step" && return 0
  check_stop
  if [ -f "$out" ]; then mark_ok "$step" "sha256=$(sha_of "$out") (file already present)"; return 0; fi
  for attempt in 1 2 3 4; do
    "$collect"
    run_select select "${SEL_ARGS[@]}"; rc=$?
    case $rc in
      0) [ -f "$out" ] || stop_chain "$step" 0 "select rc 0 but $out missing"
         mark_ok "$step" "rc=0 sha256=$(sha_of "$out")"; return 0;;
      3) [ "$attempt" -lt 4 ] || stop_chain "$step" 3 "edge runs still missing after 3 extension rounds (PREG2 5.7)"
         labels=$(jh edge "$ERRF") || stop_chain "$step" 3 "edge runs missing but the tool's labels are not parsable"
         while IFS= read -r l; do
           [ -n "$l" ] || continue
           case ";$seen;" in *";$l;"*) stop_chain "$step" 3 "edge cell requested again after its extension ran (PREG2 5.7)";; esac
           seen="$seen;$l"; n_req=$((n_req + 1))
         done < <(echo "$labels" | tr ';' '
')
         [ "$n_req" -le 3 ] || stop_chain "$step" 3 "more than 3 extension cells in one stage (PREG2 5.7)"
         say "$step: rc=3 edge runs requested ($(echo "$labels" | tr ';' '\n' | wc -l) cell(s)); launching RUN-2 edge"
         run_lane_fg "edge_$step" laneP2_RUN2.sh --stages edge --edge-labels "$labels"; rc=$?
         [ $rc -eq 0 ] || stop_chain "$step" $rc "RUN-2 edge lane";;
      *) stop_chain "$step" $rc "p2_select refused (re-run the plan command to read the message)";;
    esac
  done
}

# ---- the chain -------------------------------------------------------------------------------------------------------------------------
say "chain start (GPUs: $GPUS; slots per GPU: $PERGPU; runs root $RUNS)"
if ! step_ok wait_run1; then
  while ! run1_done; do check_stop; run1_failed && stop_chain wait_run1 1 "a RUN-1 job failed (marker .fail)"; sleep "$POLLS"; done
  mark_ok wait_run1 "RUN-1 lane state complete"
fi
if ! step_ok wait_anchor; then wait_cond wait_anchor anchor anchor_done; mark_ok wait_anchor "REEVAL_RECORD sha256=$(sha_of "$ANCHOR/REEVAL_RECORD.json")"; fi
do_select c1_select collect_c1 "$SEL_C1"

C1PID=""
c1_markers_ok() { local j; for j in c1main.FULL.2026 c1main.FULL.2027 c1main.FROZEN.2026 c1main.FROZEN.2027; do [ -f "$STATE/$j.ok" ] || return 1; done; }
c1_running() { { [ -n "$C1PID" ] && kill -0 "$C1PID" 2>/dev/null; } || lane_alive RUN3_c1; }
ensure_c1() {   # the C1 MAIN lane runs in the background; (re)start it only if it is neither finished nor alive and was not started by this chain run
  c1_markers_ok && return 0
  c1_running && return 0
  [ -z "$C1PID" ] || stop_chain c1_main 1 "the C1 MAIN lane ended without its four ok markers"
  wait_no_h; check_stop
  if command -v setsid > /dev/null 2>&1; then
    setsid nohup bash "$LANE_DIR/laneP2_RUN3.sh" --launch --gpus "$GPUS" --slots-per-gpu "$PERGPU" --parts c1 >> "$LOGS/chain_c1_main.log" 2>&1 < /dev/null & C1PID=$!
  else
    nohup bash "$LANE_DIR/laneP2_RUN3.sh" --launch --gpus "$GPUS" --slots-per-gpu "$PERGPU" --parts c1 >> "$LOGS/chain_c1_main.log" 2>&1 < /dev/null & C1PID=$!
  fi
  say "c1_main: lane started in the background"
}
ensure_c1
step_ok c1_main || mark_ok c1_main "launched (sha256 of the C1 selection $(sha_of "$SEL_C1"))"

do_select wa_select collect_wa "$SEL_A"; ensure_c1

if ! step_ok futility; then
  check_stop
  run_select futility --wa-selection "$SEL_A" --c1-selection "$SEL_C1"; rc=$?
  [ $rc -eq 0 ] || stop_chain futility $rc "p2_select futility refused"
  case $SEL_OUT in
    *FUTILITY_SKIP_WB_WC*) ( set -C; echo FUTILITY_SKIP_WB_WC > "$CS/futility.skip" ) 2>/dev/null || true; mark_ok futility "FUTILITY_SKIP_WB_WC";;
    *CONTINUE_WB_WC*) mark_ok futility "CONTINUE_WB_WC";;
    *) stop_chain futility 0 "unrecognised futility decision";;
  esac
fi

if ! step_ok run2; then
  if [ -f "$CS/futility.skip" ]; then mark_ok run2 "skipped: FUTILITY_SKIP_WB_WC (A* = W*)"
  else
    run_lane_fg run2 laneP2_RUN2.sh --stages "wb wc"; rc=$?
    [ $rc -eq 0 ] || stop_chain run2 $rc "RUN-2 lane (WB / WC)"
    mark_ok run2 "rc=0"
  fi
fi

ensure_c1
do_select final_select collect_final "$SEL_FINAL"; ensure_c1

if ! step_ok run3; then
  run_lane_fg run3 laneP2_RUN3.sh --parts "wstar dp ft"; rc=$?
  [ $rc -eq 0 ] || stop_chain run3 $rc "RUN-3 lane (4 = O1 ruling needed, 3 = yielded, 1 = failed jobs)"
  # the background C1 MAIN lane: wait for it, then check its four markers
  while ! c1_markers_ok; do ensure_c1; check_stop; sleep "$POLLS"; done
  mark_ok run3 "rc=0 sha256(final)=$(sha_of "$SEL_FINAL")"
fi
say "chain complete"
exit 0
