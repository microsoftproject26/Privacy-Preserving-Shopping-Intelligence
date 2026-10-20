#!/usr/bin/env bash
# laneP2_RUN3.sh - PPSI V4 Phase 2, RUN-3 driver, the MAIN runs (P2-CHAIN). Spec: contracts/PREG2.md (REGISTERED) 4.2 (C1 MAIN x 2 variants
# x seeds 2026 / 2027), 5.5 (WD: S_CAL(W*) -> FA1024 / DP8 / DP1), 9.1 Q5 + 9.5 (FT_P2_WSTAR x 2); handoffs P2-05 6.1(c) (C1 MAIN),
# P2-WIRE 6 (production chain), P2-DEVLOCK 2 (launch_device_p2). The name MUST stay laneP2_RUN3.sh (ML-1M driver yields to laneP2*.sh).
#
#   laneP2_RUN3.sh [--plan] [options]      PLAN MODE (default): every job with exact command / GPU / output dir; writes nothing
#   laneP2_RUN3.sh --launch [options]      run the selected parts (resume semantics as RUN-1)
# options: --gpus "0 1" (default)  --slots-per-gpu N (1..16, default 16 = every independent job at once)
#          --parts "c1 wstar dp ft" (default all four)  --serial-dp (DP8 waits for FA1024 and DP1 for DP8, same seed; default: all start after S_CAL)
# Parts and gates (a part whose gate file is missing is a refusal, exit 2):
#   c1     C1 MAIN x4 (FULL / FROZEN x 2026 / 2027). Gate: contracts/fl/P2_C1_SELECTION.json. Peak LR per variant = that file's
#          variants.<V>.winner_peak_lr; the file's sha256 is passed as --c1-selection-sha256. Independent of W*.
#   wstar  W* MAIN x2 (launch_fl --arm P2_W --p2-selection <FINAL>; run id P2_WSTAR_SMALL_s<seed>). Gate: P2_WARM_SELECTION_FINAL.json.
#   dp     S_CAL(W*) [P2_WSTAR_S_CAL, seed 2026, 20 rounds -> S_RECORD_P2.json] -> FA1024 x2, DP8 x2, DP1 x1 (seed 2026 only, 9.5), each with
#          --clip-record <that S_RECORD_P2.json>. S_CAL needs only the FINAL file (W*'s recipe), not W*'s MAIN run.
#   ft     FT_P2_WSTAR x2 (launch_device_p2.py --role MAIN --base <W* run>/ckpt/P2_WSTAR_SMALL_s<seed>/best.pt); each waits for its W* seed.
# O1 (P2-WIRE 3.3 / P2-DEVLOCK 5): if W* is WB3 (delta_rows) the dp part is NOT run; if W* is WC5 (personal head) neither dp nor ft is run.
# The skipped jobs get a marker O1_RULING_NEEDED in _lane_state and the lane exits 4; nothing is improvised.
# Run dirs: runs_p2/C1/P2_C1_FT_CENTRAL_<V>_MAIN_s<S>, runs_p2/MAIN/<run id>. Code tree: code_p2r2. Exit: 0 ok, 1 failed jobs, 2 refusal,
# 3 yielded (H step / STOP), 4 O1 ruling needed. S8: MAIN completion is read from STATUS.json only; no sealed/ file is opened; no value printed.
set -u
LIBDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$LIBDIR/p2_lane_lib.sh"
PARTS="c1 wstar dp ft"; SERIAL=0
while [ $# -gt 0 ]; do
  case $1 in
    --plan) MODE=plan;; --launch) MODE=launch;;
    --gpus|--gpu) GPUS=$2; shift;;
    --slots-per-gpu) PERGPU=$2; shift;;
    --parts) PARTS=$2; shift;;
    --serial-dp) SERIAL=1;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
  shift
done
lib_validate
for s in $PARTS; do case $s in c1|wstar|dp|ft) ;; *) echo "refused: unknown part '$s'" >&2; exit 2;; esac; done
has_part() { case " $PARTS " in *" $1 "*) return 0;; esac; return 1; }
LANE=RUN3_$(echo $PARTS | tr ' ' '\n' | sort | tr '\n' '_' | sed 's/_$//')

# ---- gates and the W* structure (O1) ---------------------------------------------------------------------------------------------------
if [ "$MODE" = launch ]; then
  has_part c1 && [ ! -f "$SEL_C1" ] && { echo "refused: part c1 needs $SEL_C1 (P2_C1_SELECTION), which does not exist" >&2; exit 2; }
  for p in wstar dp ft; do has_part $p && [ ! -f "$SEL_FINAL" ] && { echo "refused: part $p needs $SEL_FINAL (P2_WARM_SELECTION_FINAL), which does not exist" >&2; exit 2; }; done
fi
O1_DP=0; O1_FT=0; WSTAR_LABEL="<W*>"
if [ -f "$SEL_FINAL" ]; then
  WJ=$(jh get "$SEL_FINAL" winner_recipe) || { echo "refused: cannot read winner_recipe of $SEL_FINAL" >&2; exit 2; }
  WSTAR_LABEL=$(jh label "$WJ")
  case $WSTAR_LABEL in *"|up=delta_rows"*) O1_DP=1;; esac
  case $WSTAR_LABEL in *"|pf") O1_DP=1; O1_FT=1;; esac
fi

# ---- jobs ----------------------------------------------------------------------------------------------------------------------------
c1_main_sha=""; [ -f "$SEL_C1" ] && c1_main_sha=$(sha_of "$SEL_C1")
add_c1main() {
  local v s lr rid rd f a i g
  for v in FULL FROZEN; do
    f=""; [ "$v" = FROZEN ] && f="--p2-freeze-item-tables "
    lr="<variants.$v.winner_peak_lr of $SEL_C1>"; [ -f "$SEL_C1" ] && lr=$(jh get "$SEL_C1" "variants.$v.winner_peak_lr")
    for s in 2026 2027; do
      rid="P2_C1_FT_CENTRAL_${v}_MAIN_s$s"; rd=$RUNS/C1/$rid
      job "c1main.$v.$s" main "$rd" "" c1 "about 6-8 GPU-h each (INFERENCE)" "C1 MAIN $v seed $s; its sealed/ is never opened by this lane"
      i=${J_IDX[c1main.$v.$s]}; g=${J_GPU[$i]}
      a="--role MAIN --peak-lr $lr ${f}--init-from $INIT --c1-selection $SEL_C1 --c1-selection-sha256 ${c1_main_sha:-<sha256 of $SEL_C1>} -- $LAUNCHER_ARGS --seed $s --gpu $g --run-dir $rd --allow-shared-gpu --coschedule-06g"
      J_PRE[$i]="$ENVPFX $PY -B $CODE/v4int/central_ft.py --dry-run-out $DRY/C1_MAIN_${v}_s$s.json $a"
      J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/central_ft.py $a"
    done
  done
}
FLC="--allow-shared-gpu --history-policy $HP --init-from $INIT"
add_wstar() {
  local s rd i
  for s in 2026 2027; do
    rd=$RUNS/MAIN/P2_WSTAR_SMALL_s$s
    job "wstar.$s" main "$rd" "" wstar "13,654 rounds on MAIN (INFERENCE 6-9 GPU-h)" "W* MAIN seed $s, bound to P2_WARM_SELECTION_FINAL.json by sha; its sealed/ is never opened by this lane"
    i=${J_IDX[wstar.$s]}
    J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/launch_fl.py --arm P2_W --seed $s --gpu ${J_GPU[$i]} $FLC --p2-selection $SEL_FINAL --run-dir $rd"
  done
}
SCAL_RD=$RUNS/MAIN/P2_WSTAR_S_CAL_SMALL_s2026
add_dp() {
  local arm=$1 nm=$2 s=$3 need=$4 est=$5 rd i
  rd=$RUNS/MAIN/${arm}_SMALL_s$s
  job "$nm" main "$rd" "$need" dp "$est" "WD arm $arm seed $s (z / passes 2 / batch 16 / T 384 fixed by the arm, PREG2 5.5)"
  i=${J_IDX[$nm]}
  J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/launch_fl.py --arm $arm --seed $s --gpu ${J_GPU[$i]} $FLC --p2-selection $SEL_FINAL --clip-record $SCAL_RD/S_RECORD_P2.json --run-dir $rd"
}
add_dpchain() {
  local i
  job scal scal "$SCAL_RD" "" dp "0.5 GPU-h (INFERENCE)" "S_CAL of W*'s configuration (20 rounds, seed 2026); writes S_RECORD_P2.json once in its run dir"
  i=${J_IDX[scal]}
  J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/launch_fl.py --arm P2_WSTAR_S_CAL --seed 2026 --gpu ${J_GPU[$i]} $FLC --p2-selection $SEL_FINAL --run-dir $SCAL_RD"
  add_dp P2_WSTAR_FA1024 fa1024.2026 2026 scal "about 16 GPU-h (INFERENCE)"
  add_dp P2_WSTAR_FA1024 fa1024.2027 2027 scal "about 16 GPU-h (INFERENCE)"
  if [ $SERIAL = 1 ]; then
    add_dp P2_WSTAR_DP8 dp8.2026 2026 "scal fa1024.2026" "about 16 GPU-h (INFERENCE)"
    add_dp P2_WSTAR_DP8 dp8.2027 2027 "scal fa1024.2027" "about 16 GPU-h (INFERENCE)"
    add_dp P2_WSTAR_DP1 dp1.2026 2026 "scal dp8.2026" "about 8 GPU-h (INFERENCE)"
  else
    add_dp P2_WSTAR_DP8 dp8.2026 2026 scal "about 16 GPU-h (INFERENCE)"
    add_dp P2_WSTAR_DP8 dp8.2027 2027 scal "about 16 GPU-h (INFERENCE)"
    add_dp P2_WSTAR_DP1 dp1.2026 2026 scal "about 8 GPU-h (INFERENCE)"
  fi
}
add_ft() {
  local s rd i base
  for s in 2026 2027; do
    rd=$RUNS/MAIN/FT_P2_WSTAR_s$s; base=$RUNS/MAIN/P2_WSTAR_SMALL_s$s/ckpt/P2_WSTAR_SMALL_s$s/best.pt
    job "ft.$s" ft "$rd" "wstar.$s" ft "device fine-tune on MAIN, about 1 GPU-h (INFERENCE)" "FT_P2_WSTAR (Q5) seed $s: base = that seed's W* best.pt (any other base is refused by the tool); no smoke (S8)"
    i=${J_IDX[ft.$s]}; J_BASEF[$i]=$base
    J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/launch_device_p2.py --arm FT_P2_WSTAR --seed $s --gpu ${J_GPU[$i]} --run-dir $rd --history-policy $HP --role MAIN --base $base --allow-shared-gpu"
  done
}
add_c1main; add_wstar; add_dpchain; add_ft
for i in "${!J_NAME[@]}"; do J_SEL[$i]=0; if has_part "${J_PART[$i]}"; then J_SEL[$i]=1; fi; done
if [ $O1_DP = 1 ] || [ $O1_FT = 1 ]; then
  for i in "${!J_NAME[@]}"; do
    { [ "${J_PART[$i]}" = dp ] && [ $O1_DP = 1 ]; } || { [ "${J_PART[$i]}" = ft ] && [ $O1_FT = 1 ]; } || continue
    J_NOTE[$i]="O1: NOT RUN (W* = $WSTAR_LABEL is refused by this arm; ruling needed). ${J_NOTE[$i]}"
    [ "${J_SEL[$i]}" = 1 ] && J_SEL[$i]=2
  done
fi
lane_write_recipes() { :; }                  # RUN-3 reads the registered selection files; no recipe files
GPUARR_N=${#GPUARR[@]}

if [ "$MODE" = plan ]; then
  h_step_running && echo "# WARNING: an H step (H_test.sh) is running; --launch would refuse"
  echo "# P2 RUN-3 lane plan: ${#J_NAME[@]} jobs, parts: $PARTS, GPUs: $GPUS, slots per GPU: $PERGPU, runs root $RUNS, code tree $CODE"
  echo "# gates: c1 -> $SEL_C1 ($([ -f "$SEL_C1" ] && echo present || echo absent)); wstar / dp / ft -> $SEL_FINAL ($([ -f "$SEL_FINAL" ] && echo present || echo absent)); W* label: $WSTAR_LABEL"
  [ $O1_DP = 1 ] && echo "# O1: the dp part is NOT run for this W* (needs an Ops / Fable ruling)"
  [ $O1_FT = 1 ] && echo "# O1: the ft part is NOT run for this W* (personal head; needs an Ops / Fable ruling)"
  echo "# S7: never beside an H step; the ML-1M driver yields to this lane. selected=2 means selected but blocked by O1"
  [ -d "$CODE" ] || echo "# NOTE: $CODE does not exist here (plan mode only; --launch refuses)"
  print_jobs
  echo "# GPU-h (INFERENCE, PREG2 5.9 / P2-WIRE 6): C1 MAIN 4 x 6-8, W* 2 x 6-9, S_CAL 0.5, FA1024 2 x 16, DP8 2 x 16, DP1 8, FT 2 x 1; cap 250 for Phase 2; drops are Ops' decision"
  exit 0
fi

# O1-blocked jobs are not launched: marker + exit 4 at the end
O1_BLOCKED=0
for i in "${!J_NAME[@]}"; do if [ "${J_SEL[$i]}" = 2 ]; then O1_BLOCKED=$((O1_BLOCKED + 1)); fi; done
if [ $O1_BLOCKED -gt 0 ]; then
  mkdir -p "$STATE"; ( set -C; echo "W* = $WSTAR_LABEL: dp / ft jobs not run; ruling needed (P2-WIRE O1, P2-DEVLOCK Q1)" > "$STATE/O1_RULING_NEEDED" ) 2>/dev/null || true
  for i in "${!J_NAME[@]}"; do [ "${J_SEL[$i]}" = 2 ] && J_SEL[$i]=0; done
  # a selected job must not need a blocked job
  for i in "${!J_NAME[@]}"; do
    [ "${J_SEL[$i]}" = 1 ] || continue
    for n in ${J_NEED[$i]}; do
      s=${J_IDX[$n]}; { [ "${J_SEL[$s]}" = 1 ] || kind_done "$s"; } || { J_SEL[$i]=0; say "${J_NAME[$i]}: not run (O1: needs $n)"; }
    done
  done
fi
lane_launch "$LANE" v4int/launch_fl.py v4int/central_ft.py v4int/launch_device_p2.py
rc=$?
[ $rc -ne 0 ] && exit $rc
[ $O1_BLOCKED -eq 0 ] || { say "lane $LANE: $O1_BLOCKED job(s) not run (O1 ruling needed)"; exit 4; }
exit 0
