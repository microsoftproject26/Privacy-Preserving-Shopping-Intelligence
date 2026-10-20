#!/usr/bin/env bash
# laneP2_RUN2.sh - PPSI V4 Phase 2, RUN-2 driver (P2-CHAIN). Spec: contracts/PREG2.md (REGISTERED) 5.3 (WB: 3 cells), 5.4 (WC: 5 cells),
# 5.7 (edge extensions: at most 3 per stage); handoffs P2-WIRE (WB3 / LoRA launch flags, code_p2r2), P2-06 (selection files).
# The file name MUST stay laneP2_RUN2.sh (the ML-1M GPU driver yields to any process matching laneP2*.sh; PREG2 S7).
#
#   laneP2_RUN2.sh [--plan] [options]      PLAN MODE (default): every job with exact command, GPU, output dir; writes nothing
#   laneP2_RUN2.sh --launch [options]      run the jobs (resume semantics as RUN-1)
# options: --gpus "0 1" (default)  --slots-per-gpu N (1..16, default 16 = all jobs at once)
#          --stages "wb wc edge" (default "wb wc")     --edge-labels "<label>;<label>"  (needed by, and only used with, stage edge)
# Stages: wb = WB1..WB3, wc = WC1..WC5 at the A* recipe read from contracts/fl/P2_WARM_SELECTION_A.json (winner_recipe: server
# optimiser, server LR, client LR; every other field is the registered cell). edge = the 5.7 extension runs a selection tool asked
# for ("edge runs missing"): WA edge cells (frozen, dense; dir WA/), FINAL edge cells (WB2 item-LR 0.001 -> WB/, LoRA rank 64 / 1 -> WC/),
# C1 edge cells (peak LR 3e-6 / 1e-3, CALIB; dir C1/, dry run first). At most 3 edge cells per call (5.7). Labels are the tool's own labels.
# Recipe files: runs_p2/recipes/P2_W_<TAG>.json (write-once, deterministic). Run dirs: runs_p2/{WB,WC,WA,C1}/<run id>. Code tree: code_p2r2.
# S8: no sealed/ file is opened, no value is printed. Never launches while H_test.sh runs (exit 2; exit 3 if one appears or STOP exists).
set -u
LIBDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$LIBDIR/p2_lane_lib.sh"
STAGES="wb wc"; EDGE_LABELS=""
while [ $# -gt 0 ]; do
  case $1 in
    --plan) MODE=plan;; --launch) MODE=launch;;
    --gpus|--gpu) GPUS=$2; shift;;
    --slots-per-gpu) PERGPU=$2; shift;;
    --stages) STAGES=$2; shift;;
    --edge-labels) EDGE_LABELS=$2; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
  shift
done
lib_validate
for s in $STAGES; do case $s in wb|wc|edge) ;; *) echo "refused: unknown stage '$s'" >&2; exit 2;; esac; done
has_stage() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
if has_stage edge && [ -z "$EDGE_LABELS" ]; then echo "refused: stage edge needs --edge-labels" >&2; exit 2; fi
if ! has_stage edge && [ -n "$EDGE_LABELS" ]; then echo "refused: --edge-labels is only used with stage edge" >&2; exit 2; fi

# ---- A* and the registered WB / WC cells ----------------------------------------------------------------------------------------------
WB_CELLS=('WB1|"items": "unfrozen", "item_lr_mult": 0.1, "upload": "dense"'
          'WB2|"items": "unfrozen", "item_lr_mult": 0.01, "upload": "dense"'
          'WB3|"items": "unfrozen", "item_lr_mult": 0.1, "upload": "delta_rows"')
WC_CELLS=('WC1|"lora_rank": 4, "lora_lr_factor": 10' 'WC2|"lora_rank": 16, "lora_lr_factor": 10'
          'WC3|"lora_rank": 4, "lora_lr_factor": 1' 'WC4|"lora_rank": 16, "lora_lr_factor": 1' 'WC5|"personal_head": true')
ASTAR_JSON=""; ASTAR_KNOWN=1
if has_stage wb || has_stage wc; then
  if [ -f "$SEL_A" ]; then
    ASTAR_JSON=$(jh get "$SEL_A" winner_recipe) || { echo "refused: cannot read winner_recipe of $SEL_A" >&2; exit 2; }
  else
    ASTAR_KNOWN=0; ASTAR_JSON='{"server_opt": "<A*>", "server_lr": 0, "client_lr": 0}'
    [ "$MODE" = plan ] || { echo "refused: $SEL_A (P2_WARM_SELECTION_A) does not exist: select --stage WA first" >&2; exit 2; }
  fi
fi
cell_json() {  # extra-fields-json-fragment -> full recipe json of A* + the cell
  local base; base=$(printf '%s' "$ASTAR_JSON" | "$JPY" -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k: d[k] for k in ("server_opt","server_lr","client_lr")})[1:-1])')
  printf '{%s, %s}' "$base" "$1"
}
declare -A REC_JSON
add_recipe_job() {  # job_prefix stage_dir cell_name json part est note
  local pre=$1 sdir=$2 rj=$3 part=$4 est=$5 note=$6 tag rid rd rf i
  if [ $ASTAR_KNOWN = 1 ]; then tag=$(jh tag "$rj"); else tag="${7:-X}"; fi
  rid="P2_W_${tag}_CALIB_s2026"; rd=$RUNS/$sdir/$rid; rf=$RECIPES/P2_W_$tag.json
  job "$pre.$tag" calib "$rd" "" "$part" "$est" "$note"
  i=${J_IDX[$pre.$tag]}
  J_REC[$i]=$rf
  J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/launch_fl.py --arm P2_W --role CALIBRATION --seed 2026 --gpu ${J_GPU[$i]} --allow-shared-gpu --history-policy $HP --init-from $INIT --p2-recipe $rf --run-dir $rd"
  [ $ASTAR_KNOWN = 1 ] && REC_JSON[$rf]=$(jh file "$rj")
}
add_cells() {  # stage array-name
  local -n arr=$2; local x name frag rj
  for x in "${arr[@]}"; do
    name=${x%%|*}; frag=${x#*|}
    if [ $ASTAR_KNOWN = 1 ]; then rj=$(cell_json "$frag"); else rj=""; fi
    if [ $ASTAR_KNOWN = 1 ]; then add_recipe_job "$1" "$(echo "$1" | tr a-z A-Z)" "$rj" "$1" "1.5-2.5 GPU-h (INFERENCE)" "$name at A*"
    else add_recipe_job "$1" "$(echo "$1" | tr a-z A-Z)" "$rj" "$1" "1.5-2.5 GPU-h (INFERENCE)" "$name at A* (A* is read from $SEL_A at launch)" "<A*>_$name"; fi
  done
}
has_stage wb && add_cells wb WB_CELLS
has_stage wc && add_cells wc WC_CELLS

add_edge() {
  local lab=$1 p k v lr t rid rd a i g rf rj
  p=$(jh parse "$lab") || { echo "refused: edge label '$lab' is not parsable" >&2; exit 2; }
  case $p in
    C1\ *)
      read -r _ v lr <<< "$p"; t=$(ltag "$lr")
      rid="P2_C1_FT_CENTRAL_${v}_CALIB_lr${t}_s2026"; rd=$RUNS/C1/$rid
      job "edge.c1.$v.$t" calib "$rd" "" edge "0.25-0.4 GPU-h (INFERENCE)" "5.7 edge cell of C1 ($lab)"
      i=${J_IDX[edge.c1.$v.$t]}; g=${J_GPU[$i]}
      k=""; [ "$v" = FROZEN ] && k="--p2-freeze-item-tables "
      a="--role CALIBRATION --peak-lr $lr ${k}--init-from $INIT -- $LAUNCHER_ARGS --seed 2026 --gpu $g --run-dir $rd --allow-shared-gpu --coschedule-06g"
      J_PRE[$i]="$ENVPFX $PY -B $CODE/v4int/central_ft.py --dry-run-out $DRY/$rid.json $a"
      J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/central_ft.py $a";;
    *)
      local kd; kd=$(jh kind "$p"); t=$(jh tag "$p")
      case $kd in wa) rj=WA;; wb) rj=WB;; *) rj=WC;; esac
      job "edge.$t" calib "$RUNS/$rj/P2_W_${t}_CALIB_s2026" "" edge "1.5-2.5 GPU-h (INFERENCE)" "5.7 edge cell ($lab)"
      i=${J_IDX[edge.$t]}; rf=$RECIPES/P2_W_$t.json; rd=${J_RD[$i]}
      J_REC[$i]=$rf; REC_JSON[$rf]=$(jh file "$p")
      J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/launch_fl.py --arm P2_W --role CALIBRATION --seed 2026 --gpu ${J_GPU[$i]} --allow-shared-gpu --history-policy $HP --init-from $INIT --p2-recipe $rf --run-dir $rd";;
  esac
}
if has_stage edge; then
  IFS=';' read -r -a ELABS <<< "$EDGE_LABELS"
  [ ${#ELABS[@]} -le 3 ] || { echo "refused: at most 3 edge runs per stage (PREG2 5.7), got ${#ELABS[@]}" >&2; exit 2; }
  for l in "${ELABS[@]}"; do add_edge "$l"; done
fi
lane_write_recipes() { local f; for f in "${!REC_JSON[@]}"; do write_recipe "$f" "${REC_JSON[$f]}"; done; }

if [ "$MODE" = plan ]; then
  h_step_running && echo "# WARNING: an H step (H_test.sh) is running; --launch would refuse"
  echo "# P2 RUN-2 lane plan: ${#J_NAME[@]} jobs, stages: $STAGES, GPUs: $GPUS, slots per GPU: $PERGPU, runs root $RUNS, code tree $CODE"
  echo "# A* source: $SEL_A ($([ -f "$SEL_A" ] && echo present || echo absent)); every run goes to runs_p2 with the read-only tree code_p2r2; nothing is written in plan mode"
  echo "# S7: Phase 2 never shares a GPU with an H step; the ML-1M driver yields to this lane"
  [ -d "$CODE" ] || echo "# NOTE: $CODE does not exist here (plan mode only; --launch refuses)"
  print_jobs
  echo "# GPU-h (INFERENCE, PREG2 5.9): WB 3 + WC 5 = 8 x 1.5-2.5 = 12-20; edge cells 1.5-2.5 each (C1 edge 0.25-0.4); drop order WC3/WC4 -> WB2 -> WC1/WC2 -> WB3 is Ops' decision, not this lane's"
  exit 0
fi
lane_launch RUN2 v4int/launch_fl.py v4int/central_ft.py
