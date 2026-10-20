#!/usr/bin/env bash
# laneX_B.sh - X-REES46 part B (POST_HOC_EXPLORATORY, DL 539): REES46 warm FL with the FedAdam tau fix, GPU 0 only.
#   M  measure: two CALIB smokes (65 rounds, no eval) of the W* recipe (frozen items, eta 3e-6, c 0.1) at tau 1e-3 (= W*)
#      and tau 1e-6, every round logged to X_TAU_DIAG.jsonl
#   G  grid: CALIB (3,414 rounds = 6 EFE, the P2 CALIB budget), warm from theta_pre, items UNFROZEN (ilr 0.1), c 0.1,
#      tau 1e-6, FedAdam eta in {3e-6, 1e-5, 3e-5}, concurrently on GPU 0
#   S  select: x_select.py (the registered P2 CALIB rule); one upper-edge extension (eta 1e-4) if required, then re-select
#   R  MAIN: the selected cell at seeds 2026 + 2027 (13,654 rounds = 6 EFE, = W*'s budget), concurrently on GPU 0
# No TEST role anywhere. Resumable by markers in runs_x_rees46/_state; stops at the first failure (marker FAIL).
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; CODE=$V/code_x1; X=$V/x_rees46; R=$V/runs_x_rees46; ST=$R/_state
G=0
mkdir -p "$ST" "$R/logs"; LOG=$R/lane.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
fail() { touch "$ST/FAIL"; say "FAIL: $*"; exit 1; }
[ -f "$ST/FAIL" ] && { say "FAIL marker present: refuse"; exit 2; }
export X_CODE=$CODE PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1
unset CUDA_VISIBLE_DEVICES
INIT=$V/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026/PRETRAIN_FOR_FL_INIT.json
C=(--gpu $G --allow-shared-gpu --history-policy USER_HISTORY_L50 --init-from "$INIT")
cd "$CODE"
# run <name> <args...> : background-safe single job with marker
job() { local n=$1; shift; [ -f "$ST/$n.ok" ] && { say "$n: already ok"; return 0; }; say "$n: start"
  if "$PY" -B "$X/launch_fl_x.py" "$@" >> "$R/logs/$n.out" 2>&1; then touch "$ST/$n.ok"; say "$n: ok"; else say "$n: rc=$?"; touch "$ST/$n.fail"; return 1; fi; }
par() { local pids=() rc=0; for spec in "$@"; do eval "$spec" & pids+=($!); done; for p in "${pids[@]}"; do wait "$p" || rc=1; done; return $rc; }

# ---------------- M: tau measurement (W* recipe, frozen) ----------------
par "job M_tau1e-3 --x-tau 1e-3 --x-diag-first 100 --arm P2_W --role CALIBRATION --seed 2026 ${C[*]} --p2-recipe $R/recipes/WSTAR_FROZEN_eta3e-6.json --smoke --smoke-rounds 65 --run-dir $R/M/WSTAR_tau1e-3" \
    "job M_tau1e-6 --x-tau 1e-6 --x-diag-first 100 --arm P2_W --role CALIBRATION --seed 2026 ${C[*]} --p2-recipe $R/recipes/WSTAR_FROZEN_eta3e-6.json --smoke --smoke-rounds 65 --run-dir $R/M/WSTAR_tau1e-6" \
  || fail "M smokes"

# ---------------- G: CALIB grid ----------------
gjob() { echo "job G_$1 --x-tau 1e-6 --arm P2_W --role CALIBRATION --seed 2026 ${C[*]} --p2-recipe $R/recipes/UNF_eta$1.json --run-dir $R/G/UNF_eta$1_tau1e-6"; }
par "$(gjob 3e-06)" "$(gjob 1e-05)" "$(gjob 3e-05)" || fail "G grid"

# ---------------- S: selection (+ one upper-edge extension) ----------------
GR=("$R/G/UNF_eta3e-06_tau1e-6" "$R/G/UNF_eta1e-05_tau1e-6" "$R/G/UNF_eta3e-05_tau1e-6")
set +e; "$PY" -B "$X/x_select.py" "$R/X_SELECTION_G.json" "${GR[@]}" >> "$LOG" 2>&1; src=$?; set -e
if [ $src -eq 3 ]; then
  [ -f "$R/recipes/UNF_eta0.0001.json" ] || "$PY" -B -c "import json,sys;sys.path.insert(0,'.');from v4int import launch_fl as F;open('$R/recipes/UNF_eta0.0001.json','w').write(json.dumps(F.p2_recipe_file_obj({'server_opt':'fedadam','server_lr':1e-4,'client_lr':0.1,'items':'unfrozen','item_lr_mult':0.1}),sort_keys=True,indent=1))"
  par "$(gjob 0.0001)" || fail "G edge 1e-4"
  GR+=("$R/G/UNF_eta0.0001_tau1e-6")
  set +e; "$PY" -B "$X/x_select.py" "$R/X_SELECTION_FINAL.json" "${GR[@]}" >> "$LOG" 2>&1; src=$?; set -e
  [ $src -eq 0 ] || [ $src -eq 3 ] || fail "select final rc=$src"
else
  [ $src -eq 0 ] || fail "select rc=$src"
  cp "$R/X_SELECTION_G.json" "$R/X_SELECTION_FINAL.json"
fi
ETA=$("$PY" -c "import json;print(format(json.load(open('$R/X_SELECTION_FINAL.json'))['winner']['eta'],'g'))")
say "selected eta $ETA (X_SELECTION_FINAL sha $(sha256sum "$R/X_SELECTION_FINAL.json" | cut -c1-16))"
RF=$R/recipes/UNF_eta$ETA.json; [ -f "$RF" ] || fail "no recipe file $RF"

# ---------------- R: MAIN seeds 2026 + 2027 ----------------
mjob() { echo "job R_s$1 --x-tau 1e-6 --arm P2_W --seed $1 ${C[*]} --p2-recipe $RF --run-dir $R/MAIN/X_FA_UNF_eta${ETA}_tau1e-6_s$1"; }
par "$(mjob 2026)" "$(mjob 2027)" || fail "MAIN runs"
say "lane done"
