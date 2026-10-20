#!/bin/bash
# Ops (Claude Opus, 2026-09-29) — LANES v2: the WHOLE server (owner DL 168): BOTH A40s, ≤ ~12 GPU processes + CPU jobs.
# Order = PREG1_AMENDMENT_06D §8 (MUST first); commands = MASTER_PLAN_V5 §4–§8. Usage: bash setup_lanes_v1.sh <code tree dir name>
# Registration gates (Ops writes a DECISION_LOG row, then creates the file): contracts/dp/S_RECORD.json before any FA_1024/DP;
# contracts/device/SELECTION_FT_C.json / SELECTION_FT_FA.json before any FT application. Lanes wait for them.
set -uo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
TREE=${1:?code tree name, e.g. code_ds3r3}
shift; LANES="${*:?lane names}"
R=$V/runs_ds3; Q=$R/queues
[ -d "$V/$TREE/v4int" ] || { echo "no tree $TREE"; exit 2; }
C=$Q/_common_$TREE.sh
for k in $LANES; do [ -e "$Q/$k.log" ] && { echo "REFUSED: $k already ran"; exit 3; }; done
mkdir -p "$Q" "$R/FL" "$R/DEV" "$R/CFULL" "$R/BASE" "$R/TRUNC" "$V/contracts/fl" "$V/contracts/device" "$V/contracts/dp"

cat > "$C" <<EOF
V=$V
R=\$V/runs_ds3; Q=\$R/queues; CR=\$V/$TREE; PY=/home/jovyan/venvs/ppsi-gpu/bin/python
SEL=\$V/contracts/fl/LR_SELECTION.json; SREC=\$V/contracts/dp/S_RECORD.json
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=\$CR PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "\$CR" || exit 90
EOF
cat >> "$C" <<'EOF'
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
step() { echo "$(ts) $*" >> "$LOG"; }
fl()  { $PY -B v4int/launch_fl.py --gpu "$G" --history-policy USER_HISTORY_L50 --allow-shared-gpu "$@"; }
dev() { $PY -B v4int/launch_device.py --gpu "$G" --history-policy USER_HISTORY_L50 --allow-shared-gpu "$@"; }
job() {  # job <name> <cmd...>; the lane stops on failure (resume = same command + --resume)
  local name=$1; shift; step "JOB_START $name gpu=$G"
  "$@" > "$Q/$name.log" 2>&1; local rc=$?
  step "JOB_DONE $name rc=$rc"; [ $rc -eq 0 ] || { step "LANE_STOP rc=$rc"; exit $rc; }
}
wait_file() { for _ in $(seq 1 8640); do [ -e "$1" ] && return 0; sleep 60; done; step "LANE_STOP timeout $1"; exit 5; }
wait_grep() { for _ in $(seq 1 8640); do grep -q "$2" "$1" 2>/dev/null && return 0; grep -q LANE_STOP "$1" 2>/dev/null && { step "LANE_STOP upstream $1"; exit 4; }; sleep 60; done; step "LANE_STOP timeout $2"; exit 5; }
EOF

mk() {  # mk <lane> <gpu> <body>; only lanes being launched now are (re)written — never touch a running lane script
  case " $LANES " in *" $1 "*) ;; *) return 0;; esac
  { echo '#!/bin/bash'; echo "LOG=$Q/$1.log; G=$2"; echo "source $C"; printf '%s\n' "$3"; echo 'step "QUEUE_DONE rc=0"'; } > "$Q/$1.sh"
}

# lane0 (no GPU): LR selection by the registered CALIB rule once the screen is complete (06B §1)
mk lane0 none '
P1=$R/phase1_r2/phase1_r2.log
wait_grep "$P1" PHASE1_DONE
test ! -e "$SEL" || { step "LANE_STOP LR_SELECTION exists"; exit 6; }
job LR_SELECT $PY -B v4int/lr_screen.py select --runs $R/LRSCREEN/x0.5 $R/LRSCREEN/x1 $R/LRSCREEN/x2 --out $SEL'
LR='wait_grep $Q/lane0.log "JOB_DONE LR_SELECT rc=0"'

# ---------------- GPU 0: federated ----------------
mk lane1 0 "$LR"'
for A in FA FA_Q8; do job ${A}_s2026 fl --arm $A --seed 2026 --run-dir $R/FL/${A}_s2026 --lr-selection $SEL; done'
mk lane2 0 "$LR"'
for A in FA FA_Q8; do job ${A}_s2027 fl --arm $A --seed 2027 --run-dir $R/FL/${A}_s2027 --lr-selection $SEL; done'
mk lane3 0 "$LR"'
for S in 2026 2027; do job FP_s$S fl --arm FP --seed $S --run-dir $R/FL/FP_s$S --lr-selection $SEL; done'
mk lane4 0 "$LR"'
job S_CAL fl --arm S_CAL --seed 2026 --run-dir $R/FL/S_CAL --lr-selection $SEL
wait_file $SREC
for S in 2026 2027; do job FA_1024_s$S fl --arm FA_1024 --seed $S --run-dir $R/FL/FA_1024_s$S --lr-selection $SEL --clip-record $SREC; done'
mk lane5 0 "$LR"'
wait_file $SREC
job DP8_s2026 fl --arm DP --epsilon 8 --seed 2026 --run-dir $R/FL/DP8_s2026 --lr-selection $SEL --clip-record $SREC
job DP8_s2027 fl --arm DP --epsilon 8 --seed 2027 --run-dir $R/FL/DP8_s2027 --lr-selection $SEL --clip-record $SREC
job DP1_s2026 fl --arm DP --epsilon 1 --seed 2026 --run-dir $R/FL/DP1_s2026 --lr-selection $SEL --clip-record $SREC'

mk laneP 0 "$LR"'
for S in 2026 2027; do job PF_s$S fl --arm PF --seed $S --run-dir $R/FL/PF_s$S --lr-selection $SEL; done'

# ---------------- GPU 1: C_FULL + device ----------------
mk lane6 1 '
job CFULL_DRYRUN $PY -B -m v4data.run_bounded --max-rss-mib 24576 --log $R/CFULL/rss_dryrun_s2026.jsonl -- $PY -B v4int/c_full.py --dry-run-out $R/CFULL/C_FULL_DRYRUN_s2026.json -- --family SASREC --seed 2026 --phase MAIN --recipe-id 02B_SASREC --gpu 1 --run-dir $R/CFULL/C_FULL_02B_SASREC_s2026 --history-policy USER_HISTORY_L50
job C_FULL_s2026 $PY -B v4int/c_full.py --ext-registry $V/contracts/manifest_registry/REGISTRY_V2_FULL_EXT.json -- --family SASREC --seed 2026 --phase MAIN --recipe-id 02B_SASREC --gpu 1 --run-dir $R/CFULL/C_FULL_02B_SASREC_s2026 --history-policy USER_HISTORY_L50'
mk lane7 1 '
job SEL_FT_C dev --arm FT_C --seed 2026 --role UNSEEN_VAL --select --base $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --base-kind central --run-dir $R/DEV/SEL_FT_C
wait_file $V/contracts/device/SELECTION_LO.json
job LO_s2026 dev --arm LO --seed 2026 --role MAIN --run-dir $R/DEV/LO_s2026 --n-shards 64 --selection $V/contracts/device/SELECTION_LO.json
wait_file $V/contracts/device/SELECTION_FT_C.json
job FT_C_s2026 dev --arm FT_C --seed 2026 --role MAIN --run-dir $R/DEV/FT_C_s2026 --base $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --base-kind central --selection $V/contracts/device/SELECTION_FT_C.json --n-shards 64'
mk lane8 1 '
job SEL_LO dev --arm LO --seed 2026 --role UNSEEN_VAL --select --run-dir $R/DEV/SEL_LO
wait_file $V/contracts/device/SELECTION_LO.json
job LO_s2027 dev --arm LO --seed 2027 --role MAIN --run-dir $R/DEV/LO_s2027 --n-shards 64 --selection $V/contracts/device/SELECTION_LO.json
wait_file $V/contracts/device/SELECTION_FT_C.json
job FT_C_s2027 dev --arm FT_C --seed 2027 --role MAIN --run-dir $R/DEV/FT_C_s2027 --base $V/runs_ds/C_SMALL_s2027/sealed/checkpoints/best.pt --base-kind central --selection $V/contracts/device/SELECTION_FT_C.json --n-shards 64'
mk lane9 1 '
wait_grep $Q/lane1.log "JOB_DONE FA_s2026 rc=0"
job SEL_FT_FA dev --arm FT_FA --seed 2026 --role UNSEEN_VAL --select --base $R/FL/FA_s2026/ckpt/best.pt --base-kind fl --run-dir $R/DEV/SEL_FT_FA
wait_file $V/contracts/device/SELECTION_FT_FA.json
job FT_FA_s2026 dev --arm FT_FA --seed 2026 --role MAIN --run-dir $R/DEV/FT_FA_s2026 --base $R/FL/FA_s2026/ckpt/best.pt --base-kind fl --selection $V/contracts/device/SELECTION_FT_FA.json --n-shards 64
wait_grep $Q/lane2.log "JOB_DONE FA_s2027 rc=0"
job FT_FA_s2027 dev --arm FT_FA --seed 2027 --role MAIN --run-dir $R/DEV/FT_FA_s2027 --base $R/FL/FA_s2027/ckpt/best.pt --base-kind fl --selection $V/contracts/device/SELECTION_FT_FA.json --n-shards 64'

# ---------------- CPU lane (nice): baselines + T41 ----------------
mk laneC none '
job KNN_EMB_BIG nice -n 15 $PY -B -m v4base.run --baseline KNN_EMB --phase MAIN --allow-main-validation --item-table $V/runs_p3/SASREC/02B_SASREC_s2026/sealed/checkpoints/best.pt --family SASREC --table-role BIG --out $R/BASE/KNN_EMB_BIG
job KNN_EMB_SMALL nice -n 15 $PY -B -m v4base.run --baseline KNN_EMB --phase MAIN --allow-main-validation --item-table $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --family SASREC --table-role SMALL --out $R/BASE/KNN_EMB_SMALL
'
mk laneT none '
job T41_WT nice -n 15 $PY -B v4int/launch_trunc.py --study T41 --panel WARM_TRUNCATION --run-dir $R/TRUNC/T41_WT
job T41_CH nice -n 15 $PY -B v4int/launch_trunc.py --study T41 --panel COLD_H --run-dir $R/TRUNC/T41_CH'

for k in $LANES; do sed -i 's/\r$//' "$Q/$k.sh"; bash -n "$Q/$k.sh" || { echo "SYNTAX $k"; exit 7; }; done
sed -i 's/\r$//' "$C"; bash -n "$C" || exit 8
for k in $LANES; do (nohup setsid bash "$Q/$k.sh" > "$Q/$k.nohup" 2>&1 < /dev/null &); done
sleep 5; date -u; echo "LANES_V2 launched from $TREE"; ps -eo pid,cmd | grep "queues/lane" | grep -v grep | wc -l
