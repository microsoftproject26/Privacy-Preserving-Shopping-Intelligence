#!/bin/bash
# H_test.sh <step> <gpu|-> <freeze_sha256> <lock_sha256> --i-confirm — review §1 Phase H / MASTER_PLAN §11: THE SINGLE TEST
# LOOK (owner informed before it starts). FINAL r7 form (OPS-ENDGAME-2), NOT RUN. Executes ONLY the plan frozen in ALL_METHODS_LOCK
# (all_methods_lock.test_plan): every locked checkpoint + sha, device commands, baselines, ONNX flag.
# Steps (each refuses unless its predecessor marker exists; markers $Q/endgame/ok/H_<step>_<tag>.ok):
#   gate   11.1 gate check (d9_gate OPEN; register chain verifies; entry 5 binds the lock)    needs ok/G_<tag>.ok + lock + register
#   build  11.1 job_test_build.sh, r7 copy (CPU)                                               needs H_gate
#   final  11.1 job_test_final.sh, r7 copy -> REGISTRY_V2_TEST_EXT.json + DATA09_SERVER_RECEIPT needs H_build
#   (DL 327 gap / README Q12 RESOLVED: the DATA-09 jobs hard-code cd "$W/code" = the old tree; H runs byte-for-byte
#    copies of the r7 files (sha = SYNC_MANIFEST) whose only change is "$W/code -> "$W/code_ds3r7, written once, a-w)
#   reeval_dry / reeval  11.2 every locked checkpoint on MAIN_TEST (+ the TEST panels) (GPU) needs H_final / H_reeval_dry
#   dev    11.3 device arms on MAIN_TEST ×2 seeds + UNSEEN_HOLDOUT_TEST s2026 (GPU, long)      needs H_reeval
#   base   11.4 baselines on MAIN_TEST (CPU; may run beside dev)                                needs H_final
#   onnx   11.5 dump_rows MAIN_TEST -> INT8 bench (CPU; only if K16 ran on VALIDATION)         needs H_final
#   analysis 11.6 TEST index + retention_report --split TEST                                   needs H_reeval H_dev H_base H_onnx
# Interrupted computations resume with identical hashes only (E10.3): a DEV run dir without RUN_SUMMARY gets --resume.
STEP=$1; G=$2
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/endgame_H.log
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r7.sh
source $Q/endgame/_endgame_lib.sh
confirm "$5"
source $Q/endgame/_endgame_ana_args.sh
source $Q/endgame/_endgame_freeze_check.sh
freeze_check "$3"
LSHA=$4; [[ "$LSHA" =~ ^[0-9a-f]{64}$ ]] || eg_refuse "lock sha256 argument missing"
need $OK/G_$ANA_TAG.ok; need $LOCK; need $REGISTER
[ "$(sha $LOCK)" = "$LSHA" ] || eg_refuse "server ALL_METHODS_LOCK.json sha != the DL-registered $LSHA"
X=$ANA_TEST_EXT                                          # = contracts/manifest_registry/REGISTRY_V2_TEST_EXT.json
# r7 copy of a DATA-09 job: source sha == SYNC_MANIFEST (the deployed bytes); only '"$W/code' -> '"$W/code_ds3r7' changes,
# on exactly the expected number of lines (build 1: the cd; final 3: cd, cd tests_data09, cd); written once, read-only.
r7job() { local n=$1 want=$2 src=$CR/v4data09/$1 dst=$EG/${1%.sh}_r7.sh m
  m=$($PY -B -c "import json,sys; print(json.load(open(sys.argv[1]))['files']['code/v4data09/'+sys.argv[2]])" $R/deploy_r7/SYNC_MANIFEST.json $n)
  [ "$(sha $src)" = "$m" ] || eg_refuse "$src is not the r7 SYNC_MANIFEST bytes"
  if [ ! -e $dst ]; then sed 's#"\$W/code#"$W/code_ds3r7#g' $src > $dst.tmp && mv -n $dst.tmp $dst && chmod a-w $dst; fi
  [ "$(diff $src $dst | grep -c '^>')" = "$want" ] || eg_refuse "$dst: $want changed line(s) expected"
  ! grep -qE '"\$W/code("|/)' $dst || eg_refuse "$dst still names the old \$W/code tree"
  step "R7_JOB $dst $(sha $dst | cut -c1-16) (from ${m:0:16})"; echo $dst; }
mk() { echo $OK/H_${1}_$ANA_TAG.ok; }
pre() { for s in "$@"; do need $(mk $s); done; }
case "$STEP" in
 gate) ;; build) pre gate;; final) pre build;; reeval_dry) pre final;; reeval) pre reeval_dry;; dev) pre reeval;;
 base|onnx) pre final;; analysis) pre reeval dev base onnx;;
 *) eg_refuse "unknown step $STEP";;
esac
once $(mk $STEP)
case "$STEP" in reeval_dry|reeval|dev) [[ "$G" =~ ^[01]$ ]] || eg_refuse "step $STEP needs gpu 0|1";; esac
step "PHASE_START H $STEP gpu=$G lock=${LSHA:0:16}"
# the plan: one line per job, fields separated by \x1f (name, args...) — read from fd 3 (job stdin untouched)
plan() { LOCKF=$LOCK WHAT=$1 TAGM=${2:-} XR=$X $PY -B - <<'PY'
import json, os
E = os.environ; L = json.load(open(E["LOCKF"]))["all_methods_lock"]; tp = L["test_plan"]; US = "\x1f"
if E["WHAT"] == "reeval":
    dry = E["TAGM"] == "DRY"
    for r in tp["reevals"]:
        out = r["out"] + ("_DRY" if dry else "")
        a = ["--ckpt", r["ckpt"], "--ckpt-kind", r["kind"], "--expect-sha256", r["sha"], "--variant", r["variant"],
             "--recipe-id", r["recipe"], "--seed", str(r["seed"]), "--role", r["role"], "--registry-ext", E["XR"],
             "--arm-label", r["arm_label"], "--selection-rule", r["rule"], "--checkpoint-id", r["ckpt_id"], "--out-dir", out]
        if r["kind"] == "pf": a[-2:-2] = ["--pf-store", r["pf_store"], "--expect-pf-store-sha256", r["pf_sha"]]   # --out-dir stays last
        print(US.join(["EGH_RV_" + os.path.basename(out)] + a))
elif E["WHAT"] == "dev":
    for d in tp["device"]:
        print(US.join(["EGH_DEV_" + os.path.basename(d["args"][d["args"].index("--run-dir") + 1]), d.get("base_sha") or "-"] + d["args"]))
elif E["WHAT"] == "base":
    for b in tp["baselines"]: print(US.join(["EGH_BASE_" + os.path.basename(b[b.index("--out") + 1])] + b))
elif E["WHAT"] == "onnx":
    print("1" if tp["onnx"] else "0"); print(L["checkpoints"]["C_SMALL_s2026"]["PB"]["ckpt"])
    print(tp.get("onnx_first_users") or "")
    for k in ("onnx_tree", "onnx_tree_aggregate_sha256", "onnx_dump_rows_retro_sha256", "onnx_bench_sha256"): print(tp.get(k) or "")
PY
}
case "$STEP" in
gate)
  # d9_finalize g0 compares the build's counters with the G0 census (v4data09.d9_common L69-70); MEASURED 10-02: the server
  # has no g0/data_census/ -> Ops uploads SPLIT_COUNTS.json + CLIENT_CENSUS.json from the local g0/data_census, sha-checked
  for f in $V/g0/data_census/SPLIT_COUNTS.json $V/g0/data_census/CLIENT_CENSUS.json; do need $f; done
  ejob EGH_REGISTER_VERIFY $PY -B $TOOLS/test_exposure_register.py verify
  LOCKSHA=$LSHA $PY -B - $REGISTER <<'PY' >> "$LOG" 2>&1 || eg_refuse "no MANIFEST_BUILD / RETRO_TEST register entry binds this lock"
import json, os, sys
e = [x for x in json.load(open(sys.argv[1]))["entries"] if x.get("access_purpose") == "MANIFEST_BUILD"
     and x.get("split") == "RETRO_TEST" and x.get("all_methods_lock_sha256") == os.environ["LOCKSHA"]]
print("REGISTER_ENTRY_BINDS_LOCK", [x["entry_id"] for x in e]); sys.exit(0 if e else 1)
PY
  ( cd $CR && PYTHONPATH=$CR $PY -B -m v4data09.d9_gate ) > $Q/EGH_D9_GATE.log 2>&1 || { step "LANE_STOP d9_gate rc != 0"; exit 97; }
  grep -q '"gate": "OPEN"' $Q/EGH_D9_GATE.log || { step "LANE_STOP d9_gate not OPEN"; exit 97; } ;;
build)
  mkdir -p $V/data_test/_run; JB=$(r7job job_test_build.sh 1) || exit 92
  ejob EGH_TEST_BUILD bash $JB
  grep -q 'JOB_TEST_BUILD_DONE rc=0' $Q/EGH_TEST_BUILD.log || { step "LANE_STOP no JOB_TEST_BUILD_DONE rc=0"; exit 97; } ;;
final)
  JF=$(r7job job_test_final.sh 3) || exit 92
  ejob EGH_TEST_FINAL bash $JF
  grep -q 'JOB_TEST_FINAL_DONE registry_rc=0 verify_rc=0 g0_rc=0 tests_rc=0 final_rc=0' $Q/EGH_TEST_FINAL.log || { step "LANE_STOP job_test_final not all rc 0"; exit 97; }
  need $X
  $PY -B -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('overall_pass') is True else 1)" $V/data_test/receipts/DATA09_SERVER_RECEIPT.json || { step "LANE_STOP DATA09 receipt overall_pass != true"; exit 97; }
  step "TEST_EXT $(sha $X | cut -c1-16) — Ops: fetch, DL row, INTEGRITY_CHECK register entry (COUNTS_ONLY, NO_FEEDBACK)" ;;
reeval_dry|reeval)
  M=REAL; EXTRA=""; [ $STEP = reeval_dry ] && { M=DRY; EXTRA="--dry-run"; }
  plan reeval $M > $EG/H_plan_reeval_$M.txt || { step "LANE_STOP plan"; exit 97; }
  RV="$PY -B v4int/reeval.py --history-policy USER_HISTORY_L50 --gpu $G --allow-shared-gpu $EXTRA"
  while IFS=$'\x1f' read -r -u 3 -a A; do N=${A[0]}; ARGS=("${A[@]:1}"); OUT=${ARGS[-1]}
    if [ $M = REAL ] && [ -e "$OUT" ]; then
      [ "$($PY -B -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" $OUT/REEVAL_RECORD.json 2>/dev/null)" = COMPLETE ] \
        && { step "SKIP_COMPLETE $N"; continue; } || { step "LANE_STOP $OUT exists incomplete (manual decision, E10.3)"; exit 7; }
    fi
    ejob $N $RV "${ARGS[@]}"
  done 3< $EG/H_plan_reeval_$M.txt ;;
dev)
  plan dev > $EG/H_plan_dev.txt || { step "LANE_STOP plan"; exit 97; }
  while IFS=$'\x1f' read -r -u 3 -a A; do N=${A[0]}; BS=${A[1]}; ARGS=("${A[@]:2}")
    RD=${ARGS[$(( $(printf '%s\n' "${ARGS[@]}" | grep -nx -- '--run-dir' | cut -d: -f1) ))]}
    if [ "$BS" != - ]; then BP=${ARGS[$(( $(printf '%s\n' "${ARGS[@]}" | grep -nx -- '--base' | cut -d: -f1) ))]}
      [ "$(sha $BP)" = "$BS" ] || { step "LANE_STOP $N base sha != the lock record"; exit 6; }; fi
    RES=(); if [ -e "$RD" ]; then [ -f "$RD/RUN_SUMMARY.json" ] && { step "SKIP_DONE $N"; continue; }; RES=(--resume); fi
    ejob $N dev "${ARGS[@]}" --registry-ext $X "${RES[@]}"
  done 3< $EG/H_plan_dev.txt ;;
base)
  export CUDA_VISIBLE_DEVICES=-1
  plan base > $EG/H_plan_base.txt || { step "LANE_STOP plan"; exit 97; }
  while IFS=$'\x1f' read -r -u 3 -a A; do N=${A[0]}; ARGS=("${A[@]:1}"); O=${ARGS[-1]}
    [ -f "$O/RECEIPT.json" ] && { step "SKIP_DONE $N"; continue; }
    ejob $N nice -n 15 $PY -B -m v4base.run --phase MAIN_TEST --n-threads 8 "${ARGS[@]}"      # TEST_EXT is the default registry
  done 3< $EG/H_plan_base.txt ;;
onnx)
  mapfile -t P < <(plan onnx); if [ "${P[0]}" != 1 ]; then step "ONNX_NOT_IN_LOCK (K16 declared not run)"; else
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    # FIX-1: only these two commands run from code_ds3r7b, after the deployed tree is verified against the lock fields
    TB=$V/${P[3]}; [ "${P[3]}" = code_ds3r7b ] || eg_refuse "lock onnx_tree '${P[3]}' is not code_ds3r7b"
    $PY -B - $R/deploy_r7b/SYNC_MANIFEST.json $TB "${P[4]}" "${P[5]}" "${P[6]}" <<'PY' >> "$LOG" 2>&1 || eg_refuse "code_ds3r7b != the lock (see $LOG)"
import hashlib, json, sys
from pathlib import Path
m = json.load(open(sys.argv[1])); tb = Path(sys.argv[2]); agg, dsha, bsha = sys.argv[3:6]
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
bad = [r for r, x in m["files"].items() if h(tb / r[len("code/"):]) != x]
a2 = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in sorted(m["files"].items())).encode()).hexdigest()
ok = not bad and a2 == m["aggregate_sha256"] == agg and m["files"]["code/v4int/dump_rows_retro.py"] == dsha \
     and m["files"]["code/v4dev/bench.py"] == bsha
print("R7B_VERIFIED_FOR_ONNX" if ok else f"R7B_MISMATCH files={bad[:5]} agg={a2[:12]}"); sys.exit(0 if ok else 1)
PY
    r7b_run() { (cd "$TB" && PYTHONPATH="$TB" "$@"); }
    O=$R/TEST/ONNX; once $O; mkdir -p $O
    FU=(); [ -n "${P[2]}" ] && FU=(--first-users "${P[2]}")          # the VALIDATION form: ONNX_R7_U16384 -> 16384 (8.2)
    ejob EGH_ONNX_DUMP r7b_run nice -n 15 $PY -B v4int/dump_rows_retro.py --role MAIN_TEST --history-policy USER_HISTORY_L50 --registry-ext $X --out $O/rows.npz --ckpt "${P[1]}" --family SASREC "${FU[@]}"   # DEV-05 (r7b): view binding
    # Q13 (RESOLVED by DEV-05 in tree r7b, DL 358; H must run on r7b; the check below stays as a guard): the r7 bench refuses rows without meta.batch_view for C_SMALL's derived view (DEV-04
    # check_rows_view); dump_rows_retro (INT-07, unchanged in r7) drops the non-array name fields and has no --ckpt
    # binding. Stop here, value-free, unless the retro dump carries the binding (needs an INT-07 / DEV-04 fix + review).
    $PY -B -c "import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m.get('batch_view') and m.get('feature_view_binding') else 1)" $O/rows.meta.json \
      || { step "LANE_STOP K16 TEST: rows.meta.json has no batch_view / feature_view_binding (dump_rows_retro r7; README Q13)"; exit 6; }
    MN=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['manifest_n'])" $O/rows.meta.json) || { step "LANE_STOP no manifest_n"; exit 6; }
    # the VALIDATION bench form of laneD3_INT8_r7 (DL 343), unchanged: same checkpoint, variant, tag, --quiet
    ejob EGH_ONNX_BENCH r7b_run nice -n 15 $PY -B -m v4dev.bench --ckpt "${P[1]}" --family SASREC --variant SASREC_D64_B2 \
      --v1-root /home/jovyan/shared_data/rees46_nextitem_v1 --out $O --rows $O/rows.npz --manifest-n $MN --tag onnx_int8 --quiet
  fi ;;
analysis)
  BT=$ANA_BLOCK_TEST; OT=$ANA_OUT_TEST; once $BT; once $OT
  mapfile -t IT < <(ana_index_test); mapfile -t RT < <(ana_report_test "$RSHA")     # one definition (_endgame_ana_args.sh)
  ejob EGH_INDEX_TEST "${IT[@]}"                                                    # Q14 RESOLVED: no --fl on TEST
  chmod -R a-w $BT
  ejob EGH_REPORT_TEST "${RT[@]}"
  need $OT/RETENTION_RESULT.json; chmod -R a-w $OT
  # FIX-5: the registered DESCRIPTIVE full metric table of the TEST block (frozen metric_table.py; index sha of $BT, which
  # the report just read); values copied from the sealed rows, never computed; a refusal stops only this table.
  MTT=$ANA/metric_table_test_$ANA_TAG; once $MTT; step "JOB_START EGH_METRIC_TABLE"
  $PY -B $EG/metric_table.py --block $BT --split TEST --expect-block-index-sha256 "$(sha $BT/BLOCK_INDEX.json)" --out $MTT > $Q/EGH_METRIC_TABLE.log 2>&1
  rc=$?; step "JOB_DONE EGH_METRIC_TABLE rc=$rc"; [ $rc -eq 0 ] && chmod -R a-w $MTT || step "METRIC_TABLE_NOT_WRITTEN rc=$rc (see the job log; DL row)"
  step "TEST_RESULT $(sha $OT/RETENTION_RESULT.json | cut -c1-16) — Ops: register entry 6 (RESULT), DL row, fetch, thesis batch" ;;
esac
date -u +%FT%TZ > $(mk $STEP); step "QUEUE_DONE H $STEP rc=0"
