#!/usr/bin/env bash
# laneP2_SIT.sh - PPSI V4, THE single TEST sitting T0a: Phase-1 H plan, then Phase-2 scoring, then both analyses (P2-SITTING).
# Spec: contracts/PREG2.md (REGISTERED) 6.3 S1-S8, 9.5 S7, 13 items 5-6; reviews/FABLE_P2_SCRIPT_FREEZE_20261004.md (S8 strict;
# M5: p2_report stdout goes to a log read only after both RETENTION_RESULT files exist); endgame H_test.sh (the Phase-1 plan of
# ALL_METHODS_LOCK, run unchanged). The name MUST stay laneP2_SIT.sh (the ML-1M driver yields to laneP2*.sh).
#
#   laneP2_SIT.sh [--plan] --lock-sha <P2_METHODS_LOCK sha256> --register-sha <TEST_EXPOSURE_REGISTER sha256> [options]
#   laneP2_SIT.sh --launch --lock-sha ... --register-sha ... [options] --i-confirm      (the TEST look: --i-confirm LAST)
# options: --gpus "0 1" (default)   --slots-per-gpu N (1..16, default 4: Phase-2 scoring jobs per GPU)
#          --h-gpu G (H reeval_dry / reeval; default the first of --gpus)   --h-dev-gpu G (H dev; default --h-gpu)
#          --record KEY=file (repeatable; E1_MIA, E4_SECAGG, DEVICES, D1_EXT: DESCRIPTIVE records copied into the P2 TEST block)
#          --report-code-root DIR (p2_report --code-root; default the scoring tree)   --ana-tag TAG (default v3)
#          pins, default = the registered values: --h-sha (H_test.sh, DL 442) --ana-freeze-sha (ANA_SCRIPT_FREEZE_v3, DL 399)
#          --aml-sha (ALL_METHODS_LOCK, DL 402) --p2-freeze-sha (P2_SCRIPT_FREEZE, DL 476)
#          (--score-worker is internal: step 8 re-invokes this file; it refuses outside a live sitting or before the h_dev marker)
# Steps in the registered order (markers $RUNS/_sit_state/<step>.ok, write-once; a restart resumes at the first step without one):
#   1 preflight    P2_METHODS_LOCK sha = --lock-sha and p2_lock verify; the S1 free fields present; code_p2r3 digest = the lock's
#                  code tree; P2_SCRIPT_FREEZE sha = the pin and it binds the running p2_report / p2_block_index / retention_report
#                  / ana_block_index; ALL_METHODS_LOCK sha = the pin; every sealed checkpoint file present; W* not a PF recipe
#   2 register     TEST_EXPOSURE_REGISTER sha = --register-sha, chain verified by the unchanged tool, the two LOCKED_EVALUATION /
#                  RETRO_TEST entries (Phase-1 lock; P2 lock naming P2_METHODS_LOCK, code_sha256 = the lock's tree), no RETRO_TEST
#                  look (observations METRICS) registered yet (S4: no second sitting)
#   3 p2_dry       every Phase-2 TEST command of step 8 with --dry-run (CPU; the TEST gate of reeval_p2 / launch_device_p2 passes,
#                  nothing written; outputs under $RUNS/TEST/dry) - a refusal stops the sitting before H is restored
#   4 h_restore    H_test.sh chmod 555 and its sha256 = the pin (else chmod 000 again and STOP)
#   5 h_reeval_dry H_test.sh reeval_dry <h-gpu>
#   6 h_scoring    H_test.sh reeval <h-gpu>  ||  base -  ||  onnx -   (the CPU steps beside the GPU step; all must end rc 0)
#   7 h_dev        H_test.sh dev <h-dev-gpu>                          (end of all Phase-1 scoring, S3)
#   8 p2_scoring   every sealed P2 (arm, seed, rule) of the lock on MAIN_TEST with reeval_p2 (code_p2r3) + FT_P2_WSTAR x seeds with
#                  launch_device_p2, all at once over --gpus x --slots-per-gpu (the lane scheduler; never beside an H step, S7)
#   9 h_analysis   H_test.sh analysis - (TEST index + retention_report --split TEST); then H_test.sh is locked again (chmod 000)
#  10 p2_index     tools/p2_block_index.py --split TEST (Phase-1 TEST block for S5, the P2 REEVAL / DEV dirs, the 4 selection
#                  records of the lock + --record files, --p2-lock)
#  11 p2_report    tools/p2_report.py --split TEST with the freeze (script sha from P2_SCRIPT_FREEZE) -> p2_out_test_v1
#  12 done         prints the sha256 of both RETENTION_RESULT files: Ops writes the DL row FIRST, then fetches (S3)
# Any refusal, failed job or unexpected rc writes $RUNS/_sit_state/STOP (step, rc, what) and exits 1; nothing is improvised. A STOP
# file ($RUNS/_sit_state/STOP or $RUNS/_lane_state/STOP) at start: exit 3. Refused at start (exit 2): bad options, no --i-confirm,
# an H step already running, a live Phase-2 lane / chain lock, a Phase-2 training job still running.
# Output: step names, rc, seconds, sha256 and counts only. Every TEST output stays under $RUNS/TEST, $V/runs_ds3/TEST and
# $V/runs_ds3/ANA (never fetched or opened before both RETENTION_RESULT files exist, S3); tool output goes to log files there.
# S8: nothing here opens a sealed/ file, RUN_SUMMARY, rows or metrics; job completion = REEVAL_RECORD / STATUS status words only.
# Exit: 0 sitting complete, 1 STOP, 2 refusal at start, 3 STOP file present / yielded.
set -u
LIBDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF=$LIBDIR/$(basename "${BASH_SOURCE[0]}")
. "$LIBDIR/p2_lane_lib.sh"
LANE_KIND=scoring                                  # set AFTER sourcing: the lib resets it to 'training'
CODE=${SIT_CODE:-$V/code_p2r3}                     # SIT_CODE exists only so the code-tree refusal can be tested
PERGPU=4
ATAG=v3; LOCKSHA=""; REGSHA=""; HG=""; HDG=""; RCR=""; CONFIRM=0; RECORDS=()
H_SHA=11c04583635b3a87a10d09a266cbba3cece8b8e6f645a54a0c4f3ca2272755ae      # H_test.sh, content unchanged when locked (DL 442)
ANA_F=59c83c861ff104d65292a15658a38a927626a286f9e628a1d0814c2de01d8134      # ANA_SCRIPT_FREEZE_v3 (DL 399)
AML_SHA=04a3cd196149e2acf2ab5bb004f58c997199aa52e32f3c11c9ae647255beed23    # ALL_METHODS_LOCK (DL 402)
P2F_SHA=3077963f6221f3471a424a60adbad983085c8e69d7044a92227918e5a3850896    # P2_SCRIPT_FREEZE (DL 476)
while [ $# -gt 0 ]; do
  case $1 in
    --plan) MODE=plan;; --launch) MODE=launch;; --score-worker) MODE=worker;;
    --lock-sha) LOCKSHA=$2; shift;; --register-sha) REGSHA=$2; shift;;
    --gpus|--gpu) GPUS=$2; shift;;
    --slots-per-gpu) PERGPU=$2; shift;;
    --h-gpu) HG=$2; shift;; --h-dev-gpu) HDG=$2; shift;;
    --record) RECORDS+=("$2"); shift;;
    --report-code-root) RCR=$2; shift;;
    --ana-tag) ATAG=$2; shift;;
    --h-sha) H_SHA=$2; shift;; --ana-freeze-sha) ANA_F=$2; shift;; --aml-sha) AML_SHA=$2; shift;; --p2-freeze-sha) P2F_SHA=$2; shift;;
    --i-confirm) [ $# -eq 1 ] || { echo "refused: --i-confirm must be the LAST argument" >&2; exit 2; }; CONFIRM=1;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
  shift
done
lib_validate
hex64() { [[ "$1" =~ ^[0-9a-f]{64}$ ]]; }
pid_alive() { local p; p=$(cat "$1" 2>/dev/null); [[ "$p" =~ ^[1-9][0-9]*$ ]] && kill -0 "$p" 2>/dev/null; }   # a pid file -> 0 if alive
for x in "$H_SHA" "$ANA_F" "$AML_SHA" "$P2F_SHA"; do hex64 "$x" || { echo "refused: a pin is not a sha256" >&2; exit 2; }; done
if [ "$MODE" != plan ]; then
  hex64 "$LOCKSHA" || { echo "refused: --lock-sha <the P2_METHODS_LOCK file sha256 of its DL row> is required" >&2; exit 2; }
  hex64 "$REGSHA" || { echo "refused: --register-sha <the TEST_EXPOSURE_REGISTER file sha256 of its DL row> is required" >&2; exit 2; }
fi
[ "$MODE" != launch ] || [ $CONFIRM = 1 ] || { echo "refused: the sitting is the TEST look; re-run with --i-confirm as the LAST argument" >&2; exit 2; }
case $ATAG in ''|*[!a-z0-9]*) echo "refused: bad --ana-tag" >&2; exit 2;; esac
HG=${HG:-${GPUARR[0]}}; HDG=${HDG:-$HG}
for g in "$HG" "$HDG"; do case $g in 0|1) ;; *) echo "refused: --h-gpu / --h-dev-gpu must be 0 or 1 (H_test.sh)" >&2; exit 2;; esac; done
RCR=${RCR:-$CODE}
for r in ${RECORDS[@]+"${RECORDS[@]}"}; do
  case ${r%%=*} in E1_MIA|E4_SECAGG|DEVICES|D1_EXT) ;; *) echo "refused: --record $r (KEY one of E1_MIA E4_SECAGG DEVICES D1_EXT)" >&2; exit 2;; esac
  case $r in *=?*) ;; *) echo "refused: --record $r is not KEY=file" >&2; exit 2;; esac
done

GATES=$V/company/gates; LOCKF=$GATES/P2_METHODS_LOCK.json; FRZ=$GATES/P2_SCRIPT_FREEZE.json; AML=$GATES/ALL_METHODS_LOCK.json
CONTRASTS=$GATES/P2_CONTRASTS_v3.json; REG=$V/company/registers/TEST_EXPOSURE_REGISTER.json; TOOLS=$V/tools
X=$V/contracts/manifest_registry/REGISTRY_V2_TEST_EXT.json; PIN=$V/contracts/eval_device/EVAL_DEVICE_PIN_A40.json
HR=$V/runs_ds3; HQ=$HR/queues; HT=$HQ/endgame/H_test.sh; HOK=$HQ/endgame/ok
P1BLOCK=$HR/ANA/block_test_$ATAG; P1RES=$HR/ANA/out_test_$ATAG/RETENTION_RESULT.json
T=$RUNS/TEST; RVD=$T/REEVAL; DVD=$T/DEV; TDRY=$T/dry; TLOG=$T/logs
P2BLOCK=$T/p2_block_test_v1; P2OUT=$T/p2_out_test_v1; P2LOG=$TLOG/p2_report_test_v1.log
SS=$RUNS/_sit_state
STEPS="preflight register p2_dry h_restore h_reeval_dry h_scoring h_dev p2_scoring h_analysis p2_index p2_report done"

# ---- helper (stdlib only): reads the P2 lock, the freeze, the register and file bytes for sha256; never a metric -----------------
IFS= read -r -d '' SIT_PY <<'PYEOF' || true
import hashlib, importlib.util, json, sys
from pathlib import Path
US = "\x1f"
PB, CB = "PRACTICAL_BEST", "CONTROLLED_BUDGET"
TAG = {PB: "PB", CB: "CB"}
STEM = {"best.pt": "best", "endpoint_6.0.pt": "endpoint_6.0"}
HP = "USER_HISTORY_L50"
sys.stdout.reconfigure(newline="\n")                              # LF lines for bash on every host
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def j(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def die(m): print("REFUSED: " + m); sys.exit(2)
def tool(tools):
    s = importlib.util.spec_from_file_location("sit_p2_lock", str(Path(tools) / "p2_lock.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def rp(v, s):
    q = Path(str(s)); return q if q.is_absolute() else Path(v) / q
def ckpt_of(v, e):
    rd, ck = rp(v, e["run_dir"]), e["checkpoint"]
    if e.get("source_kind") == "FL": return rd / "ckpt" / e["run_id"] / ck, "fl"
    if e.get("source_kind") in ("C1", "C2"):                      # = tools/p2_lock._central_ckpt_dir
        d = rd / "sealed" / "checkpoints"
        return (d if d.is_dir() else rd / "checkpoints") / ck, "central"
    die(f"sealed {e.get('arm')} s{e.get('seed')}: unknown source_kind {e.get('source_kind')!r}")
def jobs(v, lock, runs, x):
    by = {(e["arm"], e["seed"], e["rule"]): e for e in lock["sealed_arms"]}
    out = []
    for e in sorted(lock["sealed_arms"], key=lambda e: (e["arm"] != "FT_P2_WSTAR", e["arm"], e["seed"], e["rule"])):
        a, s, r = e["arm"], int(e["seed"]), e["rule"]
        if a == "FT_P2_WSTAR":
            w = by.get(("P2_WSTAR", s, PB))
            if w is None or w.get("checkpoint_sha256") != e.get("base_checkpoint_sha256") or w.get("run_id") != e.get("base_run_id"):
                die(f"FT_P2_WSTAR s{s}: its base is not the sealed P2_WSTAR s{s} best.pt")
            base = rp(v, w["run_dir"]) / "ckpt" / w["run_id"] / "best.pt"
            o = f"{runs}/TEST/DEV/FT_P2_WSTAR_MAIN_TEST_s{s}"
            args = ["--arm", "FT_P2_WSTAR", "--seed", str(s), "--gpu", "@GPU@", "--run-dir", o, "--history-policy", HP,
                    "--role", "MAIN_TEST", "--base", base.as_posix(), "--base-sha256", e["base_checkpoint_sha256"],
                    "--registry-ext", x, "--allow-shared-gpu"]
            out.append([f"sit.ft.{s}", "ft", o, base.as_posix(), "launch_device_p2.py"] + args)
            continue
        path, kind = ckpt_of(v, e)
        name = f"{a}_s{s}_{TAG[r]}"
        o = f"{runs}/TEST/REEVAL/{name}"
        args = ["--ckpt", path.as_posix(), "--ckpt-kind", kind, "--expect-sha256", e["checkpoint_sha256"], "--variant",
                "SASREC_D64_B2", "--recipe-id", "SMALL_SASREC_D64_B2", "--seed", str(s), "--role", "MAIN_TEST",
                "--registry-ext", x, "--arm-label", a, "--selection-rule", r, "--checkpoint-id",
                f"{e['run_id']}:{STEM[e['checkpoint']]}", "--history-policy", HP, "--gpu", "@GPU@", "--allow-shared-gpu",
                "--out-dir", o]
        out.append([f"sit.rv.{name}", "reeval", o, "-", "reeval_p2.py"] + args)
    return out
c = sys.argv[1]
if c == "preflight":
    v, lockf, locksha, frz, frzsha, aml, amlsha, code, tools = sys.argv[2:11]
    PL = tool(tools)
    if not Path(lockf).is_file(): die(f"{lockf} is not on this host (S2: the P2 lock precedes the sitting)")
    if sha(lockf) != locksha: die(f"{lockf} sha256 is not the DL-registered --lock-sha {locksha[:12]}")
    lock = j(lockf)
    try: s = PL.verify_lock(lock)
    except PL.LockRefused as e: die(f"P2_METHODS_LOCK does not verify ({e})")
    miss = [k for k in ("e1_validity_reading", "part_b_not_run_analyses") if k not in lock]
    if miss: die(f"P2_METHODS_LOCK lacks the S1 free fields {miss} (P2_SCRIPT_FREEZE ruling, DL 475 condition iii)")
    ct = lock.get("code_tree") or {}
    if ct.get("name") != "code_p2r3" or Path(code).name != "code_p2r3": die(f"the lock's code tree is {ct.get('name')!r}, not code_p2r3")
    try: d = PL.code_tree_digest(code)
    except PL.LockRefused as e: die(str(e))
    if d["aggregate_sha256"] != ct.get("aggregate_sha256"): die(f"{code} is not the code tree P2_METHODS_LOCK binds (digest differs)")
    if not Path(frz).is_file() or sha(frz) != frzsha: die(f"{frz} is absent or not the registered P2_SCRIPT_FREEZE {frzsha[:12]}")
    fz = j(frz); files = fz.get("files_sha256") or {}
    if fz.get("format") != "PPSI_V4_P2_SCRIPT_FREEZE_1" or len(files) != 4: die("P2_SCRIPT_FREEZE is not a 4-file PPSI_V4_P2_SCRIPT_FREEZE_1")
    bad = sorted(r for r, h in files.items() if not (Path(v) / r).is_file() or sha(Path(v) / r) != h)
    if bad: die(f"the running {bad} differ from P2_SCRIPT_FREEZE (S2)")
    if not Path(aml).is_file() or sha(aml) != amlsha: die(f"{aml} is absent or not the registered ALL_METHODS_LOCK {amlsha[:12]}")
    gone = []
    for e in lock["sealed_arms"]:
        if e["arm"] == "FT_P2_WSTAR": continue
        p, _k = ckpt_of(v, e)
        if not p.is_file(): gone.append(f"{e['arm']} s{e['seed']} {e['rule']}")
    if gone: die(f"sealed checkpoint files missing on this host: {gone}")
    print(f"PREFLIGHT ok: P2 lock {locksha[:12]} ({s['n_sealed']} sealed, {s['n_declared_not_run']} declared not run, "
          f"{s['n_phase1_reused']} Phase-1 reused); code_p2r3 = the lock's tree ({d['n_files']} files); P2_SCRIPT_FREEZE "
          f"{frzsha[:12]} binds the running tools; ALL_METHODS_LOCK {amlsha[:12]}")
elif c == "regcheck":
    reg, regsha, amlsha, lockf, locksha = sys.argv[2:7]
    if not Path(reg).is_file() or sha(reg) != regsha: die(f"{reg} is absent or not the DL-registered --register-sha {regsha[:12]}")
    ents = [e for e in j(reg).get("entries") or [] if isinstance(e, dict)]
    looks = [e["entry_id"] for e in ents if e.get("split") == "RETRO_TEST" and e.get("observations") == "METRICS"]
    if looks: die(f"a RETRO_TEST look (observations METRICS, entries {looks}) is already registered: no second sitting (S4)")
    le = [e for e in ents if e.get("access_purpose") == "LOCKED_EVALUATION" and e.get("split") == "RETRO_TEST"]
    p1 = [e for e in le if e.get("all_methods_lock_sha256") == amlsha]
    p2 = [e for e in le if e.get("all_methods_lock_sha256") == locksha]
    if not p1: die("no LOCKED_EVALUATION / RETRO_TEST entry for the Phase-1 ALL_METHODS_LOCK (S2)")
    if not p2: die("no LOCKED_EVALUATION / RETRO_TEST entry whose all_methods_lock_sha256 is the P2_METHODS_LOCK sha (S2)")
    if any("P2_METHODS_LOCK" not in str(e.get("observations_description") or "") for e in p2):
        die("the Phase-2 LOCKED_EVALUATION entry does not name P2_METHODS_LOCK (PREG2 §13 item 6)")
    if any(e.get("observations") != "NONE" or e.get("decision") != "NO_FEEDBACK" for e in p1 + p2):
        die("a LOCKED_EVALUATION entry is not observations NONE / decision NO_FEEDBACK")
    agg = (j(lockf).get("code_tree") or {}).get("aggregate_sha256")
    if any(e.get("code_sha256") != agg for e in p2): die("the Phase-2 entry's code_sha256 is not the lock's code-tree digest")
    print(f"REGISTER ok: {regsha[:12]}; LOCKED_EVALUATION entries {[e['entry_id'] for e in p1]} (Phase-1 lock) and "
          f"{[e['entry_id'] for e in p2]} (P2 lock); {len(ents)} entries")
elif c == "jobs":
    v, lockf, runs, x = sys.argv[2:6]
    for row in jobs(v, j(lockf), runs, x): print(US.join(row))
elif c == "selrecords":
    v, lockf = sys.argv[2:4]
    for k, rec in sorted((j(lockf).get("selection_files") or {}).items()): print(f"{k}={rp(v, rec['path']).as_posix()}")
elif c == "frzsha":
    print((j(sys.argv[2]).get("files_sha256") or {})[sys.argv[3]])
else: raise SystemExit("unknown helper command")
PYEOF
sp() { "$JPY" -c "$SIT_PY" "$@"; }

# ---- the Phase-2 scoring jobs (from P2_METHODS_LOCK sealed_arms; the lib scheduler) -----------------------------------------------
build_jobs() {   # -> 0 when the lock is present and every line parsed
  local line i g a rows
  [ -f "$LOCKF" ] || return 1
  rows=$(sp jobs "$V" "$LOCKF" "$RUNS" "$X") || { echo "$rows"; return 2; }
  while IFS=$'\x1f' read -r -a A; do
    [ ${#A[@]} -ge 6 ] || continue
    local name=${A[0]} kind=${A[1]} out=${A[2]} basef=${A[3]} script=${A[4]}
    if [ "$kind" = ft ]; then
      job "$name" ft "$out" "" p2 "device fine-tune on MAIN_TEST, about 1 GPU-h (INFERENCE)" "FT_P2_WSTAR on MAIN_TEST from the sealed W* best.pt (Q5)"
    else
      job "$name" reeval "$out" "" p2 "about 3-4 min per checkpoint alone (INFERENCE, INT-07)" "MAIN_TEST re-evaluation of a sealed checkpoint (reeval_p2 TEST gate)"
    fi
    i=${J_IDX[$name]}; g=${J_GPU[$i]}
    a="${A[*]:5}"; a=${a//@GPU@/$g}
    J_CMD[$i]="$ENVPFX $PY -B $CODE/v4int/$script $a"
    J_PRE[$i]=""; J_SEL[$i]=1
    [ "$basef" = - ] || J_BASEF[$i]=$basef
  done <<< "$rows"
  return 0
}
lane_write_recipes() { :; }                         # the sitting writes no recipe files

if [ "$MODE" = worker ]; then                       # step 8 only (started by the lane itself); the lib scheduler and its lock
  # S3: only a live sitting may start it, and only after all Phase-1 scoring (h_dev marker)
  pid_alive "$SS/SIT.lock/pid" || { echo "refused: --score-worker runs only inside a live sitting" >&2; exit 2; }
  [ -f "$SS/h_dev.ok" ] || { echo "refused: Phase-2 scoring starts only after all Phase-1 scoring (S3; no $SS/h_dev.ok)" >&2; exit 2; }
  LOGS=$TLOG
  build_jobs || { echo "refused: the scoring jobs cannot be built from $LOCKF" >&2; exit 2; }
  mkdir -p "$RVD" "$DVD" "$TLOG"
  lane_launch SIT_P2 v4int/reeval_p2.py v4int/reeval.py v4int/launch_device_p2.py
  exit $?
fi

step_ok() { [ -f "$SS/$1.ok" ]; }
state_of() { if step_ok "$1"; then echo ok; else echo pending; fi; }
hmk() { echo "$HOK/H_${1}_$ATAG.ok"; }
HCMD="ANA_TAG=$ATAG bash $HT <step> <gpu|-> $ANA_F $AML_SHA --i-confirm"

# ---- plan -------------------------------------------------------------------------------------------------------------------------
if [ "$MODE" = plan ]; then
  [ -e "$SS/STOP" ] && echo "# WARNING: $SS/STOP exists; --launch would exit 3"
  h_step_running && echo "# WARNING: an H step (H_test.sh) is running; --launch would refuse"
  echo "# P2 SITTING lane plan (T0a, PREG2 6.3): 12 steps in the registered order; GPUs: $GPUS, slots per GPU: $PERGPU, H GPU $HG / dev $HDG"
  echo "# runs root $RUNS, scoring tree $CODE, H plan $HT (ANA_TAG=$ATAG), state $SS; nothing under $T, $HR/TEST or $HR/ANA is fetched before both RETENTION_RESULT files exist (S3)"
  echo "# pins: H_test.sh ${H_SHA:0:12}, ANA_SCRIPT_FREEZE ${ANA_F:0:12}, ALL_METHODS_LOCK ${AML_SHA:0:12}, P2_SCRIPT_FREEZE ${P2F_SHA:0:12}; --lock-sha ${LOCKSHA:-<required>} --register-sha ${REGSHA:-<required>}"
  n=0
  for s in $STEPS; do
    n=$((n + 1)); printf 'step %s %s state=%s\n' "$n" "$s" "$(state_of "$s")"
    case $s in
      preflight) echo "    check: sha256($LOCKF) = --lock-sha; $PY -B $TOOLS/p2_lock.py verify (via the helper); e1_validity_reading + part_b_not_run_analyses in the lock"
                 echo "    check: code-tree digest of $CODE = lock code_tree; sha256($FRZ) = ${P2F_SHA:0:12} and its 4 files = the running tools; sha256($AML) = ${AML_SHA:0:12}; sealed checkpoints present";;
      register) echo "    check: sha256($REG) = --register-sha; $PY -B $TOOLS/test_exposure_register.py verify; LOCKED_EVALUATION / RETRO_TEST for ${AML_SHA:0:12} and for the P2 lock; no METRICS entry on RETRO_TEST";;
      p2_dry) echo "    cmd: every step-8 command with --dry-run, one at a time (CPU); stdout -> $TDRY/<job>.json";;
      h_restore) echo "    cmd: chmod 555 $HT; sha256 must be ${H_SHA:0:12} (else chmod 000 and STOP)";;
      h_reeval_dry) echo "    cmd: ${HCMD/<step> <gpu|->/reeval_dry $HG}   (marker $(hmk reeval_dry))";;
      h_scoring) echo "    cmd: ${HCMD/<step> <gpu|->/reeval $HG}  ||  ${HCMD/<step> <gpu|->/base -}  ||  ${HCMD/<step> <gpu|->/onnx -}";;
      h_dev) echo "    cmd: ${HCMD/<step> <gpu|->/dev $HDG}";;
      p2_scoring) echo "    cmd: bash $SELF --score-worker (the lib scheduler; lock $STATE/DRIVER_SIT_P2.lock; logs $TLOG); jobs below";;
      h_analysis) echo "    cmd: ${HCMD/<step> <gpu|->/analysis -}; needs $P1RES; then chmod 000 $HT";;
      p2_index) echo "    cmd: cd $CODE && $ENVPFX $PY -B $TOOLS/p2_block_index.py --split TEST --device-class NVIDIA_A40-FP32-DET --pin $PIN --phase1-block $P1BLOCK --reeval $RVD --p2-dev $DVD --records <lock selection files>${RECORDS[@]+ ${RECORDS[*]}} --p2-lock $LOCKF --registry-ext $X --out $P2BLOCK";;
      p2_report) echo "    cmd: cd $CODE && $ENVPFX $PY -B $TOOLS/p2_report.py --split TEST --block $P2BLOCK --contrasts $CONTRASTS --out $P2OUT --p2-lock $LOCKF --p2-freeze $FRZ --all-methods-lock $AML --test-exposure-register $REG --registry-ext $X --phase1-result $P1RES --code-root $RCR --expect-script-sha256 <P2_SCRIPT_FREEZE tools/p2_report.py>  > $P2LOG";;
      done) echo "    print: sha256 of $P1RES and $P2OUT/RETENTION_RESULT.json -> DL row first, fetch second";;
    esac
  done
  if build_jobs; then
    echo "# step 8 jobs: ${#J_NAME[@]} (from P2_METHODS_LOCK sealed_arms; FT first, then the reevals; GPUs round robin)"
    print_jobs
  else
    echo "# step 8 jobs: built from $LOCKF at launch (the lock is not on this host yet)"
  fi
  exit 0
fi

# ---- launch: refusals at start ------------------------------------------------------------------------------------------------------
mkdir -p "$SS/logs" "$STATE"
if [ -e "$SS/STOP" ] || [ -e "$STATE/STOP" ]; then echo "STOP file present ($SS/STOP or $STATE/STOP); nothing run. Review it, move it aside, relaunch." >&2; exit 3; fi
if h_step_running; then echo "refused: an H step (H_test.sh) is already running; the sitting starts it itself" >&2; exit 2; fi
if psl | grep -E 'code_p2r[0-9]+/v4int/(launch_fl|central_ft|c_full_small|launch_device_p2|reeval_p2)\.py' | grep -v grep > /dev/null; then
  echo "refused: a Phase-2 job (code_p2r*) is still running (S7: no Phase-2 job beside an H step; every arm is sealed first)" >&2; exit 2
fi
for lk in "$STATE"/DRIVER_*.lock "$RUNS/_chain_state/CHAIN.lock"; do
  [ -d "$lk" ] || continue
  if pid_alive "$lk/pid"; then echo "refused: a Phase-2 lane holds $lk" >&2; exit 2; fi
done
SLOCK=$SS/SIT.lock
if ! mkdir "$SLOCK" 2>/dev/null; then
  if pid_alive "$SLOCK/pid"; then echo "refused: sitting lock held by pid $(cat "$SLOCK/pid")" >&2; exit 2; fi
  rm -rf "$SLOCK"; mkdir "$SLOCK" || { echo "refused: cannot take the sitting lock" >&2; exit 2; }
fi
echo $$ > "$SLOCK/pid"; trap '[ "$BASHPID" = "$$" ] && rm -rf "$SLOCK"' EXIT
export V PY JPY P2_RUNS="$RUNS" P2_INIT="$INIT" P2_SELDIR="$SELDIR" V4_DATASET_ROOT="$DATA" POLL="${POLL:-20}"

mark_ok() { ( set -C; echo "$2" > "$SS/$1.ok" ) 2>/dev/null || true; say "$1: ok $2"; }
stop_sit() {     # step rc what : STOP marker (write-once), no values
  ( set -C; printf 'step=%s rc=%s what=%s utc=%s\n' "$1" "$2" "$3" "$(date -u +%FT%TZ)" > "$SS/STOP" ) 2>/dev/null || true
  say "$1: STOP rc=$2 ($3)"; exit 1
}
check_stop() { if [ -e "$SS/STOP" ] || [ -e "$STATE/STOP" ]; then say "STOP file present; sitting paused"; exit 3; fi; }
run_h() {        # step gpu -> rc (0 only with the H marker); skipped when the H marker exists
  local st=$1 g=$2 t0 rc
  if [ -f "$(hmk "$st")" ]; then say "H $st: marker present, skipped"; return 0; fi
  t0=$(date +%s); say "H $st: start (gpu $g)"
  ANA_TAG=$ATAG bash "$HT" "$st" "$g" "$ANA_F" "$AML_SHA" --i-confirm > "$SS/logs/H_$st.out" 2>&1; rc=$?
  if [ $rc -eq 0 ] && [ ! -f "$(hmk "$st")" ]; then rc=98; fi
  say "H $st: rc=$rc sec=$(( $(date +%s) - t0 ))"; return $rc
}
say "sitting start (GPUs: $GPUS; slots per GPU: $PERGPU; H GPU $HG / dev $HDG; lock ${LOCKSHA:0:12}; register ${REGSHA:0:12})"

# 1 preflight ------------------------------------------------------------------------------------------------------------------------
if ! step_ok preflight; then
  check_stop
  out=$(sp preflight "$V" "$LOCKF" "$LOCKSHA" "$FRZ" "$P2F_SHA" "$AML" "$AML_SHA" "$CODE" "$TOOLS" 2>&1) || stop_sit preflight 2 "${out##*REFUSED: }"
  for f in "$CONTRASTS" "$X" "$PIN" "$INIT" "$HT"; do [ -e "$f" ] || stop_sit preflight 2 "$f is missing"; done
  [ -f "$CODE/v4int/reeval_p2.py" ] && [ -f "$CODE/v4int/launch_device_p2.py" ] || stop_sit preflight 2 "the scoring tree lacks reeval_p2 / launch_device_p2"
  if [ -f "$SEL_FINAL" ]; then
    WJ=$(jh get "$SEL_FINAL" winner_recipe) || stop_sit preflight 2 "cannot read winner_recipe of $SEL_FINAL"
    case $(jh label "$WJ") in *"|pf") stop_sit preflight 2 "W* has a personal head (WC5): the PF scoring form is not planned (ruling needed)";; esac
  fi
  mark_ok preflight "${out#PREFLIGHT ok: }"
fi
# 2 register -------------------------------------------------------------------------------------------------------------------------
if ! step_ok register; then
  check_stop
  ( cd "$CODE" && "$PY" -B "$TOOLS/test_exposure_register.py" verify ) > "$SS/logs/register_verify.out" 2>&1 || stop_sit register 1 "test_exposure_register.py verify failed (chain)"
  out=$(sp regcheck "$REG" "$REGSHA" "$AML_SHA" "$LOCKF" "$LOCKSHA" 2>&1) || stop_sit register 2 "${out##*REFUSED: }"
  mark_ok register "${out#REGISTER ok: }"
fi
build_jobs || stop_sit register 2 "the scoring jobs cannot be built from the lock"
NJOBS=${#J_NAME[@]}
# 3 p2_dry ---------------------------------------------------------------------------------------------------------------------------
if ! step_ok p2_dry; then
  check_stop
  mkdir -p "$TDRY" "$TLOG"
  for i in "${!J_NAME[@]}"; do
    n=${J_NAME[$i]}
    ( cd "$CODE" && eval "${J_CMD[$i]} --dry-run" ) > "$TDRY/$n.json" 2> "$TLOG/dry_$n.err" || stop_sit p2_dry 1 "dry run of $n refused (log $TLOG/dry_$n.err, read after the sitting or by the reviewer)"
  done
  mark_ok p2_dry "$NJOBS dry runs rc 0"
fi
# 4 h_restore ------------------------------------------------------------------------------------------------------------------------
if ! step_ok h_restore; then
  check_stop
  chmod 555 "$HT" 2>/dev/null || stop_sit h_restore 1 "chmod 555 $HT failed"
  s=$(sha_of "$HT")
  if [ "$s" != "$H_SHA" ]; then chmod 000 "$HT" 2>/dev/null; stop_sit h_restore 2 "$HT sha256 ${s:0:12} is not the pin ${H_SHA:0:12} (locked again)"; fi
  mark_ok h_restore "H_test.sh 555 sha256 ${H_SHA:0:12}"
fi
# 5-7 Phase-1 scoring (the H plan, unchanged) ----------------------------------------------------------------------------------------
if ! step_ok h_reeval_dry; then check_stop; run_h reeval_dry "$HG" || stop_sit h_reeval_dry $? "H reeval_dry (log $SS/logs/H_reeval_dry.out)"; mark_ok h_reeval_dry "marker $(basename "$(hmk reeval_dry)")"; fi
if ! step_ok h_scoring; then
  check_stop
  run_h base - & pb=$!
  run_h onnx - & po=$!
  run_h reeval "$HG"; rr=$?
  wait $pb; rb=$?; wait $po; ro=$?
  [ $rr -eq 0 ] || stop_sit h_scoring $rr "H reeval (log $SS/logs/H_reeval.out)"
  [ $rb -eq 0 ] || stop_sit h_scoring $rb "H base (log $SS/logs/H_base.out)"
  [ $ro -eq 0 ] || stop_sit h_scoring $ro "H onnx (log $SS/logs/H_onnx.out)"
  mark_ok h_scoring "H reeval + base + onnx markers present"
fi
if ! step_ok h_dev; then check_stop; run_h dev "$HDG" || stop_sit h_dev $? "H dev (log $SS/logs/H_dev.out)"; mark_ok h_dev "all Phase-1 scoring done (S3)"; fi
# 8 Phase-2 scoring ------------------------------------------------------------------------------------------------------------------
if ! step_ok p2_scoring; then
  check_stop
  h_step_running && stop_sit p2_scoring 2 "an H step is still running (S7)"
  for i in "${!J_NAME[@]}"; do      # E10.3: an existing incomplete re-evaluation dir is a manual decision (reeval refuses it anyway)
    [ "${J_KIND[$i]}" = reeval ] || continue
    if [ -e "${J_RD[$i]}" ] && ! kind_done "$i"; then stop_sit p2_scoring 7 "${J_RD[$i]} exists incomplete (manual decision, E10.3)"; fi
  done
  bash "$SELF" --score-worker --gpus "$GPUS" --slots-per-gpu "$PERGPU" --lock-sha "$LOCKSHA" --register-sha "$REGSHA"; rc=$?
  [ $rc -eq 0 ] || stop_sit p2_scoring $rc "Phase-2 scoring lane (1 = failed jobs, 2 = refused, 3 = yielded; logs $TLOG)"
  nd=0; for i in "${!J_NAME[@]}"; do kind_done "$i" && nd=$((nd + 1)); done
  [ "$nd" -eq "$NJOBS" ] || stop_sit p2_scoring 99 "$nd of $NJOBS scoring jobs finished"
  mark_ok p2_scoring "$NJOBS jobs finished ($(ls -d "$RVD"/*/ 2>/dev/null | wc -l) REEVAL, $(ls -d "$DVD"/*/ 2>/dev/null | wc -l) DEV dirs)"
fi
# 9 Phase-1 analysis -----------------------------------------------------------------------------------------------------------------
if ! step_ok h_analysis; then
  check_stop
  run_h analysis - || stop_sit h_analysis $? "H analysis (log $SS/logs/H_analysis.out)"
  [ -f "$P1RES" ] || stop_sit h_analysis 98 "$P1RES missing after H analysis"
  chmod 000 "$HT" 2>/dev/null || true
  mark_ok h_analysis "Phase-1 RETENTION_RESULT sha256 $(sha_of "$P1RES"); H_test.sh locked again"
fi
# 10 Phase-2 TEST block --------------------------------------------------------------------------------------------------------------
if ! step_ok p2_index; then
  check_stop
  [ -f "$P1BLOCK/BLOCK_INDEX.json" ] || stop_sit p2_index 2 "$P1BLOCK/BLOCK_INDEX.json missing (S5 needs the Phase-1 TEST block)"
  [ ! -e "$P2BLOCK" ] || [ -z "$(ls -A "$P2BLOCK" 2>/dev/null)" ] || stop_sit p2_index 7 "$P2BLOCK exists and is not empty (manual decision)"
  REC=(); while IFS= read -r l; do [ -n "$l" ] && REC+=("$l"); done < <(sp selrecords "$V" "$LOCKF")
  for r in ${RECORDS[@]+"${RECORDS[@]}"}; do [ -f "${r#*=}" ] || stop_sit p2_index 2 "--record file ${r#*=} missing"; REC+=("$r"); done
  ( cd "$CODE" && $ENVPFX "$PY" -B "$TOOLS/p2_block_index.py" --split TEST --device-class NVIDIA_A40-FP32-DET --pin "$PIN" \
      --phase1-block "$P1BLOCK" --reeval "$RVD" --p2-dev "$DVD" --records "${REC[@]}" --p2-lock "$LOCKF" --registry-ext "$X" \
      --out "$P2BLOCK" ) > "$TLOG/p2_block_index_test_v1.log" 2>&1; rc=$?
  [ $rc -eq 0 ] && [ -f "$P2BLOCK/BLOCK_INDEX.json" ] || stop_sit p2_index $rc "p2_block_index (log $TLOG/p2_block_index_test_v1.log)"
  chmod -R a-w "$P2BLOCK" 2>/dev/null || true
  mark_ok p2_index "BLOCK_INDEX sha256 $(sha_of "$P2BLOCK/BLOCK_INDEX.json"); ${#REC[@]} records"
fi
# 11 Phase-2 analysis ----------------------------------------------------------------------------------------------------------------
if ! step_ok p2_report; then
  check_stop
  [ -f "$P1RES" ] || stop_sit p2_report 2 "$P1RES missing (S3: the Phase-1 analysis precedes the Phase-2 analysis)"
  [ ! -e "$P2OUT" ] || [ -z "$(ls -A "$P2OUT" 2>/dev/null)" ] || stop_sit p2_report 7 "$P2OUT exists and is not empty (manual decision)"
  RSHA=$(sp frzsha "$FRZ" tools/p2_report.py) || stop_sit p2_report 2 "no tools/p2_report.py sha in $FRZ"
  ( cd "$CODE" && $ENVPFX "$PY" -B "$TOOLS/p2_report.py" --split TEST --block "$P2BLOCK" --contrasts "$CONTRASTS" --out "$P2OUT" \
      --p2-lock "$LOCKF" --p2-freeze "$FRZ" --all-methods-lock "$AML" --test-exposure-register "$REG" --registry-ext "$X" \
      --phase1-result "$P1RES" --code-root "$RCR" --expect-script-sha256 "$RSHA" ) > "$P2LOG" 2>&1; rc=$?
  [ $rc -eq 0 ] && [ -f "$P2OUT/RETENTION_RESULT.json" ] || stop_sit p2_report $rc "p2_report (log $P2LOG; read after the sitting)"
  chmod -R a-w "$P2OUT" 2>/dev/null || true
  mark_ok p2_report "Phase-2 RETENTION_RESULT sha256 $(sha_of "$P2OUT/RETENTION_RESULT.json")"
fi
# 12 done ----------------------------------------------------------------------------------------------------------------------------
if ! step_ok done; then
  mark_ok done "P1 $(sha_of "$P1RES") P2 $(sha_of "$P2OUT/RETENTION_RESULT.json")"
fi
say "SITTING COMPLETE: both RETENTION_RESULT files exist (Phase 1 $(sha_of "$P1RES" | cut -c1-16), Phase 2 $(sha_of "$P2OUT/RETENTION_RESULT.json" | cut -c1-16)). Ops: DL row FIRST (shas only), then fetch (S3); then the register look entries."
exit 0
