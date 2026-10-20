#!/usr/bin/env bash
# p2_lane_lib.sh - shared helpers of laneP2_RUN2.sh / laneP2_RUN3.sh / laneP2_CHAIN.sh (P2-CHAIN). SOURCED, never run.
# Same conventions as laneP2_RUN1.sh: plan mode by default, --launch, --gpus, --slots-per-gpu 1..16, write-once markers in
# $RUNS/_lane_state, value-free output (job name / GPU / rc / seconds), refuse while H_test.sh runs, resume with the launcher's own
# --resume, failure isolation, pid lock per lane. Code tree for new runs: $V/code_p2r2 (read-only). Runs root: $V/runs_p2.
# S8 (PREG2): nothing here opens a sealed/ file or prints a metric; MAIN completion is read from STATUS.json only.
# P2-SITTING: LANE_KIND is reset to 'training' here (an inherited value is ignored); only laneP2_SIT.sh sets it to 'scoring' AFTER
# sourcing, and only a scoring lane may run from the scoring tree code_p2r4 (training lanes: code_p2r2, unchanged).
set -u
export LC_ALL=C
LANE_KIND=training
V=${V:-/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4}
CODE=${CODE:-$V/code_p2r2}
PY=${PY:-/home/jovyan/venvs/ppsi-gpu/bin/python}
if [ -z "${JPY:-}" ]; then                # stdlib-only JSON / label helper (reads selection files; prints recipe fields only)
  for _c in python3 python "$PY"; do if "$_c" -c 'import json' > /dev/null 2>&1; then JPY=$_c; break; fi; done
  JPY=${JPY:-python3}
fi
DATA=${V4_DATASET_ROOT:-/home/jovyan/shared_data/rees46_nextitem_v1}
RUNS=${P2_RUNS:-$V/runs_p2}               # P2_RUNS exists only so the refusals can be tested
INIT=${P2_INIT:-$V/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026/PRETRAIN_FOR_FL_INIT.json}
SELDIR=${P2_SELDIR:-$V/contracts/fl}      # P2_WARM_SELECTION_A / _FINAL, P2_C1_SELECTION (the registered place; launch_fl binds FINAL by it)
SEL_A=$SELDIR/P2_WARM_SELECTION_A.json; SEL_FINAL=$SELDIR/P2_WARM_SELECTION_FINAL.json; SEL_C1=$SELDIR/P2_C1_SELECTION.json
ANCHOR=${P2_ANCHOR:-$RUNS/ANCHOR/REEVAL_A40/P2_ANCHOR_PRE_CALIB_CALIBRATION_s2026_CB}
HP=USER_HISTORY_L50
LAUNCHER_ARGS="--family SASREC --phase MAIN --recipe-id SMALL_SASREC_D64_B2 --history-policy $HP"
RECIPES=$RUNS/recipes; LOGS=$RUNS/logs; STATE=$RUNS/_lane_state; DRY=$RUNS/dry
ENVPFX="env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=$DATA"
GPUS="0 1"; PERGPU=16; MODE=plan

# ---- JSON / label helper (stdlib only). It prints hyper-parameters, labels and selection fields; never a metric. -----------------
IFS= read -r -d '' JH_PY <<'PYEOF' || true
import json, re, sys
def g(x): return format(float(x), "g")
def num(x): return None if x is None else (int(x) if float(x) == int(x) else float(x))
def norm(d):
    return {"server_opt": str(d["server_opt"]), "server_lr": float(d["server_lr"]), "client_lr": float(d["client_lr"]),
            "items": d.get("items") or "frozen", "item_lr_mult": None if d.get("item_lr_mult") is None else float(d["item_lr_mult"]),
            "upload": d.get("upload") or "dense", "lora_rank": None if d.get("lora_rank") is None else int(d["lora_rank"]),
            "lora_lr_factor": num(d.get("lora_lr_factor")), "personal_head": bool(d.get("personal_head", False))}
def tag(r):
    s = f"{r['server_opt'].upper()}_eta{g(r['server_lr'])}_c{g(r['client_lr'])}"
    if r["items"] == "unfrozen": s += f"_UNFROZEN_ilr{g(r['item_lr_mult'])}"
    if r["upload"] != "dense": s += f"_{r['upload'].upper()}"
    if r["lora_rank"] is not None: s += f"_LORA{int(r['lora_rank'])}x{g(r['lora_lr_factor'])}"
    if r["personal_head"]: s += "_PF"
    return s
def label(r):
    s = f"{r['server_opt']}|eta={g(r['server_lr'])}|c={g(r['client_lr'])}"
    if r["items"] == "unfrozen": s += f"|items=unfrozen|ilr={g(r['item_lr_mult'])}"
    if r["upload"] != "dense": s += f"|up={r['upload']}"
    if r["lora_rank"] is not None: s += f"|lora=r{int(r['lora_rank'])}|lrx={g(r['lora_lr_factor'])}"
    if r["personal_head"]: s += "|pf"
    return s
def kind(r):
    if r["lora_rank"] is not None or r["personal_head"]: return "wc"
    return "wb" if r["items"] == "unfrozen" else "wa"
def parse_label(s):
    m = re.match(r"^P2_C1_FT_CENTRAL_(FULL|FROZEN)\|lr=(\S+)$", s)
    if m: return ("C1", m.group(1), g(m.group(2)))
    p = s.split("|"); r = {"server_opt": p[0]}
    for x in p[1:]:
        if x.startswith("eta="): r["server_lr"] = float(x[4:])
        elif x.startswith("c="): r["client_lr"] = float(x[2:])
        elif x == "items=unfrozen": r["items"] = "unfrozen"
        elif x.startswith("ilr="): r["item_lr_mult"] = float(x[4:])
        elif x.startswith("up="): r["upload"] = x[3:]
        elif x.startswith("lora=r"): r["lora_rank"] = int(x[6:])
        elif x.startswith("lrx="): r["lora_lr_factor"] = float(x[4:])
        elif x == "pf": r["personal_head"] = True
        else: raise SystemExit("unparsable label")
    return norm(r)
def getp(obj, path):
    for k in path.split("."): obj = obj[k]
    return obj
c = sys.argv[1]
if c in ("tag", "label", "kind", "file"):
    r = norm(json.loads(sys.argv[2]))
    if c == "tag": print(tag(r))
    elif c == "label": print(label(r))
    elif c == "kind": print(kind(r))
    else: print(json.dumps({"format": "PPSI_V4_P2_RECIPE_1", "label": label(r), "recipe": r}))
elif c == "parse":
    r = parse_label(sys.argv[2])
    print(" ".join(r) if isinstance(r, tuple) else json.dumps(r))
elif c == "get":
    v = getp(json.load(open(sys.argv[2])), sys.argv[3])
    print(v if isinstance(v, str) else json.dumps(v))
elif c == "edge":
    t = open(sys.argv[2]).read(); i = t.find("only with them): ")
    if i < 0: raise SystemExit(1)
    print(";".join(re.sub(r"\s*\[[^\]]*\]\s*$", "", x.strip()) for x in t[i + 17:].strip().splitlines()[0].split("; ")))
else: raise SystemExit("unknown helper command")
PYEOF
jh() { "$JPY" -c "$JH_PY" "$@"; }

# ---- common option checks ---------------------------------------------------------------------------------------------------
lib_validate() {
  read -r -a GPUARR <<< "$GPUS"
  [ ${#GPUARR[@]} -ge 1 ] || { echo "refused: no GPU given" >&2; exit 2; }
  for g in "${GPUARR[@]}"; do case $g in ''|*[!0-9]*) echo "refused: bad GPU id '$g'" >&2; exit 2;; esac; done
  case $PERGPU in [1-9]|1[0-6]) ;; *) echo "refused: --slots-per-gpu must be 1..16" >&2; exit 2;; esac
  case $RUNS in */runs_ds3|*/runs_ds3/*|*/runs_bench*) echo "refused: the Phase-2 runs root must not be runs_ds3 or runs_bench*" >&2; exit 2;; esac
  if [ "$LANE_KIND" = scoring ]; then
    case $CODE in */code_p2r4|*/code_p2r4/) ;; *) echo "refused: CODE ($CODE) must be the Phase-2 scoring tree code_p2r4" >&2; exit 2;; esac
  else
    case $CODE in */code_p2r2|*/code_p2r2/) ;; *) echo "refused: CODE ($CODE) must be the Phase-2 tree code_p2r2" >&2; exit 2;; esac
  fi
}
psl() { ps -eo pid,args 2>/dev/null || ps -ef 2>/dev/null | awk '{c=$0; sub(/^ *[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +/,"",c); print $2, c}'; }
h_step_running() { psl | grep -E -e '[H]_test\.sh' | grep -v grep > /dev/null; }
tagf() { printf '%g' "$1"; }
ltag() { printf '%.0e' "$1" | sed 's/e-0/e-/'; }                 # = central_ft.lr_tag (3e-6, 1e-5, 1e-3 ...)
sha_of() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
say() { echo "[$(date -u +%FT%TZ)] $*"; }

# ---- job registry -------------------------------------------------------------------------------------------------------------
J_NAME=(); J_GPU=(); J_KIND=(); J_RD=(); J_NEED=(); J_PRE=(); J_CMD=(); J_REC=(); J_EST=(); J_NOTE=(); J_PART=(); J_SEL=(); J_BASEF=()
declare -A J_IDX
job() {  # name kind rundir needs(space separated job names) part est note
  local name=$1 idx=${#J_NAME[@]}
  [ -z "${J_IDX[$name]:-}" ] || { echo "duplicate job $name" >&2; exit 1; }
  J_IDX[$name]=$idx; J_GPU+=("${GPUARR[$(( idx % ${#GPUARR[@]} ))]}")
  J_NAME+=("$name"); J_KIND+=("$2"); J_RD+=("$3"); J_NEED+=("$4"); J_PART+=("$5"); J_EST+=("$6"); J_NOTE+=("$7")
  J_PRE+=(""); J_CMD+=(""); J_REC+=(""); J_SEL+=(1); J_BASEF+=("")
}

# ---- state helpers (value-free: grep -q on status words only) -----------------------------------------------------------------
run_done_calib() {      # as RUN-1: RUN_SUMMARY DONE_* and STATUS not RUNNING / FAILED
  local rd=$1
  [ -f "$rd/RUN_SUMMARY.json" ] || return 1
  grep -q '"status": *"DONE_\(MAIN\|CONTINUATION\)"' "$rd/RUN_SUMMARY.json" || return 1
  if [ -f "$rd/STATUS.json" ] && grep -q '"status": *"\(RUNNING\|FAILED\)"' "$rd/STATUS.json"; then return 1; fi
  return 0
}
run_done_main() { [ -f "$1/STATUS.json" ] && grep -q '"status": *"\(DONE_MAIN\|DONE_S_CAL\|DONE\|DONE_CONTINUATION\)"' "$1/STATUS.json"; }
run_started() { [ -f "$1/ARM_SPEC.json" ] || [ -f "$1/RUN_START.json" ] || [ -f "$1/STATUS.json" ] || [ -n "$(ls "$1"/LAUNCH*.json 2>/dev/null | head -n 1)" ]; }
kind_done() {           # index -> 0 when the job's run is finished on disk
  local i=$1 rd=${J_RD[$1]}
  case ${J_KIND[$i]} in
    calib) run_done_calib "$rd";;
    scal) run_done_main "$rd" && [ -f "$rd/S_RECORD_P2.json" ];;
    reeval) [ -f "$rd/REEVAL_RECORD.json" ] && grep -q '"status": *"COMPLETE"' "$rd/REEVAL_RECORD.json";;   # P2-SITTING scoring jobs
    *) run_done_main "$rd";;
  esac
}
job_state() { if kind_done "$1"; then echo done; elif run_started "${J_RD[$1]}"; then echo started; else echo new; fi; }

# ---- plan -------------------------------------------------------------------------------------------------------------------------
print_jobs() {   # same line shapes as RUN-1 (soft-after = needs)
  local i
  for i in "${!J_NAME[@]}"; do
    printf '%-30s gpu=%s soft-after=[%s] est=%s\n' "${J_NAME[$i]}" "${J_GPU[$i]}" "${J_NEED[$i]}" "${J_EST[$i]}"
    printf '    part: %s selected=%s\n' "${J_PART[$i]}" "${J_SEL[$i]}"
    [ -z "${J_REC[$i]}" ] || printf '    recipe: %s\n' "${J_REC[$i]}"
    [ -z "${J_PRE[$i]}" ] || printf '    dry: %s\n' "${J_PRE[$i]}"
    printf '    cmd: %s\n' "${J_CMD[$i]}"
    printf '    out: %s\n' "${J_RD[$i]}"
    printf '    state: %s (resume is automatic: a started run is continued with --resume)\n' "$(job_state "$i")"
    [ -z "${J_NOTE[$i]}" ] || printf '    note: %s\n' "${J_NOTE[$i]}"
  done
}

# ---- launch ---------------------------------------------------------------------------------------------------------------------------
# lane_launch LANE REQUIRED_FILE... : H-step / code-tree refusals, lock, scheduler over the SELECTED jobs (J_SEL=1).
write_recipe() {  # file json  (write-once; a different existing file is a refusal)
  local f=$1 j=$2
  if [ -e "$f" ]; then printf '%s\n' "$j" | cmp -s - "$f" || { echo "refused: recipe file $f exists with different content" >&2; exit 2; }; return 0; fi
  ( set -C; printf '%s\n' "$j" > "$f" ) || { echo "refused: cannot write recipe file $f" >&2; exit 2; }
  chmod a-w "$f" 2>/dev/null || true
}
ok_marker()   { ( set -C; echo "$2" > "$STATE/$1.ok" ) 2>/dev/null || true; }
fail_marker() { echo "$2" > "$STATE/$1.fail"; }
settled() { [ -f "$STATE/$1.ok" ] || [ -f "$STATE/$1.fail" ]; }
declare -A PIDG
used_gpu() { local p t=0; for p in "${!PIDG[@]}"; do if kill -0 "$p" 2>/dev/null; then [ "${PIDG[$p]}" = "$1" ] && t=$((t + 1)); else unset 'PIDG[$p]'; fi; done; echo $t; }
used_all() { local p t=0; for p in "${!PIDG[@]}"; do if kill -0 "$p" 2>/dev/null; then t=$((t + 1)); else unset 'PIDG[$p]'; fi; done; echo $t; }
want_yield() { [ -e "$STATE/STOP" ] && return 0; h_step_running && return 0; return 1; }

need_state() {   # job name -> ok | failed | wait   (in this lane: marker; otherwise on-disk state)
  local n=$1 j
  [ -f "$STATE/$n.ok" ] && { echo ok; return; }
  [ -f "$STATE/$n.fail" ] && { echo failed; return; }
  j=${J_IDX[$n]:-}
  if [ -n "$j" ] && kind_done "$j"; then echo ok; return; fi
  echo wait
}

run_job() {      # background subshell; output = names, rc, seconds only
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
  if run_started "${J_RD[$i]}" && ! kind_done "$i"; then cmd="$cmd --resume"; fi
  ( cd "$CODE" && eval "$cmd" ) >> "$log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && kind_done "$i"; then
    ok_marker "$name" "rc=0"; say "$name: ok rc=0 sec=$(( $(date +%s) - t0 ))"
  else
    [ $rc -ne 0 ] || rc=99                                    # rc 0 but not finished: a relaunch resumes it
    fail_marker "$name" "rc=$rc"; say "$name: FAILED rc=$rc sec=$(( $(date +%s) - t0 ))"
  fi
  echo "$name	$g	$rc	$(( $(date +%s) - t0 ))" >> "$LOGS/lane_summary.tsv"
  return 0
}

lane_launch() {  # lane_name required_code_file...
  local lane=$1; shift
  local f i n s
  if h_step_running; then echo "refused: an H step (H_test.sh) is running; Phase 2 never shares a GPU with an H step (PREG2 S7)" >&2; exit 2; fi
  [ -d "$CODE" ] || { echo "refused: the Phase-2 code tree $CODE is missing" >&2; exit 2; }
  for f in "$@"; do [ -f "$CODE/$f" ] || { echo "refused: $CODE/$f is missing" >&2; exit 2; }; done
  [ -f "$INIT" ] || { echo "refused: the PRE init record $INIT is missing" >&2; exit 2; }
  if [ -n "$(find "$CODE" -type f -perm /222 -print -quit 2>/dev/null)" ]; then echo "refused: $CODE is not read-only (a writable file exists); chmod -R a-w it after the review" >&2; exit 2; fi
  case $(uname -s) in MINGW*|MSYS*|CYGWIN*) ;; *)
    if [ -n "$(find "$CODE" -type d -perm /222 -print -quit 2>/dev/null)" ]; then echo "refused: $CODE is not read-only (a writable directory exists)" >&2; exit 2; fi;; esac
  # every need of a selected job must be selected too, or finished on disk (otherwise the lane would wait forever)
  for i in "${!J_NAME[@]}"; do
    [ "${J_SEL[$i]}" = 1 ] || continue
    for n in ${J_NEED[$i]}; do
      s=${J_IDX[$n]:-}
      if [ -z "$s" ] || { [ "${J_SEL[$s]}" != 1 ] && ! kind_done "$s"; }; then
        echo "refused: ${J_NAME[$i]} needs $n, which is neither in this lane's parts nor finished" >&2; exit 2
      fi
    done
  done
  mkdir -p "$LOGS" "$STATE" "$RECIPES" "$DRY" "$RUNS/WA" "$RUNS/WB" "$RUNS/WC" "$RUNS/C1" "$RUNS/MAIN"
  LOCK=$STATE/DRIVER_$lane.lock
  if ! mkdir "$LOCK" 2>/dev/null; then
    if kill -0 "$(cat "$LOCK/pid" 2>/dev/null || echo 0)" 2>/dev/null; then echo "refused: lane lock held by pid $(cat "$LOCK/pid")" >&2; exit 2; fi
    rm -rf "$LOCK"; mkdir "$LOCK" || { echo "refused: cannot take the lane lock" >&2; exit 2; }
  fi
  echo $$ > "$LOCK/pid"; trap '[ "$BASHPID" = "$$" ] && rm -rf "$LOCK"' EXIT
  lane_write_recipes
  export PYTHONDONTWRITEBYTECODE=1
  for i in "${!J_NAME[@]}"; do          # a relaunch retries this lane's failed jobs; old failure markers are kept aside
    [ "${J_SEL[$i]}" = 1 ] || continue
    f=$STATE/${J_NAME[$i]}.fail; [ -e "$f" ] && mv "$f" "$f.$(date +%s)"
  done
  local pending=() next=() yielded=0 st blocked nfail
  for i in "${!J_NAME[@]}"; do [ "${J_SEL[$i]}" = 1 ] && pending+=("$i"); done
  say "launching ${#pending[@]} jobs (lane $lane; GPUs: $GPUS; slots per GPU: $PERGPU; logs $LOGS)"
  while [ ${#pending[@]} -gt 0 ]; do
    if want_yield; then yielded=1; say "driver: an H step or a STOP file is present; launching nothing new, waiting for running jobs"; break; fi
    next=()
    for i in "${pending[@]}"; do
      local name=${J_NAME[$i]} g=${J_GPU[$i]}
      blocked=""; local wait=0
      for n in ${J_NEED[$i]}; do
        case $(need_state "$n") in failed) blocked=$n;; wait) wait=1;; esac
      done
      if [ -n "$blocked" ]; then fail_marker "$name" "blocked: needs $blocked"; say "$name: FAILED (blocked: needs $blocked) rc=1 sec=0"; continue; fi
      [ $wait = 0 ] || { next+=("$i"); continue; }
      if [ -n "${J_BASEF[$i]}" ] && [ ! -f "${J_BASEF[$i]}" ]; then fail_marker "$name" "base file missing"; say "$name: FAILED (base file missing) rc=1 sec=0"; continue; fi
      st=$(job_state "$i")
      if [ "$st" = done ]; then say "$name: already finished, skipped"; ok_marker "$name" "already finished"; continue; fi
      if [ "$(used_all)" -ge $(( PERGPU * ${#GPUARR[@]} )) ] || [ "$(used_gpu "$g")" -ge "$PERGPU" ]; then next+=("$i"); continue; fi
      if want_yield; then next+=("$i"); continue; fi
      run_job "$i" & PIDG[$!]=$g
    done
    pending=("${next[@]}")
    [ ${#pending[@]} -gt 0 ] && sleep "${POLL:-20}"
  done
  wait
  if [ "$yielded" = 1 ]; then say "driver stopped early: ${#pending[@]} jobs not launched; relaunch later (finished runs are skipped, started runs resume)."; exit 3; fi
  nfail=0
  for i in "${!J_NAME[@]}"; do [ "${J_SEL[$i]}" = 1 ] && [ -f "$STATE/${J_NAME[$i]}.fail" ] && nfail=$((nfail + 1)); done
  say "lane $lane finished; failed jobs: $nfail."
  [ "$nfail" -eq 0 ]
}
