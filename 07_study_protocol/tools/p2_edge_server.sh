#!/usr/bin/env bash
# P2-14 / owner decision O8: SERVER_BROWSER cross-version Edge benchmark on the Linux A40 server (no root).
# Label: SERVER_BROWSER = reproducible cross-version browser benchmark on SERVER CPUs; NOT a consumer device.
# Value-free w.r.t. accuracy (latency / memory only). Results go only to the loopback sink / local files; never uploaded;
# never touches TEST data.
#
# Default = PLAN mode (prints what it would do, changes nothing). `--run` executes. `--list` prints the Edge versions
# the Microsoft repo pool offers (needs network; read-only).
#
# Usage:  tools/p2_edge_server.sh [--run | --list] [--site DIR] [--out DIR] [--cpus 0-3] [--port 8799]
#   --site  directory served by serve_coi.py; must contain p2_bench.html, p2_bench.js, queries.json,
#           models/fp32.onnx, models/int8.onnx, and (offline server) vendor/ort.min.js + the pinned ort-wasm* files.
#   --out   results dir (default ./p2_edge_results): results/<tag>.json, console logs, MANIFEST.tsv, ENV.txt
#
# Design (REV-P2-14 fixes):
#  * One model per page load AND one browser launch per (version, model, arm, session): every cold start is cold.
#  * threads= is passed explicitly to wasm-mt and equals the number of CPUs in --cpus (taskset pin, same for all versions).
#  * Every run is its own process group (setsid); the whole group is killed (TERM, then KILL) plus a pkill on the per-run
#    --user-data-dir; the script asserts no msedge of that profile remains before the next run.
#  * Liveness: if msedge dies, the run is aborted at once and counted FAILED. An error JSON counts as FAILED, never OK.
#    A version whose ldd report lists missing libs is skipped (reported, nothing installed).
#  * Capture: loopback POST sink (serve_coi.py --results-dir) is primary; the console marker P2_DEVICE_BENCH_JSON: in the
#    Chromium stderr log (--enable-logging=stderr) is the fallback.
#  * Sessions are interleaved: session outer loop; version order and the model/arm order are rotated per session.
#  * Browser flags include --no-sandbox (JupyterHub container, no root); it is recorded in ENV.txt.

set -u

# ---- pinned versions (edit; the exact URL is built from the version). ----------------------------------------------
EDGE_POOL="https://packages.microsoft.com/repos/edge/pool/main/m/microsoft-edge-stable"
EDGE_VERSIONS=(
  "154.0.4258.53-1"   # the owner's laptop version (D1): primary comparison point
  "152.0.4191.66-1"   # last 152 build in the repo pool (MEASURED by Ops 2026-10-03, DL 416)
  "150.0.4078.105-1"  # last 150 build in the repo pool (MEASURED, DL 416)
  "146.0.3856.97-1"   # last 146 build in the repo pool (MEASURED, DL 416)
)
# URL form: ${EDGE_POOL}/microsoft-edge-stable_<version>_amd64.deb

MODE=plan; SITE="./p2_site"; OUT="./p2_edge_results"; CPUS="0-3"; PORT=8799; TIMEOUT_S=1800
NTIMED=1000; WARMUP=20; SESSIONS=3; COOLDOWN_S=120
# model:arm combinations; the order is rotated by session number
COMBOS=("fp32:wasm-1t" "int8:wasm-1t" "fp32:wasm-mt" "int8:wasm-mt")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE_COI="${HERE}/../code/v4dev/browser/serve_coi.py"

while [ $# -gt 0 ]; do
  case "$1" in
    --run) MODE=run ;;
    --list) MODE=list ;;
    --site) SITE="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --cpus) CPUS="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    -h|--help) sed -n 1,30p "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '%s\n' "$*"; }
SAFE_RE='^[A-Za-z0-9_./-]+$'
for v in "$SITE" "$OUT" "$CPUS"; do
  printf '%s' "$v" | grep -Eq '^[A-Za-z0-9_./,-]+$' || { say "unsafe characters in argument: $v" >&2; exit 2; }
done

if [ "$MODE" = list ]; then
  say "Edge stable versions in the pool (read-only listing):"
  curl -fsSL "${EDGE_POOL}/" | grep -o 'microsoft-edge-stable_[^"]*_amd64.deb' | sort -uV
  exit 0
fi

# number of CPUs in the taskset list ("0-3" -> 4, "0-3,8" -> 5): wasm-mt threads= is set to this
count_cpus() {
  local total=0 part a b
  IFS=',' read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then a="${part%-*}"; b="${part#*-}"; total=$((total + b - a + 1)); else total=$((total + 1)); fi
  done
  echo "$total"
}
THREADS="$(count_cpus "$CPUS")"

say "mode=$MODE site=$SITE out=$OUT cpus=$CPUS threads(wasm-mt)=$THREADS port=$PORT sessions=$SESSIONS n=$NTIMED warmup=$WARMUP"
say "label=SERVER_BROWSER (server CPU; not a consumer device); combos: ${COMBOS[*]} (rotated per session)"
say "versions: ${EDGE_VERSIONS[*]}"

DEBS="${OUT}/debs"; EXTRACT="${OUT}/edge"; RES="${OUT}/results"; SINK="${RES}/sink"
MANIFEST="${OUT}/MANIFEST.tsv"
run() { if [ "$MODE" = run ]; then say "+ $*"; "$@"; else say "[plan] $*"; fi; }

# ---- 1. download + sha256 manifest (append) + extract without root --------------------------------------------------
run mkdir -p "$DEBS" "$EXTRACT" "$RES" "$SINK"
PKGS_URL="https://packages.microsoft.com/repos/edge/dists/stable/main/binary-amd64/Packages.gz"
PKGS="${OUT}/Packages.gz"
if [ "$MODE" = run ]; then
  curl -fsSL --retry 3 -o "$PKGS" "$PKGS_URL" || { say "cannot fetch $PKGS_URL; refusing to run unverified binaries"; exit 4; }
  [ -s "$MANIFEST" ] || printf 'version	url	sha256_actual	sha256_repo	bytes
' > "$MANIFEST"
else
  say "[plan] curl -fsSL -o $PKGS $PKGS_URL   (repo SHA256 per version; mismatch => refuse)"
fi
declare -A SKIP
for V in "${EDGE_VERSIONS[@]}"; do
  URL="${EDGE_POOL}/microsoft-edge-stable_${V}_amd64.deb"
  DEB="${DEBS}/microsoft-edge-stable_${V}_amd64.deb"
  say "== $V  $URL"
  if [ "$MODE" = run ]; then
    [ -s "$DEB" ] || curl -fL --retry 3 -o "$DEB" "$URL" || say "DOWNLOAD FAILED for $V"
    if [ -s "$DEB" ]; then
      VOUT="$(python3 "${HERE}/p2_packages_sha.py" verify "$PKGS" "$V" "$DEB")"; VRC=$?
      REPO_SHA="$(printf '%s' "$VOUT" | sed -n 's/.*repo=\([^ ]*\) .*/\1/p')"
      ACT_SHA="$(printf '%s' "$VOUT" | sed -n 's/.*actual=\([^ ]*\).*/\1/p')"
      printf '%s	%s	%s	%s	%s
' "$V" "$URL" "$ACT_SHA" "$REPO_SHA" "$(stat -c %s "$DEB")" >> "$MANIFEST"
      if [ "$VRC" -ne 0 ]; then
        say "SHA256 MISMATCH or version absent from Packages for $V (${VOUT}); REFUSING this .deb"; rm -f "$DEB"; SKIP[$V]=1
      else
        mkdir -p "${EXTRACT}/${V}" && dpkg-deb -x "$DEB" "${EXTRACT}/${V}"
      fi
    else
      SKIP[$V]=1
    fi
  else
    say "[plan] curl -fL --retry 3 -o $DEB $URL ; p2_packages_sha.py verify (refuse on mismatch) ; manifest ; dpkg-deb -x $DEB ${EXTRACT}/${V}"
  fi
done

# ---- 2. missing shared libs: report only (never apt install); versions with missing libs are skipped ----------------
for V in "${EDGE_VERSIONS[@]}"; do
  [ -n "${SKIP[$V]:-}" ] && continue
  B="${EXTRACT}/${V}/opt/microsoft/msedge/msedge"
  if [ "$MODE" = run ]; then
    if [ ! -x "$B" ]; then say "MISSING BINARY for $V"; SKIP[$V]=1; continue; fi
    MISSING="$(LD_LIBRARY_PATH="$(dirname "$B")" ldd "$B" 2>&1 | grep 'not found' || true)"
    printf '%s\n' "${MISSING:-none}" > "${RES}/${V}.ldd_missing.txt"
    say "ldd missing libs for $V: ${MISSING:-none}"
    [ -n "$MISSING" ] && { say "SKIPPING $V (missing shared libs; no root, not installing)"; SKIP[$V]=1; }
    "$B" --version > "${RES}/${V}.msedge_version.txt" 2>&1 || true
  else
    say "[plan] ldd ${B} | grep 'not found'  (report; skip version if any; no install); ${B} --version"
  fi
done

# ---- 3. loopback result sink + static server --------------------------------------------------------------------------
SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; }
if [ "$MODE" = run ]; then
  if curl -sf "http://127.0.0.1:${PORT}/" > /dev/null 2>&1; then say "port $PORT already serves something; refusing"; exit 3; fi
  python3 "$SERVE_COI" --root "$SITE" --host 127.0.0.1 --port "$PORT" --results-dir "$SINK" > "${OUT}/serve_coi.log" 2>&1 &
  SRV_PID=$!
  trap cleanup EXIT
  sleep 2
  curl -sf -o /dev/null "http://127.0.0.1:${PORT}/p2_bench.html" || { say "static server not reachable / p2_bench.html missing in $SITE"; exit 3; }
else
  say "[plan] python3 $SERVE_COI --root $SITE --host 127.0.0.1 --port $PORT --results-dir $SINK &   (loopback-only sink)"
fi

# ---- 4. helpers ---------------------------------------------------------------------------------------------------------
classify() {  # $1 = sink json (may not exist), $2 = chromium stderr log, $3 = out json ; prints OK | ERR:<msg> | NONE
  python3 - "$1" "$2" "$3" <<'PY'
import json, os, sys
sink, log, out = sys.argv[1:4]
doc = None
if os.path.exists(sink):
    try:
        doc = json.load(open(sink))
    except Exception:
        doc = None
if doc is None and os.path.exists(log):
    marker = "P2_DEVICE_BENCH_JSON:"
    for line in open(log, errors="replace"):
        i = line.find(marker)
        if i < 0:
            continue
        body = line[i + len(marker):]
        j = body.rfind('", source:')
        if j >= 0:
            body = body[:j]
        try:
            doc = json.loads(body.rstrip().rstrip('"'))
            break
        except Exception:
            continue
if doc is None:
    print("NONE")
elif doc.get("error") or not doc.get("models"):
    print("ERR:" + str(doc.get("error", "no models")))
else:
    json.dump(doc, open(out, "w"), indent=1)
    print("OK")
PY
}

kill_group() {  # $1 = pgid, $2 = profile dir
  kill -TERM -- "-$1" 2>/dev/null; sleep 2; kill -KILL -- "-$1" 2>/dev/null
  pkill -KILL -f -- "--user-data-dir=$2" 2>/dev/null
  sleep 1
  if pgrep -f -- "--user-data-dir=$2" > /dev/null; then say "WARNING: msedge processes survive for $2"; return 1; fi
  return 0
}

N_OK=0; N_FAIL=0
run_one() {  # version session model arm
  local V="$1" S="$2" M="$3" ARM="$4"
  local B="${EXTRACT}/${V}/opt/microsoft/msedge/msedge"
  local TAG="${V}_${M}_${ARM}_s${S}"
  local PROFILE="${OUT}/profiles/${TAG}" LOG="${RES}/${TAG}.console.log"
  local MODEL_DESC="$(printf '%s' "${MODEL_DESC_BASE}" | tr -c 'A-Za-z0-9._/ -' '_'); edge ${V}; cpus ${CPUS}"
  local ENCDESC="${MODEL_DESC// /%20}"
  local THR=1; [ "$ARM" = wasm-mt ] && THR="$THREADS"
  local URLQ="http://127.0.0.1:${PORT}/p2_bench.html?auto=1&arm=${ARM}&threads=${THR}&model=${M}&session=${S}&label=SERVER_BROWSER&id=S-${V}&model_desc=${ENCDESC}&power=server&sink=${TAG}&vendor=1&fp32=models/fp32.onnx&int8=models/int8.onnx&queries=queries.json&n=${NTIMED}&warmup=${WARMUP}"
  local FLAGS=(--headless=new --no-sandbox --disable-gpu --disable-dev-shm-usage --no-first-run --no-default-browser-check
               --disable-background-networking --disable-component-update --disable-sync --disable-extensions
               --password-store=basic "--user-data-dir=${PROFILE}" --enable-logging=stderr --v=0)
  if [ "$MODE" != run ]; then
    say "[plan] setsid taskset -c ${CPUS} ${B} ${FLAGS[*]} '${URLQ}'  # one launch, timeout ${TIMEOUT_S}s"
    say "[plan]   wait for ${SINK}/${TAG}.json (or console marker); abort if msedge dies; kill process group; cool-down ${COOLDOWN_S}s"
    return
  fi
  rm -f "${SINK}/${TAG}.json"; mkdir -p "$PROFILE"
  say "+ ${TAG}  (threads=${THR})"
  setsid taskset -c "$CPUS" "$B" "${FLAGS[@]}" "$URLQ" > "$LOG" 2>&1 &
  local PID=$! PGID t=0 state=WAIT
  PGID="$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')"; PGID="${PGID:-$PID}"
  while [ "$t" -lt "$TIMEOUT_S" ]; do
    if [ -s "${SINK}/${TAG}.json" ]; then state=DONE; break; fi
    if grep -q 'P2_DEVICE_BENCH_JSON:' "$LOG" 2>/dev/null; then sleep 3; state=DONE; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then state=DIED; break; fi    # liveness: msedge exited before producing a result
    sleep 1; t=$((t + 1))
  done
  [ "$state" = WAIT ] && state=TIMEOUT
  kill_group "$PGID" "$PROFILE" || N_FAIL=$((N_FAIL + 1))
  local RESULT
  if [ "$state" = DONE ]; then RESULT="$(classify "${SINK}/${TAG}.json" "$LOG" "${RES}/${TAG}.DEVICE_BENCH_P2.json")"
  else RESULT="ERR:${state}"; fi
  case "$RESULT" in
    OK) N_OK=$((N_OK + 1)); say "  OK -> ${RES}/${TAG}.DEVICE_BENCH_P2.json" ;;
    *) N_FAIL=$((N_FAIL + 1)); say "  FAILED (${RESULT}); see ${LOG}" ;;
  esac
  rm -rf "$PROFILE"
  sleep "$COOLDOWN_S"
}

MODEL_DESC_BASE="$( (grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //') 2>/dev/null || echo unknown-cpu )"

# ---- 5. interleaved sessions: session outer; version order and combo order rotated per session --------------------------
NV=${#EDGE_VERSIONS[@]}; NC=${#COMBOS[@]}
for S in $(seq 1 "$SESSIONS"); do
  for ((i = 0; i < NV; i++)); do
    V="${EDGE_VERSIONS[$(( (i + S - 1) % NV ))]}"
    [ -n "${SKIP[$V]:-}" ] && continue
    for ((j = 0; j < NC; j++)); do
      C="${COMBOS[$(( (j + S - 1) % NC ))]}"
      run_one "$V" "$S" "${C%%:*}" "${C##*:}"
    done
  done
done

say "done: OK=${N_OK} FAILED=${N_FAIL}. Manifest: ${MANIFEST}; results: ${RES}"
if [ "$MODE" = run ]; then
  { echo "flags: --headless=new --no-sandbox (container, no root) taskset -c ${CPUS} wasm-mt threads=${THREADS}"
    uname -a; lscpu; uptime
    echo "vendor sha256:"; (cd "$SITE" && sha256sum vendor/* 2>/dev/null); } > "${OUT}/ENV.txt" 2>&1
fi
[ "$N_FAIL" -eq 0 ]
