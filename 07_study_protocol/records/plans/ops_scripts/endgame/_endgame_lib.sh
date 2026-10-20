# _endgame_lib.sh — shared guards for the REES46 end-game scripts (OPS-ENDGAME, 2026-10-02; FINAL r7 form OPS-ENDGAME-2).
# Sourced AFTER $Q/_common_code_ds3r7.sh (which defines V R Q CR PY ts step job wait_file fl dev and cd's into $CR).
# Server location: $V/runs_ds3/queues/endgame/_endgame_lib.sh. Every script sets LOG=$Q/endgame_<phase>.log first.
# Authority: reviews/FABLE_ENDGAME_REVIEW_20261001.md §1-§5; DL 315-321, 327, 331, 336, 342, 343, 351; PREG1_AMENDMENT_06P.
# EG_ROOT / ANA_ROOT override $Q/endgame / $R/ANA for the value-free validation harness ONLY (README §6); never in a phase.
[ -n "$CR" ] && [ -n "$LOG" ] || { echo "REFUSED_START: source _common_code_ds3r7.sh and set LOG first" >&2; exit 90; }
[ "$CR" = "$V/code_ds3r7" ] || { echo "REFUSED_START: CR=$CR is not code_ds3r7" >&2; exit 91; }
EG=${EG_ROOT:-$Q/endgame}; OK=$EG/ok; ANA=${ANA_ROOT:-$R/ANA}; mkdir -p "$OK" "$ANA"
ANA_TAG=${ANA_TAG:-v1}   # freeze tag: the VALIDATION dirs / markers of one freeze (a new freeze after a refusal = a new tag)
PIN=$V/contracts/eval_device/EVAL_DEVICE_PIN_A40.json
CONTRASTS=$V/company/gates/ANA_CONTRASTS_v4.json
CONTRASTS_SHA=f1769f0b8a0f11dc3861abdff02d9668449c3a47aa114462335462b653354515   # DL 336 (06P P1-P7 + ERRATUM 1)
# README Q19: freeze v1 = ANA_SCRIPT_FREEZE.json; a later freeze (new tag) = ANA_SCRIPT_FREEZE_<tag>.json (its own DL row).
# The lock name is FIXED: v4data09.d9_gate DEFAULT_LOCK = company/gates/ALL_METHODS_LOCK.json (MEASURED, d9_gate.py L42).
FREEZE=$V/company/gates/ANA_SCRIPT_FREEZE$([ "$ANA_TAG" = v1 ] || echo "_$ANA_TAG").json
LOCK=$V/company/gates/ALL_METHODS_LOCK.json
REGISTER=$V/company/registers/TEST_EXPOSURE_REGISTER.json
TOOLS=$V/tools
eg_refuse() { step "REFUSED_START $*"; echo "REFUSED_START: $*" >&2; exit 92; }
need()    { [ -e "$1" ] || eg_refuse "missing predecessor output $1"; }        # predecessor gate
once()    { [ ! -e "$1" ] || eg_refuse "already done ($1 exists); outputs are never overwritten"; }
confirm() { [ "$1" = "--i-confirm" ] || eg_refuse "this phase reads or locks evidence: re-run with --i-confirm as the LAST argument"; }
sha()     { sha256sum "$1" | cut -d' ' -f1; }
# ejob <name> <cmd...>: job() + stop on any REFUSED line in the job log (job() already stops on rc != 0)
ejob() { local n=$1; job "$@"; if grep -qE '(^|[^A-Z_])REFUSED([: ]|$)' "$Q/$n.log"; then step "LANE_STOP REFUSED in $n"; exit 97; fi; }
# status of a run dir (reads ONLY the 'status' key of RUN_SUMMARY.json; never a metric)
run_status() { [ -f "$1/RUN_SUMMARY.json" ] || { echo MISSING; return; }
  $PY -B -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" "$1/RUN_SUMMARY.json"; }
need_done() { local s; s=$(run_status "$1"); [ "$s" = DONE_MAIN ] || eg_refuse "$1 is $s, not DONE_MAIN"; }
# the one FL ckpt/<run_id>/ dir of a run (refuses 0 or >1); <run_id> = launch_fl's run id (CheckpointManager(root, run_id))
fl_ckdir() { local n; n=$(ls -d "$1"/ckpt/*/ 2>/dev/null | wc -l); [ "$n" -eq 1 ] || eg_refuse "$1/ckpt has $n run-id dirs"; ls -d "$1"/ckpt/*/ | sed 's#/$##'; }
# central sealed/CHECKPOINTS.json record -> sha256 of <best|endpoint_6.0> (MEASURED keys: best / endpoint_6.0, each
# {bytes, file, meta, sha256}); the reeval itself re-hashes the file against --expect-sha256 (reeval.run L969)
central_sha() { $PY -B -c "import json,sys; d=json.load(open(sys.argv[1]))[sys.argv[2]]; print(d['sha256'] if isinstance(d,dict) else d)" "$1" "$2"; }
# FL checkpoint record (built by seal_fl_ckpts) -> sha256 of a checkpoint path
fl_sha() { $PY -B -c "import json,sys; print(json.load(open(sys.argv[1]))['files'][sys.argv[2]]['file_sha256'])" "$1" "$2"; }
# seal_fl_ckpts <record.json> <ckpt.pt>... — README Q1 (RESOLVED). FL runs write no file-sha256 record; the run's own
# record is ckpt/<run_id>/events.jsonl, one line per save {bytes, content_digest, disk, saved, utc} (fl.checkpoint
# CheckpointManager._save, MEASURED on FL_T/FA_s2026). For each checkpoint, in this order:
#   (1) the file lives in ckpt/<run_id>/ of its run and the payload's run_id is that <run_id>;
#   (2) file bytes == the LAST events.jsonl entry whose "saved" is the file name;
#   (3) fl.checkpoint.load_checkpoint re-verifies payload_digest == __content_digest__, and that digest == the entry's
#       content_digest (so the file is exactly what the run saved last under that name);
#   (4) THEN the file sha256 is recorded, and it must equal every DEV LAUNCH start.file_sha256 whose start.path names the
#       same file (launch_device writes start {path, file_sha256, ...}; MEASURED on DEV/FT_FA_s2026, FT_C_s2026).
# The reeval then checks the same sha256 again (--expect-sha256). First call writes the record O_EXCL + a-w; a later call
# recomputes and must be identical.
seal_fl_ckpts() { local rec=$1; shift
  R="$R" EG_REC="$rec" $PY -B - "$@" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from fl.checkpoint import load_checkpoint
rec = Path(os.environ["EG_REC"]); R = Path(os.environ["R"])
def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
dev_bases = {}
for L in sorted((R / "DEV").glob("*/LAUNCH_*.json")):
    sb = (json.loads(L.read_text()).get("start") or {})
    if sb.get("path") and sb.get("file_sha256"):
        dev_bases.setdefault(str(Path(sb["path"]).resolve()), set()).add((sb["file_sha256"], str(L)))
files = {}
for a in sys.argv[1:]:
    p = Path(a).resolve()
    if p.parent.parent.name != "ckpt":
        sys.exit(f"REFUSED: {p} is not in a <run>/ckpt/<run_id>/ dir")
    ev = [json.loads(x) for x in (p.parent / "events.jsonl").read_text().splitlines() if x.strip()]
    last = [e for e in ev if e.get("saved") == p.name]
    if not last: sys.exit(f"REFUSED: no events.jsonl entry for {p}")
    last = last[-1]
    if p.stat().st_size != int(last["bytes"]): sys.exit(f"REFUSED: {p} bytes differ from events.jsonl")
    obj = load_checkpoint(p)                                  # verifies payload_digest == __content_digest__
    if obj.get("run_id") != p.parent.name:
        sys.exit(f"REFUSED: {p} payload run_id {obj.get('run_id')!r} is not its ckpt dir {p.parent.name}")
    if obj.get("__content_digest__") != last["content_digest"]:
        sys.exit(f"REFUSED: {p} content digest differs from the run's events.jsonl record")
    s = fsha(p)
    for want, src in dev_bases.get(str(p), ()):
        if want != s: sys.exit(f"REFUSED: {p} sha256 differs from {src} start.file_sha256")
    files[str(p)] = {"file_sha256": s, "bytes": int(last["bytes"]), "content_digest": last["content_digest"],
                     "events_utc": last.get("utc"), "dev_launch_crosschecks": sorted(x[1] for x in dev_bases.get(str(p), ()))}
    del obj
doc = {"format": "PPSI_V4_ENDGAME_FL_CKPT_RECORD_1",
       "rule": "ckpt/<run_id> + payload run_id; events.jsonl bytes + content_digest verified; then file sha256 (= DEV start.file_sha256)",
       "files": files}
if rec.exists():
    old = json.loads(rec.read_text())
    for k, v in files.items():
        if old["files"].get(k, {}).get("file_sha256") != v["file_sha256"]:
            sys.exit(f"REFUSED: {k} differs from the sealed record {rec}")
    print("FL_CKPT_RECORD_REVERIFIED", len(files))
else:
    fd = os.open(rec, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as f: json.dump(doc, f, indent=1, sort_keys=True)
    print("FL_CKPT_RECORD_WRITTEN", len(files))
PY
}
# tree_digest <dir>: sha256 over sorted "relpath\0sha256\n" of every file (paths + hashes only)
tree_digest() { (cd "$1" && find . -type f | LC_ALL=C sort | while IFS= read -r f; do
  printf '%s\0%s\n' "${f#./}" "$(sha256sum "$f" | cut -d' ' -f1)"; done | sha256sum | cut -d' ' -f1); }
# K16 (DL 343): laneD3_INT8_r7 writes $R/ONNX_R7 (full MAIN) or, after the MASTER_PLAN 8.2 2-h fallback, $R/ONNX_R7_U16384
# (--first-users 16384). The r6 $R/ONNX (view mismatch, DL 320) is never indexed. Prints the one dir holding
# BENCH_RESULT.json, or nothing (K16 not run); refuses when both hold one.
onnx_val_dir() { local a=$R/ONNX_R7/BENCH_RESULT.json b=$R/ONNX_R7_U16384/BENCH_RESULT.json
  if [ -f "$a" ] && [ -f "$b" ]; then echo BOTH; return; fi         # the caller refuses (a subshell cannot)
  if [ -f "$b" ]; then echo "$R/ONNX_R7_U16384"; elif [ -f "$a" ]; then echo "$R/ONNX_R7"; fi; }
# smoke-named dirs under the indexed roots: ana_block_index refuses the WHOLE build on any of them
# (check_smoke_name, SMOKE_WORDS SMOKE / LRSCREEN / LR_SCREEN / CALIB; L320, applied to every --reeval / --fl / --dev dir)
smoke_dirs() { local r d; for r in "$@"; do for d in "$r"/*/; do [ -d "$d" ] || continue; d=$(basename "$d")
  case "${d^^}" in *SMOKE*|*LRSCREEN*|*LR_SCREEN*|*CALIB*) echo "$r/$d";; esac; done; done; }
