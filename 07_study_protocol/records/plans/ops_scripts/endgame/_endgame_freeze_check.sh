# _endgame_freeze_check.sh — sourced by E / F / G / H after _endgame_lib.sh + _endgame_ana_args.sh. FINAL r7 form.
# freeze_check <freeze file sha256 from its DL row>: the uploaded freeze is the registered one, and every frozen identity
# (tools, reeval, contrasts, §9 command text) still equals the deployed files. Sets RSHA = the frozen retention_report sha.
freeze_check() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]] || eg_refuse "freeze sha256 argument missing / malformed"
  need $FREEZE; [ "$(sha $FREEZE)" = "$1" ] || eg_refuse "server ANA_SCRIPT_FREEZE.json sha != the DL-registered $1"
  local txt; txt=$(ana_cmd_text)
  FREEZE="$FREEZE" CR="$CR" V="$V" CONTRASTS="$CONTRASTS" CMDTXT="$txt" $PY -B - <<'PY' >> "$LOG" 2>&1 || eg_refuse "freeze identities differ from the deployed files (see $LOG)"
import hashlib, json, os, sys
from pathlib import Path
E = os.environ; fz = json.loads(Path(E["FREEZE"]).read_text(encoding="utf-8")); V = Path(E["V"]); CR = Path(E["CR"])
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
bad = [rel for rel, h in fz["scripts"].items() if sha(V / rel) != h]
if sha(CR / "v4int/reeval.py") != fz["reeval_sha256"]: bad.append("reeval.py")
if sha(E["CONTRASTS"]) != fz["contrasts"]["file_sha256"]: bad.append("ANA_CONTRASTS_v4.json")
if str(CR) != fz["code_root"]: bad.append("code_root")
if hashlib.sha256((E["CMDTXT"] + "\n").encode()).hexdigest() != fz["section9_commands_sha256"]: bad.append("section9 command text")
print("FREEZE_CHECK", "OK" if not bad else f"MISMATCH {bad}"); sys.exit(1 if bad else 0)
PY
  RSHA=$($PY -B -c "import json,sys; print(json.load(open(sys.argv[1]))['retention_report_sha256'])" $FREEZE)
  [[ "$RSHA" =~ ^[0-9a-f]{64}$ ]] || eg_refuse "freeze has no retention_report_sha256"
}
