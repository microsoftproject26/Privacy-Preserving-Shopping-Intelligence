set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
SRC=$V/runs_ds3/T3_A40
DST=$V/runs_ds3/T3_A40_r2
test ! -e "$DST"
mkdir -p "$DST/code" "$DST/contracts" "$DST/out"
cp -a "$SRC/code/." "$DST/code/"
cp -a "$SRC/contracts/PREG1_LOCK_VALUES.json" "$DST/contracts/"
export T3_BUNDLE_ROOT="$DST"
/home/jovyan/venvs/ppsi-gpu/bin/python -B - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['T3_BUNDLE_ROOT'])
p = root / 'code/BUNDLE_MANIFEST.json'
obj = json.loads(p.read_text())
obj['files'] = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in obj['files']}
p.write_text(json.dumps(obj, indent=1, sort_keys=True) + '\n', encoding='utf-8')
print('T3_A40_R2_BUNDLE', hashlib.sha256(p.read_bytes()).hexdigest(), len(obj['files']))
PY
