set -euo pipefail
# FIX-1 (SERVER): deploy code_ds3r7b BESIDE code_ds3r7 (r7 untouched; tools/ untouched; no new _common file — only H
# `onnx` uses r7b, by cd + PYTHONPATH, after verifying it against the lock). Model: deploy_r7.tmpl.sh.
# Template: __CODE_SHA__ = deploy_archive/r7b/SYNC_MANIFEST.json tar_sha256 (filled by sed before upload); upload
# code_ds3r7b.tar + SYNC_MANIFEST.json to $R/deploy_r7b/ first. Output: $R/deploy_r7b/R7B_DEPLOYED.ok (traj_cmp7b predecessor).
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
D=$V/runs_ds3/deploy_r7b; Q=$V/runs_ds3/queues; PY=/home/jovyan/venvs/ppsi-gpu/bin/python
LOG=$Q/endgame_C.log; echo "$(date -u +%FT%TZ) PHASE_START C deploy_r7b" >> $LOG
test -e "$V/code_ds3r7"; test ! -e "$V/code_ds3r7b"; test ! -e "$D/extract"; test ! -e "$D/R7B_DEPLOYED.ok"
echo "__CODE_SHA__  $D/code_ds3r7b.tar" | sha256sum -c -
export DEPLOY_ROOT="$D" V
$PY -B - <<'PY'
import hashlib, json, os, tarfile
from pathlib import PurePosixPath, Path
d = Path(os.environ['DEPLOY_ROOT']); v = Path(os.environ['V'])
m = json.loads((d/'SYNC_MANIFEST.json').read_text()); assert m['tar_sha256'] == '__CODE_SHA__'
with tarfile.open(d/'code_ds3r7b.tar') as t:
    for x in t:
        p = PurePosixPath(x.name)
        assert not p.is_absolute() and '..' not in p.parts, x.name
        assert x.isdir() or (x.isfile() and x.name in m['files']), x.name
m7 = json.loads((v/'runs_ds3/deploy_r7/SYNC_MANIFEST.json').read_text())
assert m['base']['aggregate_sha256'] == m7['aggregate_sha256'], 'r7b base is not the deployed r7'
diff = sorted(k for k in set(m['files']) | set(m7['files']) if m['files'].get(k) != m7['files'].get(k))
assert diff == sorted(m['delta_reviewed']), diff
for k, h in m['delta_reviewed'].items(): assert m['files'][k] == h, k
for k, h in m7['files'].items():                        # the deployed r7 tree is still its manifest
    if k not in m['delta_reviewed']:
        assert hashlib.sha256((v/'code_ds3r7'/k[len('code/'):]).read_bytes()).hexdigest() == h, k
print('R7B_MANIFEST_VERIFIED', len(m['files']), 'delta', len(diff))
PY
mkdir "$D/extract"; tar -xf "$D/code_ds3r7b.tar" -C "$D/extract"
$PY -B - <<'PY'
import hashlib, json, os
from pathlib import Path
d = Path(os.environ['DEPLOY_ROOT']); m = json.loads((d/'SYNC_MANIFEST.json').read_text())
for rel, h in m['files'].items(): assert hashlib.sha256((d/'extract'/rel).read_bytes()).hexdigest() == h, rel
agg = hashlib.sha256(''.join(f'{k}\0{v}\n' for k, v in sorted(m['files'].items())).encode()).hexdigest()
assert agg == m['aggregate_sha256']; print('PACK_HASHES_VERIFIED', len(m['files']), agg[:8])
PY
mv "$D/extract/code" "$V/code_ds3r7b"; chmod -R a-w "$V/code_ds3r7b"
(cd $V/code_ds3r7b && PYTHONPATH=$V/code_ds3r7b PYTHONDONTWRITEBYTECODE=1 $PY -B -c "import v4eval; s=v4eval.evaluator_code_sha256(); print('EVALUATOR', s); assert s.startswith('61b64e61')")
date -u +%FT%TZ > $D/R7B_DEPLOYED.ok
echo "$(date -u +%FT%TZ) CODE_DS3R7B_DEPLOYED" >> $LOG; echo CODE_DS3R7B_DEPLOYED
