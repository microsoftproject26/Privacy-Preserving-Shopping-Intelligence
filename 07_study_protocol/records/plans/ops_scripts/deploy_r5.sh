set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
D=$V/runs_ds3/deploy_r5
test ! -e "$V/code_ds3r5"; test ! -e "$D/extract"
echo "8654dbc1381a3eb4b9ca3efe9a714ec4580fd6629de7b74c8449bd5ea61cf137  $D/code_ds3r5.tar" | sha256sum -c -
export DEPLOY_ROOT="$D"
/home/jovyan/venvs/ppsi-gpu/bin/python -B - <<'PY'
import hashlib, json, os, tarfile
from pathlib import PurePosixPath, Path
d=Path(os.environ['DEPLOY_ROOT']); m=json.loads((d/'SYNC_MANIFEST.json').read_text())
assert m['tar_sha256']=='8654dbc1381a3eb4b9ca3efe9a714ec4580fd6629de7b74c8449bd5ea61cf137'
with tarfile.open(d/'code_ds3r5.tar') as t:
    for x in t:
        p=PurePosixPath(x.name)
        assert not p.is_absolute() and '..' not in p.parts, x.name
        assert x.isdir() or (x.isfile() and x.name in m['files']), x.name
print('TAR_NAMES_VERIFIED', len(m['files']))
PY
mkdir "$D/extract"; tar -xf "$D/code_ds3r5.tar" -C "$D/extract"
/home/jovyan/venvs/ppsi-gpu/bin/python -B - <<'PY'
import hashlib, json, os
from pathlib import Path
d=Path(os.environ['DEPLOY_ROOT']); m=json.loads((d/'SYNC_MANIFEST.json').read_text())
for rel, h in m['files'].items(): assert hashlib.sha256((d/'extract'/rel).read_bytes()).hexdigest()==h, rel
agg=hashlib.sha256(''.join(f'{k}\0{v}\n' for k,v in sorted(m['files'].items())).encode()).hexdigest()
assert agg==m['aggregate_sha256']; print('PACK_HASHES_VERIFIED', len(m['files']), agg[:8])
PY
mv "$D/extract/code" "$V/code_ds3r5"; chmod -R a-w "$V/code_ds3r5"
echo CODE_DS3R5_DEPLOYED
