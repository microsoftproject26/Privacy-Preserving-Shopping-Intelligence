set -euo pipefail
# template: __CODE_SHA__ / __TOOLS_SHA__ are filled from deploy_archive/r6 manifests before upload (sed)
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
D=$V/runs_ds3/deploy_r6
test ! -e "$V/code_ds3r6"; test ! -e "$D/extract"
test ! -e "$V/tools/ana_block_index.py"; test ! -e "$V/tools/retention_report.py"
echo "__CODE_SHA__  $D/code_ds3r6.tar" | sha256sum -c -
echo "__TOOLS_SHA__  $D/tools_r6.tar" | sha256sum -c -
export DEPLOY_ROOT="$D"
/home/jovyan/venvs/ppsi-gpu/bin/python -B - <<'PY'
import json, os, tarfile
from pathlib import PurePosixPath, Path
d = Path(os.environ['DEPLOY_ROOT'])
for tn, mn, sha in (('code_ds3r6.tar', 'SYNC_MANIFEST.json', '__CODE_SHA__'), ('tools_r6.tar', 'TOOLS_MANIFEST.json', '__TOOLS_SHA__')):
    m = json.loads((d/mn).read_text()); assert m['tar_sha256'] == sha, mn
    with tarfile.open(d/tn) as t:
        for x in t:
            p = PurePosixPath(x.name)
            assert not p.is_absolute() and '..' not in p.parts, x.name
            assert x.isdir() or (x.isfile() and x.name in m['files']), x.name
    print('TAR_NAMES_VERIFIED', tn, len(m['files']))
PY
mkdir "$D/extract" "$D/extract_tools"
tar -xf "$D/code_ds3r6.tar" -C "$D/extract"; tar -xf "$D/tools_r6.tar" -C "$D/extract_tools"
/home/jovyan/venvs/ppsi-gpu/bin/python -B - <<'PY'
import hashlib, json, os
from pathlib import Path
d = Path(os.environ['DEPLOY_ROOT'])
for ex, mn in (('extract', 'SYNC_MANIFEST.json'), ('extract_tools', 'TOOLS_MANIFEST.json')):
    m = json.loads((d/mn).read_text())
    for rel, h in m['files'].items(): assert hashlib.sha256((d/ex/rel).read_bytes()).hexdigest() == h, rel
    agg = hashlib.sha256(''.join(f'{k}\0{v}\n' for k, v in sorted(m['files'].items())).encode()).hexdigest()
    assert agg == m['aggregate_sha256']; print('PACK_HASHES_VERIFIED', mn, len(m['files']), agg[:8])
PY
mv "$D/extract/code" "$V/code_ds3r6"; chmod -R a-w "$V/code_ds3r6"
for f in ana_block_index.py retention_report.py; do cp "$D/extract_tools/tools/$f" "$V/tools/$f"; chmod a-w "$V/tools/$f"; done
sha256sum "$V/tools/ana_block_index.py" "$V/tools/retention_report.py" | cut -c1-16
sed 's#CR=$V/code_ds3r5#CR=$V/code_ds3r6#' $V/runs_ds3/queues/_common_code_ds3r5.sh > $V/runs_ds3/queues/_common_code_ds3r6.sh
grep -c code_ds3r6 $V/runs_ds3/queues/_common_code_ds3r6.sh
echo CODE_DS3R6_DEPLOYED
