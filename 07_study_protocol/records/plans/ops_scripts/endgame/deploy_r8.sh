set -euo pipefail
# SERVER: deploy tools_r8 (retention_report ANA-07 d0247370…, DL 390). r7 tools archived to tools/_archive_r7, never deleted.
# Upload tools_r8.tar + TOOLS_MANIFEST.json to $D first. Output: $D/R8_DEPLOYED.ok.
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
D=$V/runs_ds3/deploy_r8; Q=$V/runs_ds3/queues; PY=/home/jovyan/venvs/ppsi-gpu/bin/python
LOG=$Q/endgame_C.log; echo "$(date -u +%FT%TZ) PHASE_START deploy_r8" >> $LOG
test ! -e "$D/extract_tools"; test ! -e "$D/R8_DEPLOYED.ok"; test ! -e "$V/tools/_archive_r7"
echo "cd35d9e1755d33a6b4811eef07c6f1c4b35192f5bbbf290860867a47f88cec45  $D/tools_r8.tar" | sha256sum -c -
export D V
$PY -B - <<'PY'
import json, os, tarfile, hashlib
from pathlib import PurePosixPath, Path
d = Path(os.environ['D']); v = Path(os.environ['V'])
m = json.loads((d/'TOOLS_MANIFEST.json').read_text()); assert m['tar_sha256'].startswith('cd35d9e1')
with tarfile.open(d/'tools_r8.tar') as t:
    for x in t:
        p = PurePosixPath(x.name); assert not p.is_absolute() and '..' not in p.parts, x.name
        assert x.isdir() or (x.isfile() and x.name in m['files']), x.name
r7 = json.loads((v/'runs_ds3/deploy_r7/TOOLS_MANIFEST.json').read_text())['files']
for rel, h in r7.items():
    assert hashlib.sha256((v/rel).read_bytes()).hexdigest() == h, f'{rel} on the server is not the r7 tool'
print('R7_TOOLS_VERIFIED', len(r7))
PY
mkdir "$D/extract_tools"; tar -xf "$D/tools_r8.tar" -C "$D/extract_tools"
$PY -B - <<'PY'
import hashlib, json, os
from pathlib import Path
d = Path(os.environ['D']); m = json.loads((d/'TOOLS_MANIFEST.json').read_text())
for rel, h in m['files'].items(): assert hashlib.sha256((d/'extract_tools'/rel).read_bytes()).hexdigest() == h, rel
agg = hashlib.sha256(''.join(f'{k}\0{v}\n' for k, v in sorted(m['files'].items())).encode()).hexdigest()
assert agg == m['aggregate_sha256']; print('PACK_HASHES_VERIFIED', len(m['files']), agg[:8])
PY
mkdir "$V/tools/_archive_r7"
cp -p "$V/tools/retention_report.py" "$V/tools/_archive_r7/retention_report.py"
chmod u+w "$V/tools/retention_report.py"; cp "$D/extract_tools/tools/retention_report.py" "$V/tools/retention_report.py"; chmod a-w "$V/tools/retention_report.py"
chmod -R a-w "$V/tools/_archive_r7"
sha256sum $V/tools/ana_block_index.py $V/tools/retention_report.py $V/tools/test_exposure_register.py $V/tools/_archive_r7/retention_report.py | cut -c1-16
echo "d0247370f58ead3654bd0024211ae3718918b858f9bba358666f8415e2157035  $V/tools/retention_report.py" | sha256sum -c -
date -u +%FT%TZ > $D/R8_DEPLOYED.ok
echo "$(date -u +%FT%TZ) TOOLS_R8_DEPLOYED" >> $LOG; echo TOOLS_R8_DEPLOYED
