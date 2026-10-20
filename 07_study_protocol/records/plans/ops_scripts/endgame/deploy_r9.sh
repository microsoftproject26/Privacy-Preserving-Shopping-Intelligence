set -euo pipefail
# SERVER: deploy tools_r9 (retention_report ANA-08 9a3b2c2a…, DL 395). r8 tools archived to tools/_archive_r8, never deleted.
# Upload tools_r9.tar + TOOLS_MANIFEST.json to $D first. Output: $D/R9_DEPLOYED.ok.
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
D=$V/runs_ds3/deploy_r9; Q=$V/runs_ds3/queues; PY=/home/jovyan/venvs/ppsi-gpu/bin/python
LOG=$Q/endgame_C.log; echo "$(date -u +%FT%TZ) PHASE_START deploy_r9" >> $LOG
test ! -e "$D/extract_tools"; test ! -e "$D/R9_DEPLOYED.ok"; test ! -e "$V/tools/_archive_r8"
echo "0da1dbb73e23165d2b860f1cbfad07a5e055e1c9f01d8f0fa598896e57f59ee1  $D/tools_r9.tar" | sha256sum -c -
export D V
$PY -B - <<'PY'
import json, os, tarfile, hashlib
from pathlib import PurePosixPath, Path
d = Path(os.environ['D']); v = Path(os.environ['V'])
m = json.loads((d/'TOOLS_MANIFEST.json').read_text()); assert m['tar_sha256'].startswith("0da1dbb7")
with tarfile.open(d/'tools_r9.tar') as t:
    for x in t:
        p = PurePosixPath(x.name); assert not p.is_absolute() and '..' not in p.parts, x.name
        assert x.isdir() or (x.isfile() and x.name in m['files']), x.name
r8 = json.loads((v/'runs_ds3/deploy_r8/TOOLS_MANIFEST.json').read_text())['files']
for rel, h in r8.items():
    assert hashlib.sha256((v/rel).read_bytes()).hexdigest() == h, f'{rel} on the server is not the r8 tool'
print('R8_TOOLS_VERIFIED', len(r8))
PY
mkdir "$D/extract_tools"; tar -xf "$D/tools_r9.tar" -C "$D/extract_tools"
$PY -B - <<'PY'
import hashlib, json, os
from pathlib import Path
d = Path(os.environ['D']); m = json.loads((d/'TOOLS_MANIFEST.json').read_text())
for rel, h in m['files'].items(): assert hashlib.sha256((d/'extract_tools'/rel).read_bytes()).hexdigest() == h, rel
agg = hashlib.sha256(''.join(f'{k}\0{v}\n' for k, v in sorted(m['files'].items())).encode()).hexdigest()
assert agg == m['aggregate_sha256']; print('PACK_HASHES_VERIFIED', len(m['files']), agg[:8])
PY
mkdir "$V/tools/_archive_r8"
cp -p "$V/tools/retention_report.py" "$V/tools/_archive_r8/retention_report.py"
chmod u+w "$V/tools/retention_report.py"; cp "$D/extract_tools/tools/retention_report.py" "$V/tools/retention_report.py"; chmod a-w "$V/tools/retention_report.py"
chmod -R a-w "$V/tools/_archive_r8"
sha256sum $V/tools/ana_block_index.py $V/tools/retention_report.py $V/tools/test_exposure_register.py $V/tools/_archive_r8/retention_report.py | cut -c1-16
echo "9a3b2c2a918ef2baacc5299426941f1eb94c366f5bb6aa7c29b4bfdbca0afe58  $V/tools/retention_report.py" | sha256sum -c -
date -u +%FT%TZ > $D/R9_DEPLOYED.ok
echo "$(date -u +%FT%TZ) TOOLS_R9_DEPLOYED" >> $LOG; echo TOOLS_R9_DEPLOYED
