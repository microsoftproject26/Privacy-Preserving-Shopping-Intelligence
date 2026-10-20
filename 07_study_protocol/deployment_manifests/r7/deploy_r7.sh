set -euo pipefail
# C2 (SERVER): deploy code_ds3r7 BESIDE code_ds3r6 (m6; in-flight r6 runs finish and resume on r6 — fl_code_sha256 changes,
# FL-16b) + tools r7 (r6 tools archived, never deleted). Model: ../deploy_r6.tmpl.sh. PREPARED, NOT RUN.
# Template: c9e20797be4b669eb5c8d31c0acc4a782547126ac341f31a3a335dba67c0876a / 653d507cf12624daefb28e0dd31dc8983ce35606a76a315786332ca8ebc0166f / 3893a49a71be49f5baf338b93195e2e885eecb9326705902913bd4d5e1abb44d are filled from deploy_archive/r7 manifests + pack_r7.sh output (sed)
# before upload; upload code_ds3r7.tar, tools_r7.tar, SYNC_MANIFEST.json, TOOLS_MANIFEST.json to $R/deploy_r7/ first.
# Output: $R/deploy_r7/R7_DEPLOYED.ok (the traj_cmp7.sh predecessor) + $Q/_common_code_ds3r7.sh.
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
D=$V/runs_ds3/deploy_r7; Q=$V/runs_ds3/queues; PY=/home/jovyan/venvs/ppsi-gpu/bin/python
LOG=$Q/endgame_C.log; echo "$(date -u +%FT%TZ) PHASE_START C deploy_r7" >> $LOG
test -e "$V/code_ds3r6"; test ! -e "$V/code_ds3r7"; test ! -e "$D/extract"; test ! -e "$D/R7_DEPLOYED.ok"
test ! -e "$Q/_common_code_ds3r7.sh"; test ! -e "$V/tools/_archive_r6"; test ! -e "$V/tools/test_exposure_register.py"
echo "c9e20797be4b669eb5c8d31c0acc4a782547126ac341f31a3a335dba67c0876a  $D/code_ds3r7.tar" | sha256sum -c -
echo "653d507cf12624daefb28e0dd31dc8983ce35606a76a315786332ca8ebc0166f  $D/tools_r7.tar" | sha256sum -c -
echo "3893a49a71be49f5baf338b93195e2e885eecb9326705902913bd4d5e1abb44d  $V/tools/g2_run.py" | sha256sum -c -          # retention_report hashes tools/g2_run.py (A-F14)
export DEPLOY_ROOT="$D" V
$PY -B - <<'PY'
import json, os, tarfile, hashlib
from pathlib import PurePosixPath, Path
d = Path(os.environ['DEPLOY_ROOT']); v = Path(os.environ['V'])
for tn, mn, sha in (('code_ds3r7.tar', 'SYNC_MANIFEST.json', 'c9e20797be4b669eb5c8d31c0acc4a782547126ac341f31a3a335dba67c0876a'), ('tools_r7.tar', 'TOOLS_MANIFEST.json', '653d507cf12624daefb28e0dd31dc8983ce35606a76a315786332ca8ebc0166f')):
    m = json.loads((d/mn).read_text()); assert m['tar_sha256'] == sha, mn
    with tarfile.open(d/tn) as t:
        for x in t:
            p = PurePosixPath(x.name)
            assert not p.is_absolute() and '..' not in p.parts, x.name
            assert x.isdir() or (x.isfile() and x.name in m['files']), x.name
    print('TAR_NAMES_VERIFIED', tn, len(m['files']))
r6 = json.loads((v/'runs_ds3/deploy_r6/TOOLS_MANIFEST.json').read_text())['files']   # the tools being archived ARE r6's
for rel, h in r6.items():
    assert hashlib.sha256((v/rel).read_bytes()).hexdigest() == h, f'{rel} on the server is not the r6 tool'
print('R6_TOOLS_VERIFIED', len(r6))
PY
mkdir "$D/extract" "$D/extract_tools"
tar -xf "$D/code_ds3r7.tar" -C "$D/extract"; tar -xf "$D/tools_r7.tar" -C "$D/extract_tools"
$PY -B - <<'PY'
import hashlib, json, os
from pathlib import Path
d = Path(os.environ['DEPLOY_ROOT'])
for ex, mn in (('extract', 'SYNC_MANIFEST.json'), ('extract_tools', 'TOOLS_MANIFEST.json')):
    m = json.loads((d/mn).read_text())
    for rel, h in m['files'].items(): assert hashlib.sha256((d/ex/rel).read_bytes()).hexdigest() == h, rel
    agg = hashlib.sha256(''.join(f'{k}\0{v}\n' for k, v in sorted(m['files'].items())).encode()).hexdigest()
    assert agg == m['aggregate_sha256']; print('PACK_HASHES_VERIFIED', mn, len(m['files']), agg[:8])
PY
mv "$D/extract/code" "$V/code_ds3r7"; chmod -R a-w "$V/code_ds3r7"
mkdir "$V/tools/_archive_r6"
for f in ana_block_index.py retention_report.py; do
  cp -p "$V/tools/$f" "$V/tools/_archive_r6/$f"; chmod u+w "$V/tools/$f"; cp "$D/extract_tools/tools/$f" "$V/tools/$f"; chmod a-w "$V/tools/$f"
done
cp "$D/extract_tools/tools/test_exposure_register.py" "$V/tools/test_exposure_register.py"; chmod a-w "$V/tools/test_exposure_register.py"
chmod -R a-w "$V/tools/_archive_r6"
sha256sum $V/tools/ana_block_index.py $V/tools/retention_report.py $V/tools/test_exposure_register.py | cut -c1-16
(cd $V/code_ds3r7 && PYTHONPATH=$V/code_ds3r7 PYTHONDONTWRITEBYTECODE=1 $PY -B -c "import v4eval; s=v4eval.evaluator_code_sha256(); print('EVALUATOR', s); assert s.startswith('61b64e61')")
sed 's#CR=$V/code_ds3r6#CR=$V/code_ds3r7#' $Q/_common_code_ds3r6.sh > $Q/_common_code_ds3r7.sh
[ "$(grep -c 'CR=$V/code_ds3r7' $Q/_common_code_ds3r7.sh)" = 1 ]; ! grep -q code_ds3r6 $Q/_common_code_ds3r7.sh
chmod a-w $Q/_common_code_ds3r7.sh
date -u +%FT%TZ > $D/R7_DEPLOYED.ok
echo "$(date -u +%FT%TZ) CODE_DS3R7_DEPLOYED" >> $LOG; echo CODE_DS3R7_DEPLOYED
