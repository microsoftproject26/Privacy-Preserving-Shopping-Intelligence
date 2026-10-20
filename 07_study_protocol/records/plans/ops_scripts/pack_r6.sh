#!/bin/bash
# pack local ppsi_v4/code into code_ds3r6.tar + SYNC_MANIFEST.json (and ppsi_v4/tools ANA files into tools_r6.tar);
# first restore every */receipts/* file to its r5 bytes; then list every file that differs from r5.
set -euo pipefail
cd "/f/Queens University/Microsoft Project/eda/.local_ai/ppsi_v4"
OUT=deploy_archive/r6; mkdir -p $OUT
python - <<'PY'
import hashlib, json, tarfile, pathlib
m = json.load(open('deploy_archive/r5/SYNC_MANIFEST.json'))['files']
n = 0
with tarfile.open('deploy_archive/r5/code_ds3r5.tar') as tf:
    for k, h in m.items():
        if '/receipts/' not in k: continue
        p = pathlib.Path(k)
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(tf.extractfile(k).read()); n += 1
print('RECEIPTS_RESTORED', n)
PY
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' --exclude='.train03r1_tmp' --exclude='_pytest_tmp' --exclude='*.bak' --exclude='*.orig' --exclude='*_pre_*' -cf $OUT/code_ds3r6.tar code
tar -cf $OUT/tools_r6.tar tools/ana_block_index.py tools/retention_report.py
python - <<'PY'
import hashlib, json, tarfile
def man(t):
    files = {}
    with tarfile.open(t) as tf:
        for m in tf:
            if m.isfile(): files[m.name] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    tsha = hashlib.sha256(open(t, 'rb').read()).hexdigest()
    agg = hashlib.sha256(''.join(f'{k}\0{v}\n' for k, v in sorted(files.items())).encode()).hexdigest()
    return {'tar_sha256': tsha, 'aggregate_sha256': agg, 'files': files}
c = man('deploy_archive/r6/code_ds3r6.tar'); json.dump(c, open('deploy_archive/r6/SYNC_MANIFEST.json', 'w'), indent=1, sort_keys=True)
t = man('deploy_archive/r6/tools_r6.tar'); json.dump(t, open('deploy_archive/r6/TOOLS_MANIFEST.json', 'w'), indent=1, sort_keys=True)
old = json.load(open('deploy_archive/r5/SYNC_MANIFEST.json'))['files']
print('CODE TAR', c['tar_sha256']); print('CODE AGG', c['aggregate_sha256']); print('N r5', len(old), 'r6', len(c['files']))
for k in sorted(set(old) | set(c['files'])):
    a, b = old.get(k), c['files'].get(k)
    if a != b: print('  ', 'NEW' if a is None else 'GONE' if b is None else 'CHANGED', k, (b or '')[:8])
print('TOOLS TAR', t['tar_sha256']); [print('  ', k, v[:8]) for k, v in sorted(t['files'].items())]
PY
