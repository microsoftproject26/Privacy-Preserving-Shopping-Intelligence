#!/bin/bash
# LOCAL (Git Bash): tools_r8 = r7 tools with retention_report.py replaced by ANA-07 (d0247370…; DL 390, REV-ANA-07).
# ana_block_index (6e7043bb…) and test_exposure_register (7d59e828…) must be byte-identical to r7. Code tree unchanged (r7 / r7b).
set -euo pipefail
cd "/f/Queens University/Microsoft Project/eda/.local_ai/ppsi_v4"
OUT=deploy_archive/r8
[ ! -e $OUT ] || { echo "REFUSED: $OUT exists (a pack is never overwritten)"; exit 2; }
mkdir -p $OUT
tar -cf $OUT/tools_r8.tar tools/ana_block_index.py tools/retention_report.py tools/test_exposure_register.py
python - <<'PY'
import hashlib, json, sys, tarfile
t = 'deploy_archive/r8/tools_r8.tar'; files = {}
with tarfile.open(t) as tf:
    for m in tf:
        if m.isfile(): files[m.name] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
agg = hashlib.sha256(''.join(f'{k}\0{v}\n' for k, v in sorted(files.items())).encode()).hexdigest()
man = {'tar_sha256': hashlib.sha256(open(t, 'rb').read()).hexdigest(), 'aggregate_sha256': agg, 'files': files}
EXP = {'tools/retention_report.py': 'd0247370f58ead3654bd0024211ae3718918b858f9bba358666f8415e2157035',
       'tools/ana_block_index.py': '6e7043bbe42967b0de313206496de38fb0cf20cc85aaa34945dfdd19863cb662',
       'tools/test_exposure_register.py': '7d59e828f0d8514541ac9cd68393053e064f9c70df3e3655e9c4f0a34eae2622'}
if files != EXP: print('REFUSED: tools differ from the reviewed set', files); sys.exit(5)
json.dump(man, open('deploy_archive/r8/TOOLS_MANIFEST.json', 'w', newline='\n'), indent=1, sort_keys=True)
print('TOOLS TAR', man['tar_sha256']); print('AGG', agg); print('PACK_R8_OK')
PY
