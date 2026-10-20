#!/bin/bash
# pack local ppsi_v4/code into code_ds3r5.tar + SYNC_MANIFEST.json; compare file set with code_ds3r3.tar
set -euo pipefail
cd "/f/Queens University/Microsoft Project/eda/.local_ai/ppsi_v4"
OUT=deploy_archive/r5; mkdir -p $OUT
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' --exclude='.train03r1_tmp' --exclude='_pytest_tmp' --exclude='*.bak' --exclude='*.orig' --exclude='*_pre_*' -cf $OUT/code_ds3r5.tar code
tar -tf $OUT/code_ds3r5.tar | grep -v '/$' | sort > $OUT/cur.list
tar -tf deploy_archive/r4/code_ds3r4.tar | grep -v '/$' | sort > $OUT/prev.list
echo "files r3=$(wc -l < $OUT/prev.list) r4=$(wc -l < $OUT/cur.list)"
echo "--- only in r4:"; comm -13 $OUT/prev.list $OUT/cur.list
echo "--- only in r3:"; comm -23 $OUT/prev.list $OUT/cur.list
python - <<'PY'
import hashlib,json,tarfile
t='deploy_archive/r5/code_ds3r5.tar'
files={}
with tarfile.open(t) as tf:
    for m in tf:
        if m.isfile(): files[m.name]=hashlib.sha256(tf.extractfile(m).read()).hexdigest()
tsha=hashlib.sha256(open(t,'rb').read()).hexdigest()
agg=hashlib.sha256(''.join(f'{k}\0{v}\n' for k,v in sorted(files.items())).encode()).hexdigest()
json.dump({'tar_sha256':tsha,'aggregate_sha256':agg,'files':files},open('deploy_archive/r5/SYNC_MANIFEST.json','w'),indent=1,sort_keys=True)
print('TAR',tsha); print('AGG',agg); print('N',len(files))
PY
