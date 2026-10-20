#!/bin/bash
# C1 (LOCAL, Git Bash): pack ppsi_v4/code -> deploy_archive/r7/code_ds3r7.tar + SYNC_MANIFEST.json, and ppsi_v4/tools ->
# tools_r7.tar + TOOLS_MANIFEST.json. Model: ../pack_r6.sh. PREPARED, NOT RUN.
# r7 = r6 + ANA-06 (reeval, tests) + FL-16b (launch_fl c9fdc8c0…, test_fl16b 0dbdbe0f…) + DEV-04 (v4dev, tests_dev)
#      + BM-02 v4bench / tests_bench; tools = ana_block_index, retention_report, test_exposure_register (new on the server).
# Order: restore every */receipts/* file to its r6 bytes FIRST (FL-16b: 0 / 37 differ), then pack, then diff vs r6.
# Refuses (exit 3) when a file outside the expected r7 change set differs from r6, (exit 4) when v4eval changed.
# Run only after: REV-ANA-06 PASS, REV-FL-16b PASS (DL 309), DEV-04 + BM-02 reviews, ANA_CONTRASTS_v4 sha in a DL row.
set -euo pipefail
cd "/f/Queens University/Microsoft Project/eda/.local_ai/ppsi_v4"
OUT=deploy_archive/r7
[ ! -e $OUT ] || { echo "REFUSED: $OUT exists (a pack is never overwritten; use r7b)"; exit 2; }
mkdir -p $OUT
python - <<'PY'
import hashlib, json, tarfile, pathlib
m = json.load(open('deploy_archive/r6/SYNC_MANIFEST.json'))['files']
n = 0
with tarfile.open('deploy_archive/r6/code_ds3r6.tar') as tf:
    for k, h in m.items():
        if '/receipts/' not in k: continue
        p = pathlib.Path(k)
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(tf.extractfile(k).read()); n += 1
print('RECEIPTS_RESTORED_TO_R6', n)
PY
# BM-W2 (code/v4bench_w2, tests_bench_w2) is NOT part of r7 (DL 321; separate server dirs) — TO-CONFIRM README Q8
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' --exclude='.train03r1_tmp' --exclude='_pytest_tmp' \
    --exclude='*.bak' --exclude='*.orig' --exclude='*_pre_*' --exclude='code/v4bench_w2' --exclude='code/tests_bench_w2' --exclude='code/v4bench' --exclude='code/tests_bench' \
    -cf $OUT/code_ds3r7.tar code
tar -cf $OUT/tools_r7.tar tools/ana_block_index.py tools/retention_report.py tools/test_exposure_register.py
python - <<'PY'
import hashlib, json, sys, tarfile
def man(t):
    files = {}
    with tarfile.open(t) as tf:
        for m in tf:
            if m.isfile(): files[m.name] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    tsha = hashlib.sha256(open(t, 'rb').read()).hexdigest()
    agg = hashlib.sha256(''.join(f'{k}\0{v}\n' for k, v in sorted(files.items())).encode()).hexdigest()
    return {'tar_sha256': tsha, 'aggregate_sha256': agg, 'files': files}
c = man('deploy_archive/r7/code_ds3r7.tar'); json.dump(c, open('deploy_archive/r7/SYNC_MANIFEST.json', 'w'), indent=1, sort_keys=True)
t = man('deploy_archive/r7/tools_r7.tar'); json.dump(t, open('deploy_archive/r7/TOOLS_MANIFEST.json', 'w'), indent=1, sort_keys=True)
old = json.load(open('deploy_archive/r6/SYNC_MANIFEST.json'))['files']
print('CODE TAR', c['tar_sha256']); print('CODE AGG', c['aggregate_sha256']); print('N r6', len(old), 'r7', len(c['files']))
# expected r7 change set (owners: ANA-06, FL-16b, DEV-04, BM-02); anything else is printed UNEXPECTED and refuses
ALLOWED = ('code/v4int/reeval.py', 'code/v4int/launch_fl.py', 'code/tests_int/', 'code/tests_ana/', 'code/v4dev/',
           'code/tests_dev/')  # v4bench excluded: benchmarks run from code_bench3 (DL 341)
# Fable audit M4 (DL 331 / 341): assert the reviewed bytes of every changed core file
EXPECT = {'code/v4int/launch_fl.py': 'c9fdc8c083e834f875b5d9c48d1144da0740137374fc5219c18a623ee8927032', 'code/v4int/reeval.py': '833f0c981b0622e717d3309fe9a782d46905f67da45c2c74fa8c063fc7ec09f1',
          'code/v4dev/checkpointio.py': 'e84a6a5b6098833f7a41afa3d09915fdff653cf4b3697edcfb60a2bf50a62b4b',
          'code/v4dev/scoring.py': '45ee04c3844c6269df4d5a624367fb9efd6ef65e106259c90495a79a6ef7ef6e',
          'code/v4dev/dump_rows.py': '50c1a25d675af548de0817d74b4fd187c90424c962b407338578dc32ff45aae0',
          'code/v4dev/bench.py': '0b74e77a1d4d0b6a3bbc0e673ad7ea29baa3da679c5723817a6d99e5de633c41',
          'code/v4dev/export.py': 'fbb78997f4b6ec7867b9af6a243d10856a244e6f95faaaf754ec20a8966b46cb',
          'code/v4dev/browser_queries.py': '25a806f43b8abe0a1e77916d05a30acd37f2ce1f1a26dd5ebbcca2b1d21bc6e0'}
TEXPECT = {'tools/retention_report.py': '1d50f7f2b75b0750ee6f91910c9d9cbd665462e5fdc5c8c7b04870ea1074de81', 'tools/ana_block_index.py': '6e7043bbe42967b0de313206496de38fb0cf20cc85aaa34945dfdd19863cb662'}
bad = [k for k, h in EXPECT.items() if c['files'].get(k) != h] + [k for k, h in TEXPECT.items() if t['files'].get(k) != h]
if bad: print('REFUSED: reviewed-sha mismatch', bad); sys.exit(5)
print('REVIEWED_SHAS_OK', len(EXPECT) + len(TEXPECT))
unexpected = evalchg = 0
for k in sorted(set(old) | set(c['files'])):
    a, b = old.get(k), c['files'].get(k)
    if a == b: continue
    tag = 'NEW' if a is None else 'GONE' if b is None else 'CHANGED'
    ok = k.startswith(ALLOWED) and tag != 'GONE'
    unexpected += not ok; evalchg += k.startswith('code/v4eval/')
    print('  ', tag, k, (b or '')[:8], '' if ok else '<-- UNEXPECTED')
rcpt = [k for k in c['files'] if '/receipts/' in k and old.get(k) not in (None, c['files'][k])]
print('RECEIPTS_DIFFERING_FROM_R6', len(rcpt))
for k in ('code/v4int/launch_fl.py', 'code/v4int/reeval.py', 'code/tests_int/test_fl16.py', 'code/tests_int/test_fl16b.py'):
    print('KEY', k, c['files'].get(k, 'ABSENT')[:16])
print('TOOLS TAR', t['tar_sha256']); [print('  ', k, v[:16]) for k, v in sorted(t['files'].items())]
print('LOCAL tools/g2_run.py', hashlib.sha256(open('tools/g2_run.py', 'rb').read()).hexdigest())
if not c['files'].get('code/v4int/launch_fl.py', '').startswith('c9fdc8c0'): print('REFUSED: launch_fl is not FL-16b c9fdc8c0'); sys.exit(3)
if not c['files'].get('code/tests_int/test_fl16b.py', '').startswith('0dbdbe0f'): print('REFUSED: test_fl16b is not 0dbdbe0f'); sys.exit(3)
if evalchg: print('REFUSED: v4eval changed (evaluator_code_sha256 61b64e61… must be unchanged)'); sys.exit(4)
if rcpt: print('REFUSED: receipts differ from r6'); sys.exit(3)
if unexpected: print('REFUSED:', unexpected, 'unexpected file(s); review before deploy'); sys.exit(3)
print('PACK_R7_OK')
PY
# then: fill deploy_r7.tmpl.sh -> deploy_r7.sh (README §3 C2) and upload the tars + manifests to $R/deploy_r7/
