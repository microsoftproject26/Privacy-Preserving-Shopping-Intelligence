"""Compare protected copied files to the pre-work metadata inventory.

This checks names, sizes, nanosecond mtimes, inodes and symlink targets.
It is explicitly not a complete rehash of the 125 GB copy. Restored test
receipts receive an additional SHA256 comparison against the copy manifest.
"""
import hashlib
import json
from datetime import datetime, timezone
import ppsi_paths

base = ppsi_paths.SERVER_BASE
review = ppsi_paths.DELIVERY_REVIEW_ROOT
v = base / 'Privacy-Preserving-Shopping-Intelligence' / '.local_ai' / 'ppsi_v4'
before = json.loads((review / 'evidence/PROTECTED_FILES_BEFORE.json').read_text())
trees = sorted(p for p in v.iterdir() if p.is_dir() and (p.name == 'code' or p.name.startswith('code_')))
trees += [v / name for name in ['runs_ds3', 'runs_p2', 'runs_bench', 'runs_bench_gpu',
                               'runs_bench_x/TEST', 'runs_x_rees46/TEST']]
paths = set()
for tree in trees:
    # Match the original os.walk file-entry inventory: directory symlinks are
    # not file entries; ordinary file links and broken links are included.
    paths.update(p for p in tree.rglob('*') if p.is_file() or (p.is_symlink() and not p.is_dir()))
paths.update((v / 'company/gates').glob('*LOCK*.json'))
paths.update((v / 'company/registers').glob('*TEST_EXPOSURE_REGISTER*.json'))
after = {}
for p in sorted(paths):
    st = p.lstat()
    after[p.relative_to(v).as_posix()] = [st.st_size, st.st_mtime_ns, st.st_ino,
                                         str(p.readlink()) if p.is_symlink() else None]
added = sorted(after.keys() - before.keys())
removed = sorted(before.keys() - after.keys())
changed = [k for k in before.keys() & after.keys() if before[k] != after[k]]
restored = json.loads((review / 'evidence/LEGACY_TEST_RECEIPTS_RESTORED.json').read_text())
hash_checks = []
for item in restored['restored']:
    actual = hashlib.sha256((v / item['path']).read_bytes()).hexdigest()
    hash_checks.append(dict(path=item['path'], sha256=actual, expected=item['snapshot_sha256'],
                            passed=actual == item['snapshot_sha256']))
report = dict(utc=datetime.now(timezone.utc).isoformat(), before_files=len(before), after_files=len(after),
              added=added, removed=removed, changed=changed, restored_receipt_hashes=hash_checks,
              scope='protected file metadata and names; restored receipt SHA256s; not a complete tree rehash',
              success=not (added or removed or changed) and all(c['passed'] for c in hash_checks))
(review / 'evidence/PROTECTED_FILES_AFTER.json').write_text(json.dumps(after, indent=2)+'\n')
(review / 'evidence/PROTECTED_INVENTORY_COMPARISON.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(0 if report['success'] else 1)
