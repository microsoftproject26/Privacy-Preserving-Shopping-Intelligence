"""Full byte inventory of archived code, experiments, contracts and TEST gates.

Symlinks are inventoried as links without following them into external trees.
Decision logs are excluded because the authorized work appends new entries.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import ppsi_paths

BASE = ppsi_paths.SERVER_BASE
REVIEW = ppsi_paths.DELIVERY_REVIEW_ROOT
SOURCE = BASE / 'Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4'


def roots():
    code = [p for p in SOURCE.iterdir() if p.is_dir() and
            (p.name == 'code' or p.name.startswith('code_'))]
    names = ['runs_ds3', 'runs_p2', 'runs_bench', 'runs_bench_gpu',
             'runs_bench_x', 'runs_x_rees46', 'runs_exp_pb100',
             'company/gates', 'company/registers', 'contracts']
    return sorted(code + [SOURCE / n for n in names if (SOURCE / n).exists()])


def entry(p):
    st = p.lstat()
    info = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino,
                mode=st.st_mode)
    if p.is_symlink():
        info['symlink'] = str(p.readlink())
    else:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(8 << 20), b''):
                h.update(b)
        info['sha256'] = h.hexdigest()
        end = p.lstat()
        if (end.st_size, end.st_mtime_ns, end.st_ino) != (st.st_size, st.st_mtime_ns, st.st_ino):
            raise RuntimeError('Archive changed while hashing: ' + str(p))
    return str(p.relative_to(SOURCE)), info


def write_once(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write('\n')


def inventory(out):
    start = time.monotonic()
    rs = roots()
    paths = sorted({p for r in rs for p in r.rglob('*') if p.is_file() or p.is_symlink()})
    print(json.dumps(dict(status='HASHING', entries=len(paths), threads=4)), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(entry, paths))
    obj = dict(format='PPSI_PROTECTED_BYTE_INVENTORY_1', utc=datetime.now(timezone.utc).isoformat(),
               root=str(SOURCE), roots=[str(p.relative_to(SOURCE)) for p in rs],
               scope='Every regular file SHA256 plus symlink identity, inode, size, mode, mtime; no symlink target traversal',
               files=files, entries=len(files), elapsed_s=time.monotonic()-start)
    write_once(out, obj)
    print(json.dumps(dict(status='COMPLETE', entries=len(files), elapsed_s=obj['elapsed_s'],
                         sha256=hashlib.sha256(Path(out).read_bytes()).hexdigest())), flush=True)


def compare(before, after, out):
    a, b = [json.loads(Path(p).read_text()) for p in [before, after]]
    if a['root'] != b['root'] or a['roots'] != b['roots']:
        raise RuntimeError('Inventory scopes differ')
    x, y = a['files'], b['files']
    added, removed = sorted(y.keys()-x.keys()), sorted(x.keys()-y.keys())
    changed = sorted(k for k in x.keys() & y.keys() if x[k] != y[k])
    result = dict(status='PASS' if not (added or removed or changed) else 'FAIL',
                  utc=datetime.now(timezone.utc).isoformat(), added=added, removed=removed,
                  changed=changed, entries_before=len(x), entries_after=len(y),
                  before=str(Path(before)), after=str(Path(after)),
                  before_sha256=hashlib.sha256(Path(before).read_bytes()).hexdigest(),
                  after_sha256=hashlib.sha256(Path(after).read_bytes()).hexdigest(),
                  scope=a['scope'])
    write_once(out, result)
    print(json.dumps(result))
    if result['status'] != 'PASS':raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='op', required=True)
    q = sub.add_parser('inventory');q.add_argument('--out', required=True)
    q = sub.add_parser('compare');q.add_argument('--before', required=True)
    q.add_argument('--after', required=True);q.add_argument('--out', required=True)
    a = p.parse_args()
    if a.op == 'inventory':inventory(a.out)
    else:compare(a.before, a.after, a.out)
