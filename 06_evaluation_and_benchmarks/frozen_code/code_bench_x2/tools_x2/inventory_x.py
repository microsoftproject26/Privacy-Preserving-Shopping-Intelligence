"""Inventory of the confirmatory benchmark trees (Fable T1 condition 5, DL 540 pattern). Reads metadata only.

  python -B tools_x2/inventory_x.py snap --label before      -> runs_bench_x/x2_state/INVENTORY_CONF_before.json (write-once)
  python -B tools_x2/inventory_x.py snap --label after
  python -B tools_x2/inventory_x.py compare before after     -> exit 0 if identical, 1 with the differing paths

For every regular file under runs_bench/ and runs_bench_gpu/ (symlinks not followed): (relative path, size in bytes,
st_mtime_ns). digest = sha256 of the sorted lines "path\\tsize\\tmtime_ns\\n"; also one digest per <root>/<dataset>.
No file content is opened. The exploratory TEST look must leave both trees bit-for-bit unchanged in this metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(os.path.dirname(HERE))
ROOTS = ("runs_bench", "runs_bench_gpu")
STATE = os.path.join(V4, "runs_bench_x", "x2_state")


def snap() -> dict:
    rows = []
    for r in ROOTS:
        base = os.path.join(V4, r)
        for dp, dns, fns in os.walk(base, followlinks=False):
            dns.sort()
            for fn in sorted(fns):
                p = os.path.join(dp, fn)
                st = os.lstat(p)
                if not os.path.isfile(p) or os.path.islink(p):
                    continue
                rows.append((os.path.relpath(p, V4).replace(os.sep, "/"), st.st_size, st.st_mtime_ns))
    rows.sort()
    lines = [f"{p}\t{s}\t{m}\n" for p, s, m in rows]
    per = {}
    for (p, _s, _m), ln in zip(rows, lines):
        k = "/".join(p.split("/")[:2])
        per.setdefault(k, hashlib.sha256())
        per[k].update(ln.encode())
    return {"format": "X_CONF_INVENTORY_1", "roots": list(ROOTS), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_files": len(rows), "total_bytes": int(sum(s for _p, s, _m in rows)),
            "digest": hashlib.sha256("".join(lines).encode()).hexdigest(),
            "per_dir": {k: h.hexdigest() for k, h in sorted(per.items())},
            "rows": [[p, s, m] for p, s, m in rows]}


def path_of(label: str) -> str:
    return os.path.join(STATE, f"INVENTORY_CONF_{label}.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snap")
    s.add_argument("--label", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "snap":
        obj = snap()
        p = path_of(a.label)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            os.write(fd, (json.dumps(obj, indent=0, sort_keys=True) + "\n").encode())
        finally:
            os.close(fd)
        print(f"INVENTORY {a.label}: {obj['n_files']} files, {obj['total_bytes']} bytes, digest {obj['digest']} -> {p}")
        return 0
    A, B = json.load(open(path_of(a.a))), json.load(open(path_of(a.b)))
    if A["digest"] == B["digest"]:
        print(f"IDENTICAL {A['digest']} ({A['n_files']} files)")
        return 0
    ra, rb = {r[0]: r for r in A["rows"]}, {r[0]: r for r in B["rows"]}
    diff = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
    print(f"DIFFERS: {len(diff)} paths, e.g. {diff[:10]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
