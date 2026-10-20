"""code_sync.py (owner: Operations Director, OPS-07) — verified one-way sync of ppsi_v4/code + ppsi_v4/contracts to the server.

  pack    (local)   build <out>/code_sync.tar (no __pycache__ / .pytest_cache / *.pyc) + <out>/SYNC_MANIFEST.json
                    {rel_path: sha256}; contracts/DATA_PROTOCOL_LOCK.json is never packed (read-only on both sides;
                    its hash is checked instead). Also checks the local contracts against the lock's contracts_sha256.
  preflight (server) compare the manifest with the server tree: files present on both sides with DIFFERENT content
                    are listed and the sync is refused (no blind overwrite); server-only files are listed and kept.
  apply   (server)  extract (refuses absolute paths, '..', links; --keep-server files are not extracted and must stay
                    byte-identical), then re-hash every manifest file, verify the
                    snapshot (ppsi_env.verify_snapshot) and the lock hash; writes a receipt JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path
import ppsi_paths

V = ppsi_paths.PROJECT_ROOT.resolve()   # the ppsi_v4 workspace
ROOTS = ("code", "contracts")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
LOCK_REL = "contracts/DATA_PROTOCOL_LOCK.json"
LOCK_FILE_SHA256 = "ebf4e86f6beb82e98283108a2dad9082727e8559b81301cd46f35f60c79fd7c0"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def walk(base: Path, exclude=()):
    for root in ROOTS:
        for dp, dns, fns in os.walk(base / root):
            dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
            for fn in sorted(fns):
                p = Path(dp) / fn
                rel = p.relative_to(base).as_posix()
                if fn.endswith(".pyc") or rel == LOCK_REL or p.is_symlink() or any(rel.startswith(e) for e in exclude):
                    continue
                yield rel, p


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pack(out: Path, exclude=()) -> int:
    out.mkdir(parents=True, exist_ok=True)
    lock = json.loads((V / LOCK_REL).read_text(encoding="utf-8"))
    if sha(V / LOCK_REL) != LOCK_FILE_SHA256:
        raise SystemExit("local DATA_PROTOCOL_LOCK hash differs from the recorded value")
    bad = {n: h for n, h in lock["contracts_sha256"].items() if sha(V / "contracts" / n) != h}
    if bad:
        raise SystemExit(f"local contracts differ from the lock's contracts_sha256: {sorted(bad)}")
    files = {rel: sha(p) for rel, p in walk(V, exclude)}
    with tarfile.open(out / "code_sync.tar", "w", format=tarfile.PAX_FORMAT) as t:
        for rel in files:
            ti = t.gettarinfo(str(V / rel), arcname=rel)
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.mode = 0o644
            with open(V / rel, "rb") as f:
                t.addfile(ti, f)
    man = {"document": "SYNC_MANIFEST", "packed_utc": utc(), "roots": list(ROOTS), "n_files": len(files),
           "total_bytes": sum((V / r).stat().st_size for r in files), "files": files,
           "excluded_prefixes": list(exclude), "lock_file_sha256": LOCK_FILE_SHA256, "tar_sha256": sha(out / "code_sync.tar")}
    (out / "SYNC_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("n_files", "total_bytes", "tar_sha256")}))
    return 0


def _load(man_path: Path) -> dict:
    return json.loads(man_path.read_text(encoding="utf-8"))


def preflight(man_path: Path, target: Path) -> dict:
    man = _load(man_path)
    differ, same, new = [], 0, 0
    for rel, h in man["files"].items():
        p = target / rel
        if p.is_file():
            if sha(p) == h:
                same += 1
            else:
                differ.append(rel)
        else:
            new += 1
    server_only = sorted(rel for rel, _ in walk(target) if rel not in man["files"])   # kept (e.g. files owned by active workers)
    return {"same": same, "new": new, "differ": differ, "server_only": server_only}


def apply(man_path: Path, tar_path: Path, target: Path, receipt: Path, allow_differ: list, keep_server: list) -> int:
    man = _load(man_path)
    if sha(tar_path) != man["tar_sha256"]:
        raise SystemExit("tar hash differs from the manifest")
    pre = preflight(man_path, target)
    bad_keep = sorted(set(keep_server) - set(pre["differ"]))
    if bad_keep:
        raise SystemExit(f"--keep-server names files that do not differ: {bad_keep}")
    kept = {rel: sha(target / rel) for rel in keep_server}      # server copy stays byte-identical
    unexpected = sorted(set(pre["differ"]) - set(allow_differ) - set(keep_server))
    if unexpected:
        raise SystemExit(f"REFUSED: server files differ from local and are not allow-listed: {unexpected}")
    lock_before = sha(target / LOCK_REL)
    if lock_before != LOCK_FILE_SHA256:
        raise SystemExit("server DATA_PROTOCOL_LOCK hash differs from the recorded value")
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            n = m.name
            if not m.isfile() or n.startswith("/") or ".." in Path(n).parts or n not in man["files"]:
                raise SystemExit(f"REFUSED tar member {n!r}")
        # write only files that are new or allow-listed; byte-identical files (e.g. the read-only snapshot) are untouched
        to_write = {rel for rel in man["files"] if not (target / rel).is_file()} | (set(allow_differ) & set(pre["differ"]))
        members = [m for m in t.getmembers() if m.name in to_write and m.name not in kept]
        if sys.version_info >= (3, 12):
            t.extractall(str(target), members=members, filter="data")
        else:
            t.extractall(str(target), members=members)
    post_bad = sorted(rel for rel, h in man["files"].items() if sha(target / rel) != kept.get(rel, h))
    sys.path.insert(0, str(target / "code"))
    import ppsi_env  # noqa: E402
    try:
        sm = ppsi_env.verify_snapshot()   # raises on any drift
        snap = {"unchanged": True, "n_files": len(sm["files"]),
                "manifest_sha256": sha(target / "code" / "snap_v1" / "SNAPSHOT_MANIFEST.json")}
    except Exception as e:  # noqa: BLE001 - recorded, and the receipt fails
        snap = {"unchanged": False, "error": repr(e)}
    lock_after = sha(target / LOCK_REL)
    rec = {"document": "OPS07_CODE_SYNC_RECEIPT", "applied_utc": utc(), "host": os.uname().nodename if hasattr(os, "uname") else "",
           "target": str(target), "manifest_sha256": sha(man_path), "tar_sha256": man["tar_sha256"], "n_files": man["n_files"],
           "preflight": {k: (v if isinstance(v, int) else len(v)) for k, v in pre.items()}, "n_written": len(members), "differ_overwritten": sorted(set(pre["differ"]) - set(kept)), "kept_server_copy_sha256": kept,
           "server_only_kept": pre["server_only"], "post_hash_mismatches": post_bad, "snapshot": snap,
           "lock_file_sha256_before": lock_before, "lock_file_sha256_after": lock_after,
           "pass": (not post_bad) and lock_after == LOCK_FILE_SHA256 and snap["unchanged"] is True}
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, receipt)
    print(json.dumps({k: rec[k] for k in ("n_files", "preflight", "post_hash_mismatches", "pass")}, default=str))
    return 0 if rec["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="op", required=True)
    a = sp.add_parser("pack"); a.add_argument("--out", required=True)
    a.add_argument("--exclude", nargs="*", default=[], help="rel-path prefixes not packed (dirs owned by active workers)")
    b = sp.add_parser("preflight"); b.add_argument("--manifest", required=True); b.add_argument("--target", required=True)
    c = sp.add_parser("apply"); c.add_argument("--manifest", required=True); c.add_argument("--tar", required=True)
    c.add_argument("--target", required=True); c.add_argument("--receipt", required=True)
    c.add_argument("--allow-differ", nargs="*", default=[])
    c.add_argument("--keep-server", nargs="*", default=[], help="differing files whose SERVER copy is kept (not extracted)")
    x = ap.parse_args()
    if x.op == "pack":
        return pack(Path(x.out), tuple(x.exclude))
    if x.op == "preflight":
        r = preflight(Path(x.manifest), Path(x.target))
        print(json.dumps(r, indent=1))
        return 0
    return apply(Path(x.manifest), Path(x.tar), Path(x.target), Path(x.receipt), x.allow_differ, x.keep_server)


if __name__ == "__main__":
    raise SystemExit(main())
