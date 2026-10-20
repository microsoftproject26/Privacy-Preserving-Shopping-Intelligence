"""verify_frozen.py — read-only integrity re-check of the frozen dataset (Sonnet A, Wave 1).

Launch gate 1 / FULL_RUN_PROTOCOL.md #10.1: sha256 every file listed in `FILES.json` and
compare against the recorded digest. Never writes inside `/home/jovyan/shared_data/`. Caps
pyarrow/OS thread usage is not relevant here (pure hashlib + file IO), but this script still
avoids spawning unbounded parallelism.

Output JSON: {verified, missing, mismatched, total_bytes, elapsed_s} to stdout AND to
`$W/tests_full/A/verify_frozen.json` (the caller passes --out for the latter).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root: Path, files_json: Path | None = None) -> dict:
    root = Path(root)
    files_json = files_json or (root / "FILES.json")
    manifest = json.loads(files_json.read_text(encoding="utf-8"))
    entries = manifest["files"]

    t0 = time.time()
    verified = 0
    missing: list[str] = []
    mismatched: list[dict] = []
    total_bytes = 0

    for entry in entries:
        rel = entry["path"]
        expected_sha = entry["sha256"]
        expected_bytes = entry.get("bytes")
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        actual_bytes = p.stat().st_size
        actual_sha = sha256_file(p)
        total_bytes += actual_bytes
        ok = (actual_sha == expected_sha) and (expected_bytes is None or actual_bytes == expected_bytes)
        if ok:
            verified += 1
        else:
            mismatched.append({
                "path": rel,
                "expected_sha256": expected_sha, "actual_sha256": actual_sha,
                "expected_bytes": expected_bytes, "actual_bytes": actual_bytes,
            })

    elapsed_s = time.time() - t0
    return {
        "root": str(root),
        "total_files_listed": len(entries),
        "verified": verified,
        "missing": missing,
        "mismatched": mismatched,
        "total_bytes": total_bytes,
        "elapsed_s": elapsed_s,
        "all_ok": verified == len(entries) and not missing and not mismatched,
    }


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/home/jovyan/shared_data/rees46_nextitem_v1")
    ap.add_argument("--out", default=None, help="also write JSON result here")
    args = ap.parse_args()

    result = verify(Path(args.root))
    print(json.dumps({k: v for k, v in result.items() if k not in ("missing", "mismatched")}, indent=2))
    if result["missing"]:
        print(f"MISSING ({len(result['missing'])}):", result["missing"][:20])
    if result["mismatched"]:
        print(f"MISMATCHED ({len(result['mismatched'])}):", result["mismatched"][:5])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")

    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
