"""X-REES46 inventory digest of the confirmatory TEST / ANA trees (read-only; POST_HOC_EXPLORATORY guard): per tree the
P2-lock code-tree rule (sha256 of compact sort_keys JSON {posix relpath: file sha256}; __pycache__ / *.pyc skipped),
file count and bytes; symlinks are listed, never followed. Written once to <out> (refused if it exists).
  python x_inventory.py <out.json> <tree> [<tree> ...]      (trees relative to ppsi_v4)"""
import hashlib
import json
import sys
import time
from pathlib import Path

V = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


out = Path(sys.argv[1])
if out.exists():
    sys.exit(f"{out} exists")
rec = {"format": "PPSI_V4_X_INVENTORY_1", "label": "POST_HOC_EXPLORATORY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "rule": __doc__.split("\n  python")[0], "trees": {}}
for t in sys.argv[2:]:
    root = V / t
    files, links, nbytes = {}, [], 0
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(x in ("__pycache__", ".pytest_cache") for x in rel.parts) or p.suffix == ".pyc":
            continue
        if p.is_symlink():
            links.append(rel.as_posix())
        elif p.is_file():
            files[rel.as_posix()] = sha(p)
            nbytes += p.stat().st_size
    rec["trees"][t] = {"aggregate_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                       "n_files": len(files), "bytes": nbytes, "symlinks": links}
rec["combined_sha256"] = hashlib.sha256(json.dumps({k: v["aggregate_sha256"] for k, v in rec["trees"].items()},
                                                   sort_keys=True).encode()).hexdigest()
out.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
print(json.dumps({k: (v["aggregate_sha256"][:16], v["n_files"]) for k, v in rec["trees"].items()}), rec["combined_sha256"])
