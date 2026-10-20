"""v4data.build_all (owner: Data Lead, DATA-04) - the one entry point for the freeze-step build.

Order: snapshot check -> TRANSFORM_STATE_CLEAN_V2.json -> manifests + bootstrap + receipts -> independent re-read
verification of every written manifest -> snapshot check again. Every step is fail-closed; nothing is written
outside data_v2/. Run under the RSS guard:

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 \
    python -m v4data.run_bounded --max-rss-mib 8192 --log data_v2/receipts/rss.jsonl -- python -m v4data.build_all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .common import (CODE, atomic_write_json, data_v2_root, dataset_root, sha256_file, table_content_sha256,
                     utc_now)
from .manifests import PRE_LOCK_IDS, ManifestBuilder, reread_columns
from .transform_state import load_state, write_state


def snapshot_check() -> dict:
    """verify_snapshot() without importing the snapshot (keeps the build's RSS small): every file of
    snap_v1/SNAPSHOT_MANIFEST.json must hash to its recorded sha256."""
    snap = CODE / "snap_v1"
    man = json.loads((snap / "SNAPSHOT_MANIFEST.json").read_text())
    bad = [rel for rel, h in man["files"].items() if sha256_file(snap / rel) != h]
    return {"n_files": len(man["files"]), "mismatched": bad, "unchanged": not bad,
            "manifest_sha256": sha256_file(snap / "SNAPSHOT_MANIFEST.json")}


def verify_written(out_root: Path) -> dict:
    """Independent re-read of every manifest: marker present, file sha256 as recorded, content hash recomputed."""
    res = {}
    mdir = out_root / "manifests"
    for mid in PRE_LOCK_IDS:
        marker = json.loads((mdir / f"{mid}.OK").read_text())
        header = json.loads((mdir / f"{mid}.header.json").read_text(),
                            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        ok = True
        for rel, h in marker["verified_sha256"].items():
            ok &= sha256_file(out_root / rel) == h
        for t in header["tables"]:
            tbl = pq.ParquetFile(mdir / f"{mid}.{t['table']}.parquet").read()
            ok &= table_content_sha256(reread_columns(tbl)) == t["table_content_sha256"] \
                and tbl.num_rows == t["row_count"]
            del tbl
        ok &= all(e["match"] for e in header["expected_counts_check"]) and all(header["assertions"].values())
        res[mid] = ok
    return {"all_verified": all(res.values()), "per_manifest": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DATA-04 freeze-step build")
    ap.add_argument("--out", default=None)
    ap.add_argument("--buckets", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else data_v2_root()
    ds = dataset_root()
    t0 = time.time()
    snap0 = snapshot_check()
    if not snap0["unchanged"]:
        raise SystemExit(f"snapshot drift before the build: {snap0['mismatched']}")
    p, st = write_state(out, ds)
    load_state(p)
    buckets = [int(x) for x in a.buckets.split(",")] if a.buckets else None
    build = ManifestBuilder(ds_root=ds, out_root=out, buckets=buckets).run()
    ver = verify_written(out)
    snap1 = snapshot_check()
    rec = {"receipt": "BUILD_ALL", "utc": utc_now(), "dataset_root": str(ds), "wall_s": round(time.time() - t0, 1),
           "transform_state_sha256": st["transform_state_sha256"],
           "transform_state_file_sha256": sha256_file(p), "snapshot_before": snap0, "snapshot_after": snap1,
           "verify": ver, "build_receipt": {k: build[k] for k in ("full_build", "manifest_bytes_total",
                                                                   "reconciliation_all_match", "wall_s")},
           "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                   "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")}}
    atomic_write_json(out, "receipts/BUILD_ALL.json", rec)
    ok = ver["all_verified"] and snap1["unchanged"] and build["reconciliation_all_match"]
    print(json.dumps({"ok": ok, **{k: rec[k] for k in ("wall_s", "transform_state_sha256")},
                      "verify": ver["all_verified"], "snapshot_unchanged": snap1["unchanged"]}))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
