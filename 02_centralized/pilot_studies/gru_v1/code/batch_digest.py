"""Cross-platform batch digest (T-F).

Picks the 256 smallest ``decision_id``s from TRAIN in user_bucket 0 and 1
(deterministic: a sort, no RNG), collates them through the vendored
``DecisionWindowLoader``, and hashes every array of the resulting batch dict
with blake2b in a fixed (sorted) key order, folding each array's dtype and
shape into the digest input alongside its bytes.

Run once on LOCAL and once on LOBOT against their respective copies of the
frozen dataset; the two overall digests must be byte-for-byte identical. If
they are not, the per-key digests recorded in the JSON output pinpoint which
key first diverges.

Usage:
    python batch_digest.py --root <dataset_root> --label <local|lobot> --out <artifacts_dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402

from rees46_eda.loaders import DecisionWindowLoader  # noqa: E402
from index_utils import read_index_projected  # noqa: E402

CONTEXT_LENGTH = 50
SPLIT = "TRAIN"
BUCKETS = (0, 1)
N_PICK = 256
# Minimal columns collate() actually reads off each record dict (see
# loaders.py:collate): user_bucket, decision_id, target_class, loss_mask,
# eval_mask, and (user_id / client_idx) for client_idx resolution.
RECORD_COLUMNS = ["decision_id", "user_id", "client_idx",
                  "target_class", "loss_mask", "eval_mask"]


def pick_256_smallest(dataset_root: Path, layout: str) -> list[dict]:
    """The 256 globally-smallest decision_ids across TRAIN buckets 0 and 1.

    Deterministic: a stable sort on decision_id, no randomness anywhere.
    """
    t = read_index_projected(dataset_root, layout, SPLIT, RECORD_COLUMNS, buckets=BUCKETS)
    cols = t.column_names
    recs = [{c: t[c][i].as_py() for c in cols} for i in range(t.num_rows)]
    recs.sort(key=lambda d: d["decision_id"])
    if len(recs) < N_PICK:
        raise RuntimeError(f"only {len(recs)} TRAIN rows in buckets {BUCKETS}, need {N_PICK}")
    return recs[:N_PICK]


def digest_batch(batch: dict) -> tuple[str, dict]:
    """blake2b over every array in `batch`, in sorted key order.

    Returns (overall_hexdigest, {key: {sha: per-key-hexdigest, dtype, shape}}).
    Each array's dtype and shape are folded into BOTH the overall digest and
    the per-key digest, so a shape/dtype mismatch is caught even if the raw
    bytes happen to coincide.
    """
    overall = hashlib.blake2b()
    per_key = {}
    for key in sorted(batch.keys()):
        arr = np.ascontiguousarray(batch[key])
        dtype_s = str(arr.dtype)
        shape_s = str(tuple(int(x) for x in arr.shape))
        key_h = hashlib.blake2b()
        for h in (overall, key_h):
            h.update(key.encode("utf-8"))
            h.update(dtype_s.encode("utf-8"))
            h.update(shape_s.encode("utf-8"))
            h.update(arr.tobytes())
        per_key[key] = {"sha": key_h.hexdigest(), "dtype": dtype_s, "shape": shape_s}
    return overall.hexdigest(), per_key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="frozen dataset root")
    ap.add_argument("--label", required=True, help="local|lobot (free text host label)")
    ap.add_argument("--out", required=True, help="artifacts directory to write JSON into")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DecisionWindowLoader(root, CONTEXT_LENGTH)
    layout = loader.layout
    records = pick_256_smallest(root, layout)
    decision_ids = [r["decision_id"] for r in records]
    assert decision_ids == sorted(decision_ids)
    assert len(set(decision_ids)) == N_PICK

    batch = loader.collate(records)
    overall, per_key = digest_batch(batch)

    result = {
        "label": args.label,
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "root": str(root),
        "layout": layout,
        "context_length": CONTEXT_LENGTH,
        "split": SPLIT,
        "buckets": list(BUCKETS),
        "n_picked": N_PICK,
        "decision_id_min": decision_ids[0],
        "decision_id_max": decision_ids[-1],
        "decision_ids_sha256": hashlib.sha256(
            ",".join(str(d) for d in decision_ids).encode()).hexdigest(),
        "overall_digest_blake2b": overall,
        "keys_sorted": sorted(batch.keys()),
        "per_key": per_key,
    }
    out_path = out_dir / f"batch_digest_{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"label={args.label} host={platform.node()} layout={layout}")
    print(f"decision_id range=[{decision_ids[0]}, {decision_ids[-1]}] n={len(decision_ids)}")
    print(f"OVERALL_DIGEST={overall}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
