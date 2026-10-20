"""v4base.strata — TRAIN-period user strata (contracts/PREG1_AMENDMENT_04.md §3; DOWNSTREAM_FINAL_PLAN_FABLE.md
(2)): category diversity (number of distinct main categories, terciles), brand loyalty (share of views on the
user's top brand, terciles), persona = diversity x loyalty (9 cells), and activity quantiles (n_train). Per MAIN
user, from TRAIN-period events ONLY.

Ordering rule (task packet's own wording, verbatim): "Cut points are computed on MAIN TRAIN and written once to a
JSON (O_EXCL) with their hashes BEFORE any value is read". Enforced structurally here: `compute_cutpoints`
(`build_registry`) never hands back a registry object that can be read without `write_registry` having just written
it (O_EXCL: an existing file is never silently reused/overwritten); `load_registry` is the only sanctioned read
path afterward and it verifies the self-hash before returning anything, refusing a tampered/hand-edited file.

**Key space (review fix, HIGH)**: the registry is keyed by `client_key_hex`, the ONLY safe cross-manifest join key
(contracts/MANIFEST_SPEC_V4.json: "Cross-manifest joins use client_key_hex only"). `client_ord` (a TRAIN manifest's
own dense rank) and `eval_user_key` (an evaluation manifest's own, separately-densified rank) are DIFFERENT
per-manifest integer spaces — the SAME dense integer can, and in general does, name a DIFFERENT real user in each
one. `scan_user_train_stats`/`main` therefore require the caller to supply `user_keys` = `client_key_hex` values
(one per canonical TRAIN position; see `v4base.identity.train_client_key_hex_by_canonical_position`), never a raw
`client_ord`. A user absent from the registry (no MAIN-TRAIN row under this manifest — e.g. a cold/U_NEW eval user)
gets the explicit `NO_TRAIN_STRATUM` sentinel from `assign_stratum_for_user`, and is NEVER bucketed by inventing a
value for it.

De-duplication (review fix, MED): a user's TRAIN decisions' windows overlap (a later decision's window is a
superset of an earlier one's, up to the L-cap), so scanning every decision's own window would count the SAME real
event once per later decision that still covers it (up to ~L times), inflating `top_brand_share`.
`scan_user_train_stats` uses `sessions.scan_best_window_per_user` to read each user's single longest window exactly
once (category diversity, a set union, was never biased by this; brand loyalty, a view COUNT, was).

Judgment calls:
  1. "Activity quantiles" = QUARTILES (4 bins; Ops confirmed this choice). No existing "activity quantile" BIN
     scheme was found anywhere in this codebase to reuse (searched v4eval, census_v2, v4data06, company/reviews):
     only fixed activity BANDS (`census_v2.core.stratum_codes`: B/C/D by raw n, not a quantile split) and a generic
     percentile-REPORTING grid (`census_v2.core.QUANTILE_POINTS`, min/p10/.../max, not a bin scheme either) exist.
  2. Only KNOWN (>= 3, `sessions.ITEM_TOKEN_OFFSET`) `main_category_tokens`/`brand_tokens` positions count toward
     diversity/loyalty (a MISSING/OOV/PAD event contributes to n_train activity but is not a "view of a
     category/brand"); a user with zero known-category views gets diversity 0 and loyalty share 0.0 (never a
     divide-by-zero).
  3. Percentile method: `numpy.percentile(..., method="linear")` — the repo-wide convention (census_v2/core.py,
     featstat_v1/fs_core.py, v4eval/bootstrap.py all use it).
  4. Bucket boundary rule: `numpy.searchsorted(cuts, value, side="right")`, i.e. a value exactly AT a cut point
     falls into the LOWER bin. An explicit, arbitrary-but-documented choice; ties are common for n_train /
     n_distinct_main_categories (small integers).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .guard import RssGuard
from .sessions import ITEM_TOKEN_OFFSET, scan_best_window_per_user

KNOWN_START = ITEM_TOKEN_OFFSET                 # known >= 3 (PAD 0 / MISSING 1 / OOV 2), the universal Clean V2 rule
DEFAULT_FIT_CHUNK = 2048
DIVERSITY_N_BINS = 3                            # terciles
LOYALTY_N_BINS = 3                              # terciles
ACTIVITY_N_BINS = 4                              # quartiles (judgment call 1, module docstring; Ops accepted)
FORMAT = "PPSI_V4_BASE02_STRATA_V1"
SELF_KEY = "registry_self_sha256"
NO_TRAIN_STRATUM = "NO_TRAIN"                    # a user with no row in THIS registry's source TRAIN manifest


def _percentile_cuts(values: np.ndarray, n_bins: int) -> list:
    """`n_bins - 1` interior cut points at the equally-spaced percentiles (linear method, the repo convention)."""
    if values.size == 0:
        return [0.0] * (n_bins - 1)
    pts = [100.0 * i / n_bins for i in range(1, n_bins)]
    return [float(v) for v in np.percentile(values.astype(np.float64), pts, method="linear")]


def bucket(value: float, cuts) -> int:
    """0-based bin index: value <= cuts[i] -> bin i (searchsorted 'right' side; see module docstring, judgment 4)."""
    return int(np.searchsorted(np.asarray(list(cuts), dtype=np.float64), float(value), side="right"))


def scan_user_train_stats(provider, *, user_keys, chunk: int = DEFAULT_FIT_CHUNK,
                          guard: Optional[RssGuard] = None) -> Dict[str, dict]:
    """One user-de-duplicated MAIN-TRAIN scan: {client_key_hex: {"n_train", "n_distinct_main_categories",
    "top_brand_share"}}. `user_keys` MUST be `client_key_hex` values (one per canonical TRAIN position — see
    `v4base.identity.train_client_key_hex_by_canonical_position`), never a raw `client_ord` (module docstring, key
    space). `n_train` is the user's own count of TRAIN decisions this provider serves (never de-duplicated: it is a
    count of DECISIONS, not events). Diversity/loyalty count only KNOWN (>= 3) `main_category_tokens`/`brand_tokens`
    positions of the user's SINGLE longest TRAIN window (`sessions.scan_best_window_per_user`)."""
    best = scan_best_window_per_user(provider, user_keys=user_keys,
                                     extra_fields=("main_category_tokens", "brand_tokens"), chunk=chunk, guard=guard)
    out: Dict[str, dict] = {}
    for u, row in best.items():
        L_valid = int(row["lengths"])
        cats = [c for c in row["main_category_tokens"][:L_valid] if c >= KNOWN_START]
        brs = [b for b in row["brand_tokens"][:L_valid] if b >= KNOWN_START]
        top = 0
        if brs:
            counts: Dict[int, int] = {}
            for b in brs:
                counts[b] = counts.get(b, 0) + 1
            top = max(counts.values())
        out[u] = {"n_train": int(row["n_train"]), "n_distinct_main_categories": len(set(cats)),
                  "top_brand_share": (top / len(brs)) if brs else 0.0}
    return out


def _self_sha256(obj: dict) -> str:
    body = {k: v for k, v in obj.items() if k != SELF_KEY}
    text = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_exclusive(path, text: str) -> str:
    """Create `path` with O_EXCL (an existing file is never overwritten), LF, fsync; returns the file sha256 — the
    same convention `v4models/variants.py` and v4train's other O_EXCL writers already use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(data).hexdigest()


def build_registry(user_stats: Dict[str, dict], *, source_manifest_id: Optional[str] = None) -> dict:
    """The frozen object: cut points (computed on MAIN TRAIN, once) + the per-user (`client_key_hex`-keyed) table +
    a self-hash."""
    n_train_vals = np.asarray([v["n_train"] for v in user_stats.values()], dtype=np.float64)
    div_vals = np.asarray([v["n_distinct_main_categories"] for v in user_stats.values()], dtype=np.float64)
    loy_vals = np.asarray([v["top_brand_share"] for v in user_stats.values()], dtype=np.float64)
    cuts = {"activity_quantile_cuts": _percentile_cuts(n_train_vals, ACTIVITY_N_BINS),
           "category_diversity_tercile_cuts": _percentile_cuts(div_vals, DIVERSITY_N_BINS),
           "brand_loyalty_tercile_cuts": _percentile_cuts(loy_vals, LOYALTY_N_BINS)}
    obj = {"format": FORMAT, "task": "BASE-02 item 3", "source_manifest_id": source_manifest_id,
          "user_key_field": "client_key_hex", "n_users": len(user_stats),
          "n_bins": {"activity": ACTIVITY_N_BINS, "category_diversity": DIVERSITY_N_BINS,
                     "brand_loyalty": LOYALTY_N_BINS},
          "percentile_method": "linear", "cuts": cuts, "users": {str(u): v for u, v in sorted(user_stats.items())}}
    obj[SELF_KEY] = _self_sha256(obj)
    return obj


def write_registry(path, user_stats: Dict[str, dict], **kwargs) -> dict:
    obj = build_registry(user_stats, **kwargs)
    text = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    file_sha256 = write_exclusive(path, text)
    return {"path": str(path), "file_sha256": file_sha256, "registry_self_sha256": obj[SELF_KEY]}


def load_registry(path) -> dict:
    """Read + verify the self-hash BEFORE any cut point / user row is used (the task packet's own ordering rule)."""
    p = Path(path)
    obj = json.loads(p.read_bytes().decode("utf-8"))
    if obj.get("format") != FORMAT:
        raise ValueError(f"{p}: not a {FORMAT} strata registry")
    want, got = obj.get(SELF_KEY), _self_sha256(obj)
    if want != got:
        raise ValueError(f"{p}: self-hash mismatch (want {want}, recomputed {got}) — refusing a tampered registry")
    return obj


def assign_stratum_from_values(registry: dict, *, n_train: int, n_distinct_main_categories: int,
                               top_brand_share: float) -> dict:
    """Given one user's own three MAIN-TRAIN numbers and an already-loaded (self-hash-verified) registry, return
    the stratum labels."""
    cuts = registry["cuts"]
    act = bucket(n_train, cuts["activity_quantile_cuts"])
    div = bucket(n_distinct_main_categories, cuts["category_diversity_tercile_cuts"])
    loy = bucket(top_brand_share, cuts["brand_loyalty_tercile_cuts"])
    return {"status": "BUCKETED", "activity_quantile": act, "category_diversity_tercile": div,
           "brand_loyalty_tercile": loy, "persona": div * LOYALTY_N_BINS + loy}


def assign_stratum_for_user(registry: dict, client_key_hex: str) -> dict:
    """Convenience over the registry's OWN per-user table, keyed by `client_key_hex` (module docstring, key space;
    NEVER a raw `client_ord`/`eval_user_key` — translate first via `v4base.identity`). A user with no row in this
    registry (no MAIN-TRAIN decision under the source manifest this registry was built from — e.g. cold/U_NEW) gets
    the explicit `NO_TRAIN_STRATUM` sentinel, never a bucketed guess."""
    row = registry["users"].get(str(client_key_hex))
    if row is None:
        return {"status": NO_TRAIN_STRATUM, "activity_quantile": None, "category_diversity_tercile": None,
               "brand_loyalty_tercile": None, "persona": None}
    return assign_stratum_from_values(registry, n_train=row["n_train"],
                                      n_distinct_main_categories=row["n_distinct_main_categories"],
                                      top_brand_share=row["top_brand_share"])


def main(argv=None) -> int:
    """`python -B -m v4base.strata --out <json>` (task item 4). MAIN still needs `--allow-main-validation`
    (Ops passes it only from P3 on), even though this CLI only ever reads MAIN's TRAIN split."""
    import argparse

    from . import identity
    from .guard import configure_low_priority
    from .run import REGISTRY_V2_PATH, check_main_gate, resolve_phase_manifests

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-main-validation", action="store_true")
    ap.add_argument("--history-policy", default="USER_HISTORY_L50")
    ap.add_argument("--registry", default=str(REGISTRY_V2_PATH))
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--fit-chunk", type=int, default=None)
    args = ap.parse_args(argv)

    check_main_gate("MAIN", args.allow_main_validation)

    configure_low_priority(args.n_threads)
    guard = RssGuard()

    import v4train  # noqa: F401  (ppsi_env first, before v4int's torch/store imports)
    from v4int.providers import StoreTrainProvider
    from v4int.verify import DataPaths

    registry_path = Path(args.registry)
    train_id = args.train_manifest
    if train_id is None:
        train_id, _ = resolve_phase_manifests("MAIN", registry_path)

    paths = DataPaths.default()
    train_provider = StoreTrainProvider(train_id, history_policy=args.history_policy, paths=paths,
                                        manifest_registry=registry_path)
    user_keys = identity.train_client_key_hex_by_canonical_position(train_provider, manifest_dir=paths.manifest_dir)
    kwargs = {} if args.fit_chunk is None else {"chunk": args.fit_chunk}
    stats = scan_user_train_stats(train_provider, user_keys=user_keys, guard=guard, **kwargs)
    receipt = write_registry(args.out, stats, source_manifest_id=train_id)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["KNOWN_START", "DIVERSITY_N_BINS", "LOYALTY_N_BINS", "ACTIVITY_N_BINS", "NO_TRAIN_STRATUM",
          "scan_user_train_stats", "bucket", "build_registry", "write_registry", "load_registry",
          "assign_stratum_from_values", "assign_stratum_for_user", "write_exclusive", "main"]
