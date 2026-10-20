"""T3 parity test for the window cache (Task T-H).

Takes 512 random rows of the TRAIN manifest, reads them from the built
`CacheReader`, and independently collates the SAME `decision_id`s through a
fresh `DecisionWindowLoader` (never touching the cache). Every one of the 19
cached arrays must be EXACTLY equal (after `window_cache`'s documented
int32->int64 cast) -- not approximately.

A deliberately-broken variant reads the cache offset by one row while the
reference collation still targets the correct decisions, and is marked
`xfail(strict=True)`: it must actually fail, proving this test has power to
catch a real row-alignment bug.

Run on Lobot (needs the frozen dataset + a built TRAIN cache):
    CACHE_DIR=<...>/cache/train \\
    MANIFEST_PATH=<...>/artifacts/train_pilot_manifest.parquet \\
    DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1 \\
    <lobot python> -m pytest code/test_cache_parity.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import pyarrow.parquet as pq

CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rees46_eda.loaders import DecisionWindowLoader, detect_layout  # noqa: E402
from window_cache import ARRAY_NAMES, CacheReader, collate_rows  # noqa: E402

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/home/jovyan/shared_data/rees46_nextitem_v1"))
CACHE_DIR = Path(os.environ.get(
    "CACHE_DIR",
    "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1/cache/train"))
MANIFEST_PATH = Path(os.environ.get(
    "MANIFEST_PATH",
    "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1/artifacts/train_pilot_manifest.parquet"))
SPLIT = "TRAIN"
N_SAMPLE = 512
SEED = 20260922  # distinct from the manifest-construction seed (2026), fixed for reproducibility


def _load_manifest():
    t = pq.read_table(MANIFEST_PATH, columns=["decision_id", "user_bucket"])
    return (t["decision_id"].to_numpy(zero_copy_only=False),
            t["user_bucket"].to_numpy(zero_copy_only=False))


def _sample_rows(n_total: int):
    rng = np.random.default_rng(SEED)
    rows = rng.choice(n_total, N_SAMPLE, replace=False)
    rows.sort()
    return rows


def _reference_collate(dec_ids: np.ndarray, buckets: np.ndarray) -> dict[str, np.ndarray]:
    """Independently collate `dec_ids` (grouped by `buckets`) through a FRESH
    `DecisionWindowLoader`, never via the cache or `window_cache.build_cache`.
    Output arrays are placed at the LOCAL sample position (not the manifest
    row index), so they line up 1:1 with `CacheReader.read(rows)`'s output.
    """
    layout = detect_layout(DATASET_ROOT)
    loader = DecisionWindowLoader(DATASET_ROOT, 50, layout=layout)

    n = dec_ids.shape[0]
    out: dict[str, np.ndarray] = {}
    by_bucket: dict[int, list[int]] = {}
    for i, b in enumerate(buckets):
        by_bucket.setdefault(int(b), []).append(i)

    for b, local_idxs in by_bucket.items():
        local_idxs = np.asarray(local_idxs, dtype=np.int64)
        batch = collate_rows(loader, DATASET_ROOT, layout, SPLIT, b, dec_ids[local_idxs])
        for name in ARRAY_NAMES:
            if name not in out:
                sample = batch[name]
                out[name] = np.zeros((n,) + sample.shape[1:], dtype=sample.dtype)
            out[name][local_idxs] = batch[name]
        loader._buckets.pop(b, None)
    return out


@pytest.fixture(scope="module")
def sample():
    dec_all, buck_all = _load_manifest()
    rows = _sample_rows(dec_all.shape[0])
    dec_ids = dec_all[rows]
    buckets = buck_all[rows]
    reference = _reference_collate(dec_ids, buckets)
    return rows, dec_ids, reference, dec_all.shape[0]


def test_cache_parity_exact(sample):
    rows, dec_ids, reference, _n_total = sample
    reader = CacheReader(CACHE_DIR)
    cached = reader.read(rows)

    # Row alignment sanity: the cache's own decision_id channel must match
    # the sampled manifest decision_ids exactly, BEFORE comparing content.
    assert np.array_equal(cached["decision_id"].numpy(), dec_ids.astype(np.int64))

    compared = 0
    for name in ARRAY_NAMES:
        c = cached[name].numpy()
        d = reference[name]
        assert c.shape == d.shape, f"{name}: shape mismatch {c.shape} != {d.shape}"
        if name in reader.header.get("int32_cast_names", []):
            d = d.astype(np.int64)
        assert c.dtype == d.dtype, f"{name}: dtype mismatch {c.dtype} != {d.dtype}"
        assert np.array_equal(c, d), f"{name}: cached and directly-collated arrays differ"
        compared += 1

    assert compared == len(ARRAY_NAMES) == 19, (
        f"expected to compare exactly 19 arrays, compared {compared}")


@pytest.mark.xfail(strict=True, reason=(
    "deliberately broken: cache reader offset by one row while the reference "
    "still targets the correct decisions -- must fail, proving this test can fail"))
def test_cache_parity_broken_offset_xfail(sample):
    rows, dec_ids, reference, n_total = sample
    reader = CacheReader(CACHE_DIR)
    broken_rows = (rows + 1) % n_total  # off-by-one row alignment bug, injected on purpose
    cached = reader.read(broken_rows)

    for name in ARRAY_NAMES:
        c = cached[name].numpy()
        d = reference[name]
        if name in reader.header.get("int32_cast_names", []):
            d = d.astype(np.int64)
        assert np.array_equal(c, d), f"{name}: cached and directly-collated arrays differ"
