"""DEV-02 tests: `v4dev.dump_rows`'s pure / provider-duck-typed core, against synthetic `v4eval.manifest.EvalManifest`
fixtures (no Clean V2 store, no server -- `build_eval_provider` itself is real-server-only and is not exercised
here, same discipline as DEV-01's onnxruntime-session code)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from dev_testkit import assert_raises, nc
from v4dev.bench import load_rows_npz, validate_rows_layout
from v4dev.dump_rows import (DumpRowsError, dump_from_provider, first_users_selection,
                             population_selection_first_users_mask, write_rows)
from v4eval.manifest import build_eval_manifest

K = 32
L = 6


def _synthetic_manifest(*, n_users=6, rows_per_user=3, seed=0, some_non_eval=False, manifest_id="SYN_EVAL_TEST"):
    """A real, validated `EvalManifest`: `n_users` users interleaved (arbitrary but deterministic) over
    ascending decision_id, every row eval_mask=True (rankable, non-OOV) unless `some_non_eval` marks a few rows
    non-evaluated (to check `manifest_row` uses full-manifest, not stream, positions)."""
    rng = np.random.default_rng(seed)
    n = n_users * rows_per_user
    decision_id = np.arange(1000, 1000 + n, dtype=np.int64)
    euk_pool = np.repeat(np.arange(n_users, dtype=np.int64), rows_per_user)
    rng.shuffle(euk_pool)                                   # interleaved: manifest (decision_id) order != user order
    eval_mask = np.ones(n, dtype=bool)
    if some_non_eval:
        eval_mask[1] = False
        eval_mask[4] = False
    target_class = np.where(eval_mask, rng.integers(0, K, size=n), -1).astype(np.int64)
    target_oov = np.zeros(n, dtype=bool)
    censor_class = np.where(eval_mask, "NONE", "CEN_NO_NEXT")
    header = {"catalogue_K": K, "manifest_id": manifest_id}
    columns = {"decision_id": decision_id, "eval_user_key": euk_pool, "eval_mask": eval_mask,
              "target_class": target_class, "target_oov": target_oov, "censor_class": censor_class}
    return build_eval_manifest(header, columns)


def _synthetic_feature_batch(manifest, *, seed=0):
    """Per-E2E-row synthetic feature arrays (right-padded, `lengths >= 1` by construction), in manifest order,
    plus the manifest's own `decision_id` / `target_class` on those same rows (module docstring: the provider's
    batch already carries them)."""
    rng = np.random.default_rng(seed)
    stream = np.flatnonzero(manifest.eval_mask)
    n = int(stream.size)
    lengths = rng.integers(1, L + 1, size=n).astype(np.int64)
    ar = np.arange(L)[None, :]
    attention_mask = (ar < lengths[:, None]).astype(np.int64)
    item_tokens = rng.integers(1, 50, size=(n, L)).astype(np.int64)
    return {"decision_id": manifest.decision_id[stream].copy(), "target_class": manifest.target_class[stream].copy(),
            "lengths": lengths, "attention_mask": attention_mask, "item_tokens": item_tokens}


class _FakeEvalProvider:
    """Duck-types `v4int.providers.StoreEvalProvider`'s public surface `dump_from_provider` reads."""

    def __init__(self, manifest, feature_batch, *, eval_batch=4, manifest_content_sha256="deadbeef" * 8,
                history_policy="USER_HISTORY_L50"):
        self.manifest = manifest
        self._feature_batch = feature_batch
        self.eval_batch = eval_batch
        self.manifest_binding = {"content_sha256": manifest_content_sha256}
        self.history_policy = history_policy
        self.manifest_id = manifest.manifest_id

    def batches(self, bs):
        n = next(iter(self._feature_batch.values())).shape[0]
        for start in range(0, n, bs):
            end = min(n, start + bs)
            yield {k: v[start:end] for k, v in self._feature_batch.items()}


INPUT_KEYS = ("item_tokens", "lengths", "attention_mask")


def _hexes(n_users):
    """A synthetic client_key_hex per eval_user_key: sha256 digests in a deliberately v1-order-independent way (the
    hex order of the digests is NOT the selection_key_v2 order, which is what the tests below pin)."""
    return [hashlib.sha256(f"synthetic-user-{i}".encode()).hexdigest() for i in range(n_users)]


def _v2_order(euks, hexes):
    from census_v2.core import selection_key_v2
    return sorted(euks, key=lambda u: (selection_key_v2(bytes.fromhex(hexes[u])), u))


# --------------------------------------------------------------------------------------------- population selection
def test_population_selection_first_users_mask_none_keeps_everything():
    euk = np.array([2, 0, 1, 0, 2])
    mask = population_selection_first_users_mask(euk, None)
    assert mask.all()


def test_population_selection_first_users_mask_selects_by_selection_key_v2():
    hexes = _hexes(3)
    euk = np.array([2, 0, 1, 0, 2])
    first = _v2_order([0, 1, 2], hexes)[0]
    mask = population_selection_first_users_mask(euk, 1, hexes)
    assert list(mask) == [bool(u == first) for u in euk]


def test_population_selection_first_users_mask_zero_selects_nothing():
    mask = population_selection_first_users_mask(np.array([0, 1, 2]), 0, _hexes(3))
    assert not mask.any()


def test_population_selection_first_users_mask_refuses_negative():
    assert_raises(DumpRowsError, population_selection_first_users_mask, np.array([0, 1]), -1, _hexes(2))


def test_first_users_refuses_missing_client_key_hex():
    assert_raises(DumpRowsError, population_selection_first_users_mask, np.array([0, 1]), 1, None)


def test_first_users_selection_order_is_the_v2_key_order_not_the_eval_user_key_order():
    hexes = _hexes(40)
    euk = np.repeat(np.arange(40), 2)
    sel = first_users_selection(euk, hexes, 10)
    assert sel["keys"].tolist() == _v2_order(range(40), hexes)[:10]
    assert sel["keys"].tolist() != list(range(10))                   # the old `eval_user_key < N` rule differs
    assert sel["n_available"] == 40


# --------------------------------------------------------------------------------------------------- dump_from_provider
def test_dump_from_provider_layout_roundtrip_through_validate_rows_layout():
    manifest = _synthetic_manifest()
    provider = _FakeEvalProvider(manifest, _synthetic_feature_batch(manifest, seed=1))
    rows, meta = dump_from_provider(provider)
    validate_rows_layout(rows, INPUT_KEYS)                          # must not raise
    assert meta["n_rows"] == manifest.n == rows["manifest_row"].shape[0]
    assert meta["manifest_n"] == manifest.n
    assert meta["n_users"] == 6
    assert meta["first_users"] is None
    assert meta["manifest_id"] == "SYN_EVAL_TEST"
    assert meta["manifest_content_sha256"] == "deadbeef" * 8
    assert meta["history_policy"] == "USER_HISTORY_L50"


def test_dump_from_provider_roundtrips_through_disk(tmp_path):
    manifest = _synthetic_manifest(seed=2)
    provider = _FakeEvalProvider(manifest, _synthetic_feature_batch(manifest, seed=2))
    rows, meta = dump_from_provider(provider)
    result = write_rows(rows, meta, tmp_path / "out" / "rows.npz")
    assert result["rows_path"].is_file()
    assert result["meta_path"].name == "rows.meta.json"
    assert result["meta_path"].is_file()
    loaded = load_rows_npz(result["rows_path"])
    validate_rows_layout(loaded, INPUT_KEYS)
    assert np.array_equal(loaded["manifest_row"], rows["manifest_row"])
    assert result["meta"]["rows_npz_sha256"]


def test_manifest_row_uses_full_manifest_positions_not_stream_positions():
    manifest = _synthetic_manifest(some_non_eval=True)
    provider = _FakeEvalProvider(manifest, _synthetic_feature_batch(manifest, seed=3))
    rows, _ = dump_from_provider(provider)
    expected = np.flatnonzero(manifest.eval_mask)                   # full-manifest positions of the E2E rows
    assert np.array_equal(rows["manifest_row"], expected)
    assert 1 not in rows["manifest_row"].tolist()                   # the manually-marked non-eval rows
    assert 4 not in rows["manifest_row"].tolist()


def test_dump_from_provider_refuses_mismatched_batch_fields():
    manifest = _synthetic_manifest(seed=4)
    feat = _synthetic_feature_batch(manifest, seed=4)

    class _BadProvider(_FakeEvalProvider):
        def batches(self, bs):
            gen = super().batches(bs)
            first = next(gen)
            yield first
            second = next(gen)
            del second["item_tokens"]                                # a later batch drops a field
            yield second

    provider = _BadProvider(manifest, feat, eval_batch=4)
    assert_raises(DumpRowsError, dump_from_provider, provider)


# --------------------------------------------------------------------------------------------------- --first-users
def test_first_users_prefix_property_and_meta():
    manifest = _synthetic_manifest(n_users=6, rows_per_user=3, seed=5)
    hexes = _hexes(6)
    provider = _FakeEvalProvider(manifest, _synthetic_feature_batch(manifest, seed=5))
    rows_small, meta_small = dump_from_provider(provider, first_users=2, client_key_hex=hexes)
    rows_big, meta_big = dump_from_provider(provider, first_users=4, client_key_hex=hexes)

    assert meta_small["n_users"] == 2 and meta_big["n_users"] == 4
    assert set(rows_small["manifest_row"].tolist()).issubset(set(rows_big["manifest_row"].tolist()))
    order = _v2_order(range(6), hexes)
    stream = np.flatnonzero(manifest.eval_mask)
    euk_stream = manifest.eval_user_key[stream]
    assert np.array_equal(rows_small["manifest_row"], stream[np.isin(euk_stream, order[:2])])
    for meta, n in ((meta_small, 2), (meta_big, 4)):
        assert meta["first_users"] == n and meta["n_e2e_users_available"] == 6
        assert "selection_key_v2" in meta["first_users_rule"]
        want = hashlib.sha256("\n".join(sorted(hexes[u] for u in order[:n])).encode()).hexdigest()
        assert meta["first_users_set_sha256"] == want
    assert meta_small["first_users_set_sha256"] != meta_big["first_users_set_sha256"]


def test_first_users_count_is_exactly_n_when_available_and_users_without_e2e_rows_are_skipped():
    n_users = 30
    manifest = _synthetic_manifest(n_users=n_users, rows_per_user=2, seed=11, some_non_eval=True)
    hexes = _hexes(n_users)
    provider = _FakeEvalProvider(manifest, _synthetic_feature_batch(manifest, seed=11))
    stream = np.flatnonzero(manifest.eval_mask)
    with_e2e = set(np.unique(manifest.eval_user_key[stream]).tolist())
    for n in (1, 7, 16):
        rows, meta = dump_from_provider(provider, first_users=n, client_key_hex=hexes)
        assert meta["n_users"] == n
        kept = set(np.unique(manifest.eval_user_key[rows["manifest_row"]]).tolist())
        assert kept == set(_v2_order(sorted(with_e2e), hexes)[:n])
        assert meta["n_e2e_users_available"] == len(with_e2e)


def test_first_users_exceeding_population_keeps_everything():
    manifest = _synthetic_manifest(n_users=6, rows_per_user=3, seed=6)
    provider = _FakeEvalProvider(manifest, _synthetic_feature_batch(manifest, seed=6))
    rows_all, meta_all = dump_from_provider(provider, first_users=None)
    rows_over, meta_over = dump_from_provider(provider, first_users=1000, client_key_hex=_hexes(6))
    assert meta_over["n_users"] == meta_all["n_users"] == 6
    assert np.array_equal(rows_all["manifest_row"], rows_over["manifest_row"])


@nc("--first-users must NOT be the eval_user_key < N rule (that is the v1 digest order, not selection_key_v2)")
def test_nc_first_users_would_be_eval_user_key_prefix():
    manifest = _synthetic_manifest(n_users=40, rows_per_user=2, seed=12)
    provider = _FakeEvalProvider(manifest, _synthetic_feature_batch(manifest, seed=12))
    rows, _ = dump_from_provider(provider, first_users=5, client_key_hex=_hexes(40))
    kept = set(np.unique(manifest.eval_user_key[rows["manifest_row"]]).tolist())
    assert kept == {0, 1, 2, 3, 4}


# --------------------------------------------------------------------------------------------------- determinism
def test_dump_from_provider_is_deterministic():
    manifest = _synthetic_manifest(seed=7)
    feat = _synthetic_feature_batch(manifest, seed=7)
    hx = _hexes(6)
    rows_a, meta_a = dump_from_provider(_FakeEvalProvider(manifest, feat), first_users=3, client_key_hex=hx)
    rows_b, meta_b = dump_from_provider(_FakeEvalProvider(manifest, feat), first_users=3, client_key_hex=hx)
    assert meta_a == meta_b
    assert set(rows_a) == set(rows_b)
    for k in rows_a:
        assert np.array_equal(rows_a[k], rows_b[k])


def test_write_rows_content_is_identical_but_npz_sha_is_only_a_file_binding(tmp_path):
    """np.savez zip timestamps make rows_npz_sha256 non-reproducible across runs (disclosed); the ARRAYS are equal."""
    manifest = _synthetic_manifest(seed=8)
    feat = _synthetic_feature_batch(manifest, seed=8)
    rows, meta = dump_from_provider(_FakeEvalProvider(manifest, feat), first_users=3, client_key_hex=_hexes(6))
    r1 = write_rows(rows, meta, tmp_path / "a" / "rows.npz")
    r2 = write_rows(rows, meta, tmp_path / "b" / "rows.npz")
    a, b = load_rows_npz(r1["rows_path"]), load_rows_npz(r2["rows_path"])
    assert set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
    assert r1["meta"]["rows_npz_sha256"] and r2["meta"]["rows_npz_sha256"]


# --------------------------------------------------------------------------------------------------- negative control
@nc("a lengths=0 row must be refused by validate_rows_layout, not silently accepted, even after a full dump_rows "
    "round trip through disk")
def test_nc_dump_rows_would_let_a_length_zero_row_through():
    manifest = _synthetic_manifest(seed=9)
    feat = _synthetic_feature_batch(manifest, seed=9)
    feat["lengths"][0] = 0
    feat["attention_mask"][0, :] = 0                                 # keep padding contiguous (all-pad) for that row
    provider = _FakeEvalProvider(manifest, feat)
    rows, _ = dump_from_provider(provider)
    try:
        validate_rows_layout(rows, INPUT_KEYS)
        refused = False
    except ValueError:
        refused = True
    assert not refused, "validate_rows_layout correctly refused the length-0 row (this NC claims the opposite)"


# --------------------------------------------------------------------------------------------------- GATE-B B2
class _ProvenanceProvider(_FakeEvalProvider):
    """A real StoreEvalProvider batch also carries provenance fields (DEV-04 / DL 209): a str scalar, a tuple of channel
    names, a dict of counts, a str ndarray, a 0-d numpy scalar, a python int; plus one torch tensor array field."""

    def batches(self, bs):
        for b in super().batches(bs):
            n = int(b["lengths"].shape[0])
            yield dict(b, feature_view="S_STAR_V2", categorical_channel_names=("brand", "category"),
                       clean_v2_rc_counts={"rc0": 3, "rc1": 0}, channel_labels=np.array(["a"] * n),
                       schema_version=np.int64(2), n_channels=36,
                       session_offset_tokens=torch.arange(n, dtype=torch.long))


TEXT_FIELDS = ("categorical_channel_names", "channel_labels", "clean_v2_rc_counts", "feature_view", "n_channels",
               "schema_version")


def check_text_fields_dropped(dump=dump_from_provider):
    """B2 key property: a batch with text / tuple / dict / scalar fields dumps; exactly the numeric array fields are kept
    (plus manifest_row) and the dropped names are recorded; the kept arrays equal the clean provider's."""
    manifest = _synthetic_manifest(seed=11, some_non_eval=True)
    feat = _synthetic_feature_batch(manifest, seed=11)
    try:
        rows, meta = dump(_ProvenanceProvider(manifest, feat, eval_batch=4))
    except Exception as e:                                          # noqa: BLE001  (the crash IS the failure)
        raise AssertionError(f"dump_from_provider crashed on a batch with text fields: {type(e).__name__}: {e}")
    assert set(rows) == set(feat) | {"session_offset_tokens", "manifest_row"}
    assert meta["dropped_non_array_fields"] == sorted(TEXT_FIELDS)
    clean, clean_meta = dump_from_provider(_FakeEvalProvider(manifest, feat, eval_batch=4))
    assert all(np.array_equal(rows[k], clean[k]) for k in clean) and "dropped_non_array_fields" not in clean_meta
    assert rows["session_offset_tokens"].dtype == np.int64
    validate_rows_layout(rows, INPUT_KEYS)
    return rows, meta


def test_dump_from_provider_keeps_numeric_array_fields_only():
    check_text_fields_dropped()


def test_numeric_array_field_rule():
    from v4dev.dump_rows import numeric_array_field as ok
    assert ok(np.zeros(3, dtype=np.int64)) and ok(np.zeros((2, 3), dtype=np.float32)) and ok(np.zeros(2, dtype=bool))
    assert ok(torch.zeros(2, dtype=torch.long)) and ok(torch.zeros(2, 2))
    for bad in ("abc", b"ab", ("a", "b"), {"a": 1}, 3, 2.5, np.int64(2), np.array(5), np.array(["a", "b"]),
                np.array([{"a": 1}], dtype=object), [1, 2, 3], None):
        assert not ok(bad), bad


def test_validation_and_retro_dump_share_one_schema():
    """The same provider through v4int.dump_rows_retro.ArrayFieldsView (the RETRO_TEST wrapper) and directly through
    v4dev.dump_rows gives the same NPZ field set and arrays (one schema for VALIDATION and TEST)."""
    from v4int.dump_rows_retro import ArrayFieldsView
    manifest = _synthetic_manifest(seed=12)
    feat = _synthetic_feature_batch(manifest, seed=12)
    direct, _ = dump_from_provider(_ProvenanceProvider(manifest, feat, eval_batch=4))
    view = ArrayFieldsView(_ProvenanceProvider(manifest, feat, eval_batch=4))
    retro, _ = dump_from_provider(view)
    assert set(direct) == set(retro) and all(np.array_equal(direct[k], retro[k]) for k in direct)


def test_dump_refuses_a_batch_without_numeric_fields():
    manifest = _synthetic_manifest(seed=13)

    class _TextOnly(_FakeEvalProvider):
        def batches(self, bs):
            for _b in super().batches(bs):
                yield {"feature_view": "S_STAR_V2", "names": ("a",)}

    assert_raises(DumpRowsError, dump_from_provider, _TextOnly(manifest, _synthetic_feature_batch(manifest, seed=13)))


@nc("without the numeric-array filter (every batch field concatenated, the pre-B2 dump_rows) a batch with text fields "
    "must crash, not dump")
def test_nc_unfiltered_dump_accepts_text_fields(monkeypatch):
    import v4dev.dump_rows as DR
    monkeypatch.setattr(DR, "numeric_array_field", lambda v: True)
    check_text_fields_dropped(DR.dump_from_provider)
