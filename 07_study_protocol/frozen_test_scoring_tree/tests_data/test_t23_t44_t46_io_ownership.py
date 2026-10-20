"""T23 (client ownership), T44 (path escape), T45 (invalid JSON / NaN leaves no marker), T46 (atomic write under
interruption), and the TEST-manifest refusal (MANIFEST_SPEC global_rules.test_gate; T48-style).

Proves: a client-scoped view reads only its own contiguous rows; output paths cannot leave the workspace (including
sibling-prefix and symlink escapes); NaN/inf is refused before any file exists; an interrupted write leaves the
previous file intact and no temporary behind; no TEST manifest can be built and no TEST row is materialised.
Does not prove: physical or cryptographic isolation (one orchestrator reads all data; CLIENT_CONTRACT C5.6).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from data_testkit import assert_raises, nc
from v4data import common
from v4data.common import (PathEscapeError, TestGateError, atomic_write_bytes, atomic_write_json, safe_path,
                           write_marker)
from v4data.manifests import TEST_MANIFEST_IDS, main as build_main, read_non_test, refuse_test_manifest
from v4data.store import ClientView, CleanV2Store, OwnershipError


# ------------------------------------------------------------------------------------------------ T23
def _m01_tables(fake_build):
    m = fake_build["man"]["M01_COHORT_TRAIN"]["tables"]
    return m["clients"], m["decisions"]


def test_t23_client_view_reads_only_its_rows(fake_build):
    cli, dec = _m01_tables(fake_build)
    for i, key in enumerate(cli["client_key_hex"].tolist()):
        v = ClientView(None, cli, dec, key)
        ids = v.decision_ids()
        lo, n = int(cli["row_start"][i]), int(cli["row_count"][i])
        assert np.array_equal(ids, dec["decision_id"][lo:lo + n])
        assert np.all(dec["client_ord"][lo:lo + n] == i)
        others = np.setdiff1d(dec["decision_id"], ids)
        if others.size:
            assert_raises(OwnershipError, v.check_owned, others[:1])


def test_t23_unknown_or_duplicate_client_refused(fake_build):
    cli, dec = _m01_tables(fake_build)
    assert_raises(OwnershipError, ClientView, None, cli, dec, "0" * 64)
    bad = dict(cli)
    bad["row_count"] = cli["row_count"].copy()
    bad["row_count"][0] += 1                                   # range now runs into the next client
    if len(cli["client_ord"]) > 1:
        assert_raises(OwnershipError, ClientView, None, bad, dec, cli["client_key_hex"][0])


@nc("a bucket-scoped reader hands a client other users' rows (buckets are not clients)")
def test_nc_t23_bucket_view(fake_build):
    cli, dec = _m01_tables(fake_build)
    key = cli["client_key_hex"][0]
    bucket_rows = dec["decision_id"]                            # every client of the bucket
    own = ClientView(None, cli, dec, key).decision_ids()
    assert np.array_equal(np.sort(bucket_rows), np.sort(own))


# ------------------------------------------------------------------------------------------------ T44
def test_t44_path_escape_refused(tmp_path):
    root = tmp_path / "data_v2"
    root.mkdir()
    for bad in ("../x.json", "a/../../x.json", "/etc/passwd", "C:/Windows/x", "\\\\server\\share\\x"):
        assert_raises(PathEscapeError, safe_path, root, bad)
    (tmp_path / "data_v2_evil").mkdir()
    assert_raises(PathEscapeError, safe_path, root, "../data_v2_evil/x.json")
    assert safe_path(root, "manifests/x.json").parent == Path(os.path.realpath(root / "manifests"))


def test_t44_symlink_escape_refused(tmp_path):
    root = tmp_path / "data_v2"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, root / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this host")
    assert_raises(PathEscapeError, safe_path, root, "link/x.json")


@nc("a containment test without symlink resolution (abspath, not realpath) accepts an escaping symlink")
def test_nc_t44_symlink_naive(tmp_path):
    root = tmp_path / "data_v2"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, root / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this host")
    cand = os.path.abspath(root / "link" / "x.json")
    naive_inside = os.path.commonpath([os.path.abspath(root), cand]) == os.path.abspath(root)
    assert not naive_inside


@nc("a string-prefix containment test accepts the sibling directory data_v2_evil")
def test_nc_t44_prefix_check(tmp_path):
    root = tmp_path / "data_v2"
    cand = os.path.realpath(tmp_path / "data_v2_evil" / "x.json")
    naive_ok = cand.startswith(os.path.realpath(root))
    assert not naive_ok


# ------------------------------------------------------------------------------------------------ T45
def test_t45_nan_refused_before_any_file(tmp_path):
    for bad in (float("nan"), float("inf")):
        assert_raises(ValueError, atomic_write_json, tmp_path, "h.json", {"x": bad})
    assert not list(tmp_path.iterdir()), "a file or temp file was left behind"
    assert_raises(ValueError, write_marker, tmp_path, "m.OK", {})
    assert not (tmp_path / "m.OK").exists()


@nc("default json.dumps writes NaN, which is not strict JSON")
def test_nc_t45_default_json(tmp_path):
    txt = json.dumps({"x": float("nan")})
    json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(c)))


# ------------------------------------------------------------------------------------------------ T46
def test_t46_interrupted_write_keeps_previous_file(tmp_path, monkeypatch):
    atomic_write_bytes(tmp_path, "t.bin", b"old-content")

    def boom(*a, **k):
        raise KeyboardInterrupt("simulated interruption during rename")

    monkeypatch.setattr(common.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_bytes(tmp_path, "t.bin", b"new-content-that-must-not-appear")
    monkeypatch.undo()
    assert (tmp_path / "t.bin").read_bytes() == b"old-content"
    assert [p.name for p in tmp_path.iterdir()] == ["t.bin"], "a temporary file survived the interruption"


@nc("a non-atomic in-place write truncates the previous file when interrupted")
def test_nc_t46_in_place_write(tmp_path):
    p = tmp_path / "t.bin"
    p.write_bytes(b"old-content")
    try:
        with open(p, "wb") as fh:
            fh.write(b"new")
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert p.read_bytes() == b"old-content"


# ------------------------------------------------------------------------------------------------ TEST gate
@pytest.mark.parametrize("mid", TEST_MANIFEST_IDS + ("M99_ANY_TEST_THING",))
def test_test_manifest_refused(mid):
    assert_raises(TestGateError, refuse_test_manifest, mid)
    assert_raises(TestGateError, refuse_test_manifest, mid, Path("ALL_METHODS_LOCK.json"))


def test_test_manifest_refused_by_cli():
    assert_raises(TestGateError, build_main, ["--manifest", "M07_TEST_WARM"])


def test_store_refuses_test_split():
    assert_raises(TestGateError, CleanV2Store.build, splits=("TRAIN", "TEST"), buckets=[0])


def test_scan_never_materialises_test_rows(fake_build):
    t = read_non_test(fake_build["root"] / "decisions" / "user_bucket=0" / "ledger.parquet", ["split", "decision_id"])
    assert "TEST" not in set(t["split"].to_pylist())
    full = pq.ParquetFile(fake_build["root"] / "decisions" / "user_bucket=0" / "ledger.parquet").read(
        columns=["split", "decision_id"])
    test_ids = set(np.asarray(full["decision_id"])[np.asarray(full["split"].to_pylist()) == "TEST"].tolist())
    assert test_ids, "fixture must contain TEST rows"
    for mid, m in fake_build["man"].items():
        for tbl in m["tables"].values():
            for col in ("decision_id", "query_decision_id", "support_decision_id"):
                if col in tbl:
                    assert not (set(np.asarray(tbl[col]).tolist()) & test_ids), f"{mid} holds a TEST decision"


@nc("an unfiltered read materialises the TEST rows")
def test_nc_unfiltered_read(fake_build):
    full = pq.ParquetFile(fake_build["root"] / "decisions" / "user_bucket=0" / "ledger.parquet").read(columns=["split"])
    assert "TEST" not in set(full["split"].to_pylist())
