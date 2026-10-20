"""TRAIN-02R2 F3 and F4 (v4int).

  * F3: `bind_data_protocol_lock_v2` turns a v1 or otherwise-shaped lock into LaunchRefused (never a KeyError), and a
    non-smoke run binds only the lock pinned at G1 (the recomputed lock self-hash must equal G1_APPROVED.json's
    `lock_v2_self`; a missing gate file / field refuses); a self-consistent forged lock (edited and re-hashed) passed
    with --lock-path is refused;
  * F4: under LOCK_V2 the R4 population_hash needs population_sha256_selection_key_v2 (no fallback to the per-manifest
    population_sha256); a v1-shaped lock keeps the fallback.
Negative controls: a non-smoke bind without the G1 pin; a population_hash that falls back under LOCK_V2.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import v4int.launch_central as L
import v4int.providers as P
from intkit import assert_raises, nc
from test_lock_v2_bindings import _r4_providers
from v4int.providers import ProviderError, comparison_key_base
from v4int.verify import DataPaths, lock_self_sha256, verify_lock


def _gate(tmp_path, **fields) -> Path:
    p = tmp_path / "G1_APPROVED.json"
    p.write_text(json.dumps({"gate": "G1", **fields}), encoding="utf-8")
    return p


def _forged_lock(tmp_path) -> DataPaths:
    paths = DataPaths.default()
    lock = json.loads(Path(paths.lock_path).read_text(encoding="utf-8"))
    lock["data_version"]["name"] = lock["data_version"]["name"] + "_FORGED"
    lock["lock_self_sha256"] = lock_self_sha256(lock)                              # self-consistent
    p = tmp_path / "LOCK_FORGED.json"
    p.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    return replace(paths, lock_path=p)


def _check_lock_pin(tmp_path):
    paths = DataPaths.default()
    _, lself, _ = verify_lock(paths)
    ok = _gate(tmp_path, lock_v2_self=lself)
    ctx = L.bind_data_protocol_lock_v2(paths, smoke=False, gate_path=ok)
    assert ctx["lock_pinned_at_g1"] == lself
    forged = _forged_lock(tmp_path)
    assert verify_lock(forged)[1] != lself, "the forged lock is self-consistent but another lock"
    assert_raises(L.LaunchRefused, L.bind_data_protocol_lock_v2, forged, smoke=False, gate_path=ok)
    assert_raises(L.LaunchRefused, L.bind_data_protocol_lock_v2, paths, smoke=False, gate_path=_gate(tmp_path))
    assert_raises(L.LaunchRefused, L.bind_data_protocol_lock_v2, paths, smoke=False, gate_path=tmp_path / "no.json")


def test_f3_non_smoke_binds_only_the_pinned_lock(tmp_path):
    _check_lock_pin(tmp_path)


@nc("a non-smoke bind that accepts any self-consistent lock (no G1 pin)")
def test_nc_bind_without_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "check_lock_pin", lambda lock_self, **kw: lock_self)                   # MUTANT
    _check_lock_pin(tmp_path)


def test_f3_v1_or_other_lock_is_launch_refused(tmp_path):
    hist = DataPaths.historical()
    for smoke in (True, False):
        try:
            L.bind_data_protocol_lock_v2(hist, smoke=smoke, gate_path=_gate(tmp_path, lock_v2_self=verify_lock(hist)[1]))
        except L.LaunchRefused:
            pass
        else:
            raise AssertionError("a v1 lock must be refused")
    bad = tmp_path / "BAD_LOCK.json"
    bad.write_text("{}", encoding="utf-8")
    assert_raises(L.LaunchRefused, L.bind_data_protocol_lock_v2, replace(DataPaths.default(), lock_path=bad),
                  smoke=True)


def test_r3_smoke_is_a_required_keyword():
    """TRAIN-02R3 item 3: bind_data_protocol_lock_v2 has no smoke default (a caller must state it)."""
    import inspect
    prm = inspect.signature(L.bind_data_protocol_lock_v2).parameters["smoke"]
    assert prm.kind is inspect.Parameter.KEYWORD_ONLY and prm.default is inspect.Parameter.empty
    assert_raises(TypeError, L.bind_data_protocol_lock_v2, DataPaths.default())


def _check_population_hash():
    lock, tp, ep = _r4_providers()
    assert comparison_key_base(DataPaths.default(), tp, ep)["population_hash"] == "POPHASH_R4_TEST"
    tp.verified.header = {"population_sha256": "PER_MANIFEST_POPULATION"}           # no selection key under LOCK_V2
    assert_raises(ProviderError, comparison_key_base, DataPaths.default(), tp, ep)
    assert_raises(ProviderError, P.population_hash_of, {"population_sha256": "X"}, lock)
    v1_shaped = {"features_and_transforms": {"schema_self_sha256": "s", "transform_state_self_sha256": "t"}}
    assert P.population_hash_of({"population_sha256": "X"}, v1_shaped) == "X", "the v1 route keeps its fallback"
    assert_raises(ProviderError, P.population_hash_of, {}, v1_shaped)


def test_f4_population_hash_fallback_only_on_a_v1_lock():
    _check_population_hash()


@nc("a population_hash that falls back to population_sha256 under LOCK_V2 (the pre-R2 behaviour)")
def test_nc_population_fallback_under_v2(monkeypatch):
    monkeypatch.setattr(P, "population_hash_of",                                                 # MUTANT
                        lambda h, lock: h.get(P.POPULATION_HEADER_KEY) or h.get("population_sha256") or
                        (_ for _ in ()).throw(ProviderError("none")))
    _check_population_hash()
