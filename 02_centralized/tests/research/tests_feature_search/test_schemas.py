"""SCHEMA-01 derived schemas and registries (contracts/derived_schemas; v4fs.schemas).

  * every written schema and the registry verify their self-hashes; the registry's file sha256 and schema self-hash
    match the files; tampering (a byte of a schema file, the registry body, a swapped path) is refused;
  * L0 keeps every Clean V2 channel (53 information channels); L3 keeps exactly the FEATSTAT ladder's L3 lists
    (30 channels) and its widths; the base line names FEATURE_SCHEMA_CLEAN_V2 2.1.0 with the verified self-hash;
  * the 14 binding FS groups are an exact partition of L3's inputs minus item_tokens (28 channels; the ladder's
    "30" counts item_tokens and the output catalogue);
  * derive("L3", groups): the id rule, the dropped channels, refusals (ITEM_ID, unknown / absent groups);
  * the theta0 registry merge refuses a changed value and keeps both torch builds apart.
"""
from __future__ import annotations

import copy
from pathlib import Path
import json
import shutil

import pytest

import ppsi_paths
from fskit import BASE, L0, L0S, L3, L3S, assert_raises, nc
from v4fs import schemas as S
from v4fs.importance import check_groups
from v4models.derived import full_keep, self_hash
from v4models.schema import SchemaError


def test_written_schemas_and_registry_self_hashes():
    reg = S.read_registry()
    assert reg["self_sha256"] == self_hash(reg)
    assert sorted(reg["schemas"]) == ["L0S_CLEAN_V2_SESSION", "L0_CLEAN_V2", "L3S_NEAR_DUPLICATE_SESSION",   # + S* (FS stage B, DL 52)
                                      "L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7", "L3_NEAR_DUPLICATE"]
    assert reg["base"] == f"FEATURE_SCHEMA_CLEAN_V2 2.1.0 self {BASE.self_sha256}"
    assert BASE.self_sha256.startswith("55a244e1")
    for sid, ent in reg["schemas"].items():
        obj = json.loads((S.DERIVED_DIR / ent["path"]).read_text(encoding="utf-8"))
        assert obj["self_sha256"] == self_hash(obj) == ent["self_sha256"]
        assert S._sha(S.DERIVED_DIR / ent["path"]) == ent["file_sha256"]
        assert obj["base"] == reg["base"] and obj["schema_id"] == sid
        assert S.load_registered(sid).self_sha256 == ent["self_sha256"]
        der = obj.get("source", {}).get("derived_from")                 # S* (FS stage B, DL 52): derived by drop, not a rung
        fresh = S.derived_obj(S.derive(der["schema_id"], obj["source"]["dropped_groups"])) if der else S.schema_obj(sid)
        assert obj == fresh, f"{sid}: the file differs from a fresh generation"


def test_l0_is_the_full_clean_v2_and_l3_is_the_ladder_rung():
    assert L0.keep == {k: tuple(v) for k, v in full_keep(BASE).items()} and L0.n_information_channels == 53
    assert L0.projection.is_identity and L0.widths == BASE.widths
    r = S.rung("L3_NEAR_DUPLICATE")
    for blk, key in S.LADDER_BLOCKS.items():
        assert list(L3.keep[key]) == r["channels"][blk]
    assert L3.n_information_channels == r["n_information_channels"] == 30
    assert L3.raw["widths"] == {"tokens": 4, "numeric_dim": 10, "flags_dim": 3, "user_context_dim": 8,
                                "user_mask_dim": 4}
    assert L3.raw["source"]["ladder"]["sha256"] == S._sha(S.LADDER_PATH)


def test_the_fs_groups_partition_l3_minus_item_tokens():
    from v4fs.importance import groups_for
    gs = check_groups(L3S, groups_for(L3S))
    assert len(gs) == 15 and sorted(c for m in gs.values() for c in m) == sorted(
        c for c in L3S.channel_names if c not in ("item_tokens", "session_offset_tokens"))
    g = check_groups(L3, groups_for(L3))
    names = [c for m in g.values() for c in m]
    assert len(g) == 14 and len(names) == len(set(names)) == 28
    assert sorted(names) == sorted(c for c in L3.channel_names if c != "item_tokens")
    assert len(L3.channel_names) == 29 and L3.n_information_channels == 30          # + item_tokens + the catalogue


def test_registry_tampering_is_refused(tmp_path):
    d = tmp_path / "ds"
    shutil.copytree(S.DERIVED_DIR, d)
    assert S.load_registered("L3_NEAR_DUPLICATE", d / "REGISTRY.json").schema_id == "L3_NEAR_DUPLICATE"
    p = d / "L3_NEAR_DUPLICATE.json"
    p.write_bytes(p.read_bytes().replace(b'"title"', b'"title" ', 1))            # same JSON, other bytes
    assert_raises(SchemaError, S.load_registered, "L3_NEAR_DUPLICATE", d / "REGISTRY.json")
    shutil.copy(S.DERIVED_DIR / "L3_NEAR_DUPLICATE.json", p)
    reg = json.loads((d / "REGISTRY.json").read_text(encoding="utf-8"))
    reg["schemas"]["L3_NEAR_DUPLICATE"]["path"] = "L0_CLEAN_V2.json"                # swapped path, stale hash
    (d / "REGISTRY.json").write_text(json.dumps(reg), encoding="utf-8")
    assert_raises(SchemaError, S.read_registry, d / "REGISTRY.json")
    reg["self_sha256"] = self_hash(reg)                                             # consistent hash, wrong file
    (d / "REGISTRY.json").write_text(json.dumps(reg), encoding="utf-8")
    assert_raises(SchemaError, S.load_registered, "L3_NEAR_DUPLICATE", d / "REGISTRY.json")
    assert_raises(SchemaError, S.load_registered, "L9_ID_ONLY_ANCHOR")               # not registered


def test_derive_l3_minus_groups():
    d = S.derive("L3_NEAR_DUPLICATE", ["G12c", "G8"])
    assert d.schema_id == "L3_NEAR_DUPLICATE-minus-G12c+G8"
    gone = {"recent_mean_log_price", "current_minus_recent_mean_log_price", "recent_price_supported",
            "log1p_segment_elapsed_s"}
    assert set(L3.channel_names) - set(d.channel_names) == gone and d.n_information_channels == 26
    assert sorted(d.raw["source"]["dropped_channels"]) == sorted(gone)
    assert d.raw["source"]["derived_from"] == {"schema_id": "L3_NEAR_DUPLICATE", "self_sha256": L3.self_sha256}
    assert S.derive("L3_NEAR_DUPLICATE", ["G8", "G12c"]).self_sha256 == d.self_sha256        # order-free
    assert S.derive("L3_NEAR_DUPLICATE", []).channel_names == L3.channel_names
    every = S.derive("L3_NEAR_DUPLICATE", [g for g in S.FS_GROUPS if g not in ("G1", "G13")])   # G1 REQUIRED; G13: L3S
    assert every.channel_names == ("item_tokens", "category_tokens") and every.n_information_channels == 3
    assert_raises(SchemaError, S.derive, "L3_NEAR_DUPLICATE", ["G13"])                     # not an L3 input
    f = S.derive("L3_NEAR_DUPLICATE", ["USER_MATCH"])                                     # a FEATSTAT group
    assert "recent_brand_match_share" not in f.channel_names
    assert_raises(SchemaError, S.derive, "L3_NEAR_DUPLICATE", ["ITEM_ID"])
    assert_raises(SchemaError, S.derive, "L3_NEAR_DUPLICATE", ["G99"])
    assert_raises(SchemaError, S.derive, "L3_NEAR_DUPLICATE", ["SIMULTANEOUS_BLOCK"])      # absent from L3


def test_session_schemas_l0s_l3s():
    """SCHEMA-01R1 (DECISION_LOG 47 / 48): L0S = L0 + session_offset_tokens + G13; L3S = L3 + the same five; the
    extension contract is hashed with each schema; L0S is not `is_full` (the L0 guard is L0's)."""
    ext = ["session_offset_tokens", "session_mode_main_category_tokens", "current_matches_session_mode",
           "session_mode_share", "current_matches_session_mode_available"]
    for s_, b_ in ((L0S, L0), (L3S, L3)):
        assert sorted(set(s_.channel_names) - set(b_.channel_names)) == sorted(ext)
        assert set(b_.channel_names) <= set(s_.channel_names) and not s_.is_full
        assert s_.n_information_channels == b_.n_information_channels + 5
        assert sorted(s_.raw["extension_channels"]) == sorted(ext)
        assert s_.keep["categorical"][-2:] == ("session_offset_tokens", "session_mode_main_category_tokens")
        assert s_.keep["user_context"][-2:] == ("current_matches_session_mode", "session_mode_share")
        assert s_.keep["user_context_masks"][-1] == "current_matches_session_mode_available"
    assert (L0S.n_information_channels, L3S.n_information_channels) == (58, 35)
    assert L0.is_full and "extension_channels" not in L0.raw and "extension_channels" not in L3.raw
    obj = copy.deepcopy(L3S.raw)
    obj["extension_channels"]["session_offset_tokens"]["vocab"] = 10                  # a different contract
    obj["self_sha256"] = self_hash(obj)
    from v4models.derived import derived_from_obj
    assert_raises(SchemaError, derived_from_obj, obj, BASE)
    obj = copy.deepcopy(L3S.raw)                                                       # a value without its mask
    obj["keep"]["user_context_masks"] = obj["keep"]["user_context_masks"][:-1]
    obj["widths"]["user_mask_dim"] -= 1
    obj["n_information_channels"] -= 1
    del obj["extension_channels"]["current_matches_session_mode_available"]
    obj["self_sha256"] = self_hash(obj)
    assert_raises(SchemaError, derived_from_obj, obj, BASE)


def test_required_channels_are_never_dropped_and_g13_is_droppable():
    assert S.REQUIRED_CHANNELS == ("item_tokens", "category_tokens", "session_offset_tokens")
    for g in ("G1", "ITEM_ID", "ITEM_METADATA_TOKENS"):
        assert_raises(SchemaError, S.derive, "L3S_NEAR_DUPLICATE_SESSION", [g])
        assert_raises(SchemaError, S.derive, "L3S_NEAR_DUPLICATE_SESSION", [g, "G8"])
    d = S.derive("L3S_NEAR_DUPLICATE_SESSION", ["G13"])
    assert d.schema_id == "L3S_NEAR_DUPLICATE_SESSION-minus-G13" and d.n_information_channels == 31
    assert not (set(S.FS_GROUPS["G13"]) & set(d.channel_names))
    assert {"item_tokens", "category_tokens", "session_offset_tokens"} <= set(d.channel_names)
    every = S.derive("L3S_NEAR_DUPLICATE_SESSION", [g for g in S.FS_GROUPS if g != "G1"])
    assert every.channel_names == ("item_tokens", "category_tokens", "session_offset_tokens")


@nc("a derive without the REQUIRED-channel guard drops category_tokens")
def test_nc_dropping_a_required_channel(monkeypatch):
    monkeypatch.setattr(S, "REQUIRED_CHANNELS", ("item_tokens",))
    d = S.derive("L3S_NEAR_DUPLICATE_SESSION", ["G1"])
    assert "category_tokens" in d.channel_names, "a REQUIRED channel was dropped"


def test_theta0_registry_merge():
    rec = {"schema_id": "L3_NEAR_DUPLICATE", "schema_self_sha256": L3.self_sha256, "torch": "9.9.9+x",
           "family": "GRU", "seed": 2026, "init_sha256": "a" * 64, "raw_init_sha256": "a" * 64,
           "manifest_digest": "b" * 64, "shared_numel": 20961310, "K": 158486, "host": "h", "utc": "t"}
    reg = S.theta0_obj(None, [rec])
    assert reg["self_sha256"] == self_hash(reg)
    ent = reg["schemas"]["L3_NEAR_DUPLICATE"]["builds"]["9.9.9+x"]["GRU"]["2026"]
    assert ent["init_sha256"] == "a" * 64
    other = dict(rec, torch="1.0.0+y", init_sha256="c" * 64)
    reg2 = S.theta0_obj(reg, [other])
    assert sorted(reg2["schemas"]["L3_NEAR_DUPLICATE"]["builds"]) == ["1.0.0+y", "9.9.9+x"]
    assert_raises(SchemaError, S.theta0_obj, reg2, [dict(rec, init_sha256="d" * 64)])
    assert S.theta0_obj(reg2, [dict(rec, utc="later")])["schemas"] == reg2["schemas"]      # same value: no change


@pytest.mark.skipif(not S.THETA0_PATH.is_file(), reason="THETA0_REGISTRY.json not written yet")
def test_the_committed_theta0_registry_verifies():
    reg = S.read_theta0()
    for sid, ent in reg["schemas"].items():
        assert S.load_registered(sid).self_sha256 == ent["self_sha256"]
        for build, fams in ent["builds"].items():
            for fam, seeds in fams.items():
                assert sorted(seeds) == ["2026", "2027"], (sid, build, fam)
    assert {"L0S_CLEAN_V2_SESSION", "L3S_NEAR_DUPLICATE_SESSION"} <= set(reg["schemas"])


V1_ROOTS = [Path(r"F:\Queens University\Microsoft Project\eda\rees46_nextitem_v1"),
            ppsi_paths.DATA_ROOT]
V1 = next((r for r in V1_ROOTS if (r / "mappings" / "catalogue.parquet").is_file()), None)


@pytest.mark.skipif(V1 is None or not S.THETA0_PATH.is_file(), reason="no v1 mapping root or no theta0 registry")
@pytest.mark.parametrize("family", ["GRU", "SASREC"])
@pytest.mark.parametrize("sid", ["L3_NEAR_DUPLICATE", "L3S_NEAR_DUPLICATE_SESSION", "L0S_CLEAN_V2_SESSION"])
def test_registered_theta0_equals_the_running_build_full_k(family, sid):
    """The registry value for the RUNNING torch build is what v4models.build gives (both seeds, full K)."""
    import gc

    import torch
    from v4models.build import build
    torch.set_num_threads(4)
    ds = S.load_registered(sid)
    for seed in (2026, 2027):
        want = S.registered_theta0(sid, ds.self_sha256, family, seed, torch.__version__)
        if want is None:
            pytest.skip(f"no {sid} theta0 registered for torch {torch.__version__}")
        bm = build(family, ds, seed, "cpu", v1_root=V1)
        got = (bm.init_sha256, bm.raw_init_sha256, bm.manifest_digest, bm.manifest.shared_numel)
        del bm
        gc.collect()
        assert got == (want["init_sha256"], want["raw_init_sha256"], want["manifest_digest"], want["shared_numel"])
