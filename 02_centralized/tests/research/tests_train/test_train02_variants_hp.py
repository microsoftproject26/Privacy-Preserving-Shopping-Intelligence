"""TRAIN-02 items 0a, 0b, 0c, 1, 2 (config rules) and 3 — model variants, theta0 per variant, the HP registry, the
RunConfig rules and the S1 schedule.

  * 0a: MODEL_VARIANT_REGISTRY.json (O_EXCL, self-hash, equal to the v4models.variants table; tampering refused); the
    current builds GRU_D128 / SASREC_D256_B2 are bit-identical to today's (the variant=None build, the direct module
    constructor, and THETA0_REGISTRY.json's 2.9.1+cpu numel / theta0 / raw / manifest digest at the full K); the
    ContextGRUD mirror at d = 128 equals ContextGRU bitwise (base, L3, L0S schemas; parent source pinned); the new
    variants have the registered shapes; per-variant adapter equivalence (scores == module forward; topology derived
    from the module); theta0 determinism (two builds) and seed dependence; dropout via the build kwargs only;
  * 0b: v4train.theta0_variants: determinism, the guard against THETA0_REGISTRY (EQUAL at the full K for the current
    builds; a DIFFERENT value blocks the write), O_EXCL, a K-override file refused as a production registry, the
    read-only files unchanged; derived_schemas/REGISTRY.json and THETA0_REGISTRY.json byte-identical;
  * 0c: RunConfig carries model_variant / theta0_init_sha256 (and the other TRAIN-02 keys) and leaves them out of the
    config hash while None (the TRAIN-01R2 field set);
  * 1 (TRAIN-02R1, PREG1_AMENDMENT_02R2): HP_REGISTRY_P2R2.json: self-hash, trial sha256, readable ids, the §1 LR
    grid at full precision, the §2 cells identical across size levels (byte-identical multisets), every (size, p)
    pair, SASRec p counts 3 / 6 / 3, no control, O_EXCL; tampering refused; the superseded HP_REGISTRY_P2.json refused
    by the loader, validate() and the driver; the --verify CLI (F4) exits 0 / 1;
  * 2: RunConfig.validate accepts off-recipe LR / dropout / schedule / variant ONLY as the trial named by hp_id; role
    TUNING (COHORT / CALIBRATION, hp_id required, seeds 2026 / 2027, theta0 required without test_overrides) -> label
    HP_SEARCH, registry role CALIBRATION;
  * 3: S1 = the 02R1 knots; S0's planned_lr_table bit-identical to the TRAIN-01R2 implementation (reference copy +
    pinned hashes); T2 pins P1-P5 for S1 (fl.schedule interpolation parity on knots + 10,000 grid + tolerance zone).
Negative controls: a build that ignores the variant; a variant-registry loader without the code-table check; a theta0
writer that ignores the guard; an HP loader without verification; a loader that accepts the stale registry; a validate() without the HP rule; mutated S1 knots.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import v4train  # noqa: F401
from trainkit import KEY, SCHEMA, assert_raises, nc, raises_with, registry_keys
from fl.numerics import seeded_global_rng, state_digest
from v4models import build, gru_v2
from v4models import variants as V
from v4models.build import _default_vocab, make_adapter, module_forward
from v4models.monitors import common_mode_report
from v4models.synthetic import synthetic_examples_v2
from v4train import hp_registry as H
from v4train import schedule as SCH
from v4train import theta0_variants as T0
from v4train.recipe import RecipeError, RunConfig
import ppsi_paths

V4 = ppsi_paths.PROJECT_ROOT.resolve()       # the ppsi_v4 workspace (contracts, data)
REG_JSON = V4 / "contracts" / "derived_schemas" / "REGISTRY.json"
THETA0_JSON = V4 / "contracts" / "derived_schemas" / "THETA0_REGISTRY.json"
REG_JSON_SHA = "c23454cb7d34886e0374349b73a4d153907ebc56b17061926e193e221e4a0bc6"      # before TRAIN-02 (unchanged)
THETA0_JSON_SHA = "e4d8eb86bd4badfc3ab7671b132a49e68f8711617e618d6e76ba5e513aee369c"   # before TRAIN-02 (unchanged)
K_SMALL, FULL_K = 64, 158_486
FAMS = ("GRU", "SASREC")
TRAIN01R2_FIELDS = {"run_id", "family", "seed", "exposure_point", "cuda_visible_devices", "determinism_mode",
                    "manifest_registry", "data_version", "schema_registry", "role", "phase", "population", "peak_lr",
                    "effective_batch", "microbatch", "log_every", "checkpoint_every", "eval_config",
                    "continuation_budget", "calibration_record", "expected_init_sha256", "key", "test_overrides",
                    "recipe"}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _derived(sid):
    from v4fs.schemas import load_registered
    return load_registered(sid)


def _batch(n=6, seed=2, K=K_SMALL, session=False):
    b = synthetic_examples_v2(n, K, SCHEMA.widths, torch.Generator().manual_seed(seed),
                              with_session_offset=session, with_session_intent=session)
    return b


# ================================================================================================ 0a variants
def test_variant_registry_file_and_o_excl(tmp_path):
    r = V.load_variant_registry()
    assert r["variants"] == V.VARIANTS and sorted(V.VARIANTS) == sorted(
        ["GRU_D128", "GRU_D256", "SASREC_D256_B2", "SASREC_D256_B3", "SASREC_D384_B2", "SASREC_D64_B2"])   # Ops: + TRAIN-04 SMALL variant
    obj = json.loads(Path(r["path"]).read_text(encoding="utf-8"))
    assert obj["self_sha256"] == V.self_sha256(obj) == r["self_sha256"]
    assert b"\r" not in Path(r["path"]).read_bytes(), "LF line endings"
    assert V.DEFAULT_VARIANT == {"GRU": "GRU_D128", "SASREC": "SASREC_D256_B2"}
    assert V.VARIANTS["SASREC_D384_B2"]["sasrec_entry"]["d_model"] == 384 and \
        V.VARIANTS["SASREC_D384_B2"]["sasrec_entry"]["heads"] == 4
    p = tmp_path / "MV.json"
    V.write_registry(p)
    assert V.load_variant_registry(p)["self_sha256"] == r["self_sha256"]
    assert_raises(FileExistsError, V.write_registry, p)                           # O_EXCL: never overwritten


def _drifted(tmp_path):
    obj = V.registry_obj()
    obj["variants"]["GRU_D256"]["d"] = 192                                        # a drifted definition, re-hashed
    obj["self_sha256"] = V.self_sha256(obj)
    p = tmp_path / "drift.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _check_variant_registry_refusals(tmp_path):
    obj = V.registry_obj()
    obj["variants"]["GRU_D256"]["d"] = 192                                        # edited, stale self-hash
    p = tmp_path / "edited.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    raises_with(V.VariantError, "self-hash mismatch", V.load_variant_registry, p)
    raises_with(V.VariantError, "differ from the v4models.variants code table", V.load_variant_registry,
                _drifted(tmp_path))


def test_variant_registry_tampering_refused(tmp_path):
    _check_variant_registry_refusals(tmp_path)


@nc("a variant-registry loader that checks only the self-hash (not the code table)")
def test_nc_variant_registry_without_code_check(tmp_path, monkeypatch):
    real = V.load_variant_registry

    def lax(path=V.REGISTRY_PATH):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("self_sha256") != V.self_sha256(obj):
            raise V.VariantError("self-hash mismatch")
        return {"path": str(path), "file_sha256": _sha(path), "self_sha256": obj["self_sha256"],
                "variants": obj["variants"]}
    monkeypatch.setattr(V, "load_variant_registry", lax)                              # MUTANT
    assert real is not lax
    _check_variant_registry_refusals(tmp_path)


@pytest.mark.parametrize("family", FAMS)
def test_current_builds_bit_identical_small_k(family):
    cur = V.DEFAULT_VARIANT[family]
    a = build(family, SCHEMA, 2026, K=K_SMALL)
    b = build(family, SCHEMA, 2026, K=K_SMALL, variant=cur)
    c = build(family, SCHEMA, 2026, K=K_SMALL, variant=cur, dropout=V.VARIANTS[cur]["dropout_default"])
    assert a.init_sha256 == b.init_sha256 == c.init_sha256 and a.raw_init_sha256 == b.raw_init_sha256
    assert a.manifest_digest == b.manifest_digest and type(a.module) is type(b.module)
    if family == "GRU":
        assert type(a.module) is gru_v2.ContextGRU, "GRU_D128 keeps the ContextGRU class itself"
        with seeded_global_rng(2026, torch.device("cpu")):
            ref = gru_v2.build_model(SCHEMA.widths.gru_spec(), _default_vocab(K_SMALL),
                                     product_idx_of_class=np.arange(3, 3 + K_SMALL))
        assert state_digest(make_adapter(ref, SCHEMA.widths).broadcast_state()) == a.init_sha256
    else:
        from fl.families import SASREC_ENTRY
        assert V.VARIANTS[cur]["sasrec_entry"] == SASREC_ENTRY
        assert a.module.cfg == b.module.cfg


def test_current_builds_bit_identical_to_theta0_registry_full_k():
    """GRU_D128 / SASREC_D256_B2 at the full catalogue against THETA0_REGISTRY.json (this torch build's entries)."""
    reg = json.loads(THETA0_JSON.read_text(encoding="utf-8"))
    tv = torch.__version__
    checked = 0
    for sid, seeds in (("L3_NEAR_DUPLICATE", (2026, 2027)), ("L0S_CLEAN_V2_SESSION", (2026,))):
        ds = _derived(sid)
        builds = reg["schemas"][sid]["builds"]
        assert tv in builds, f"THETA0_REGISTRY has no {tv} entry for {sid}"
        for fam in FAMS:
            for seed in seeds:
                bm = build(fam, ds, seed, K=FULL_K, variant=V.DEFAULT_VARIANT[fam])
                e = builds[tv][fam][str(seed)]
                assert (bm.init_sha256, bm.raw_init_sha256, bm.manifest_digest, bm.manifest.shared_numel) == \
                    (e["init_sha256"], e["raw_init_sha256"], e["manifest_digest"], e["shared_numel"]), (sid, fam, seed)
                checked += 1
                del bm
    assert checked == 6


@pytest.mark.parametrize("sid", ["BASE", "L3_NEAR_DUPLICATE", "L0S_CLEAN_V2_SESSION"])
def test_contextgrud_mirror_at_d128_is_bitwise_contextgru(sid):
    assert V.parent_init_source_sha256() == V.GRU_MIRRORED_INIT_SHA256, \
        "gru_v2.ContextGRU.__init__ changed: re-check the ContextGRUD mirror"
    widths = SCHEMA.widths if sid == "BASE" else _derived(sid).widths
    vocab = _default_vocab(K_SMALL)
    vocab.setdefault("session_offset_tokens", 9)
    vocab.setdefault("session_mode_main_category_tokens", 16)
    poc = np.arange(3, 3 + K_SMALL)
    with seeded_global_rng(11, torch.device("cpu")):
        a = gru_v2.build_model(widths.gru_spec(), vocab, product_idx_of_class=poc)
    with seeded_global_rng(11, torch.device("cpu")):
        b = V.ContextGRUD(widths.gru_spec(), vocab, product_idx_of_class=poc, d=128)
    assert list(a.state_dict()) == list(b.state_dict())
    assert state_digest(a.state_dict()) == state_digest(b.state_dict())
    assert repr(a) == repr(b).replace("ContextGRUD", "ContextGRU")


def test_new_variant_shapes():
    g = build("GRU", SCHEMA, 2026, K=K_SMALL, variant="GRU_D256").module
    assert isinstance(g, V.ContextGRUD) and isinstance(g, gru_v2.ContextGRU)
    assert g.item_embed.weight.shape[1] == 256 and g.input_proj.out_features == 256
    assert tuple(g.input_norm.normalized_shape) == (256,) and (g.gru.input_size, g.gru.hidden_size) == (256, 256)
    assert g.readout_proj.out_features == 256 and g.category_embed.weight.shape[1] == 32
    s3 = build("SASREC", SCHEMA, 2026, K=K_SMALL, variant="SASREC_D256_B3").module
    assert len(s3.blocks) == 3 and s3.d == 256
    s384 = build("SASREC", SCHEMA, 2026, K=K_SMALL, variant="SASREC_D384_B2").module
    blk = s384.blocks[0]
    assert s384.d == 384 and len(s384.blocks) == 2 and (blk.h, blk.dh) == (4, 96)
    assert blk.ff1.out_features == 4 * 384 and tuple(s384.output_embed.shape) == (K_SMALL, 384)
    b2 = build("SASREC", SCHEMA, 2026, K=K_SMALL).module
    n = {v: build(V.VARIANTS[v]["family"], SCHEMA, 2026, K=K_SMALL, variant=v).manifest.shared_numel
         for v in V.VARIANTS}
    assert n["GRU_D256"] > n["GRU_D128"] and n["SASREC_D384_B2"] > n["SASREC_D256_B3"] > n["SASREC_D256_B2"]
    assert len(b2.blocks) == 2


@pytest.mark.parametrize("name", sorted(V.VARIANTS))
def test_per_variant_adapter_equivalence_and_theta0_determinism(name):
    fam = V.VARIANTS[name]["family"]
    bm = build(fam, SCHEMA, 2026, K=K_SMALL, variant=name)
    again = build(fam, SCHEMA, 2026, K=K_SMALL, variant=name)
    other = build(fam, SCHEMA, 2027, K=K_SMALL, variant=name)
    assert bm.init_sha256 == again.init_sha256 and bm.init_sha256 != other.init_sha256
    a = bm.adapter
    a.module.eval()
    b = _batch()
    with torch.no_grad():
        assert torch.equal(a.scores(b), module_forward(a, b)), "adapter scores == the module's own forward"
        q = a.query(b)
    d = V.VARIANTS[name].get("d") or V.VARIANTS[name]["sasrec_entry"]["d_model"]
    assert a.query_dim == d == q.shape[1], "topology (query dim) derived from the module"
    assert a.head_weight().shape == (K_SMALL, d) and a.manifest.shared_numel == bm.manifest.shared_numel
    if fam == "GRU":
        assert a.head_weight().data_ptr() == a.module.item_embed.weight[3].data_ptr(), "tied head"
    else:
        assert bm.recentered and common_mode_report(a)["head"]["common_mode_norm"] < 1e-6, "C7 recentered theta0"
    ds = _derived("L3S_NEAR_DUPLICATE_SESSION")                                  # a projecting derived schema too
    bd = build(fam, ds, 2026, K=K_SMALL, variant=name)
    bd.module.eval()
    bs = _batch(session=True)
    with torch.no_grad():
        assert torch.equal(bd.adapter.scores(bs), module_forward(bd.adapter, bs))


def test_dropout_through_build_kwargs_only():
    g = build("GRU", SCHEMA, 2026, K=K_SMALL, variant="GRU_D256", dropout=(0.3, 0.15))
    assert (g.module.input_dropout.p, g.module.readout_dropout.p) == (0.3, 0.15) and g.dropout == (0.3, 0.15)
    assert g.init_sha256 == build("GRU", SCHEMA, 2026, K=K_SMALL, variant="GRU_D256").init_sha256, \
        "dropout has no parameters: theta0 does not depend on it"
    s = build("SASREC", SCHEMA, 2026, K=K_SMALL, variant="SASREC_D384_B2", dropout=(0.1,))
    assert s.module.cfg.dropout == 0.1 and s.module.blocks[0].p == 0.1
    assert s.init_sha256 == build("SASREC", SCHEMA, 2026, K=K_SMALL, variant="SASREC_D384_B2").init_sha256
    assert_raises(V.VariantError, build, "GRU", SCHEMA, 2026, K=K_SMALL, dropout=(0.1,))       # 2 sites
    assert_raises(V.VariantError, build, "GRU", SCHEMA, 2026, K=K_SMALL, variant="SASREC_D384_B2")
    assert_raises(V.VariantError, build, "SASREC", SCHEMA, 2026, K=K_SMALL, variant="SASREC_D512_B2")


def _check_variant_is_built():
    bm = build("GRU", SCHEMA, 2026, K=K_SMALL, variant="GRU_D256")
    assert bm.adapter.query_dim == 256 and bm.variant == "GRU_D256"


def test_variant_is_built():
    _check_variant_is_built()


@nc("a build that ignores the variant argument (always the current module)")
def test_nc_build_ignores_variant(monkeypatch):
    import sys
    B = sys.modules["v4models.build"]
    real = B.construct_module
    monkeypatch.setattr(B, "construct_module", lambda fam, w, seed, **kw: real(fam, w, seed, **dict(kw, variant=None)))
    _check_variant_is_built()


# ================================================================================================ 0b theta0 builder
def test_theta0_variants_small_k_determinism_o_excl_and_load(tmp_path):
    res = T0.measure("L3_NEAR_DUPLICATE", [2026, 2027], list(V.VARIANTS), K=K_SMALL, schema=_derived("L3_NEAR_DUPLICATE"),
                     log=lambda *_: None)
    assert res["ok"] and len(res["records"]) == 2 * len(V.VARIANTS) and all(res["determinism"].values())
    assert {g["status"] for g in res["guard"].values()} == {"NOT_APPLICABLE_K_OVERRIDE"}
    assert all(v["unchanged"] for v in res["read_only_files"].values())
    assert res["read_only_files"]["derived_schema_registry"]["bound_by_lock_v2"] is True
    assert set(res["inventory"]) == set(V.VARIANTS)
    inv = res["inventory"]["SASREC_D384_B2"]
    assert inv["adamw_state_bytes"] == 2 * inv["shared_bytes"] and inv["query_dim"] == 384
    by = {(r["model_variant"], r["seed"]): r["init_sha256"] for r in res["records"]}
    assert by[("GRU_D128", 2026)] == build("GRU", _derived("L3_NEAR_DUPLICATE"), 2026, K=K_SMALL).init_sha256
    out = tmp_path / "THETA0_VARIANTS.json"
    w = T0.write(res, out)
    assert w["self_sha256"] == json.loads(out.read_text(encoding="utf-8"))["self_sha256"]
    assert_raises(FileExistsError, T0.write, res, out)
    raises_with(T0.Theta0VariantsError, "K-override", T0.load_theta0_variants, out)
    obj = T0.registry_obj(res)
    assert T0.lookup(obj, "L3_NEAR_DUPLICATE", torch.__version__, "SASREC_D256_B3", 2027) == \
        by[("SASREC_D256_B3", 2027)]
    raises_with(T0.Theta0VariantsError, "no registered theta0", T0.lookup, obj, "L3_NEAR_DUPLICATE",
                torch.__version__, "SASREC_D256_B3", 2030)
    if T0.OUT_PATH.exists():                           # built on the server by Ops (R5) and synced; never written here
        prod = T0.load_theta0_variants(T0.OUT_PATH)
        assert prod["K_override"] is None and {g["status"] for g in prod[
            "guard_current_builds_vs_THETA0_REGISTRY"].values()} == {"EQUAL"}


def _guarded(tmp_path, monkeypatch=None):
    res = T0.measure("L3_NEAR_DUPLICATE", [2026], ["GRU_D128", "SASREC_D256_B2"], K=FULL_K,
                     schema=_derived("L3_NEAR_DUPLICATE"), log=lambda *_: None)
    return res


def test_theta0_variants_guard_equal_at_full_k(tmp_path):
    res = _guarded(tmp_path)
    assert res["ok"] and {g["status"] for g in res["guard"].values()} == {"EQUAL"}


def _check_guard_blocks_write(tmp_path, monkeypatch):
    real = T0.registered_current

    def altered(sid, fam, seed, *a, **k):
        e = dict(real(sid, fam, seed, *a, **k))
        e["init_sha256"] = "0" * 64
        return e
    monkeypatch.setattr(T0, "registered_current", altered)
    res = T0.measure("L3_NEAR_DUPLICATE", [2026], ["GRU_D128"], K=FULL_K, schema=_derived("L3_NEAR_DUPLICATE"),
                     log=lambda *_: None)
    assert res["guard"]["GRU_D128/2026"]["status"] == "DIFFERENT" and not res["ok"]
    raises_with(T0.Theta0VariantsError, "failed a guard", T0.write, res, tmp_path / "t.json")
    assert not (tmp_path / "t.json").exists()


def test_theta0_variants_guard_blocks_write(tmp_path, monkeypatch):
    _check_guard_blocks_write(tmp_path, monkeypatch)


@nc("a theta0 writer that ignores the current-build guard")
def test_nc_theta0_writer_ignores_guard(tmp_path, monkeypatch):
    real = T0.write
    monkeypatch.setattr(T0, "write", lambda res, path=T0.OUT_PATH: real(dict(res, ok=True), path))   # MUTANT
    _check_guard_blocks_write(tmp_path, monkeypatch)


def test_derived_schema_registry_and_theta0_registry_byte_identical():
    assert _sha(REG_JSON) == REG_JSON_SHA and _sha(THETA0_JSON) == THETA0_JSON_SHA
    lock = json.loads((V4 / "contracts" / "DATA_PROTOCOL_LOCK_V2.json").read_text(encoding="utf-8"))
    assert lock["features_and_transforms"]["production_schema"]["derived_schema_registry_file_sha256"] == REG_JSON_SHA


# ================================================================================================ 0c + 2 RunConfig
BASE = dict(run_id="r", exposure_point="end", cuda_visible_devices="0", determinism_mode="STRICT", key=dict(KEY),
            **registry_keys())


def _trial(hp_id):
    return H.trial(H.load_hp_registry(), hp_id)


def _tuning(tid, **over):
    t = _trial(tid)
    kw = dict(BASE, family=t["family"], seed=2026, role="TUNING", phase="CALIBRATION", hp_id=tid,
              peak_lr=t["peak_lr"], model_variant=t["model_variant"], dropout=tuple(t["dropout"]),
              schedule=t["schedule"], theta0_init_sha256="a" * 64)
    kw.update(over)
    return RunConfig(**kw)


def test_new_keys_left_out_of_the_hash_while_none():
    c = RunConfig(family="GRU", seed=2026, **BASE)
    c.validate()
    assert set(c.as_json()) == TRAIN01R2_FIELDS, "a pre-TRAIN-02 config keeps its field set, hence its sha256"
    for f in ("model_variant", "dropout", "schedule", "theta0_init_sha256", "hp_id", "hp_registry"):
        assert RunConfig.__dataclass_fields__[f].default is None
    t = _tuning("P2_GRU_D256_S0_s3")
    t.validate()
    j = t.as_json()
    assert {"model_variant", "theta0_init_sha256", "hp_id", "dropout", "schedule"} <= set(j)
    assert t.sha256() != dataclasses.replace(t, theta0_init_sha256="b" * 64).sha256()


def test_tuning_role_derivations():
    t = _tuning("P2_SASREC_D384_B2_S1_s2")
    t.validate()
    assert (t.run_label, t.manifest_role, t.run_role, t.train_phase, t.end_efe) == \
        ("HP_SEARCH", "CALIBRATION", "TUNING", "TUNING", 6.0)
    assert (t.effective_variant, t.effective_dropout, t.effective_schedule) == ("SASREC_D384_B2", (0.2,), "S1")
    assert dataclasses.replace(t, test_overrides=True).run_label == "SMOKE_NOT_EVIDENCE"
    _tuning("P2_GRU_D128_S0_s0", seed=2027).validate()                                     # stage 2 seed


def _check_hp_rule():
    t = _tuning("P2_GRU_D256_S1_s4")
    t.validate()
    for over, field in ((dict(peak_lr=1e-3), "peak_lr"), (dict(dropout=(0.2, 0.1)), "dropout"),
                        (dict(schedule="S0"), "schedule"), (dict(model_variant="GRU_D128"), "model_variant")):
        raises_with(RecipeError, f"'{field}'", dataclasses.replace(t, **over).validate)
    for over in (dict(), dict(test_overrides=True)):                               # no hp_id: the existing rules
        for bad, text in ((dict(peak_lr=_trial("P2_GRU_D128_S0_s0")["peak_lr"]), "not in the registered set"),
                          (dict(schedule="S1"), "not the registered S0"),
                          (dict(dropout=(0.3, 0.15)), "not the executed"),
                          (dict(model_variant="GRU_D256"), "not the registered build")):
            raises_with(RecipeError, text, RunConfig(family="GRU", seed=2026, **dict(BASE, **over, **bad)).validate)


def test_hp_rule_accepts_only_the_named_trial():
    _check_hp_rule()


@nc("a validate() that accepts off-recipe values without the HP-registry rule")
def test_nc_validate_without_hp_rule(monkeypatch):
    monkeypatch.setattr(RunConfig, "_validate_train02", lambda self, fam: None)          # MUTANT
    _check_hp_rule()


def test_tuning_refusals():
    raises_with(RecipeError, "hp_id is valid for role TUNING only",
                RunConfig(family="GRU", seed=2026, hp_id="P2_GRU_D128_S0_s0", **BASE).validate)
    raises_with(RecipeError, "needs hp_id", _tuning("P2_GRU_D128_S0_s0", hp_id=None).validate)
    raises_with(RecipeError, "is a GRU trial, not SASREC", _tuning("P2_GRU_D128_S0_s0", family="SASREC",
                                                                   model_variant=None, dropout=None).validate)
    raises_with(RecipeError, "is not in the HP registry", _tuning("P2_GRU_D128_S0_s0", hp_id="P2_GRU_D512_S0_s0").validate)
    raises_with(RecipeError, "needs population COHORT and phase CALIBRATION", _tuning("P2_GRU_D128_S0_s0", phase="MAIN").validate)
    raises_with(RecipeError, "needs theta0_init_sha256", _tuning("P2_GRU_D128_S0_s0", theta0_init_sha256=None).validate)
    _tuning("P2_GRU_D128_S0_s0", theta0_init_sha256=None, test_overrides=True).validate()
    raises_with(RecipeError, "seed 2026 (stage 1) or 2027", _tuning("P2_GRU_D128_S0_s0", seed=2028).validate)
    raises_with(RecipeError, "no continuation", _tuning("P2_GRU_D128_S0_s0", continuation_budget="b.json").validate)
    raises_with(RecipeError, "not a calibration record", _tuning("P2_GRU_D128_S0_s0", calibration_record="c.json").validate)
    raises_with(RecipeError, "64-hex", _tuning("P2_GRU_D128_S0_s0", theta0_init_sha256="xyz").validate)
    raises_with(RecipeError, "differ", _tuning("P2_GRU_D128_S0_s0", expected_init_sha256="b" * 64).validate)
    raises_with(RecipeError, "effective batch", _tuning("P2_GRU_D128_S0_s0", effective_batch=32).validate)
    raises_with(RecipeError, "tuple of floats", _tuning("P2_GRU_D128_S0_s0", dropout=[0.1, 0.05]).validate)


# ================================================================================================ 1 HP registry (02R2)
STALE_P2 = V4 / "contracts" / "hp_registry" / "HP_REGISTRY_P2.json"


def test_hp_registry_design_02r2():
    """02R2 §4: identical cell multisets across the size levels of a family; every (size, p) pair; SASRec p counts
    3 / 6 / 3; the LR values are the §1 midpoints at full precision; the schedules are 02R1's."""
    r = H.load_hp_registry()
    raw = r["raw"]
    assert Path(r["path"]).name == "HP_REGISTRY_P2R2.json" and raw["format"] == "PPSI_V4_HP_REGISTRY_P2R2_1"
    assert raw["self_sha256"] == H.canonical_sha256(raw) and b"\r" not in Path(r["path"]).read_bytes()
    assert not any("control" in json.dumps(t).lower() for f in r["families"].values() for t in f), "no control"
    lo, hi = math.log10(3e-4), math.log10(3e-3)
    exact = {fam: [10.0 ** (lo + (k + 0.5) * ((hi - lo) / n)) for k in range(n)] for fam, n in (("GRU", 6), ("SASREC", 4))}
    approx = {"GRU": [3.6e-4, 5.3e-4, 7.7e-4, 1.14e-3, 1.67e-3, 2.45e-3], "SASREC": [4.0e-4, 7.1e-4, 1.26e-3, 2.24e-3]}
    for fam in ("GRU", "SASREC"):
        grid = [s["peak_lr"] for s in raw["lr_grid"][fam]["strata"]]
        assert grid == exact[fam], "full-precision midpoints (02R2 §1)"
        assert all(abs(g - a) / a < 0.02 for g, a in zip(grid, approx[fam]))
        trials = r["families"][fam]
        assert len(trials) == 12 and len({t["hp_id"] for t in trials}) == 12
        for t in trials:
            assert t["sha256"] == H.canonical_sha256(t, exclude=("sha256",))
            assert t["peak_lr"] == exact[fam][t["lr_stratum"]]
            assert t["hp_id"] == f"P2_{t['model_variant']}_{t['schedule']}_s{t['lr_stratum']}", "readable, stable id"
            assert t["dropout"] == ([t["dropout_p"], t["dropout_p"] / 2] if fam == "GRU" else [t["dropout_p"]])
        sizes = H.VARIANT_LEVELS[fam]
        ms = {v: H.cell_multiset(trials, v).encode("utf-8") for v in sizes}
        assert len(set(ms.values())) == 1, "the cell multiset is byte-identical across size levels"
        for v in sizes:
            for p in (0.1, 0.2, 0.3):
                assert any(t["model_variant"] == v and t["dropout_p"] == p for t in trials), (v, p)
    gru = {(t["schedule"], t["lr_stratum"], t["dropout_p"]) for t in r["families"]["GRU"] if t["model_variant"] == "GRU_D128"}
    assert gru == {("S0", 0, 0.1), ("S0", 3, 0.2), ("S0", 5, 0.3), ("S1", 1, 0.3), ("S1", 2, 0.2), ("S1", 4, 0.1)}
    sas = [(t["schedule"], t["lr_stratum"], t["dropout_p"]) for t in r["families"]["SASREC"]
           if t["model_variant"] == "SASREC_D384_B2"]
    assert sorted(sas) == [("S0", 0, 0.1), ("S0", 3, 0.3), ("S1", 1, 0.2), ("S1", 2, 0.2)]
    counts = {p: sum(t["dropout_p"] == p for t in r["families"]["SASREC"]) for p in (0.1, 0.2, 0.3)}
    assert counts == {0.1: 3, 0.2: 6, 0.3: 3}, "02R2 §3 family p counts"
    assert raw["schedules"]["S1"] == [[0.0, 0.0], [0.02, 1.0], [4.2, 1.0], [6.0, 0.01]]
    assert raw["schedules"]["S0"] == [list(k) for k in SCH.KNOTS]


def test_hp_registry_o_excl(tmp_path):
    p = tmp_path / "HP.json"
    w = H.write_registry(p)
    assert w["self_sha256"] == H.load_hp_registry()["self_sha256"]
    assert_raises(FileExistsError, H.write_registry, p)
    assert_raises(FileExistsError, H.write_registry, H.REGISTRY_PATH)             # the registered file is never rewritten


def _check_stale_registry_refused(tmp_path):
    assert STALE_P2.is_file(), "the 02R1 registry is kept on disk as the record"
    raises_with(H.HPRegistryError, "superseded registry", H.load_hp_registry, STALE_P2)
    raises_with(RecipeError, "superseded registry",
                _tuning("P2_GRU_D128_S0_s0", hp_registry=str(STALE_P2)).validate)
    import v4train.p2_driver as D
    import argparse as _ap
    ns = _ap.Namespace(family="GRU", stage=1, gpu=0, out_dir=tmp_path, history_policy="USER_HISTORY_L50",
                       hp_registry=STALE_P2, theta0_variants=None, dry_run=True)
    raises_with(H.HPRegistryError, "superseded registry", D.run_stage, ns, runner=None, say=lambda *_: None)


def test_stale_p2_registry_refused_for_launches(tmp_path):
    _check_stale_registry_refused(tmp_path)


@nc("a loader that accepts the superseded 02R1 registry (format check removed)")
def test_nc_stale_registry_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "STALE_REGISTRIES", {})                                          # MUTANT
    monkeypatch.setattr(H, "FORMAT", "PPSI_V4_HP_REGISTRY_P2_1")
    _check_stale_registry_refused(tmp_path)


def _tampered(tmp_path):
    obj = json.loads(H.REGISTRY_PATH.read_text(encoding="utf-8"))
    t = obj["families"]["GRU"][0]
    t["peak_lr"] = 1e-3                                                           # a hidden trial value
    t["sha256"] = H.canonical_sha256(t, exclude=("sha256",))
    obj["self_sha256"] = H.canonical_sha256(obj)
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    stale = copy.deepcopy(obj)
    stale["families"]["GRU"][1]["peak_lr"] = 2e-3                                   # edited without re-hashing
    q = tmp_path / "stale.json"
    q.write_text(json.dumps(stale), encoding="utf-8")
    return p, q


def _check_hp_tampering(tmp_path):
    p, q = _tampered(tmp_path)
    raises_with(H.HPRegistryError, "differ from the registered design", H.load_hp_registry, p)
    raises_with(H.HPRegistryError, "self-hash mismatch", H.load_hp_registry, q)
    raises_with(RecipeError, "HP registry refused", _tuning("P2_GRU_D128_S0_s0", hp_registry=str(p),
                                                            peak_lr=1e-3).validate)


def test_hp_registry_tampering_refused(tmp_path):
    _check_hp_tampering(tmp_path)


@nc("an HP registry loader that does not verify the file")
def test_nc_hp_loader_without_verification(tmp_path, monkeypatch):
    def lax(path=H.REGISTRY_PATH):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return {"path": str(path), "file_sha256": _sha(path), "self_sha256": obj["self_sha256"],
                "families": obj["families"], "lr_grid_sha256": None, "raw": obj}
    monkeypatch.setattr(H, "load_hp_registry", lax)                                  # MUTANT
    _check_hp_tampering(tmp_path)


def test_hp_verify_cli(tmp_path, monkeypatch, capsys):
    """F4: `--verify` recomputes the grid and the cells; exit 0 on the registered file, 1 on any difference."""
    assert H.main(["--verify", str(H.REGISTRY_PATH)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    p, _ = _tampered(tmp_path)
    assert H.main(["--verify", str(p)]) == 1
    res = json.loads(capsys.readouterr().out)
    assert not res["ok"] and any("trials differ" in x for x in res["problems"])
    assert H.main(["--verify", str(STALE_P2)]) == 1
    capsys.readouterr()
    assert H.main(["--verify", str(tmp_path / "missing.json")]) == 1
    capsys.readouterr()
    real = H.lr_grid                                                              # a host whose arithmetic differs
    monkeypatch.setattr(H, "lr_grid", lambda: {**real(), "GRU": {**real()["GRU"], "strata": [
        dict(s, peak_lr=s["peak_lr"] * (1 + 1e-15)) for s in real()["GRU"]["strata"]]}})
    assert H.main(["--verify", str(H.REGISTRY_PATH)]) == 1, "a recomputation that differs is reported (exit 1)"
    capsys.readouterr()
    monkeypatch.setattr(H, "lr_grid", real)
    assert H.load_hp_registry()["self_sha256"] == H.registry_obj()["self_sha256"], "the loader is not loosened"


# ================================================================================================ 3 schedule
def _ref_interp(e, knots):
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= e <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (e - x0) / (x1 - x0)
    raise ValueError


def _ref_table(peak, N, B, phase):
    """The TRAIN-01R2 planned_lr_table (S0, point "end", MAIN / CALIBRATION), transcribed verbatim."""
    end_efe = {"MAIN": 6.0, "CALIBRATION": 2.0}[phase]
    end = -(-int(round(end_efe * 2)) * N // 2)
    knots = ((0.0, 0.0), (0.02, 1.0), (2.0, 1.0), (3.0, 0.1), (5.5, 0.1), (6.0, 0.01))
    lrs, e = [], 0
    while e < end:
        n = min(B, N - e % N)
        at = (e + n) / N
        lrs.append(float(peak) * _ref_interp(min(at, 6.0), knots))
        e += n
    return hashlib.sha256(np.asarray(lrs, dtype="<f8").tobytes()).hexdigest()


# the TRAIN-01R2 algorithm's table hashes (computed with _ref_table, the verbatim transcription, 2026-09-27)
S0_PINNED = {(1e-3, 200, 32, "MAIN"): "143ed10639d25d546cfd07f3596b8a83b69fbf28013b65724fb3e61dbe6ca98f",
             (5e-4, 646_067, 256, "CALIBRATION"): "989a7115a51ac7610b7f45d86e8861f9dfd405940c0ba227702d79ce9ea595d1",
             (1e-3, 1024, 256, "MAIN"): "2567f2052932c552c5a3a2de7b491efb0d18800a2e7c98d644f4b325e35397d4",
             (1e-3, 646_067, 256, "MAIN"): "25fd198c79ba860a999ffae7762101532f7a6d8a80569174d1d8466d9340c96d"}


def test_s0_planned_table_bit_identical_to_train01r2():
    for (peak, N, B, phase), pinned in S0_PINNED.items():
        t = SCH.planned_lr_table(peak, N, B, phase, "end")
        assert t["sha256"] == _ref_table(peak, N, B, phase) == pinned, (peak, N, B, phase)
        assert t["sha256"] == SCH.planned_lr_table(peak, N, B, phase, "end", schedule="S0")["sha256"]
    t = SCH.planned_lr_table(1e-3, 646_067, 256, "MAIN", "end")
    tt = SCH.planned_lr_table(1e-3, 646_067, 256, "TUNING", "end", schedule="S0")
    assert tt["sha256"] == t["sha256"], "TUNING with S0 plans exactly the MAIN table"
    assert SCH.planned_lr_table(1e-3, 646_067, 256, "TUNING", "end", schedule="S1")["sha256"] != t["sha256"]


def _registered_s1_knots():
    text = (V4 / "contracts" / "PREG1_AMENDMENT_02R1.md").read_text(encoding="utf-8")
    m = re.search(r"S1 \(WSD, linear\) =\s*\(\((.*?)\)\)", text, flags=re.S)
    pts = re.findall(r"\(?([0-9.]+),([0-9.]+)\)?", m.group(1).replace(" ", ""))
    return tuple((float(a), float(b)) for a, b in pts)


def _check_s1_is_registered():
    assert SCH.SCHEDULES["S1"] == _registered_s1_knots() == ((0.0, 0.0), (0.02, 1.0), (4.2, 1.0), (6.0, 0.01))
    assert SCH.SCHEDULES["S0"] == SCH.KNOTS
    for e in np.linspace(0.0, 2.0, 401):
        assert SCH.s_main(float(e), "S1") == SCH.s_main(float(e), "S0"), "S0 == S1 on [0, 2]"


def test_s1_is_the_registered_schedule():
    _check_s1_is_registered()


@nc("an S1 whose knots are not the registered ones (decay starts at 4.0)")
def test_nc_s1_mutated_knots(monkeypatch):
    monkeypatch.setitem(SCH.SCHEDULES, "S1", ((0.0, 0.0), (0.02, 1.0), (4.0, 1.0), (6.0, 0.01)))     # MUTANT
    _check_s1_is_registered()


def test_t2_pins_hold_for_s1():
    import fl.schedule as FS
    # P3: fl.schedule's interpolation, on the knots, a 10,000-point grid and the 1e-12 tolerance zone
    grid = [x for x, _ in SCH.S1_KNOTS] + [float(x) for x in np.linspace(0.0, 6.0, 10_000)]
    for e in grid:
        assert SCH.s_main(e, "S1") == FS._interp(min(e, 6.0), SCH.S1_KNOTS)
    assert abs(SCH.s_main(6.0, "S1") - 0.01) < 1e-12 and SCH.s_main(6.0 + 1e-12, "S1") == SCH.s_main(6.0, "S1")
    assert_raises(SCH.ScheduleError, SCH.s_main, 6.0 + 2e-12, "S1")
    assert SCH.s_main(3.0, "S1") == 1.0 and SCH.s_main(5.1, "S1") == pytest.approx(0.505, abs=1e-12)
    # P1: the exposure point is still an explicit, checked argument
    assert_raises(SCH.ScheduleError, SCH.planned_lr_table, 1e-3, 1000, 32, "TUNING", None, schedule="S1")
    # P2: hashed table, min LR > 0, the final update ends at 6N with 0.01 x peak
    for N, B in ((646_067, 256), (1000, 32), (200, 32)):
        t = SCH.planned_lr_table(1e-3, N, B, "TUNING", "end", schedule="S1")
        assert t["min_lr"] > 0 and t["final_exposures"] == 6 * N and abs(t["last_lr"] - 1e-5) <= 1e-12 * 1e-3
        assert len(t["sha256"]) == 64 and t["schedule"] == "S1"
    # P4: one client, one round per update: FL's round LR == the central lr_for_update, exactly
    N, B, e = 1000, 32, 0
    while e < 6 * N:
        n = min(B, N - e % N)
        fl_lr = 1e-3 * FS._interp(min((e + n) / N, 6.0), SCH.S1_KNOTS)
        assert fl_lr == SCH.lr_for_update(1e-3, e, n, N, "TUNING", "end", "S1")
        e += n
    # P5: LO steps s(6 (t + 1) / T): lr(0) > 0, lr(T - 1) == 0.01 x peak (1e-12), T effective updates
    for T in (1, 2, 6, 7, 300, 10_000):
        lrs = [1e-3 * SCH.s_main(6.0 * (t + 1) / T, "S1") for t in range(T)]
        assert len(lrs) == T and lrs[0] > 0 and abs(lrs[-1] - 1e-5) <= 1e-12 * 1e-3
    assert_raises(SCH.ScheduleError, SCH.s_of, 6.5, "CONTINUATION", "S1")
    assert_raises(SCH.ScheduleError, SCH.s_main, 1.0, "S2")
    assert SCH.eval_marks("TUNING") == [2.0, 4.0, 6.0] and SCH.snapshot_marks("TUNING") == [5.0, 5.5, 6.0]
    assert SCH.snapshot_marks("MAIN") == [] and SCH.phase_end("TUNING") == 6.0
