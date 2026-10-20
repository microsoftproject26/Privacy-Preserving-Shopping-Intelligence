"""TRAIN-01R2 — the manifest-role registry, derived-schema acceptance and the native screen / smoke refusal (G1 R1, R3).

  * required keys: `manifest_registry` and `data_version` (None refused, also with test_overrides; no code default);
    an explicit `role` must agree with population and phase;
  * the HISTORICAL registry file replaces the hard-coded ids: self-hash, the old role map (M01 / M02, M05 / M06; no
    FULL_TRAIN, never built), its values equal the DATA_PROTOCOL_LOCK's; `expected_manifests` reads it;
  * registry verification: self-hash, shape, TEST ids, binding presence; the data version; the role; each provider's
    manifest id, content / header hashes, row count and fingerprint; a provider without a binding;
  * recorded: registry file sha256 / self-hash / data version / role / manifest ids and schema id / self-hash in
    RUN_START, every checkpoint (file and index), RUN_SUMMARY, key_base and every row, with the derived run_label;
  * resume refuses a changed registry and a changed schema (named cause, nothing written) and the original resumes;
    continuation needs the registry's CONTINUATION role;
  * derived schemas: accepted through the registry (SCHEMA-01's "schemas" format and a top-level map), refused when
    unregistered, hash-inconsistent, tampered or when the registry itself is tampered; the base schema needs no
    registry; SCHEMA-01's real DerivedSchema (contracts/derived_schemas, when present) trains end to end;
  * run labels: PRODUCTION / SMOKE_NOT_EVIDENCE / FEATURE_SEARCH_SCREEN / HISTORICAL_REFERENCE derived from the config;
    the FEATURE_SEARCH_SCREEN role validates the test-free settings without test_overrides; non-test runs on the CPU
    (device class patched to the frozen A16 class) write PRODUCTION, HISTORICAL_REFERENCE and FEATURE_SEARCH_SCREEN
    rows; the screen keeps no resumable and refuses continuation;
  * selection refuses every label it does not accept, per consumer (cohort contest, tie-break, calibration choice,
    calibration record, C_MATCHED) — one parametrized case per label, plus unlabelled rows; a non-test PRODUCTION
    calibration end to end is chosen through selection and consumed by a non-test main run; a non-test calibration on
    the HISTORICAL data version is refused before any arm runs.
Negative controls: a manifest id not in the registry accepted; a registry hash mismatch accepted; a schema not in the
registry accepted; a screen row accepted by selection; a resume with a changed schema accepted.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch

import v4train.registry as R
import v4train.trainer as T
from test_t22_selection import WIDE, cohort_rows, full_rows, row, summaries, tb_cohort
from trainkit import art  # noqa: E402  (TRAIN-03: sealed artifact root)
from trainkit import (BIG_DISK, DEFAULT_REGISTRY, FS_DATA_VERSION, FS_REGISTRY, HISTORICAL_ROLES, K, KEY, N,
                      NEW_DATA_VERSION,
                      SCHEMA, WIDTHS, assert_raises, cfg, derived_schema, interrupt_after, nc, providers, raises_with,
                      ramp, read, registry_keys, registry_obj, scripted, trainer, write_json_file, write_registry,
                      write_schema_registry)
from v4train import selection as SEL
from v4train.calibration import CalibrationError, run_calibration
from v4train.checkpoint import CheckpointError, CheckpointStore
from v4train.evalconfig import frozen
from v4train.provider import SyntheticEvalProvider, SyntheticTrainProvider, expected_manifests
from v4train.recipe import HISTORICAL_DATA_VERSION, RUN_LABELS, RecipeError, RunConfig
from v4train.registry import (HISTORICAL_REGISTRY_PATH, RegistryError, canonical_self_sha256, load_manifest_registry,
                              with_self_sha256)
from v4train.schedule import eval_marks
from v4train.selection import (SelectionError, accept_calibration_record, assert_c_matched_eligible, calibration_choice,
                               select_family, tiebreak_stage1)
from v4train.trainer import BUDGET_SCHEMA, InjectedInterruption, TrainerError

V4 = Path(__file__).resolve().parents[2]
MAIN = eval_marks("MAIN")
VALS = {m: 0.1 for m in MAIN}
ID_FIELDS = ("data_version", "manifest_role", "manifest_registry_sha256", "manifest_registry_self_sha256",
             "train_manifest_id", "eval_manifest_id", "schema_id", "schema_self_sha256", "run_label")
NON_TEST = dict(effective_batch=256)                       # + test_overrides=False (set by _non_test_cfg)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _snapshot(run_dir) -> dict:
    return {str(p.relative_to(run_dir)): _sha(p) for p in sorted(Path(run_dir).rglob("*")) if p.is_file()}


def _frozen_device_class(monkeypatch):
    """Non-test runs on the local CPU: the runtime device class is patched to the frozen A16 class (tests only)."""
    monkeypatch.setattr(T, "runtime_device_class", lambda device: frozen().device_class)


def _non_test_cfg(run_id, *, reg, dv, **kw):
    kw = dict(NON_TEST, **kw)
    c = cfg(run_id, kw.pop("family", "GRU"), manifest_registry=reg, data_version=dv, **kw)
    return dataclasses.replace(c, test_overrides=False)


# ------------------------------------------------------------------------------------------------ required keys
def test_required_registry_keys(work):
    base = dict(run_id="r", family="GRU", seed=2026, key=dict(KEY), exposure_point="end", cuda_visible_devices="0",
                determinism_mode="STRICT")
    for f in ("manifest_registry", "data_version", "schema_registry", "role"):
        assert RunConfig.__dataclass_fields__[f].default is None, f"{f}: no code default"
    raises_with(RecipeError, "manifest_registry is a required", RunConfig(**base, data_version="DV").validate)
    raises_with(RecipeError, "manifest_registry is a required",
                RunConfig(**base, data_version="DV", test_overrides=True).validate)                  # even in tests
    raises_with(RecipeError, "data_version is a required", RunConfig(**base, manifest_registry="r.json").validate)
    raises_with(RecipeError, "data_version is a required",
                RunConfig(**base, manifest_registry="r.json", data_version="", test_overrides=True).validate)
    RunConfig(**base, **registry_keys()).validate()
    raises_with(RecipeError, "manifest_registry is a required", trainer, work / "n",
                cfg("n", "GRU", manifest_registry=None))
    assert not (work / "n").exists()
    c = cfg("t", "GRU")
    assert c.manifest_registry == DEFAULT_REGISTRY and c.data_version == HISTORICAL_DATA_VERSION, "trainkit states them"
    assert c.sha256() != dataclasses.replace(c, data_version="OTHER").sha256()
    for bad in ({"role": "CONTINUATION"}, {"role": "LO"}, {"role": "CALIBRATION"}, {"role": "FULL_TRAIN"},
                {"role": "FEATURE_SEARCH_SCREEN", "phase": "CALIBRATION", "seed": 2026},
                {"role": "MAIN", "population": "FULL_TRAIN"}):
        assert_raises(RecipeError, RunConfig(**{**base, **registry_keys(), **bad}).validate)
    for role in ("MAIN", "FEATURE_SEARCH_SCREEN"):
        RunConfig(**base, **registry_keys(role), role=role).validate()


def test_historical_registry_file_replaces_the_hard_coded_ids():
    reg = load_manifest_registry(HISTORICAL_REGISTRY_PATH)
    assert reg.data_version == HISTORICAL_DATA_VERSION == "CLEAN_V2_HISTORICAL_262144"
    assert reg.roles == {"MAIN": {"train": "M01_COHORT_TRAIN", "eval": "M02_VALIDATION_WARM"},
                         "CONTINUATION": {"train": "M01_COHORT_TRAIN", "eval": "M02_VALIDATION_WARM"},
                         "CALIBRATION": {"train": "M05_CALIB_TRAIN", "eval": "M06_CALIB_VALIDATION_WARM"}}
    assert reg.self_sha256 == canonical_self_sha256(reg.raw) and reg.raw["status"] == "HISTORICAL_POPULATION"
    lock = json.loads((V4 / "contracts" / "DATA_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    locked = lock["population_and_manifests"]["manifest_hashes"]["manifests"]
    for mid, e in reg.manifests.items():
        assert (e["n_decisions"], e["n_users"], e["n_eval_rows"]) == \
            (locked[mid]["n_decisions"], locked[mid]["n_users"], locked[mid]["n_eval_rows"])
        assert e["content_sha256"] == locked[mid]["header_content_sha256"]
        assert e["header_sha256"] == locked[mid]["header_file_sha256"]
    assert expected_manifests("COHORT", "MAIN") == ("M01_COHORT_TRAIN", "M02_VALIDATION_WARM")
    assert expected_manifests("COHORT", "CALIBRATION") == ("M05_CALIB_TRAIN", "M06_CALIB_VALIDATION_WARM")
    assert_raises(RegistryError, expected_manifests, "FULL_TRAIN", "MAIN")          # never built: not registered
    assert expected_manifests("COHORT", "MAIN", FS_REGISTRY, role="FEATURE_SEARCH_SCREEN") == \
        ("FS_TRAIN_5_100_32768", "FS_VALIDATION_5_100_32768")
    fixture = load_manifest_registry(DEFAULT_REGISTRY)
    assert {k: v for k, v in fixture.roles.items() if k != "FULL_TRAIN"} == HISTORICAL_ROLES == reg.roles
    import v4train.provider as P
    for gone in ("TRAIN_MANIFESTS", "EVAL_MANIFESTS", "FULL_TRAIN_MANIFEST", "FULL_EVAL_MANIFEST"):
        assert not hasattr(P, gone), f"v4train.provider still hard-codes {gone}"


# ------------------------------------------------------------------------------------------------ registry checks
def _obj():
    return json.loads(Path(DEFAULT_REGISTRY).read_text(encoding="utf-8"))


def test_registry_shape_refusals():
    o = _obj()
    cases = {
        "data_version missing": {k: v for k, v in o.items() if k != "data_version"},
        "unknown role": dict(o, roles=dict(o["roles"], LO={"train": "M01_COHORT_TRAIN", "eval": "M02_VALIDATION_WARM"})),
        "must map exactly 'train' and 'eval'": dict(o, roles=dict(o["roles"], MAIN={"train": "M01_COHORT_TRAIN"})),
        "which has no manifest entry": dict(o, roles=dict(o["roles"], MAIN={"train": "MX", "eval": "M02_VALIDATION_WARM"})),
        "TEST manifest": dict(o, manifests=dict(o["manifests"], M07_TEST_WARM=o["manifests"]["M02_VALIDATION_WARM"])),
        "n_decisions must be a positive int": dict(o, manifests=dict(o["manifests"], M01_COHORT_TRAIN=dict(
            o["manifests"]["M01_COHORT_TRAIN"], n_decisions="200"))),
        "binds nothing": dict(o, manifests=dict(o["manifests"], M01_COHORT_TRAIN={"n_decisions": 200})),
    }
    for text, bad in cases.items():
        p = write_json_file(with_self_sha256(bad))
        raises_with(RegistryError, text, load_manifest_registry, p)
    p = Path(write_json_file(o))
    p.write_text(p.read_text(encoding="utf-8").replace('"n_users": 0', '"n_users": NaN', 1), encoding="utf-8")
    raises_with(RegistryError, "not strict JSON", load_manifest_registry, p)
    raises_with(RegistryError, "does not exist", load_manifest_registry, Path(DEFAULT_REGISTRY).with_name("NONE.json"))
    lock_style = dict(o, self_sha256_rule="the lock's rule")                        # FS-01 may follow the lock's rule
    lock_style["self_sha256"] = canonical_self_sha256(lock_style, exclude=("self_sha256_rule",))
    load_manifest_registry(write_json_file(lock_style))


def _check_registry_hash_mismatch_refused(work):
    o = _obj()
    o["manifests"]["M01_COHORT_TRAIN"]["n_users"] = 999               # an edit no other check sees; self-hash stale
    p = write_json_file(o, name=f"TAMPERED_{work.name}.json")
    raises_with(TrainerError, "self-hash mismatch", trainer, work / "h", cfg("h", "GRU", manifest_registry=p))


def test_registry_hash_mismatch_refused(work):
    _check_registry_hash_mismatch_refused(work)
    o = _obj()
    o["self_sha256"] = "0" * 64
    raises_with(TrainerError, "self-hash mismatch", trainer, work / "z",
                cfg("z", "GRU", manifest_registry=write_json_file(o)))


def test_data_version_and_role_refusals(work):
    raises_with(TrainerError, "differs from the manifest registry's", trainer, work / "d",
                cfg("d", "GRU", data_version="SOME_OTHER_VERSION"))
    tp, ep = providers(role="FEATURE_SEARCH_SCREEN")
    raises_with(TrainerError, "role 'FEATURE_SEARCH_SCREEN' is not in the manifest registry", trainer, work / "r",
                cfg("r", "GRU", role="FEATURE_SEARCH_SCREEN", **registry_keys()), tp, ep)     # the default registry
    raises_with(TrainerError, "role 'MAIN' is not in the manifest registry", trainer, work / "m",
                cfg("m", "GRU", **registry_keys("FEATURE_SEARCH_SCREEN")))                    # FS registry, MAIN run


def _check_unregistered_manifest_refused(work):
    tp0, ep0 = providers()
    tp = SyntheticTrainProvider(N, K, WIDTHS, seed=3, manifest_id="MX_UNREGISTERED_TRAIN")
    raises_with(TrainerError, "is not the registry's 'M01_COHORT_TRAIN'", trainer, work / "u", cfg("u", "GRU"), tp, ep0)
    ep = SyntheticEvalProvider(60, K, WIDTHS, seed=4, manifest_id="MX_UNREGISTERED_EVAL")
    raises_with(TrainerError, "is not the registry's 'M02_VALIDATION_WARM'", trainer, work / "v", cfg("v", "GRU"),
                tp0, ep)


def test_manifest_id_not_in_registry_refused(work):
    _check_unregistered_manifest_refused(work)


def test_provider_binding_mismatches_refused(work):
    _, ep = providers()
    other = SyntheticTrainProvider(N, K, WIDTHS, seed=4, manifest_id="M01_COHORT_TRAIN")     # same id, other content
    raises_with(TrainerError, "fingerprint mismatch", trainer, work / "c", cfg("c", "GRU"), other, ep)
    tp, _ = providers()
    tp.manifest_binding = dict(tp.manifest_binding, header_sha256="0" * 64)
    raises_with(TrainerError, "['header_sha256'] differ", trainer, work / "h", cfg("h", "GRU"), tp, ep)
    tp, _ = providers()
    del tp.manifest_binding
    raises_with(TrainerError, "declares no manifest_binding", trainer, work / "b", cfg("b", "GRU"), tp, ep)
    tp, _ = providers()
    tp.manifest_binding = dict(tp.manifest_binding, manifest_id="M05_CALIB_TRAIN")
    raises_with(TrainerError, "binding names 'M05_CALIB_TRAIN'", trainer, work / "i", cfg("i", "GRU"), tp, ep)
    tp, _ = providers()
    _, ep_other = providers(seed=7)
    raises_with(TrainerError, "fingerprint mismatch", trainer, work / "e", cfg("e", "GRU"), tp, ep_other)
    count = write_registry({"MAIN": (tp, ep)}, entry_extra={"M01_COHORT_TRAIN": {"n_decisions": N + 1}})
    raises_with(TrainerError, f"serves {N} decisions, the registry entry states {N + 1}", trainer, work / "n",
                cfg("n", "GRU", manifest_registry=count), tp, ep)
    fp = write_registry({"MAIN": (tp, ep)}, entry_extra={"M01_COHORT_TRAIN": {"fingerprint": "f" * 64}})
    raises_with(TrainerError, "fingerprint differs", trainer, work / "f", cfg("f", "GRU", manifest_registry=fp), tp, ep)
    # a fingerprint-only entry (SCHEMA-01's dry-run registry shape) binds by the provider fingerprint alone
    fp_only = registry_obj({"MAIN": (tp, ep)})
    for mid, prov in (("M01_COHORT_TRAIN", tp), ("M02_VALIDATION_WARM", ep)):
        fp_only["manifests"][mid] = {"n_decisions": fp_only["manifests"][mid]["n_decisions"],
                                     "content_sha256": prov.fingerprint, "header_sha256": None,
                                     "fingerprint": prov.fingerprint}
    tr = trainer(work / "ok", cfg("ok", "GRU", manifest_registry=write_json_file(with_self_sha256(fp_only))), tp, ep)
    assert tr.bindings["train"]["bound_by"] == ["fingerprint"]


# ------------------------------------------------------------------------------------------------ recorded
def _check_ids_recorded(run_dir, *, label, role="MAIN", reg=DEFAULT_REGISTRY, schema=SCHEMA):
    st = read(run_dir / "RUN_START.json")
    rec = st["manifest_registry"]
    assert rec["file_sha256"] == _sha(reg) and rec["self_sha256"] == read(reg)["self_sha256"]
    assert rec["data_version"] == st["data_version"] == read(reg)["data_version"] and rec["role"] == role
    assert rec["bindings"]["train"]["manifest_id"] == rec["train_manifest"] == st["train_provider"]["manifest_id"]
    assert st["schema"]["schema_id"] == schema.schema_id and st["schema"]["schema_self_sha256"] == schema.self_sha256
    assert st["run_label"] == label and st["manifest_role"] == role
    for f in ID_FIELDS:
        assert st["key_base"][f] is not None, f"key_base lacks {f}"
    want = {f: st["key_base"][f] for f in ID_FIELDS}
    assert want["run_label"] == label and want["manifest_registry_sha256"] == rec["file_sha256"]
    idx = read(art(run_dir) / "CHECKPOINTS.json")
    assert idx, "no checkpoints"
    for kind, e in idx.items():
        for where, meta in (("index", e["meta"]),
                            ("file", torch.load(art(run_dir) / e["file"], map_location="cpu", weights_only=True)["meta"])):
            assert meta["manifest_registry"]["file_sha256"] == rec["file_sha256"], f"{kind} {where}"
            assert meta["schema"] == {k: st["schema"][k] for k in ("schema_id", "schema_self_sha256")}, f"{kind} {where}"
            assert meta["run_label"] == label
    s = read(art(run_dir) / "RUN_SUMMARY.json")
    assert s["run_label"] == label and s["manifest_registry"]["file_sha256"] == rec["file_sha256"]
    files = [art(run_dir) / "RESULT_ROWS.json"] + sorted((art(run_dir) / "evals").glob("*.rows.json"))
    n = 0
    for f in files:
        for r in read(f)["rows"]:
            n += 1
            assert {k: r.get(k) for k in ID_FIELDS} == want, f"{f.name}: a row lacks the TRAIN-01R2 identity"
    assert n > 0


def test_ids_recorded_everywhere_real_eval(work):
    c = cfg("rec", "GRU", checkpoint_every=4)
    with pytest.raises(InjectedInterruption):
        trainer(work / "r", c, fault=interrupt_after(9)).run()                 # real v4eval at 0.5 and 1.0
    live = read(art(work / "r") / "CHECKPOINTS.json")["latest"]
    assert live["meta"]["manifest_registry"]["role"] == "MAIN" and live["meta"]["schema"]["schema_id"] == \
        SCHEMA.schema_id
    assert trainer(work / "r", c).resume()["status"] == "DONE_MAIN"
    _check_ids_recorded(work / "r", label="SMOKE_NOT_EVIDENCE")
    st = CheckpointStore(work / "S", "cfg-sha")                              # no checkpoint without the records
    det = {"mode": "STRICT", "cublas_workspace_config": ":4096:8"}
    raises_with(CheckpointError, "manifest registry record", st.save, "best",
                {"meta": {"config_sha256": "cfg-sha", "determinism": det}, "model": {}})
    raises_with(CheckpointError, "schema record", st.save, "best",
                {"meta": {"config_sha256": "cfg-sha", "determinism": det, "manifest_registry": {
                    "file_sha256": "f", "self_sha256": "s", "data_version": "D", "role": "MAIN"}}, "model": {}})


# ------------------------------------------------------------------------------------------------ resume
def test_resume_refuses_a_changed_registry(work):
    reg = write_json_file(_obj(), name=f"MUTABLE_{work.name}.json")
    orig = Path(reg).read_bytes()
    c = cfg("rreg", "GRU", checkpoint_every=4, manifest_registry=reg)
    with pytest.raises(InjectedInterruption):
        trainer(work / "B", c, fault=interrupt_after(9), eval_override=scripted(VALS)).run()
    snap = _snapshot(work / "B")
    write_json_file(with_self_sha256(dict(_obj(), note="edited after the run started")), name=Path(reg).name)
    raises_with(TrainerError, "the manifest registry changed", trainer(work / "B", c, eval_override=scripted(VALS)).resume)
    assert _snapshot(work / "B") == snap, "a refused resume writes nothing"
    Path(reg).write_bytes(orig)
    assert trainer(work / "B", c, eval_override=scripted(VALS)).resume()["status"] == "DONE_MAIN"


def _check_resume_refuses_changed_schema(work):
    sreg = write_schema_registry(derived_schema("L0_SYNTH_TEST"))
    c = cfg("rsch", "GRU", checkpoint_every=4, schema_registry=sreg)
    with pytest.raises(InjectedInterruption):
        trainer(work / "B", c, fault=interrupt_after(9), eval_override=scripted(VALS)).run()
    snap = _snapshot(work / "B")
    raises_with(TrainerError, "the schema changed", trainer(work / "B", c, schema=derived_schema("L0_SYNTH_TEST"),
                                                            eval_override=scripted(VALS)).resume)
    assert _snapshot(work / "B") == snap, "a refused resume writes nothing"
    assert trainer(work / "B", c, eval_override=scripted(VALS)).resume()["status"] == "DONE_MAIN"


def test_resume_refuses_a_changed_schema(work):
    _check_resume_refuses_changed_schema(work)


def test_continuation_needs_the_continuation_role(work):
    budget = work / "budget.json"
    budget.write_text(json.dumps({"schema": BUDGET_SCHEMA, "run_id": "cr", "max_efe": 8.0, "approved_by": "synthetic",
                                  "recorded_utc": "2026-09-26T00:00:00Z"}), encoding="utf-8")
    reg = write_registry({"MAIN": providers("MAIN")})                          # no CONTINUATION role
    c = cfg("cr", "GRU", continuation_budget=str(budget), manifest_registry=reg)
    vals = ramp(MAIN, 6.0)
    assert trainer(work / "r", c, eval_override=scripted(vals)).run()["still_improving"]
    raises_with(TrainerError, "role 'CONTINUATION' is not in the manifest registry",
                trainer(work / "r", c, eval_override=scripted(vals)).continue_run)


# ------------------------------------------------------------------------------------------------ derived schemas
@pytest.mark.parametrize("top_level", [False, True])
def test_derived_schema_accepted_through_the_registry(work, top_level):
    ds = derived_schema("L0_SYNTH_TEST")
    sreg = write_schema_registry(ds, top_level=top_level)
    tr = trainer(work / "d", cfg("d", "GRU", schema_registry=sreg), schema=ds)
    assert tr.schema_record["kind"] == "DERIVED" and tr.schema_record["schema_id"] == "L0_SYNTH_TEST"
    assert tr.schema_record["registry"]["file_sha256"] == _sha(sreg)
    assert tr.key_base["schema_id"] == "L0_SYNTH_TEST" and tr.key_base["schema_self_sha256"] == ds.self_sha256
    base = trainer(work / "b", cfg("b", "GRU", schema_registry=str(work / "no_such_registry.json")))
    assert base.schema_record["kind"] == "BASE" and base.schema_record["registry"] is None


def _check_unregistered_schema_refused(work):
    sreg = write_schema_registry(derived_schema("L0_SYNTH_TEST"))
    other = derived_schema("L9_UNREGISTERED")                                 # a self-consistent file, not listed
    raises_with(TrainerError, "is not in the derived-schema registry", trainer, work / "s",
                cfg("s", "GRU", schema_registry=sreg), schema=other)


def test_schema_not_in_registry_refused(work):
    _check_unregistered_schema_refused(work)
    raises_with(TrainerError, "does not exist", trainer, work / "x",
                cfg("x", "GRU", schema_registry=str(work / "none.json")), schema=derived_schema("L0_SYNTH_TEST"))


def test_derived_schema_tampering_refused(work):
    ds = derived_schema("L0_SYNTH_TEST")
    sreg = write_schema_registry(ds)
    c = cfg("t", "GRU", schema_registry=sreg)
    raises_with(TrainerError, "differs from the registry's", trainer, work / "a", c,
                schema=dataclasses.replace(ds, self_sha256="a" * 64))                        # another self-hash
    raw = json.loads(json.dumps(ds.raw))
    raw["keep"]["event_numeric_features"] = raw["keep"]["event_numeric_features"][:-1]
    raises_with(TrainerError, "content does not hash", trainer, work / "b", c, schema=dataclasses.replace(ds, raw=raw))
    p = Path(ds.path)
    good = p.read_bytes()
    p.write_text(json.dumps(dict(json.loads(good), title="edited"), sort_keys=True), encoding="utf-8")
    raises_with(TrainerError, "self-hash mismatch", trainer, work / "c", c, schema=ds)             # the registered file
    p.write_bytes(good)
    bad_reg = json.loads(Path(sreg).read_text(encoding="utf-8"))
    bad_reg["schemas"]["L0_SYNTH_TEST"]["note"] = "edited"
    raises_with(TrainerError, "self-hash mismatch", trainer, work / "d",
                cfg("d", "GRU", schema_registry=write_json_file(bad_reg)), schema=ds)         # the registry itself
    fsha = json.loads(Path(sreg).read_text(encoding="utf-8"))
    fsha["schemas"]["L0_SYNTH_TEST"]["file_sha256"] = "0" * 64
    raises_with(TrainerError, "file sha256", trainer, work / "e",
                cfg("e", "GRU", schema_registry=write_json_file(with_self_sha256(fsha))), schema=ds)
    trainer(work / "ok", c, schema=ds)


def test_schema01_derived_schema_trains_end_to_end(work):
    """SCHEMA-01's real DerivedSchema from contracts/derived_schemas (read-only), when that task's files exist."""
    reg = V4 / "contracts" / "derived_schemas" / "REGISTRY.json"
    S = pytest.importorskip("v4fs.schemas")
    if not reg.is_file() or not hasattr(S, "load_registered"):
        pytest.skip("SCHEMA-01's derived-schema registry is not present")
    ids = sorted(json.loads(reg.read_text(encoding="utf-8"))["schemas"])
    sid = "L3_NEAR_DUPLICATE" if "L3_NEAR_DUPLICATE" in ids else ids[-1]
    ds = S.load_registered(sid, reg)
    tr = trainer(work / "l3", cfg("l3", "GRU"), schema=ds, eval_override=scripted(VALS))      # default registry path
    assert tr.schema_record["kind"] == "DERIVED" and tr.schema_record["schema_id"] == sid
    s = tr.run()
    assert s["status"] == "DONE_MAIN" and s["schema"]["schema_id"] == sid
    assert {r["schema_id"] for r in read(art(work / "l3") / "RESULT_ROWS.json")["rows"]} == {sid}


# ------------------------------------------------------------------------------------------------ run labels
def test_run_label_and_role_derivation():
    assert RUN_LABELS == ("PRODUCTION", "SMOKE_NOT_EVIDENCE", "FEATURE_SEARCH_SCREEN", "HISTORICAL_REFERENCE",
                          "HP_SEARCH")                                          # TRAIN-02: P2 TUNING rows
    base = dict(run_id="l", family="GRU", seed=2026, exposure_point="end", cuda_visible_devices="0",
                determinism_mode="STRICT", key=dict(KEY), manifest_registry="r.json")
    cases = [(dict(data_version=NEW_DATA_VERSION), "PRODUCTION", "MAIN"),
             (dict(data_version=NEW_DATA_VERSION, phase="CALIBRATION"), "PRODUCTION", "CALIBRATION"),
             (dict(data_version=HISTORICAL_DATA_VERSION), "HISTORICAL_REFERENCE", "MAIN"),
             (dict(data_version=NEW_DATA_VERSION, population="FULL_TRAIN", calibration_record="c.json"),
              "HISTORICAL_REFERENCE", "FULL_TRAIN"),
             (dict(data_version=FS_DATA_VERSION, role="FEATURE_SEARCH_SCREEN"), "FEATURE_SEARCH_SCREEN",
              "FEATURE_SEARCH_SCREEN"),
             (dict(data_version=NEW_DATA_VERSION, test_overrides=True), "SMOKE_NOT_EVIDENCE", "MAIN"),
             (dict(data_version=FS_DATA_VERSION, role="FEATURE_SEARCH_SCREEN", test_overrides=True),
              "SMOKE_NOT_EVIDENCE", "FEATURE_SEARCH_SCREEN")]
    for over, label, role in cases:
        c = RunConfig(**{**base, **over})
        c.validate()
        assert (c.run_label, c.manifest_role) == (label, role), over
    assert "run_label" not in RunConfig.__dataclass_fields__, "the label is derived, never set by the caller"


def test_screen_role_uses_the_test_free_settings():
    base = dict(run_id="fs", family="SASREC", exposure_point="end", cuda_visible_devices="0", determinism_mode="STRICT",
                key=dict(KEY), role="FEATURE_SEARCH_SCREEN", **registry_keys("FEATURE_SEARCH_SCREEN"))
    for seed in (2026, 2027):
        c = RunConfig(**base, seed=seed)
        c.validate()                                                         # no test_overrides needed
        assert not c.test_overrides and c.effective_batch == 256 and c.peak_lr == 1e-3 and c.end_efe == 6.0
        assert c.determinism_mode == "STRICT" and c.run_label == "FEATURE_SEARCH_SCREEN"
    for text, bad in (("executed peak", dict(peak_lr=5e-4)), ("main runs use seeds", dict(seed=2028)),
                      ("effective batch is 256", dict(effective_batch=128)),
                      ("no continuation", dict(continuation_budget="b.json")),
                      ("not a calibration record", dict(calibration_record="c.json")),
                      ("determinism_mode is locked", dict(determinism_mode="WARN_ONLY")),
                      ("needs population COHORT and phase MAIN", dict(phase="CALIBRATION"))):
        raises_with(RecipeError, text, RunConfig(**{**base, "seed": 2026, **bad}).validate)


def _non_test_run(work, name, *, dv, role=None, vals=None, population="COHORT", **kw):
    tp, ep = providers(n=1024, role=role, population=population)
    reg = write_registry({(role or ("FULL_TRAIN" if population == "FULL_TRAIN" else "MAIN")): (tp, ep)},
                         data_version=dv)
    c = _non_test_cfg(name, reg=reg, dv=dv, role=role, population=population, **kw)
    tr = trainer(work / name, c, tp, ep, eval_override=scripted(vals or ramp(MAIN, 6.0)))
    return tr, tr.run(), read(art(work / name) / "RESULT_ROWS.json")["rows"]


def test_non_test_rows_carry_production_and_historical_labels(work, monkeypatch):
    _frozen_device_class(monkeypatch)
    _, s, rows = _non_test_run(work, "prod", dv=NEW_DATA_VERSION, vals=ramp(MAIN, 3.0))
    assert s["run_label"] == "PRODUCTION" and {r["run_label"] for r in rows} == {"PRODUCTION"}
    assert all(r["c_matched_eligible"] is True and r["row_status"] == "CENTRAL_CONTEST" for r in rows)
    assert_c_matched_eligible(rows[0])
    _, s, rows = _non_test_run(work, "hist", dv=HISTORICAL_DATA_VERSION, vals=ramp(MAIN, 3.0))
    assert s["run_label"] == "HISTORICAL_REFERENCE" and {r["run_label"] for r in rows} == {"HISTORICAL_REFERENCE"}
    assert all(r["c_matched_eligible"] is False for r in rows)
    assert_raises(SelectionError, assert_c_matched_eligible, rows[0])


def test_non_test_screen_run(work, monkeypatch):
    _frozen_device_class(monkeypatch)
    tr, s, rows = _non_test_run(work, "fs", dv=FS_DATA_VERSION, role="FEATURE_SEARCH_SCREEN")   # still improving
    assert not tr.cfg.test_overrides and tr.cfg.effective_batch == 256 and tr.screen
    assert s["run_label"] == "FEATURE_SEARCH_SCREEN" and s["still_improving"] and not s["latest_kept"]
    assert sorted(p.name for p in (art(work / "fs") / "checkpoints").glob("*.pt")) == ["best.pt", "endpoint_6.0.pt"]
    assert "feature-search screen" in (art(work / "fs") / "RETENTION_LOG.jsonl").read_text(encoding="utf-8")
    assert rows and all(r["run_label"] == "FEATURE_SEARCH_SCREEN" and r["row_status"] == "FEATURE_SEARCH_SCREEN"
                        and r["c_matched_eligible"] is False and r["downstream_seed_allowed"] is False
                        and r["manifest_role"] == "FEATURE_SEARCH_SCREEN" and r["data_version"] == FS_DATA_VERSION
                        for r in rows)
    _check_ids_recorded(work / "fs", label="FEATURE_SEARCH_SCREEN", role="FEATURE_SEARCH_SCREEN",
                        reg=tr.cfg.manifest_registry)
    assert_raises(SelectionError, assert_c_matched_eligible, rows[0])
    raises_with(SelectionError, "FEATURE_SEARCH_SCREEN row is refused", assert_c_matched_eligible,
                dict(rows[0], c_matched_eligible=True))                      # the label alone is enough
    raises_with(TrainerError, "has no continuation", trainer(work / "fs", tr.cfg, tr.train, tr.evalp).continue_run)


# ------------------------------------------------------------------------------------------------ selection labels
def _labelled(rows, label):
    out = []
    for r in rows:
        r = dict(r)
        if label is None:
            r.pop("run_label", None)
        else:
            r["run_label"] = label
        out.append(r)
    return out


def _arm_rows(label, v=(0.30, 0.31)):
    return {m: _labelled([row("GRU", 2026, x, rule=None, efe=2.0)], label) for m, x in zip((1.0, 0.5), v)}


def _cal_record(label, dv=NEW_DATA_VERSION, **over):
    return dict({"family": "GRU", "chosen_peak_lr": 1e-3, "run_label": label, "data_version": dv,
                 "evidence": label == "PRODUCTION"}, **over)


@pytest.mark.parametrize("label", list(RUN_LABELS) + [None])
def test_selection_refuses_each_label(label):
    """Every consumer accepts only its label (cohort / calibration / C_MATCHED: PRODUCTION; tie-break:
    HISTORICAL_REFERENCE); every other label, and an unlabelled row, is refused with a message naming it."""
    text = "unlabelled row" if label is None else f"{label} row is refused"
    cohort = _labelled(cohort_rows(), label)
    if label == "PRODUCTION":
        assert select_family(cohort, delta_bar_ci=WIDE)["status"] == "DECISIVE_STABLE"
    else:
        raises_with(SelectionError, text, select_family, cohort, delta_bar_ci=WIDE)
    full = _labelled(full_rows((0.305,), (0.300,), seeds=(2026,)), label)
    if label == "HISTORICAL_REFERENCE":
        assert tiebreak_stage1(tb_cohort(), full, summaries(), delta_ci=WIDE)["status"] == "FULL_TRAIN_TIEBREAK_STAGE1"
    else:
        raises_with(SelectionError, text, tiebreak_stage1, tb_cohort(), full, summaries(), delta_ci=WIDE)
    if label == "PRODUCTION":
        ch = calibration_choice(_arm_rows(label))
        assert ch["chosen_multiplier"] == 0.5 and ch["chosen_peak_lr"] == 5e-4
        assert accept_calibration_record(_cal_record(label), family="GRU", data_version=NEW_DATA_VERSION) == 1e-3
        assert_c_matched_eligible(_labelled([row("GRU", 2026, 0.3)], label)[0])
    else:
        raises_with(SelectionError, text, calibration_choice, _arm_rows(label))
        raises_with(SelectionError, f"run_label {label!r} refused", accept_calibration_record, _cal_record(label),
                    family="GRU", data_version=NEW_DATA_VERSION)
        assert_raises(SelectionError, assert_c_matched_eligible, _labelled([row("GRU", 2026, 0.3)], label)[0])


def test_calibration_choice_and_record_guards():
    raises_with(SelectionError, "not 1.9", calibration_choice,
                {1.0: [row("GRU", 2026, 0.3, rule=None, efe=1.9)], 0.5: [row("GRU", 2026, 0.3, rule=None, efe=2.0)]})
    mixed = _arm_rows("PRODUCTION")
    mixed[0.5] = [dict(mixed[0.5][0], data_version="OTHER")]
    raises_with(SelectionError, "arms differ", calibration_choice, mixed)
    tie = calibration_choice(_arm_rows("PRODUCTION", v=(0.3, 0.3 + 5e-9)))
    assert tie["chosen_multiplier"] == 1.0                                   # exact tie keeps the executed peak
    for text, rec in (("another family", _cal_record("PRODUCTION", family="SASREC")),
                      ("data_version", _cal_record("PRODUCTION", dv="OTHER")),
                      ("not marked as evidence", _cal_record("PRODUCTION", evidence=False))):
        raises_with(SelectionError, text, accept_calibration_record, rec, family="GRU", data_version=NEW_DATA_VERSION)


def _check_screen_rows_refused():
    rows = _labelled(cohort_rows(), "FEATURE_SEARCH_SCREEN")
    raises_with(SelectionError, "FEATURE_SEARCH_SCREEN row is refused", select_family, rows, delta_bar_ci=WIDE)


def test_screen_rows_refused_by_selection():
    _check_screen_rows_refused()


def test_non_test_calibration_is_production_and_is_consumed(work, monkeypatch):
    _frozen_device_class(monkeypatch)
    tp, ep = providers("CALIBRATION", n=1024)
    reg = write_registry({"CALIBRATION": (tp, ep)}, data_version=NEW_DATA_VERSION)
    kw = dict(key=dict(KEY), exposure_point="end", cuda_visible_devices=cfg("x", "GRU").cuda_visible_devices,
              determinism_mode="STRICT", model_kw={"K": K}, trainer_kw={"eval_override": scripted(
                  {m: 0.2 for m in eval_marks("CALIBRATION")})}, log_every=5)
    rec = run_calibration("GRU", SCHEMA, tp, ep, work / "cal", manifest_registry=reg, data_version=NEW_DATA_VERSION,
                          test_overrides=False, disk_usage_fn=BIG_DISK, **kw)
    assert rec["run_label"] == "PRODUCTION" and rec["evidence"] is True and rec["chosen_multiplier"] == 1.0   # tie
    assert rec["manifest_registry"]["file_sha256"] == _sha(reg) and rec["data_version"] == NEW_DATA_VERSION
    assert {a["run_label"] for a in rec["arms"]} == {"PRODUCTION"}
    tpm, epm = providers("MAIN", n=1024)
    mreg = write_registry({"MAIN": (tpm, epm)}, data_version=NEW_DATA_VERSION)
    good = _non_test_cfg("main", reg=mreg, dv=NEW_DATA_VERSION,
                         calibration_record=str(work / "cal" / "CALIBRATION_GRU.json"))
    start = trainer(work / "m", good, tpm, epm)._run_start_record({"state": "OK"})
    assert start["calibration_record"]["run_label"] == "PRODUCTION"
    smoke = write_json_file(dict(read(work / "cal" / "CALIBRATION_GRU.json"), run_label="SMOKE_NOT_EVIDENCE",
                                 evidence=False))
    raises_with(TrainerError, "calibration record refused", trainer(
        work / "m2", dataclasses.replace(good, calibration_record=smoke), tpm, epm)._run_start_record, {"state": "OK"})
    hreg = write_registry({"CALIBRATION": (tp, ep)}, data_version=HISTORICAL_DATA_VERSION)
    raises_with(CalibrationError, "gives HISTORICAL_REFERENCE", run_calibration, "GRU", SCHEMA, tp, ep, work / "hcal",
                manifest_registry=hreg, data_version=HISTORICAL_DATA_VERSION, test_overrides=False, **kw)
    assert not (work / "hcal").exists(), "refused before any arm runs"


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a trainer that accepts a manifest id that is not the registry's for the role (the pre-R2 open door)")
def test_nc_manifest_id_not_in_registry(work, monkeypatch):
    def lax(reg, role, kind, p):                          # MUTANT: no registry lookup of the provider's id / binding
        mid = p.manifest_id if kind == "train" else p.manifest.manifest_id
        return {"manifest_id": mid, "content_sha256": None, "header_sha256": None, "n_decisions": 0,
                "fingerprint": p.fingerprint, "bound_by": []}
    monkeypatch.setattr(T, "check_provider", lax)
    _check_unregistered_manifest_refused(work)


@nc("a registry loader that does not verify the registry's self-hash")
def test_nc_registry_hash_mismatch(work, monkeypatch):
    monkeypatch.setattr(R, "_verify_self_hash", lambda obj, what: obj.get("self_sha256"))    # MUTANT
    _check_registry_hash_mismatch_refused(work)


@nc("a schema check that trusts the schema object instead of the derived-schema registry")
def test_nc_schema_not_in_registry(work, monkeypatch):
    def trusting(schema, registry_path=None):             # MUTANT: the object's own claims, no registry lookup
        return {"kind": "DERIVED", "schema_id": schema.schema_id, "schema_self_sha256": schema.self_sha256,
                "file": str(schema.path), "file_sha256": None, "registry": None}
    monkeypatch.setattr(T, "verify_schema", trusting)
    _check_unregistered_schema_refused(work)


@nc("a cohort selection that accepts FEATURE_SEARCH_SCREEN rows as evidence")
def test_nc_screen_row_accepted_by_selection(monkeypatch):
    monkeypatch.setitem(SEL.EVIDENCE_LABELS, "COHORT", ("PRODUCTION", "FEATURE_SEARCH_SCREEN"))       # MUTANT
    _check_screen_rows_refused()


@nc("a resume that ignores a schema change")
def test_nc_resume_with_changed_schema(work, monkeypatch):
    monkeypatch.setattr(T.CentralTrainer, "_check_resume_schema", lambda self, recorded, where: None)  # MUTANT
    _check_resume_refuses_changed_schema(work)
