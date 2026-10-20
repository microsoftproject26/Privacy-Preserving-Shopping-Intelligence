"""GATE-B B1 (MULTI_EXPERT_REVIEW_20260929 §4, review 06 [5-R1]) — v4int.reeval accepts the two registered non-MAIN
central sources (CPU; the GPU / pin / data-root evaluation stays server-only, like every other reeval run):

  * C_FULL (REGISTRY_V2_FULL_EXT.json role C_FULL: FULL_TRAIN_5_100_TIEBREAK -> MAIN_VALIDATION_5_100_131072) and PRE
    (REGISTRY_V2_PRETRAIN_EXT.json role PRETRAIN: PRETRAIN_TRAIN_5_100_NONMAIN -> UNEW_VAL_VALIDATION_32768,
    endpoint_6.0 only) pass the --dry-run on MAIN and the panels, verified against the REAL extension files and the real
    REGISTRY_V2, with the checkpoint meta checked on the CPU; the planned key and the row labels are reported;
  * every unregistered / mislabelled source refuses (claimed by train manifest, run-id prefix, run label or arm label);
  * the MAIN-pair central dry run is byte-for-byte the pre-B1 record;
  * the key helpers (source train side on MAIN_TEST, the chained source-extension verify wrapper).
Negative controls (test_nc_*, strict xfail on AssertionError).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from intkit import nc
from v4int import reeval as R
import ppsi_paths

V4 = ppsi_paths.PROJECT_ROOT.resolve()       # the ppsi_v4 workspace (contracts, company records)
REG = V4 / "contracts" / "manifest_registry"
SOURCES = {k: dict(v) for k, v in R.CENTRAL_SOURCES.items()}     # the registered table at import (NC-proof fixtures)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _ext_self(name: str) -> str:
    return json.loads((REG / SOURCES[name]["ext"]).read_text(encoding="utf-8"))["self_sha256"]


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reeval_records"
REAL_START = {"C_FULL": "CFULL_s2026_RUN_START.json", "PRE": "PRE_RUN_START.json"}   # FIX-REEVAL: server copies


def real_start(name: str) -> dict:
    """The REAL RUN_START.json of the finished source run (copied read-only from the server; FIX-REEVAL)."""
    return json.loads((FIXTURES / REAL_START[name]).read_text(encoding="utf-8"))


def source_run(tmp: Path, name: str, *, ckpt: str = None, meta_over=None, start_over=None, kb_over=None,
               kind: str = "AUTO", init: bool = True, init_over=None, start: dict = None) -> Path:
    """A sealed run of the registered source `name`: its RUN_START is the REAL server record (FIX-REEVAL: config with
    config.key through the extension, key base / manifest_registry = the per-run trainer view, providers) unless
    `start` is given, with start_over / kb_over applied; a real torch {kind, meta, model} checkpoint whose meta carries
    the record's config / init sha256 (v4train.checkpoint layout: kind = the file's kind) and, for PRE, the run's
    PRETRAIN_FOR_FL_INIT.json binding that checkpoint (FIX-DELTA-03 m2)."""
    d = SOURCES[name]
    ckpt = ckpt or ("endpoint_6.0.pt" if name == "PRE" else "best.pt")
    if kind == "AUTO":
        kind = {"best.pt": "best", "endpoint_6.0.pt": "endpoint_6.0"}.get(ckpt, ckpt.rsplit(".", 1)[0])
    run = tmp / "run"
    cp = run / "sealed" / "checkpoints" / ckpt
    cp.parent.mkdir(parents=True)
    start = dict(real_start(name) if start is None else start)
    start["key_base"] = dict(start.get("key_base") or {}, **(kb_over or {}))
    start.update(start_over or {})
    rid = start.get("run_id")
    (run / "RUN_START.json").write_text(json.dumps(start), encoding="utf-8")
    meta = dict({"model_variant": d["variant"], "seed": d["seed"], "config_sha256": start.get("config_sha256"),
                 "init_sha256": start.get("init_sha256"), "efe": 6.0, "mark": 6.0}, **(meta_over or {}))
    obj = {"meta": meta, "model": {"w": torch.zeros(2)}}
    if kind is not None:
        obj["kind"] = kind
    torch.save(obj, cp)
    if name == "PRE" and init:
        rec = {"task": "DATA-10 v4int.pretrain_small", "run_id": rid, "run_label": "PRETRAIN_FOR_FL",
               "fl_init": dict({"kind": "endpoint_6.0", "path": f"sealed/checkpoints/{ckpt}", "sha256": _sha(cp),
                                "bytes": cp.stat().st_size}, **(init_over or {}))}
        (run / "PRETRAIN_FOR_FL_INIT.json").write_text(json.dumps(rec), encoding="utf-8")
    return cp


def argv(cp: Path, tmp: Path, name: str, *, dry=True, **kw):
    d = SOURCES.get(name) or SOURCES["C_FULL"]
    a = {"--ckpt": str(cp), "--ckpt-kind": "central", "--expect-sha256": _sha(cp), "--variant": d["variant"],
         "--recipe-id": d["recipe_id"], "--seed": str(d["seed"]), "--role": "MAIN", "--arm-label": d["arm_label"],
         "--selection-rule": "PRACTICAL_BEST", "--checkpoint-id": f"{name}:ckpt", "--gpu": "0",
         "--out-dir": str(tmp / "out"), "--history-policy": "USER_HISTORY_L50"}
    a.update({f"--{k.replace('_', '-')}": str(v) for k, v in kw.items()})
    return R.parse_args([x for kv in a.items() for x in kv] + (["--dry-run"] if dry else []))


# ------------------------------------------------------------------------------------------------ the dry run
def check_registered_sources_dry_run(tmp_path, run=R.run):
    """B1 key property: C_FULL (MAIN = its own eval manifest: SOURCE train side; panels: ROLE key) and PRE (MAIN and the
    panels: ROLE key) pass the dry run against the real extensions; rows will carry the source labels."""
    for name, role, planned in (("C_FULL", "MAIN", "SOURCE_TRAIN_SIDE"), ("C_FULL", "SPARSE_PANEL", "ROLE_KEY"),
                                ("PRE", "MAIN", "ROLE_KEY"), ("PRE", "HEAVY_PANEL", "ROLE_KEY"),
                                ("PRE", "UNSEEN_HOLDOUT", "ROLE_KEY")):
        t = tmp_path / f"{name}_{role}"
        cp = source_run(t, name)
        try:
            rec = run(argv(cp, t, name, role=role))
        except R.ReevalRefused as e:
            raise AssertionError(f"{name} on {role} refused: {e}")
        d = SOURCES[name]
        src = rec.get("central_source") or {}
        assert src.get("source") == name and src.get("pair") == {"train": d["train"], "eval": d["eval"]}, rec
        assert src["ext_self_sha256"] == _ext_self(name) and src["checks"]["all_ok"]
        assert rec["planned_key"].startswith(planned) and rec["arm_label"] == d["arm_label"]
        assert rec["row_labels"] == {"central_source": name, "source_train_manifest_id": d["train"],
                                     "source_eval_manifest_id": d["eval"], "source_manifest_role": d["ext_role"],
                                     "source_registry_self_sha256": _ext_self(name)}
        assert rec["checkpoint_meta"]["model_variant"] == d["variant"] and not (t / "out").exists()


def test_registered_sources_pass_the_dry_run(tmp_path):
    check_registered_sources_dry_run(tmp_path)


def test_c_full_endpoint_checkpoint_accepted(tmp_path):
    cp = source_run(tmp_path, "C_FULL", ckpt="endpoint_6.0.pt")
    assert R.run(argv(cp, tmp_path, "C_FULL", selection_rule="CONTROLLED_BUDGET"))["central_source"]["source"] == "C_FULL"


def test_main_pair_dry_run_record_is_unchanged(tmp_path):
    """A MAIN-pair central run (C_SMALL) keeps the pre-B1 dry-run record exactly (no torch load, no new key)."""
    run = tmp_path / "run"
    cp = run / "sealed" / "checkpoints" / "best.pt"
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b"synthetic-checkpoint-for-preflight")
    start = {"run_id": "C_SMALL_s2026", "config": {"family": "SASREC", "model_variant": "SASREC_D64_B2",
                                                   "recipe_id": "SMALL_SASREC_D64_B2", "seed": 2026},
             "key_base": {"split": "VALIDATION", "run_label": "PRODUCTION", "train_manifest_id": "MAIN_TRAIN_5_100_131072"},
             "train_provider": {"manifest_id": "MAIN_TRAIN_5_100_131072"},
             "eval_provider": {"manifest_id": "MAIN_VALIDATION_5_100_131072"}}
    (run / "RUN_START.json").write_text(json.dumps(start), encoding="utf-8")
    a = argv(cp, tmp_path, "C_SMALL", variant="SASREC_D64_B2", recipe_id="SMALL_SASREC_D64_B2", arm_label="C_SMALL")
    rec = R.run(a)
    assert rec == {"dry_run": True, "checkpoint_sha256": _sha(cp), "checkpoint_kind": "central",
                   "source_run": str(run), "source_run_id": "C_SMALL_s2026"}


def test_real_extensions_verify_against_the_real_registry():
    for name in R.CENTRAL_SOURCES:                                  # FIX-REEVAL: the REAL RUN_START records
        rec = R.verify_source_registry(name, real_start(name), REG / "REGISTRY_V2.json")
        assert rec["checks"]["all_ok"] and rec["base_registry_file_sha256"] == _sha(REG / "REGISTRY_V2.json")
    with pytest.raises(R.ReevalRefused, match="manifest_registry_self_sha256"):
        R.verify_source_registry("PRE", real_start("C_FULL"), REG / "REGISTRY_V2.json")
    with pytest.raises(R.ReevalRefused, match="refused"):
        R.verify_source_registry("C_FULL", real_start("C_FULL"), REG / "REGISTRY_V2.json", base_file_sha256="0" * 64)


# ------------------------------------------------------------------------------------------------ refusals
def check_unregistered_sources_refuse(tmp_path, run=R.run):
    cases = [("C_FULL", dict(start_over={"eval_provider": {"manifest_id": "UNEW_VAL_VALIDATION_32768"}}), {},
              "eval_provider"),
             ("C_FULL", dict(start_over={"run_id": "C_BIG_02B_SASREC_s2026"}), {}, "run id prefix"),
             ("C_FULL", {}, dict(seed=2027), "seed"),
             ("C_FULL", {}, dict(role="TRUE_COLD"), "not registered for the C_FULL"),
             ("C_FULL", dict(kb_over={"manifest_registry_self_sha256": "f" * 64}), {}, "manifest_registry_self"),
             ("C_FULL", dict(meta_over={"config_sha256": "d" * 64}), {}, "config_sha256"),
             ("PRE", dict(ckpt="best.pt"), {}, "checkpoint file"),
             ("PRE", dict(kb_over={"run_label": "PRODUCTION"}), {}, "run_label"),
             ("PRE", {}, dict(arm_label="C_FULL"), "more than one"),
             ("PRE", {}, dict(arm_label="FA_WARM"), "--arm-label"),
             ("PRE", {}, dict(variant="SASREC_D256_B3", recipe_id="02B_SASREC"), "variant")]
    for i, (name, run_kw, arg_kw, msg) in enumerate(cases):
        t = tmp_path / f"r{i}"
        cp = source_run(t, name, **run_kw)
        refused = None
        try:
            run(argv(cp, t, name, **arg_kw))
        except R.ReevalRefused as e:
            refused = str(e)
        assert refused is not None and msg in refused, (i, name, msg, refused)


def test_unregistered_or_mislabelled_sources_refuse(tmp_path):
    check_unregistered_sources_refuse(tmp_path)


def test_source_arm_labels_on_a_main_pair_or_fl_checkpoint_refuse(tmp_path):
    run = tmp_path / "run"
    cp = run / "sealed" / "checkpoints" / "best.pt"
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b"x")
    (run / "RUN_START.json").write_text(json.dumps({
        "run_id": "02B_SASREC_MAIN_s2026", "config": {"family": "SASREC", "model_variant": "SASREC_D256_B3",
                                                       "recipe_id": "02B_SASREC", "seed": 2026},
        "key_base": {"split": "VALIDATION", "run_label": "PRODUCTION"},
        "train_provider": {"manifest_id": "MAIN_TRAIN_5_100_131072"},
        "eval_provider": {"manifest_id": "MAIN_VALIDATION_5_100_131072"}}), encoding="utf-8")
    with pytest.raises(R.ReevalRefused, match="not the registered C_FULL"):          # a C_BIG checkpoint as C_FULL
        R.run(argv(cp, tmp_path, "C_FULL"))
    with pytest.raises(R.ReevalRefused, match="registered CENTRAL source"):
        R.run(argv(cp, tmp_path, "PRE", ckpt_kind="fl", variant="SASREC_D64_B2", recipe_id="SMALL_SASREC_D64_B2"))


@nc("reeval WITHOUT the source classification (the pre-B1 central path) must not pass the unregistered-source refusals")
def test_nc_unclassified_sources_are_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "classify_central_source", lambda start, args, ckpt: None)
    check_unregistered_sources_refuse(tmp_path)


@nc("a reeval whose registered-source table lacks PRE (an unregistered theta_pre) must not pass the dry run of PRE")
def test_nc_pre_not_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "CENTRAL_SOURCES", {k: v for k, v in R.CENTRAL_SOURCES.items() if k != "PRE"})
    check_registered_sources_dry_run(tmp_path)


# ------------------------------------------------------------------------------------------------ FIX-DELTA-03 m1 / m2
def check_source_checkpoint_files_refuse(tmp_path, run=R.run):
    """REV-DELTA-03 m1: C_FULL is best.pt (PRACTICAL_BEST) or endpoint_6.0.pt (CONTROLLED_BUDGET) only, bound to the rule
    (no endpoint_8.0 after the 06D §7 continuation, no intermediate file), and the file's v4train kind must be its
    name's. m2: PRE's endpoint carries meta mark 6.0 and is the run's PRETRAIN_FOR_FL_INIT.json fl_init (path + sha)."""
    cases = [("C_FULL", dict(ckpt="endpoint_8.0.pt"), dict(selection_rule="CONTROLLED_BUDGET"), "checkpoint file"),
             ("C_FULL", dict(ckpt="best_extended.pt"), {}, "checkpoint file"),
             ("C_FULL", dict(ckpt="best.pt"), dict(selection_rule="CONTROLLED_BUDGET"), "--selection-rule"),
             ("C_FULL", dict(ckpt="endpoint_6.0.pt"), dict(selection_rule="PRACTICAL_BEST"), "--selection-rule"),
             ("C_FULL", dict(ckpt="best.pt", kind="endpoint_6.0"), {}, "carries kind"),
             ("C_FULL", dict(ckpt="endpoint_6.0.pt", meta_over={"mark": 5.5}), dict(selection_rule="CONTROLLED_BUDGET"),
              "meta mark"),
             ("PRE", dict(meta_over={"mark": 3.0}), {}, "meta mark"),
             ("PRE", dict(meta_over={"mark": None}), {}, "meta mark"),
             ("PRE", dict(kind="best"), {}, "carries kind"),
             ("PRE", dict(init=False), {}, "PRETRAIN_FOR_FL_INIT.json"),
             ("PRE", dict(init_over={"sha256": "0" * 64}), {}, "fl_init.sha256"),
             ("PRE", dict(init_over={"path": "checkpoints/endpoint_6.0.pt"}), {}, "fl_init.path"),
             ("PRE", dict(init_over={"kind": "best"}), {}, "fl_init.kind")]
    for i, (name, run_kw, arg_kw, msg) in enumerate(cases):
        t = tmp_path / f"f{i}"
        cp = source_run(t, name, **run_kw)
        refused = None
        try:
            run(argv(cp, t, name, **arg_kw))
        except R.ReevalRefused as e:
            refused = str(e)
        assert refused is not None and msg in refused, (i, name, msg, refused)


def test_source_checkpoint_files_and_theta_pre_record_refuse(tmp_path):
    check_source_checkpoint_files_refuse(tmp_path)


def test_source_dry_run_records_the_checkpoint_file_and_the_pre_init_record(tmp_path):
    cp = source_run(tmp_path / "c", "C_FULL", ckpt="endpoint_6.0.pt")
    rec = R.run(argv(cp, tmp_path / "c", "C_FULL", selection_rule="CONTROLLED_BUDGET"))
    assert rec["checkpoint_file"] == {"file": "endpoint_6.0.pt", "kind": "endpoint_6.0", "mark": 6.0}
    assert "pre_init_record" not in rec
    cp = source_run(tmp_path / "p", "PRE")
    rec = R.run(argv(cp, tmp_path / "p", "PRE", selection_rule="CONTROLLED_BUDGET"))
    assert rec["pre_init_record"]["fl_init_sha256"] == _sha(cp) and rec["checkpoint_file"]["mark"] == 6.0
    assert R.CENTRAL_SOURCES["C_FULL"]["checkpoint_files"] == ("best.pt", "endpoint_6.0.pt")
    assert R.CENTRAL_SOURCES["C_FULL"]["rule_files"] == {"PRACTICAL_BEST": "best.pt",
                                                         "CONTROLLED_BUDGET": "endpoint_6.0.pt"}


@nc("the pre-FIX-DELTA-03 C_FULL source (checkpoint_files None, no rule binding) must not pass the C_FULL file refusals")
def test_nc_c_full_any_checkpoint_file(tmp_path, monkeypatch):
    loose = {k: dict(v) for k, v in R.CENTRAL_SOURCES.items()}
    loose["C_FULL"].update(checkpoint_files=None, rule_files=None)
    monkeypatch.setattr(R, "CENTRAL_SOURCES", loose)
    check_source_checkpoint_files_refuse(tmp_path)


@nc("a reeval that checks the file name only (no kind / meta mark check) must not pass the checkpoint refusals")
def test_nc_source_checkpoint_name_only(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_source_checkpoint", lambda name, obj, ckpt: {})
    check_source_checkpoint_files_refuse(tmp_path)


@nc("a reeval that trusts --expect-sha256 for PRE (no PRETRAIN_FOR_FL_INIT.json binding) must not pass the refusals")
def test_nc_pre_without_the_init_record(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_pre_init_record", lambda src, ckpt, got: {})
    check_source_checkpoint_files_refuse(tmp_path)


# ------------------------------------------------------------------------------------------------ key helpers
def test_source_train_side_keeps_the_c_full_train_side_on_main_test():
    base_kb = {"train_manifest_id": "FULL_TRAIN_5_100_TIEBREAK", "population_hash": "TB", "split": "VALIDATION"}
    kb = {"train_manifest_id": "MAIN_TRAIN_5_100_131072", "population_hash": "MAIN", "split": "RETRO_TEST",
          "eval_manifest_id": "MAIN_RETRO_TEST_5_100_131072", "candidate_order_hash": "C"}
    over = {"manifest_role": "MAIN_TEST", "train_manifest_id": "MAIN_TRAIN_5_100_131072", "v4_split": "RETRO_TEST"}
    kb2, over2 = R.source_train_side(kb, over, base_kb)
    assert kb2 == dict(kb, train_manifest_id="FULL_TRAIN_5_100_TIEBREAK", population_hash="TB")
    assert over2 == dict(over, train_manifest_id="FULL_TRAIN_5_100_TIEBREAK")
    assert R.CENTRAL_SOURCES["C_FULL"]["own_eval_roles"] == ("MAIN", "MAIN_TEST")
    assert R.CENTRAL_SOURCES["PRE"]["own_eval_roles"] == ()          # PRE on MAIN: the MAIN role key (ANA A-F7 allows it)


def test_source_row_labels_are_not_comparison_key_fields():
    from v4eval.rows import COMPARISON_KEY
    lab = R.source_row_labels("PRE", {"ext_self_sha256": "s"})
    assert not set(lab) & set(COMPARISON_KEY)
    rows = R.relabel([{"method": "C"}], arm_label="PRE", selection_rule="CONTROLLED_BUDGET", checkpoint_id="x",
                     checkpoint_sha256="y", kind="central", source_run_id="r", efe=6.0, extra=lab)
    assert rows[0]["method"] == "PRE" and rows[0]["source_train_manifest_id"] == "PRETRAIN_TRAIN_5_100_NONMAIN"


def test_install_source_ext_chains_one_wrapper_per_extension(tmp_path, monkeypatch):
    import v4int.verify as VF
    base_p = REG / "REGISTRY_V2.json"
    base = json.loads(base_p.read_text(encoding="utf-8"))
    calls = []

    def orig(path, lock_self, lock_file, *, lock=None, root=None):
        calls.append(Path(path).name)
        return base, _sha(base_p), base["self_sha256"], None
    monkeypatch.setattr(VF, "verify_registry", orig)
    full, pre = REG / "REGISTRY_V2_FULL_EXT.json", REG / "REGISTRY_V2_PRETRAIN_EXT.json"
    R.install_source_ext(full, base_p)
    first = VF.verify_registry
    R.install_source_ext(full, base_p)                                 # idempotent for the same path
    assert VF.verify_registry is first
    R.install_source_ext(pre, base_p)
    assert VF.verify_registry(full, "L", "F")[2] == _ext_self("C_FULL")
    assert VF.verify_registry(pre, "L", "F")[2] == _ext_self("PRE")
    assert VF.verify_registry(base_p, "L", "F")[0] is base and set(calls) == {"REGISTRY_V2.json"}
