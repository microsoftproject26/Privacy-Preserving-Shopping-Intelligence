"""ANA-05 - v4int.reeval admits C_FULL seed 2027 as a registered DESCRIPTIVE central source (06D §8 NICE seed; 06E §2 /
DL 176: the second seed of a single-seed contrast is DESCRIPTIVE; K15 / K17-K19 / K20b / K20c stay seed 2026). CPU only.

No real C_FULL s2027 record exists locally (DL 265: the s2027 dry run was refused on the server before any record was
copied). The s2027 RUN_START used here is DERIVED from the REAL s2026 record (fixtures/reeval_records, FIX-REEVAL): the
seed fields (seed, config.seed, run_id, config.run_id, key_base.initialization_block) are set to 2027 and config_sha256
is recomputed; every registry hash (config.key = the FULL extension, key_base / manifest_registry = the trainer view)
is the real record's, which is what a second c_full run through the same extension writes (INFERENCE; the server dry
run of s2027 confirms it).

  * the s2027 dry run passes the SAME checks as s2026 (classification, FIX-REEVAL binding, checkpoint file / kind /
    mark) on MAIN PB / CB and the panels, and reports central_source.seed_status DESCRIPTIVE and the DESCRIPTIVE row
    labels (comparison_status / source_seed_status / source_seed_rule);
  * the s2026 dry-run record and row labels are unchanged (no seed_status key);
  * seed 2028, PRE at seed 2027 and a seed / record mismatch refuse; the FIX-REEVAL binding still refuses on s2027.
Negative controls (test_nc_*, strict xfail on AssertionError).
"""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from intkit import nc
from test_gate_b_reeval import REG, SOURCES, argv, real_start, source_run
from v4int import reeval as R

BASE_SELF = json.loads((REG / "REGISTRY_V2.json").read_text(encoding="utf-8"))["self_sha256"]


def rehash(start: dict) -> dict:
    start["config_sha256"] = hashlib.sha256(json.dumps(start["config"], sort_keys=True, allow_nan=False,
                                                       default=list).encode("utf-8")).hexdigest()
    return start


def derived_start(seed: int = 2027) -> dict:
    """The real C_FULL s2026 RUN_START with its seed fields set to `seed` and config_sha256 recomputed."""
    st = copy.deepcopy(real_start("C_FULL"))
    rid = f"C_FULL_02B_SASREC_s{seed}"
    st["seed"] = st["config"]["seed"] = seed
    st["run_id"] = st["config"]["run_id"] = rid
    kb = st["key_base"]
    kb["initialization_block"] = kb["initialization_block"].replace("seed=2026", f"seed={seed}")
    return rehash(st)


def s2027_run(tmp, ckpt="best.pt", start=None, seed=2027):
    return source_run(tmp, "C_FULL", ckpt=ckpt, start=start or derived_start(seed), meta_over={"seed": seed})


CASES = (("MAIN", "best.pt", "PRACTICAL_BEST", "SOURCE_TRAIN_SIDE"),
         ("MAIN", "endpoint_6.0.pt", "CONTROLLED_BUDGET", "SOURCE_TRAIN_SIDE"),
         ("UNSEEN_HOLDOUT", "best.pt", "PRACTICAL_BEST", "ROLE_KEY"),
         ("SPARSE_PANEL", "best.pt", "PRACTICAL_BEST", "ROLE_KEY"),
         ("HEAVY_PANEL", "endpoint_6.0.pt", "CONTROLLED_BUDGET", "ROLE_KEY"))


def check_s2027_dry_run(tmp_path, run=R.run):
    real = real_start("C_FULL")
    for i, (role, ckpt, rule, planned) in enumerate(CASES):
        t = tmp_path / f"s27_{i}"
        cp = s2027_run(t, ckpt=ckpt)
        try:
            rec = run(argv(cp, t, "C_FULL", role=role, selection_rule=rule, seed=2027))
        except R.ReevalRefused as e:
            raise AssertionError(f"C_FULL s2027 {role} {rule} refused: {e}")
        cs = rec["central_source"]
        assert cs["source"] == "C_FULL" and cs["checks"]["all_ok"], rec
        assert cs.get("seed_status") == "DESCRIPTIVE" and cs.get("seed") == 2027, cs
        b = cs["binding"]                                             # the same registry binding as the real s2026 run
        assert b["comparison_key_registry_self_sha256"] == real["config"]["key"]["manifest_registry_self_sha256"]
        assert b["trainer_view_self_sha256"] == real["key_base"]["manifest_registry_self_sha256"]
        assert rec["planned_key"].startswith(planned) and rec["checkpoint_meta"]["seed"] == 2027
        lab = rec["row_labels"]
        assert lab["central_source"] == "C_FULL" and lab["comparison_status"] == "DESCRIPTIVE", lab
        assert lab["source_seed_status"] == "DESCRIPTIVE" and "06E §2" in lab["source_seed_rule"]
        assert not (t / "out").exists()


def test_c_full_s2027_passes_the_same_checks_as_s2026(tmp_path):
    check_s2027_dry_run(tmp_path)


def test_c_full_s2026_record_and_labels_unchanged(tmp_path):
    cp = source_run(tmp_path, "C_FULL")                                # the unmodified REAL s2026 RUN_START
    rec = R.run(argv(cp, tmp_path, "C_FULL"))
    assert "seed_status" not in rec["central_source"] and "seed" not in rec["central_source"]
    assert set(rec["row_labels"]) == {"central_source", "source_train_manifest_id", "source_eval_manifest_id",
                                      "source_manifest_role", "source_registry_self_sha256"}
    assert R.source_seed_labels("C_FULL", 2026) == {} and R.source_seed_labels(None, 2027) == {}
    assert R.source_seed_labels("PRE", 2026) == {}
    assert R.source_seeds("C_FULL") == (2026, 2027) and R.source_seeds("PRE") == (2026,)
    assert SOURCES["C_FULL"]["seed"] == 2026                           # the confirmatory seed is unchanged


def check_seed_refusals(tmp_path, run=R.run):
    cases = [("c2028", lambda t: s2027_run(t, start=derived_start(2028), seed=2028), dict(seed=2028), "--seed"),
             ("rec_vs_arg", lambda t: s2027_run(t), dict(seed=2026), "seed"),          # an s2027 run called s2026
             ("s26_as_27", lambda t: source_run(t, "C_FULL"), dict(seed=2027), "seed")]
    for tag, mk, kw, msg in cases:
        t = tmp_path / tag
        cp = mk(t)
        refused = None
        try:
            run(argv(cp, t, "C_FULL", **kw))
        except R.ReevalRefused as e:
            refused = str(e)
        assert refused is not None and msg in refused, (tag, refused)
    t = tmp_path / "pre27"                                              # PRE is ONE theta_pre (seed 2026)
    st = copy.deepcopy(real_start("PRE"))
    st["seed"] = st["config"]["seed"] = 2027
    cp = source_run(t, "PRE", start=rehash(st), meta_over={"seed": 2027})
    with pytest.raises(R.ReevalRefused, match="--seed"):
        run(argv(cp, t, "PRE", seed=2027))


def test_unregistered_seeds_refuse(tmp_path):
    check_seed_refusals(tmp_path)


def test_fix_reeval_binding_still_refuses_on_s2027(tmp_path):
    st = derived_start()
    st["config"]["key"]["manifest_registry_self_sha256"] = BASE_SELF              # the base registry, re-hashed
    cp = s2027_run(tmp_path / "a", start=rehash(st))
    with pytest.raises(R.ReevalRefused, match="FIX-REEVAL"):
        R.run(argv(cp, tmp_path / "a", "C_FULL", seed=2027))
    st = derived_start()
    st["config"]["peak_lr"] = 1.0                                                  # config edited, not re-hashed
    cp = s2027_run(tmp_path / "b", start=st)
    with pytest.raises(R.ReevalRefused, match="config_sha256"):
        R.run(argv(cp, tmp_path / "b", "C_FULL", seed=2027))
    cp = s2027_run(tmp_path / "c", ckpt="endpoint_8.0.pt")                         # file rule unchanged
    with pytest.raises(R.ReevalRefused, match="checkpoint file"):
        R.run(argv(cp, tmp_path / "c", "C_FULL", seed=2027, selection_rule="CONTROLLED_BUDGET"))


def check_row_labelling():
    """run()'s labelling: an s2027 C_FULL row is DESCRIPTIVE (relabel + source labels + seed labels), s2026 REEVAL."""
    base = [{"metric": "mrr", "k": 20, "value": 0.1}]
    src_reg = {"ext_self_sha256": "e" * 64}
    out = {}
    for seed in (2026, 2027):
        a = SimpleNamespace(role="MAIN", arm_label="C_FULL", selection_rule="PRACTICAL_BEST",
                            checkpoint_id="C_FULL:best", ckpt_kind="central")
        extra = dict(R.source_row_labels("C_FULL", src_reg), **R.source_seed_labels("C_FULL", seed))
        out[seed] = R.label_rows(base, a, checkpoint_sha256="c" * 64, source_run_id="C_FULL_02B_SASREC_s%d" % seed,
                                 efe=6.0, extra=extra)[0]
    assert out[2026]["comparison_status"] == "REEVAL" and "source_seed_status" not in out[2026]
    assert out[2027]["comparison_status"] == "DESCRIPTIVE" and out[2027]["source_seed_status"] == "DESCRIPTIVE"
    rec = R.add_source_seed_record({"source": "C_FULL"}, "C_FULL", 2027)
    assert rec["seed_status"] == "DESCRIPTIVE" and R.add_source_seed_record({"source": "C_FULL"}, "C_FULL", 2026) == \
        {"source": "C_FULL"}


def test_s2027_rows_are_labelled_descriptive():
    check_row_labelling()


def test_seed_labels_are_not_comparison_key_fields():
    from v4eval.rows import COMPARISON_KEY
    assert not set(R.source_seed_labels("C_FULL", 2027)) & set(COMPARISON_KEY)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("the pre-ANA-05 source table (C_FULL seed 2026 only) must not pass the s2027 dry run")
def test_nc_s2027_not_registered(tmp_path, monkeypatch):
    table = {k: dict(v) for k, v in R.CENTRAL_SOURCES.items()}
    table["C_FULL"]["descriptive_seeds"] = ()
    monkeypatch.setattr(R, "CENTRAL_SOURCES", table)
    check_s2027_dry_run(tmp_path)


@nc("an s2027 source whose rows are NOT labelled DESCRIPTIVE must not pass the labelling check")
def test_nc_s2027_unlabelled(monkeypatch):
    monkeypatch.setattr(R, "source_seed_labels", lambda name, seed: {})
    check_row_labelling()


@nc("a source classification that accepts any seed must not pass the unregistered-seed refusals")
def test_nc_any_seed_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "source_seeds", lambda name: (2026, 2027, 2028))
    check_seed_refusals(tmp_path)
