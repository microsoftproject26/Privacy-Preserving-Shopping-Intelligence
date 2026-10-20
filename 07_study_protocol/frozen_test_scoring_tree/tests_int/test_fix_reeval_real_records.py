"""FIX-REEVAL - v4int.reeval's registered-source binding on the REAL C_FULL / PRE server records (CPU only).

The GATE-B B1 check compared RUN_START key_base.manifest_registry_self_sha256 with the extension's self hash; the real
records carry the per-run trainer VIEW there (the trainer's own registry), and the extension's self hash in config.key
(= LAUNCH_<k>.comparison_key), so the real C_FULL dry run was refused. The fixtures under fixtures/reeval_records are
byte copies of company/evidence/reeval_records (RUN_START / LAUNCH_0 / c_full / pretrain_small SHIM_0 of
C_FULL_02B_SASREC_s2026 and PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026).

  * the real records' three registry hashes are what the code says (config.key = LAUNCH comparison key = extension;
    key_base = RUN_START.manifest_registry = SHIM trainer_view = the view recomputed from the extension);
  * the dry run of C_FULL (MAIN PB / CB, panels) and PRE (MAIN PB / CB, UNSEEN_HOLDOUT / SPARSE_PANEL / HEAVY_PANEL) on
    the real RUN_START passes and records the binding;
  * a run whose comparison key names the base REGISTRY_V2 (or whose key base / view / config hash does not match the
    extension) refuses;
  * the eval-only TEST role reads the real train binding (RUN_START.manifest_registry.bindings.train) and fails closed
    without one.
Negative controls (test_nc_*, strict xfail on AssertionError).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from intkit import nc
from test_gate_b_reeval import REG, SOURCES, _ext_self, _sha, argv, real_start, source_run
from v4int import reeval as R

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reeval_records"
EVIDENCE = Path(__file__).resolve().parents[2] / "company" / "evidence" / "reeval_records"
TAG = {"C_FULL": "CFULL_s2026", "PRE": "PRE"}
BASE_SELF = json.loads((REG / "REGISTRY_V2.json").read_text(encoding="utf-8"))["self_sha256"]


def rec(name: str, kind: str) -> dict:
    return json.loads((FIXTURES / f"{TAG[name]}_{kind}.json").read_text(encoding="utf-8"))


def rehash(start: dict) -> dict:
    """RunConfig.sha256 (v4train.recipe) of the edited config, written independently of reeval.run_config_sha256."""
    start["config_sha256"] = hashlib.sha256(json.dumps(start["config"], sort_keys=True, allow_nan=False,
                                                       default=list).encode("utf-8")).hexdigest()
    return start


# ------------------------------------------------------------------------------------------------ the real shape
def test_fixtures_are_byte_copies_of_the_evidence():
    names = sorted(p.name for p in FIXTURES.glob("*.json"))
    assert len(names) == 6
    if EVIDENCE.is_dir():
        assert names == sorted(p.name for p in EVIDENCE.glob("*.json"))
        assert all(_sha(FIXTURES / n) == _sha(EVIDENCE / n) for n in names)


def test_real_records_carry_three_registry_hashes():
    """What each record means (launch_central + the shims + v4train.trainer._data_identity_fields)."""
    for name in ("C_FULL", "PRE"):
        st, la, sh = rec(name, "RUN_START"), rec(name, "LAUNCH_0"), rec(name, "SHIM_0")
        ext = _ext_self(name)
        assert st["config"]["key"]["manifest_registry_self_sha256"] == ext
        assert la["comparison_key"] == st["config"]["key"] and la["config_sha256"] == st["config_sha256"]
        assert la["data_protocol_lock_v2"]["manifest_registry"]["self_sha256"] == ext == sh["ext_registry"]["self_sha256"]
        view = sh["trainer_view"]["self_sha256"]
        assert view != ext and view != BASE_SELF
        assert st["key_base"]["manifest_registry_self_sha256"] == view == st["manifest_registry"]["self_sha256"]
        assert la["data_protocol_lock_v2"]["trainer_registry_view"]["self_sha256"] == view
        assert st["key_base"]["manifest_registry_sha256"] == st["manifest_registry"]["file_sha256"]
        assert "manifest_binding" not in st["train_provider"]                 # v4train RUN_START shape


# ------------------------------------------------------------------------------------------------ the dry run
CASES = (("C_FULL", "MAIN", "best.pt", "PRACTICAL_BEST"), ("C_FULL", "MAIN", "endpoint_6.0.pt", "CONTROLLED_BUDGET"),
         ("C_FULL", "UNSEEN_HOLDOUT", "best.pt", "PRACTICAL_BEST"), ("C_FULL", "SPARSE_PANEL", "best.pt",
                                                                     "PRACTICAL_BEST"),
         ("C_FULL", "HEAVY_PANEL", "endpoint_6.0.pt", "CONTROLLED_BUDGET"),
         ("PRE", "MAIN", "endpoint_6.0.pt", "PRACTICAL_BEST"), ("PRE", "MAIN", "endpoint_6.0.pt", "CONTROLLED_BUDGET"),
         ("PRE", "UNSEEN_HOLDOUT", "endpoint_6.0.pt", "CONTROLLED_BUDGET"),
         ("PRE", "SPARSE_PANEL", "endpoint_6.0.pt", "CONTROLLED_BUDGET"),
         ("PRE", "HEAVY_PANEL", "endpoint_6.0.pt", "CONTROLLED_BUDGET"))


def check_real_records_pass(tmp_path, run=R.run):
    for i, (name, role, ckpt, rule) in enumerate(CASES):
        t = tmp_path / f"ok{i}"
        cp = source_run(t, name, ckpt=ckpt)                          # the unmodified REAL RUN_START
        try:
            out = run(argv(cp, t, name, role=role, selection_rule=rule))
        except R.ReevalRefused as e:
            raise AssertionError(f"{name} {role} {rule} refused on the real record: {e}")
        b = out["central_source"]["binding"]
        st, la, sh = rec(name, "RUN_START"), rec(name, "LAUNCH_0"), rec(name, "SHIM_0")
        assert b["comparison_key_registry_self_sha256"] == la["comparison_key"]["manifest_registry_self_sha256"]
        assert b["trainer_view_self_sha256"] == st["key_base"]["manifest_registry_self_sha256"] == \
            sh["trainer_view"]["self_sha256"]
        assert b["trainer_view_file_sha256"] == st["key_base"]["manifest_registry_sha256"]
        assert b["config_sha256"] == st["config_sha256"] and out["role"] == role
        assert b["ext_file_sha256"] == sh["ext_registry"]["file_sha256"] and b["view_checks"]["all_ok"]


def test_real_c_full_and_pre_records_pass_the_dry_run(tmp_path):
    check_real_records_pass(tmp_path)


# ------------------------------------------------------------------------------------------------ refusals
def _base_view(tmp: Path) -> tuple:
    """The MAIN-pair trainer view (launch_central, built from REGISTRY_V2 itself): (self, file sha256)."""
    from v4int.launch_central import build_trainer_registry_view
    v = build_trainer_registry_view(REG / "REGISTRY_V2.json", tmp, registry_rel_path="contracts/manifest_registry/"
                                    "REGISTRY_V2.json", registry_file_sha256=_sha(REG / "REGISTRY_V2.json"))
    return v["self_sha256"], _sha(v["path"])


def refusal_cases(tmp: Path) -> list:
    out = []

    def edit(name, fn, msg):
        st = copy.deepcopy(real_start(name))
        fn(st)
        out.append((name, st, msg))

    for name in ("C_FULL", "PRE"):                                   # the comparison key names the base registry
        edit(name, lambda s: rehash(s) if s["config"]["key"].update(manifest_registry_self_sha256=BASE_SELF) is None
             else None, "config.key.manifest_registry_self_sha256")
    edit("C_FULL", lambda s: s["config"]["key"].update(manifest_registry_self_sha256=BASE_SELF),   # not re-hashed
         "config.key.manifest_registry_self_sha256")
    edit("C_FULL", lambda s: s["key_base"].update(manifest_registry_self_sha256=_ext_self("C_FULL")),   # GATE-B shape
         "key_base.manifest_registry_self_sha256")
    vself, vfile = _base_view(tmp / "base_view")

    def main_view(s):
        s["key_base"].update(manifest_registry_self_sha256=vself, manifest_registry_sha256=vfile)
        s["manifest_registry"].update(self_sha256=vself, file_sha256=vfile)
    edit("C_FULL", main_view, "trainer view self")
    edit("PRE", main_view, "trainer view self")
    edit("PRE", lambda s: rehash(s) if s["config"]["key"].update(population_hash="0" * 64) is None else None,
         "config.key fields unchanged in key_base")
    edit("C_FULL", lambda s: s["config"].update(peak_lr=1.0), "config_sha256 = RunConfig.sha256")
    edit("PRE", lambda s: rehash(s) if s["config"].pop("key") else None, "config.key.manifest_registry_self_sha256")
    edit("PRE", lambda s: s.pop("manifest_registry"), "RUN_START.manifest_registry")
    return out


def check_binding_refusals(tmp_path, run=R.run):
    for i, (name, st, msg) in enumerate(refusal_cases(tmp_path)):
        t = tmp_path / f"nc{i}"
        cp = source_run(t, name, start=st)
        refused = None
        try:
            run(argv(cp, t, name))
        except R.ReevalRefused as e:
            refused = str(e)
        assert refused is not None and msg in refused and "FIX-REEVAL" in refused, (i, name, msg, refused)


def test_runs_not_built_through_the_extension_refuse(tmp_path):
    check_binding_refusals(tmp_path)


def test_cross_source_records_refuse():
    for name, other in (("C_FULL", "PRE"), ("PRE", "C_FULL")):
        with pytest.raises(R.ReevalRefused, match="config.key.manifest_registry_self_sha256"):
            R.verify_source_registry(name, real_start(other), REG / "REGISTRY_V2.json")


def test_other_checks_still_refuse_on_the_real_record(tmp_path):
    """The checkpoint sha / meta config sha / file rule checks are unchanged on the real shape."""
    cp = source_run(tmp_path / "a", "C_FULL", meta_over={"config_sha256": "d" * 64})
    with pytest.raises(R.ReevalRefused, match="config_sha256"):
        R.run(argv(cp, tmp_path / "a", "C_FULL"))
    cp = source_run(tmp_path / "b", "C_FULL")
    with pytest.raises(R.ReevalRefused, match="--expect-sha256"):
        R.run(argv(cp, tmp_path / "b", "C_FULL", expect_sha256="0" * 64))
    cp = source_run(tmp_path / "c", "PRE")
    with pytest.raises(R.ReevalRefused, match="not registered for the PRE"):
        R.run(argv(cp, tmp_path / "c", "PRE", role="TRUE_COLD"))


# ------------------------------------------------------------------------------------------------ TEST route binding
def test_central_train_binding_reads_the_real_record():
    for name in ("C_FULL", "PRE"):
        st, la = rec(name, "RUN_START"), rec(name, "LAUNCH_0")
        b = R.central_train_binding(st)
        assert b is not None and b["manifest_id"] == st["train_provider"]["manifest_id"]
        assert b["content_sha256"] == la["train_provider"]["manifest_binding"]["content_sha256"]
    st = real_start("C_FULL")
    st["manifest_registry"]["bindings"]["train"]["manifest_id"] = "OTHER"
    assert R.central_train_binding(st) is None


def test_eval_only_test_role_fails_closed_without_a_train_binding(monkeypatch):
    import v4int.providers as P
    from v4int import retro_store as RS
    monkeypatch.setattr(P, "StoreEvalProvider", lambda *a, **k: SimpleNamespace(
        manifest_id="E", manifest_binding={"content_sha256": "EVAL"}))
    monkeypatch.setattr(RS, "train_key_ref", lambda mid, **kw: SimpleNamespace(
        manifest_id=mid, manifest_binding={"content_sha256": "TRAIN"}))
    monkeypatch.setattr(P, "comparison_key_base", lambda paths, tp, ep: {"k": 1})
    monkeypatch.setattr(RS, "retro_role_key", lambda kb, rk, gate, **kw: ({"kb": 1}, {"o": 1}))
    kw = dict(gate=None, pair={"eval": "E"}, reg_kw={"history_policy": "USER_HISTORY_L50", "manifest_registry": None},
              paths=None, base_kb={}, main_train_id="MAIN_TRAIN_5_100_131072")
    args = SimpleNamespace(role="TRUE_COLD_TEST")
    for missing in (None, {}, {"manifest_id": "MAIN_TRAIN_5_100_131072"}):
        with pytest.raises(R.ReevalRefused, match="none recorded"):
            R.evaluated_test_role(args, source_train_binding=missing, **kw)
    with pytest.raises(R.ReevalRefused, match="source run's train manifest"):
        R.evaluated_test_role(args, source_train_binding={"content_sha256": "OTHER"}, **kw)
    assert R.evaluated_test_role(args, source_train_binding={"content_sha256": "TRAIN"}, **kw)[1] == {"kb": 1}
    kw["pair"] = {"train": "T", "eval": "E"}                            # a train/eval role never needs it
    assert R.evaluated_test_role(args, source_train_binding=None, **kw)[3]["manifest_id"] == "T"


# ------------------------------------------------------------------------------------------------ negative controls
def _old_rule(name, start, ext, ext_path):
    if (start.get("key_base") or {}).get("manifest_registry_self_sha256") != ext.get("self_sha256"):
        raise R.ReevalRefused(f"the {name} run's key base was not built through {SOURCES[name]['ext']} "
                              "(manifest_registry_self_sha256)")
    return {}


@nc("the pre-FIX-REEVAL rule (key_base registry self = extension self) must refuse the REAL C_FULL / PRE records")
def test_nc_old_key_base_rule_refuses_the_real_records(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_source_binding", _old_rule)
    check_real_records_pass(tmp_path)


@nc("a reeval that does not check the source binding at all must not pass the not-through-the-extension refusals")
def test_nc_binding_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_source_binding", lambda name, start, ext, ext_path: {"view_checks": {"all_ok": True}})
    check_binding_refusals(tmp_path)


@nc("a binding check on config.key only (no key base / trainer view check) must not pass the refusals")
def test_nc_config_key_only(tmp_path, monkeypatch):
    def cfg_only(name, start, ext, ext_path):
        if (start.get("config") or {}).get("key", {}).get("manifest_registry_self_sha256") != ext["self_sha256"]:
            raise R.ReevalRefused("config.key.manifest_registry_self_sha256 FIX-REEVAL")
        return {}
    monkeypatch.setattr(R, "check_source_binding", cfg_only)
    check_binding_refusals(tmp_path)


@nc("the pre-FIX-REEVAL train binding (train_provider.manifest_binding only) finds none in the real RUN_START")
def test_nc_train_binding_from_the_provider_only(monkeypatch):
    monkeypatch.setattr(R, "central_train_binding",
                        lambda start: (start.get("train_provider") or {}).get("manifest_binding"))
    test_central_train_binding_reads_the_real_record()
