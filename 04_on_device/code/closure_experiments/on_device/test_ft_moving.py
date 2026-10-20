"""Meaningful admissions, source protection, causality and FT kernel controls."""
from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import ppsi_env  # noqa: F401, E402 — isolation before reused numerical imports
import numpy as np
import pytest
import torch

from experiment_extensions.on_device import ft_moving as F
from fl.numerics import enforce_strict_fp32, state_digest
import ppsi_paths

sys.path.insert(0, str(ppsi_paths.code_dir("tests_fl")))   # the FL test kit (03_federated_learning/tests/research)
from fl_testkit import one_client, tiny  # noqa: E402


@pytest.fixture(autouse=True)
def strict():
    torch.set_num_threads(1)
    enforce_strict_fp32()


@pytest.fixture
def plan():
    return F.load_plan(F.phase_root() / "planning/FT_MOVING_MODELS_DESIGN.json")


@pytest.fixture
def entry(plan):
    return copy.deepcopy(F.select_base(plan, "cold", 2026))


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=1, allow_nan=False))


@pytest.mark.parametrize("role", ["MAIN_TEST", "TRUE_COLD_TEST", "TEST2", "CALIBRATION", "UNSEEN_VAL"])
def test_no_other_role_before_checkpoint_access(plan, role):
    with pytest.raises(F.Refused, match="MAIN validation"):
        F.select_base(plan, "cold", 2026, role=role)


def test_cross_seed_or_unknown_line_refused(plan):
    for line, seed in [("cold", 2028), ("new", 2026)]:
        with pytest.raises(F.Refused):
            F.select_base(plan, line, seed)


def test_plan_tampering_refused(tmp_path, plan):
    plan["procedure"]["passes"] = 2
    p = tmp_path / "plan.json"
    dump(p, plan)
    with pytest.raises(F.Refused, match="self SHA"):
        F.load_plan(p)
    plan["self_sha256"] = F.canonical_sha({k: v for k, v in plan.items() if k != "self_sha256"})
    dump(p, plan)
    with pytest.raises(F.Refused, match="fixed FT_C"):
        F.load_plan(p)


def test_own_real_source_bindings_pass(plan):
    for b in plan["bases"]:
        spec, launch = F.check_base(b)
        assert spec["cfg"]["seed"] == b["seed"]
        assert launch["run_id"] == b["base_run_id"]


def test_wrong_checkpoint_sha_refused_before_payload(entry):
    entry["base"]["sha256"] = "0" * 64
    with pytest.raises(F.Refused, match="bound file changed"):
        F.check_base(entry)


def test_other_seed_checkpoint_is_not_own_base(entry, plan):
    entry["base"] = F.select_base(plan, "cold", 2027)["base"]
    with pytest.raises(F.Refused, match="outside its bound"):
        F.check_base(entry)


def test_source_seed_cannot_be_relabelled(entry):
    entry["seed"] = 2027
    with pytest.raises(F.Refused, match="completed MAIN base seed"):
        F.check_base(entry)


def test_source_run_cannot_be_relabelled(entry):
    entry["base_run_id"] = "FT_P2_WSTAR_SMALL_s2026"
    with pytest.raises(F.Refused, match="run/arm mismatch"):
        F.check_base(entry)


@pytest.fixture
def payload(entry, monkeypatch):
    spec, launch = F.check_base(entry)
    state = tiny(10, tied=False).broadcast_state(clone=True)
    obj = {"kind": "weights", "run_id": entry["base_run_id"], "config_digest": launch["fl_config_digest"],
           "efe": 6.0, "theta": state}
    entry["validation_baseline"]["model_state_digest"] = state_digest(state)
    monkeypatch.setattr(F, "check_base", lambda b: (spec, launch))
    import fl.checkpoint as C
    monkeypatch.setattr(C, "load_checkpoint", lambda p: obj)
    return obj


def test_payload_and_exact_baseline_state_pass(entry, payload):
    assert F.check_checkpoint(entry)[0] is payload


@pytest.mark.parametrize("field,value", [("kind", "resumable"), ("run_id", "wrong_seed"),
                                          ("config_digest", "wrong_config"), ("efe", float("nan")),
                                          ("efe", 0.0)])
def test_wrong_payload_refused(entry, payload, field, value):
    payload[field] = value
    with pytest.raises(F.Refused):
        F.check_checkpoint(entry)


def test_baseline_of_another_state_refused(entry, payload):
    entry["validation_baseline"]["model_state_digest"] = "0" * 64
    with pytest.raises(F.Refused, match="matched|bound unchanged baseline"):
        F.check_checkpoint(entry)


def test_old_output_and_symlink_refused(tmp_path):
    phase = tmp_path / "phase"
    phase.mkdir()
    with pytest.raises(F.Refused, match="strictly inside"):
        F.output_guard(tmp_path / "old", {"test": True}, root=phase)
    old = tmp_path / "old"
    old.mkdir()
    alias = phase / "alias"
    alias.symlink_to(old, target_is_directory=True)
    with pytest.raises(F.Refused):
        F.output_guard(alias / "run", {"test": True}, root=phase)
    assert not list(old.iterdir())


def test_exact_partial_resume_and_completed_refusal(tmp_path):
    phase = tmp_path / "phase"
    phase.mkdir()
    out, binding = phase / "run", {"base": "exact", "code": "exact"}
    F.output_guard(out, binding, root=phase)
    dump(out / "STATUS.json", {"status": "PARTIAL"})
    assert F.output_guard(out, binding, resume=True, root=phase) == out
    with pytest.raises(F.Refused):
        F.output_guard(out, dict(binding, code="changed"), resume=True, root=phase)
    with pytest.raises(F.Refused):
        F.output_guard(out, binding, root=phase)
    dump(out / "STATUS.json", {"status": "DONE"})
    with pytest.raises(F.Refused, match="completed validation"):
        F.output_guard(out, binding, resume=True, root=phase)


def test_resume_does_not_create_missing_run(tmp_path):
    with pytest.raises(F.Refused, match="existing own partial"):
        F.output_guard(tmp_path / "never_created", {}, resume=True, root=tmp_path)
    assert not (tmp_path / "never_created").exists()


def test_context_restores_even_on_exception(entry):
    from v4int import launch_device as D
    original = {k: getattr(D, k) for k in ("ARMS", "load_warm_base", "write_json", "device_code_sha256", "FT_PROCEDURE")}
    with pytest.raises(RuntimeError):
        with F.device_context(entry, {"id": "toy"}):
            assert D.FT_PROCEDURE[entry["new_arm"]] == "FT_C"
            with pytest.raises(F.Refused, match="nested"):
                with F.device_context(entry, {}):
                    pass
            raise RuntimeError("synthetic exception")
    assert all(getattr(D, k) is v for k, v in original.items())


def test_shim_toy_finetune_bit_identical_to_original_kernel(entry):
    from fl.finetune import FTRecipe, FTRunner
    from v4int import launch_device as D
    base = tiny(3, tied=False).broadcast_state(clone=True)
    client = one_client("owned-toy", 37, seed=11, invalid_frac=0.1)
    client.examples["target_ts"] = torch.arange(37)
    selection = D.load_selection(D.FT_PRE_SELECTION)
    with F.device_context(entry, {"id": "toy"}):
        spec = D.device_spec(entry["new_arm"], seed=2026, peak_lr=0.05, role="MAIN",
                             start_digest=state_digest(base), selection=selection, fused=False)
        _, runner, budgets = D.make_runners(spec, tiny(9, tied=False), base)[0]
        a = runner.run(client, lambda adapter, b: state_digest(adapter.broadcast_state()), support_cutoff=1000)
        assert budgets == {4: entry["new_arm"] + "_f0.3_P4"}
    original = FTRecipe(arm="FT_C", base_digest=state_digest(base), seed=2026, peak_lr=0.05,
                        lr_frac=0.3, passes=(4,), fused=False)
    b = FTRunner(tiny(9, tied=False), base, original).run(client,
        lambda adapter, n: state_digest(adapter.broadcast_state()), support_cutoff=1000)
    assert a.record() == b.record()
    assert a.bytes_up == 0


def test_future_support_refused_by_unchanged_kernel(entry):
    from fl.finetune import CausalityError
    from v4int import launch_device as D
    base = tiny(3, tied=False).broadcast_state(clone=True)
    client = one_client("future-label", 4, seed=11)
    client.examples["target_ts"] = torch.tensor([1, 2, 3, 1000])
    with F.device_context(entry, {"id": "toy"}):
        spec = D.device_spec(entry["new_arm"], seed=2026, peak_lr=0.05, role="MAIN",
                             start_digest=state_digest(base), selection=D.load_selection(D.FT_PRE_SELECTION), fused=False)
        _, runner, _ = D.make_runners(spec, tiny(9, tied=False), base)[0]
        with pytest.raises(CausalityError):
            runner.run(client, support_cutoff=1000)


def test_oov_only_user_stays_in_macro_denominator():
    from v4eval.bootstrap import point
    m = SimpleNamespace(n=4, catalogue_K=24, eval_mask=np.array([1, 1, 1, 1], bool),
                        rankable=np.array([1, 1, 1, 0], bool), eval_user_key=np.array([0, 1, 1, 2]))
    st = F.validation_stats(m, np.array([1, 20, 20, -1], np.int32))
    assert list(st.user_keys) == [0, 1, 2]
    assert list(st.NR) == [1, 2, 0]
    assert point(st, "macro_e2e") == pytest.approx((1 + 0.05 + 0) / 3)
    assert point(st, "macro_e2e") != point(st, "micro_e2e")
    with pytest.raises(F.Refused, match="OOV"):
        F.validation_stats(m, np.array([1, 20, 20, 1], np.int32))
    with pytest.raises(F.Refused, match="rankable"):
        F.validation_stats(m, np.array([-1, 20, 20, -1], np.int32))


def test_paired_bootstrap_rejects_missing_user():
    from v4eval import bootstrap as B
    m = SimpleNamespace(n=3, catalogue_K=24, eval_mask=np.ones(3, bool),
                        rankable=np.array([1, 1, 0], bool), eval_user_key=np.array([0, 1, 2]))
    st = F.validation_stats(m, np.array([1, 20, -1], np.int32))
    sample = B.make_plan([0, 1], n_resamples=10)
    with pytest.raises(B.PairingError):
        B.contrast(sample, st, st, "macro_e2e", "wrong denominator")


def test_exact_independent_review_binding(tmp_path):
    plan_path = tmp_path / "plan"
    plan_path.write_text("owned protocol")
    r = {"status": "PASS", "protocol_sha256": F.sha(plan_path), "ft_module_sha256": F.sha(F.__file__),
         "numerical_code_sha256": "exact"}
    p = tmp_path / "review.json"
    dump(p, r)
    assert F.review_gate(p, plan_path, {"sha256": "exact"})["sha256"] == F.sha(p)
    r["ft_module_sha256"] = "other-version"
    dump(p, r)
    with pytest.raises(F.Refused, match="exact current FT"):
        F.review_gate(p, plan_path, {"sha256": "exact"})


@pytest.mark.parametrize("device", [None, "0", "0,1"])
def test_wrong_or_late_cuda_export_refuses_before_source_access(monkeypatch, device, tmp_path):
    if device is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", device)
    monkeypatch.setattr(F, "load_plan", lambda p: (_ for _ in ()).throw(AssertionError("source read before env refusal")))
    assert F.main(["--line", "cold", "--seed", "2026", "--gpu", "1", "--out", str(tmp_path / "never")]) == 2
    assert not (tmp_path / "never").exists()
