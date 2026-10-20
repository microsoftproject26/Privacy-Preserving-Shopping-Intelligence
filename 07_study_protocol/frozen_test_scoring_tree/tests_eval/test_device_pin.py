"""INT-06 pin integrity and fail-closed identity checks (CPU tests)."""
import json
import hashlib

import pytest

from evaltestkit import nc
from v4eval import device_pin as dp


def _pin(name="NVIDIA A40"):
    obj = {"format": dp.FORMAT, "device_class": "NVIDIA_A40-FP32-DET", "device_name": name,
           "driver": "test-driver", "torch": "test-torch", "cuda": "test-cuda",
           "receipts": {"t3_gpu_receipt_sha256": "a" * 64, "t3_verdict": "PASS",
                        "strict_determinism_receipt_sha256": "b" * 64},
           "authority": "PREG1_AMENDMENT_06D §1", "registered_utc": "2026-09-28T00:00:00Z"}
    obj["self_sha256"] = dp._digest(obj)
    return obj


def test_absent_pin_accepts_frozen_only(tmp_path):
    assert dp.active_pin(tmp_path / "missing.json") is None
    assert dp.accepted_classes("A16-FP32-DET", None) == {"A16-FP32-DET"}


def test_pin_roundtrip_and_tamper(tmp_path):
    p = tmp_path / "pin.json"
    obj = _pin()
    p.write_text(json.dumps(obj), encoding="utf-8")
    assert dp.load_pin(p) == obj
    obj["device_name"] = "another device"
    p.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(dp.DevicePinError, match="self hash"):
        dp.load_pin(p)


def test_live_device_name_must_match(tmp_path, monkeypatch):
    p = tmp_path / "pin.json"
    p.write_text(json.dumps(_pin("Different NVIDIA A40")), encoding="utf-8")
    monkeypatch.setattr(dp.torch.cuda, "get_device_name", lambda _dev: "NVIDIA A40")
    with pytest.raises(dp.DevicePinError, match="name differs"):
        dp.pin_for_device("cuda:0", p)


def test_pinned_class_is_extra_accepted_class():
    obj = _pin()
    assert dp.accepted_classes("A16-FP32-DET", obj) == {"A16-FP32-DET", "NVIDIA_A40-FP32-DET"}


def test_pin_uses_registered_compact_json_rule():
    obj = _pin()
    body = {k: v for k, v in obj.items() if k != "self_sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
    assert obj["self_sha256"] == expected


def test_pin_refuses_incomplete_strict_receipt(tmp_path):
    t3 = tmp_path / "t3.json"
    strict = tmp_path / "strict.json"
    t3.write_text(json.dumps({"status": "COMPLETE", "all_checks_pass": True,
                              "families": {name: {"checks_pass": True, "a16_reference": {"diff_count": 1}}
                                           for name in ("GRU", "SASREC")}}), encoding="utf-8")
    strict.write_text(json.dumps({"all_checks_pass": True, "table": [
        {"variant": "SASREC_D256_B3", "n_reps": 2, "bitwise_rep_to_rep": True}]}), encoding="utf-8")
    with pytest.raises(dp.DevicePinError, match="both registered variants"):
        dp.write_pin(device_index=0, t3_receipt=t3, strict_receipt=strict, out=tmp_path / "pin.json")


def test_write_once_after_both_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(dp.torch.cuda, "get_device_name", lambda _dev: "NVIDIA A40")
    monkeypatch.setattr(dp.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dp.torch.version, "cuda", "13.0")
    t3 = tmp_path / "t3.json"
    strict = tmp_path / "strict.json"
    t3.write_text(json.dumps({"status": "COMPLETE", "all_checks_pass": True,
                              "device": {"name": "NVIDIA A40", "eval_device_class": "NVIDIA_A40-FP32-DET"},
                              "families": {name: {"checks_pass": True, "a16_reference": {"diff_count": 1}}
                                           for name in ("GRU", "SASREC")}}), encoding="utf-8")
    strict.write_text(json.dumps({"all_checks_pass": True, "table": [
        {"variant": name, "n_reps": 2, "bitwise_rep_to_rep": True}
        for name in ("SASREC_D64_B2", "SASREC_D256_B3")]}), encoding="utf-8")
    p = tmp_path / "pin.json"
    written = dp.write_pin(device_index=0, t3_receipt=t3, strict_receipt=strict,
                           out=p, driver="test-driver")
    assert dp.load_pin(p) == written
    with pytest.raises(FileExistsError):
        dp.write_pin(device_index=0, t3_receipt=t3, strict_receipt=strict,
                     out=p, driver="test-driver")


# ------------------------------------------------------------------------------------------------ INT-06 review additions
# The evaluator identity of the A40 block: every row of the pinned retention block (the 8 central re-evaluations, the FL
# runs, every later arm) must carry ONE evaluator_code_sha256 (ANA-01 KEY_FIELDS; 06E §4). code_ds3r2 deployed v4eval
# with this identity; any byte change of v4eval/*.py splits the block (an Ops decision + re-evaluation, never silent).
A40_BLOCK_EVALUATOR_SHA256 = "61b64e61c1a2e4f5a6fecd42f0ad3b98de26562d13a07758573cc9496b8d962e"


def test_evaluator_identity_is_the_deployed_a40_block_identity():
    from v4eval import evaluator_code_sha256
    assert evaluator_code_sha256() == A40_BLOCK_EVALUATOR_SHA256, (
        "v4eval/*.py changed: the pinned A40 retention block (code_ds3r2) would split on evaluator_code_sha256")


def _pinned_predict(monkeypatch, pin):
    from test_frozen_config_t3 import _batches, _tiny_case
    from v4eval import frozen_config as fc
    from v4eval import predict as P
    from v4eval.evaluate import evaluate
    if pin is not None:
        monkeypatch.setattr(P, "pin_for_device", lambda dev: pin)
    cfg = fc.load()
    a, ex, m = _tiny_case(n=80)
    art = P.predict_adapter(a, _batches(ex, cfg.eval_batch), m, config=cfg, model_sha256="t")
    return art, evaluate(m, art, config=cfg)


def _rows(res):
    from v4eval.rows import COMPARISON_KEY
    base = {f: f"x-{f}" for f in COMPARISON_KEY if f not in ("metric", "k", "aggregation", "target_denominator",
                                                               "eval_score_block", "eval_batch", "eval_device_class")}
    from v4eval.rows import rows_from_result
    return rows_from_result(res, base, method="C", seed=2026, selection_rule="PRACTICAL_BEST", efe=3.0)


def test_pinned_device_rows_are_measured_pinned_with_the_pin_sha(monkeypatch):
    pin = _pin()
    art, res = _pinned_predict(monkeypatch, pin)
    assert art.header["eval_config"]["device_pin_sha256"] == pin["self_sha256"]
    rows = _rows(res)
    assert rows and all(r["status"] == "MEASURED_PINNED_DEVICE" and r["device_pin_sha256"] == pin["self_sha256"]
                        for r in rows)


def test_unpinned_non_frozen_device_rows_stay_rehearsal(monkeypatch):
    art, res = _pinned_predict(monkeypatch, None)
    assert "device_pin_sha256" not in art.header["eval_config"]
    assert all(r["status"] == "REHEARSAL_NON_FROZEN_DEVICE" and "device_pin_sha256" not in r for r in _rows(res))


def test_default_pin_path_is_absent_in_the_repo():
    assert dp.active_pin() is None, "a pin file in the repo would change every local digest (tests create pins in tmp)"


@nc("the DL 166 digest rule (ensure_ascii=False) reproduces the registered self_sha256 of a pin with a '§'")
def test_nc_non_ascii_digest_rule():
    obj = _pin()
    body = {k: v for k, v in obj.items() if k != "self_sha256"}
    bad = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                         .encode("utf-8")).hexdigest()
    assert bad == obj["self_sha256"]


@nc("without a pin a non-frozen (A40) class is an accepted evaluation class")
def test_nc_absent_pin_accepts_a40():
    assert "NVIDIA_A40-FP32-DET" in dp.accepted_classes("A16-FP32-DET", None)


@nc("a tampered pin (device_name edited, self hash kept) loads")
def test_nc_tampered_pin_loads(tmp_path):
    obj = _pin()
    obj["device_name"] = "NVIDIA A40 (edited)"
    p = tmp_path / "pin.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    try:
        dp.load_pin(p)
        loaded = True
    except dp.DevicePinError:
        loaded = False
    assert loaded


@nc("rows of an unpinned non-frozen device are labelled MEASURED_PINNED_DEVICE")
def test_nc_unpinned_rows_pinned(monkeypatch):
    _art, res = _pinned_predict(monkeypatch, None)
    assert all(r["status"] == "MEASURED_PINNED_DEVICE" for r in _rows(res))
