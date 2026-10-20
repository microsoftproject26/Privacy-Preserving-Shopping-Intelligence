"""FL-16b (REV-FL-16 m2 / m5) — hardening of the S-record loaders in launch_fl; misuse refusals only (CPU, the tiny
synthetic worlds of test_fl05 / test_fl12 / test_fl16; no real data, no GPU).

m2: the smoke branch of the warm S record in `_finish_smoke` — SMOKE_S_CAL_WARM[_FROZEN] writes S_RECORD_WARM[_FROZEN]
    .json labelled SMOKE_NOT_EVIDENCE (never S_RECORD.json); it binds the variant's FA_1024_WARM / DP_WARM with smoke=True
    and is refused with smoke=False and by the other variant; the cold smoke S_CAL still writes S_RECORD.json only.
m5: both loaders (load_s_record, load_warm_s_record) check the S content against the registered S rule
    (fl.dp.median_clip_norm): S finite > 0, n_norms int > 0, a median rule, rounds_0based = 0..19 (production) or
    0..k-1, 1 <= k <= 50 (smoke), label S_CALIBRATION / SMOKE_NOT_EVIDENCE; a non-object or non-canonicalisable (NaN)
    record is refused (LaunchRefused), not a crash. Every record an S calibration writes (and the registered
    contracts/dp/S_RECORD.json) is still admitted, and every pinned digest table is unchanged.
NCs (strict xfail on AssertionError): each new guard removed (or one of its clauses neutralised) admits the bad record.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import v4int.launch_fl as F
from fl.dp import median_clip_norm
from intkit import nc
from test_fl05_launch import run_tiny
from test_fl12_launch import pre, untied_world, warm_world  # noqa: F401
from test_fl16 import (FL16_NEW_ARM_TABLE_SHA, PEAK, PRE_FL16_TABLE_SHA, _cold_record, _json, _scal_for, _tsha,
                       _warm_record, new_arm_table, pre_table, wtiny)

T = F.tuning_config
REGISTERED_S_RECORD = F.V4 / "contracts" / "dp" / "S_RECORD.json"
RULE = median_clip_norm([[0, [1.0]]], rounds=[0])["rule"]          # the registered fl.dp rule string


@pytest.fixture(scope="module", autouse=True)
def _strict_numerics_module():
    from fl.numerics import enforce_strict_fp32
    torch.set_num_threads(1)
    enforce_strict_fp32()
    yield


def refused(fn, *a, **kw) -> None:
    """The call must raise LaunchRefused; any other exception (a crash) or none is an AssertionError."""
    try:
        fn(*a, **kw)
    except F.LaunchRefused:
        return
    except Exception as e:                                   # noqa: BLE001
        raise AssertionError(f"{fn.__name__} crashed with {type(e).__name__} instead of refusing") from e
    raise AssertionError(f"{fn.__name__} admitted the record")


def _resign(obj: dict, path: Path, **changes) -> Path:
    o = dict(obj, **changes)
    o["self_sha256"] = F._canon(o)
    path.write_text(json.dumps(o), encoding="utf-8")
    return path


def smoke_spec(arm, ident):
    """wtiny's spec with the registered T = 384, so that a 3-round smoke stays below the first evaluation unit."""
    s = wtiny(arm, ident)
    s["cfg"]["endpoint_rounds"] = 384
    return s


# ================================================================================================ invariance
def test_pinned_digest_tables_unchanged():
    p = pre_table()
    assert len(p) == 145 and _tsha(p) == PRE_FL16_TABLE_SHA
    t = new_arm_table()
    assert len(t) == 440 and _tsha(t) == FL16_NEW_ARM_TABLE_SHA


def test_registered_rule_and_rounds_are_what_the_guard_checks():
    assert RULE.startswith("median") and F.KNOBS["s_cal_rounds"] == 20 and F.MAX_SMOKE_ROUNDS == 50
    assert median_clip_norm([[r, [1.0]] for r in range(20)])["rounds_0based"] == list(range(20))


@pytest.mark.skipif(not REGISTERED_S_RECORD.is_file(), reason="contracts/dp/S_RECORD.json not in this tree")
def test_registered_cold_s_record_still_admitted():
    for smoke in (False, True):
        rec = F.load_s_record(REGISTERED_S_RECORD, smoke=smoke)
        assert rec["format"] == F.S_FORMAT                    # (no S value is read or printed)


# ================================================================================================ m2: the smoke record
@pytest.fixture(scope="module")
def smoke_chain(pre, tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    out = {}
    for arm in ("S_CAL_WARM", "S_CAL_WARM_FROZEN", "S_CAL"):
        spec = smoke_spec(arm, pre["ident"])
        name = f"SMOKE_{arm}"
        o = run_tiny(root, spec, name, world=warm_world(pre) if arm != "S_CAL" else untied_world(), smoke_rounds=3)
        out[arm] = {"spec": spec, "out": o, "dir": root / name}
    return out


def test_m2_smoke_s_cal_warm_writes_the_warm_smoke_record(smoke_chain, pre):
    ident = pre["ident"]
    for sarm, fname, binds, other in (("S_CAL_WARM", "S_RECORD_WARM.json", ("FA_1024_WARM", "DP_WARM"),
                                       "DP_WARM_FROZEN"),
                                      ("S_CAL_WARM_FROZEN", "S_RECORD_WARM_FROZEN.json",
                                       ("FA_1024_WARM_FROZEN", "DP_WARM_FROZEN"), "FA_1024_WARM")):
        c = smoke_chain[sarm]
        assert c["out"]["status"] == "SMOKE_DONE_NOT_EVIDENCE"
        assert not (c["dir"] / "S_RECORD.json").exists() and (c["dir"] / fname).is_file()
        assert sorted(p.name for p in c["dir"].glob("S_RECORD*.json")) == [fname]
        rec = _json(c["dir"] / fname)
        assert rec["format"] == F.S_FORMAT_WARM and rec["label"] == "SMOKE_NOT_EVIDENCE" and rec["arm"] == sarm
        assert rec["self_sha256"] == F._canon(rec) and rec["rounds_0based"] == [0, 1, 2]
        assert rec["init_from"] == ident and rec["init_sha256"] == ident["state_digest"]
        assert rec["frozen_item_tables"] == (list(F.FROZEN_ITEM_KEYS) if sarm.endswith("FROZEN") else None)
        assert rec["arm_spec_sha256"] == F.spec_digest(c["spec"])
        assert not list((c["dir"] / "ckpt").rglob("*.pt"))
        for arm in binds:                                     # binds the variant's arms on a smoke run only
            got = F.load_warm_s_record(c["dir"] / fname, arm=arm, scal_spec=c["spec"], init_identity=ident,
                                       smoke=True)
            assert got["self_sha256"] == rec["self_sha256"]
            refused(F.load_warm_s_record, c["dir"] / fname, arm=arm, scal_spec=c["spec"], init_identity=ident,
                    smoke=False)
        refused(F.load_warm_s_record, c["dir"] / fname, arm=other, scal_spec=c["spec"], init_identity=ident,
                smoke=True)                                   # the other variant's arm
        refused(F.load_s_record, c["dir"] / fname, smoke=True)    # the cold loader refuses a warm record
    cold = smoke_chain["S_CAL"]
    assert cold["out"]["status"] == "SMOKE_DONE_NOT_EVIDENCE"
    assert sorted(p.name for p in cold["dir"].glob("S_RECORD*.json")) == ["S_RECORD.json"]
    crec = F.load_s_record(cold["dir"] / "S_RECORD.json", smoke=True)
    assert crec["label"] == "SMOKE_NOT_EVIDENCE" and crec["rounds_0based"] == [0, 1, 2]
    refused(F.load_s_record, cold["dir"] / "S_RECORD.json", smoke=False)


# ================================================================================================ m5: the S content
def _bad_contents(prod: bool) -> dict:
    lab = {} if prod else {"label": "SMOKE_NOT_EVIDENCE"}
    return {"S_zero": dict(lab, S=0.0), "S_negative": dict(lab, S=-0.3), "S_bool": dict(lab, S=True),
            "S_string": dict(lab, S="0.37"), "S_missing": dict(lab, S=None), "S_inf_overflow": dict(lab, S=1e308 * 10),
            "n_zero": dict(lab, n_norms=0), "n_float": dict(lab, n_norms=20.5), "n_bool": dict(lab, n_norms=True),
            "rule_mean": dict(lab, rule="mean client-delta L2 norm"), "rule_missing": dict(lab, rule=None),
            "rounds_short": dict(lab, rounds_0based=list(range(19)) if prod else []),
            "rounds_shifted": dict(lab, rounds_0based=list(range(1, 21)) if prod else [1, 2, 3]),
            "rounds_float": dict(lab, rounds_0based=[float(r) for r in range(20 if prod else 3)]),
            "rounds_too_many": dict(lab, rounds_0based=list(range(21 if prod else 51))),
            "rounds_missing": dict(lab, rounds_0based=None)}


def test_m5_cold_loader_checks_the_s_content(tmp_path):
    good = _json(_cold_record(tmp_path / "good.json"))
    assert F.load_s_record(tmp_path / "good.json", smoke=False)["S"] == 0.5
    smoke = _resign(good, tmp_path / "sm.json", label="SMOKE_NOT_EVIDENCE", rounds_0based=[0, 1, 2])
    assert F.load_s_record(smoke, smoke=True)["rounds_0based"] == [0, 1, 2]
    _resign(good, tmp_path / "r50.json", label="SMOKE_NOT_EVIDENCE", rounds_0based=list(range(50)))
    F.load_s_record(tmp_path / "r50.json", smoke=True)                   # the smoke bound itself is admitted
    for prod in (True, False):
        for name, ch in _bad_contents(prod).items():
            p = _resign(good, tmp_path / f"{name}_{prod}.json", **ch) if not (
                isinstance(ch.get("S"), float) and not math.isfinite(ch["S"])) else None
            if p is None:                                       # inf cannot be canonicalised: written unsigned
                p = tmp_path / f"{name}_{prod}.json"
                p.write_text(json.dumps(dict(good, **ch)), encoding="utf-8")
            refused(F.load_s_record, p, smoke=not prod)
    _resign(good, tmp_path / "lab.json", label="OTHER")
    refused(F.load_s_record, tmp_path / "lab.json", smoke=True)            # an unknown label (smoke run)


def test_m5_unverifiable_records_are_refused_not_crashes(tmp_path, pre):
    good = _json(_cold_record(tmp_path / "good.json"))
    (tmp_path / "nan.json").write_text(json.dumps(dict(good, S=float("nan"))), encoding="utf-8")
    (tmp_path / "list.json").write_text(json.dumps([good]), encoding="utf-8")
    (tmp_path / "str.json").write_text(json.dumps("S_RECORD"), encoding="utf-8")
    scal = _scal_for("DP_WARM", T(), pre["ident"])
    for name in ("nan", "list", "str"):
        refused(F.load_s_record, tmp_path / f"{name}.json", smoke=True)
        refused(F.load_warm_s_record, tmp_path / f"{name}.json", arm="DP_WARM", scal_spec=scal,
                init_identity=pre["ident"], smoke=True)
    wgood = _json(_warm_record(tmp_path / "w.json", scal, pre["ident"]))
    (tmp_path / "wnan.json").write_text(json.dumps(dict(wgood, S=float("nan"))), encoding="utf-8")
    refused(F.load_warm_s_record, tmp_path / "wnan.json", arm="DP_WARM", scal_spec=scal, init_identity=pre["ident"],
            smoke=True)


def test_m5_warm_loader_checks_the_s_content(tmp_path, pre):
    ident = pre["ident"]
    for arm in ("FA_1024_WARM", "DP_WARM", "FA_1024_WARM_FROZEN", "DP_WARM_FROZEN"):
        scal = _scal_for(arm, T(), ident)
        good_p = _warm_record(tmp_path / arm / "good.json", scal, ident)
        good = _json(good_p)
        assert F.load_warm_s_record(good_p, arm=arm, scal_spec=scal, init_identity=ident, smoke=False)["S"] == 0.37
        sm = _warm_record(tmp_path / arm / "sm.json", scal, ident, smoke=True)
        smo = _json(sm)
        _resign(smo, sm, rounds_0based=[0, 1, 2])
        F.load_warm_s_record(sm, arm=arm, scal_spec=scal, init_identity=ident, smoke=True)
        for prod in (True, False):
            base = good if prod else _json(sm)
            for name, ch in _bad_contents(prod).items():
                p = tmp_path / arm / f"{name}_{prod}.json"
                if isinstance(ch.get("S"), float) and not math.isfinite(ch["S"]):
                    p.write_text(json.dumps(dict(base, **ch)), encoding="utf-8")
                else:
                    _resign(base, p, **ch)
                refused(F.load_warm_s_record, p, arm=arm, scal_spec=scal, init_identity=ident, smoke=not prod)


def test_m5_s_calibration_records_stay_admitted(pre, tmp_path):
    """Every record an S calibration writes (s_record_obj / s_record_obj_warm from fl.dp.median_clip_norm, production
    and smoke) passes the content check."""
    ident = pre["ident"]
    log = [[r, [0.1 + 0.01 * r, 0.2]] for r in range(20)]
    for smoke, rounds in ((False, range(0, 20)), (True, range(3)), (True, range(50))):
        norm = median_clip_norm(log if not smoke else [[r, [0.3]] for r in rounds], rounds=rounds)
        cold = F.s_record_obj(norm, run_id="x", seed=2026, config_digest="c" * 64, plan_digest="p" * 64,
                              init_sha256="i" * 64, smoke=smoke)
        F.check_s_content(cold, "cold")
        (tmp_path / "c.json").write_text(json.dumps(cold), encoding="utf-8")
        F.load_s_record(tmp_path / "c.json", smoke=smoke)
        scal = _scal_for("DP_WARM_FROZEN", T(), ident)
        warm = F.s_record_obj_warm(norm, spec=scal, run_id="x", seed=2026,
                                   config_digest=F.fl_run_config(scal, "x", 64).digest(), plan_digest="p" * 64,
                                   init_sha256=ident["state_digest"], smoke=smoke)
        F.check_s_content(warm, "warm")
        (tmp_path / "w.json").write_text(json.dumps(warm), encoding="utf-8")
        F.load_warm_s_record(tmp_path / "w.json", arm="DP_WARM_FROZEN", scal_spec=scal, init_identity=ident,
                             smoke=smoke)


# ================================================================================================ negative controls
@nc("m2: a warm smoke that writes its record under the cold name S_RECORD.json (FL-16 smoke branch mutated)")
def test_nc_warm_smoke_writes_cold_name(pre, tmp_path, monkeypatch):
    monkeypatch.setitem(F.S_RECORD_WARM_FILE, "FA_WARM", "S_RECORD.json")          # MUTANT
    run_tiny(tmp_path, smoke_spec("S_CAL_WARM", pre["ident"]), "SMOKE_W", world=warm_world(pre), smoke_rounds=3)
    assert not (tmp_path / "SMOKE_W" / "S_RECORD.json").exists()


@nc("m5: without the content check a re-signed warm record with S = 0 binds DP_WARM")
def test_nc_warm_s_zero_binds(pre, tmp_path, monkeypatch):
    scal = _scal_for("DP_WARM", T(), pre["ident"])
    p = _warm_record(tmp_path / "z.json", scal, pre["ident"], S=0.0)
    monkeypatch.setattr(F, "check_s_content", lambda obj, path: None)                # MUTANT
    refused(F.load_warm_s_record, p, arm="DP_WARM", scal_spec=scal, init_identity=pre["ident"], smoke=False)


@nc("m5: without the content check a re-signed cold record with n_norms = 0 binds FA_1024 / DP")
def test_nc_cold_n_zero_binds(tmp_path, monkeypatch):
    good = _json(_cold_record(tmp_path / "good.json"))
    p = _resign(good, tmp_path / "n0.json", n_norms=0)
    monkeypatch.setattr(F, "check_s_content", lambda obj, path: None)                # MUTANT
    refused(F.load_s_record, p, smoke=False)


@nc("m5: the pre-FL-16b verification (obj.get / _canon unguarded) crashes on a NaN record instead of refusing it")
def test_nc_nan_record_crashes(tmp_path, monkeypatch):
    good = _json(_cold_record(tmp_path / "good.json"))
    (tmp_path / "nan.json").write_text(json.dumps(dict(good, S=float("nan"))), encoding="utf-8")
    monkeypatch.setattr(F, "_s_verified", lambda obj, fmt: obj.get("format") == fmt and   # MUTANT
                        obj.get("self_sha256") == F._canon(obj))
    refused(F.load_s_record, tmp_path / "nan.json", smoke=True)


FIX = {"S": 0.37, "n_norms": 20, "rule": RULE, "rounds_0based": list(range(20)), "label": "S_CALIBRATION"}


@pytest.mark.parametrize("field,bad", [("S", -0.3), ("n_norms", 0), ("rule", "mean"),
                                       ("rounds_0based", list(range(1, 21))), ("label", "OTHER")])
@nc("m5: each clause of check_s_content is load-bearing (the clause's field neutralised to a good value admits it)")
def test_nc_each_content_clause(tmp_path, monkeypatch, field, bad):
    good = _json(_cold_record(tmp_path / "good.json"))
    p = _resign(good, tmp_path / "b.json", **{field: bad})
    real = F.check_s_content
    if field == "label":                                      # (label OTHER: the clause neutralised = S_CALIBRATION)
        monkeypatch.setattr(F, "check_s_content", lambda obj, path: real(dict(obj, label="S_CALIBRATION"), path))
    else:
        monkeypatch.setattr(F, "check_s_content", lambda obj, path: real(dict(obj, **{field: FIX[field]}), path))
    refused(F.load_s_record, p, smoke=True)
