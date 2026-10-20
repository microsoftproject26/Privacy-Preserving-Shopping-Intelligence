"""INT-06 scope from REV-FL-01 + amendment 06E (CPU, tiny synthetic model / clients; the INT-03 / INT-04 fixtures).

  * B1 (PF resume): the resumed leg evaluates with the run's LIVE store (run.store) — per-mark store digests, the
    evaluator's values and PF_STORE.pt equal the straight run's (a real store-dependent evaluate_pf, not file existence);
  * B2 (PF best mark): PF_STORE_best.pt holds the store of the PRACTICAL_BEST mark (half_efe, theta digest = best.pt's),
    identical straight vs resumed; RESULT_ROWS PRACTICAL_BEST rows reference it (file, sha256, store digest in the
    checkpoint id), CONTROLLED_BUDGET rows reference PF_STORE.pt;
  * M2: an FL resume leg on another device class / pin, another FL code identity, or from a pre-INT-06 ARM_SPEC is
    refused; FA stays byte-identical (no PF files, no pf fields);
  * M1: device rows on the pinned class carry the pin (MEASURED_PINNED_DEVICE; result eval_config + RUN_SUMMARY);
  * M3: every device shard records class / pin / code; resuming or merging shards of another class / pin / code refuses;
  * 06E §1: `--arm LO --role UNSEEN_VAL --select` selects the LO cell (highest E2E macro MRR@20, ties -> fewer passes)
    into SELECTION.json (O_EXCL, selection_role UNSEEN_VAL); LO rows with that selection mark the two other cells
    DESCRIPTIVE (LO_CELL_NOT_SELECTED).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import v4int.launch_device as D
import v4int.launch_fl as F
from intkit import assert_raises, nc
from test_int03_launch_fl import _tiny_spec, _world
from test_int04_device import K, PEAK, _frozen, _plans
from test_int04_device import _world as _dworld

from fl.checkpoint import load_checkpoint, sha256_file
from fl.numerics import state_digest
from fl.personal import PersonalStore
from fl.synthetic import make_tiny_adapter


# ------------------------------------------------------------------------------------------------ B1 / B2 (PF)
def _pf_leg(tmp_path, name, log, *, stale=False, **kw):
    w = _world()

    def evaluate_pf(mark, store):
        st = PersonalStore(store.query_dim) if stale else store            # stale = the pre-fix symptom
        log[round(mark, 6)] = (st.digest(), len(st))
        v = min(mark, 3.0) + 1e-9 * sum(float(st.p(k).abs().sum()) for k in st.keys())
        rows = [{"metric": "mrr", "k": 20, "value": v, "split": "VALIDATION", "method": "C",
                 "training_paradigm": "CENTRAL", "row_status": "CENTRAL_CONTEST"}]
        return v, rows, np.arange(5), {"E2E": {"n": 5}}
    return F.execute(spec=_tiny_spec("PF"), run_id=name, run_dir=tmp_path / name, say=lambda *_: None,
                     evaluate_pf=evaluate_pf, **w, **kw)


def _straight_and_resumed(tmp_path, stale=False):
    ls, lp = {}, {}
    s = _pf_leg(tmp_path, "s", ls)
    part = _pf_leg(tmp_path, "p", lp, interrupt_at=3)
    assert part["status"] == "INTERRUPTED"
    r = _pf_leg(tmp_path, "p", lp, resume=True, stale=stale)
    return s, r, ls, lp


def test_pf_resume_uses_the_live_store_and_persists_the_best_mark_store(tmp_path):
    s, r, ls, lp = _straight_and_resumed(tmp_path)
    assert s["status"] == r["status"] == "DONE_MAIN"
    assert ls == lp and max(n for _, n in ls.values()) > 0                   # B1: per-mark store digests identical
    best = s["summary"]["pf_store_best"]
    assert best["half_efe"] < 12, "the fixture must have a best mark before 6.0 (else B2 is not exercised)"
    for leg in ("s", "p"):
        run = tmp_path / leg
        end = load_checkpoint(run / "PF_STORE.pt")
        assert PersonalStore.from_state_dict(end["store"]).digest() == ls[6.0][0] == end["store_digest"]
        assert end["clients"] == ls[6.0][1] > 0 and end["half_efe"] == 12
        b = load_checkpoint(run / "PF_STORE_best.pt")
        assert b["half_efe"] == best["half_efe"] and b["store_digest"] == ls[b["mark"]][0]
        assert PersonalStore.from_state_dict(b["store"]).digest() == b["store_digest"]
        theta_best = load_checkpoint(run / "ckpt" / leg / "best.pt")["theta"]
        assert b["theta_digest"] == state_digest(theta_best)
        rows = json.loads((run / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
        pb = [x for x in rows if x["selection_rule"] == "PRACTICAL_BEST"]
        cb = [x for x in rows if x["selection_rule"] == "CONTROLLED_BUDGET"]
        assert pb and all(x["pf_store_file"] == "PF_STORE_best.pt" and x["pf_store_sha256"] ==
                          sha256_file(run / "PF_STORE_best.pt") and x["checkpoint_id"].endswith(
                              f"|pf_store={b['store_digest']}") for x in pb)
        assert cb and all(x["pf_store_file"] == "PF_STORE.pt" and x["pf_store_digest"] == end["store_digest"]
                          for x in cb)
    assert load_checkpoint(tmp_path / "s" / "PF_STORE_best.pt")["store_digest"] == \
        load_checkpoint(tmp_path / "p" / "PF_STORE_best.pt")["store_digest"]


def test_fa_has_no_pf_files_and_unchanged_rows(tmp_path):
    w = _world()
    out = F.execute(spec=_tiny_spec("FA"), run_id="fa", run_dir=tmp_path / "fa", say=lambda *_: None, **w)
    assert out["status"] == "DONE_MAIN" and "pf_store" not in out["summary"]
    assert not (tmp_path / "fa" / "PF_STORE.pt").exists() and not (tmp_path / "fa" / "PF_STORE_best.pt").exists()
    rows = json.loads((tmp_path / "fa" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert rows and not any(k.startswith("pf_store") for r in rows for k in r)
    assert not any("|pf_store=" in r["checkpoint_id"] for r in rows)


def test_pf_best_record_refuses_a_stale_best_store(tmp_path):
    ls = {}
    _pf_leg(tmp_path, "x", ls)
    run = tmp_path / "x"
    fake = type("Run", (), {})()
    fake.state = type("S", (), {"evals": [[12, 6, 3.0, False, "MAIN"]]})()
    with pytest.raises(RuntimeError, match="cannot be rebuilt"):
        F.pf_best_record(run, fake, 4, run_id="x", config_digest="OTHER", theta_state={})


@nc("the pre-fix launcher (stale store after resume) reproduces the straight run's per-mark PF store digests")
def test_nc_stale_store_after_resume(tmp_path):
    _s, _r, ls, lp = _straight_and_resumed(tmp_path, stale=True)
    assert ls == lp


# ------------------------------------------------------------------------------------------------ M2 (FL resume)
def test_fl_resume_refuses_another_device_code_or_legacy_spec(tmp_path, monkeypatch):
    w = _world()
    F.execute(spec=_tiny_spec("FA"), run_id="m", run_dir=tmp_path / "m", say=lambda *_: None, interrupt_at=2, **w)
    spec_rec = json.loads((tmp_path / "m" / "ARM_SPEC.json").read_text(encoding="utf-8"))
    assert spec_rec["device"] == {"eval_device_class": "CPU-FP32-DET", "device_pin_sha256": None}
    assert spec_rec["code_sha256"] == F.fl_code_sha256()

    def resume():
        return F.execute(spec=_tiny_spec("FA"), run_id="m", run_dir=tmp_path / "m", say=lambda *_: None, resume=True,
                         **_world())
    with monkeypatch.context() as mp:
        mp.setattr(F, "device_record", lambda a: {"eval_device_class": "NVIDIA_A40-FP32-DET", "device_pin_sha256": "p"})
        assert_raises(F.LaunchRefused, resume)
    with monkeypatch.context() as mp:
        mp.setattr(F, "fl_code_sha256", lambda: "another-tree")
        assert_raises(F.LaunchRefused, resume)
    legacy = {k: v for k, v in spec_rec.items() if k not in ("device", "code_sha256")}
    (tmp_path / "m" / "ARM_SPEC.json").write_text(json.dumps(legacy), encoding="utf-8")
    assert_raises(F.LaunchRefused, resume)
    (tmp_path / "m" / "ARM_SPEC.json").write_text(json.dumps(spec_rec), encoding="utf-8")
    assert resume()["status"] == "DONE_MAIN"


@nc("an FL resume leg on another device class is accepted")
def test_nc_resume_on_another_device(tmp_path, monkeypatch):
    F.execute(spec=_tiny_spec("FA"), run_id="n", run_dir=tmp_path / "n", say=lambda *_: None, interrupt_at=2,
              **_world())
    monkeypatch.setattr(F, "device_record", lambda a: {"eval_device_class": "NVIDIA_A40-FP32-DET",
                                                        "device_pin_sha256": "p"})
    try:
        F.execute(spec=_tiny_spec("FA"), run_id="n", run_dir=tmp_path / "n", say=lambda *_: None, resume=True,
                  **_world())
        accepted = True
    except F.LaunchRefused:
        accepted = False
    assert accepted


# ------------------------------------------------------------------------------------------------ device arms
PIN = {"self_sha256": "e" * 64, "device_class": "NVIDIA_A40-FP32-DET"}


def _kb():
    from v4eval.rows import COMPARISON_KEY
    return {f: "x" for f in COMPARISON_KEY}


def _lo(w, tmp, *, role="MAIN", select=False, selection=None, **kw):
    worker = make_tiny_adapter(K=K, seed=1, tied=False)
    start = make_tiny_adapter(K=K, seed=1, tied=False).broadcast_state(clone=True)
    spec = D.device_spec("LO", seed=2026, peak_lr=PEAK, role=role, start_digest=state_digest(start), select=select,
                         selection=selection)

    def row_fn(res, cell, selected):
        return D.device_rows(res, _kb(), {"run_label": "PRODUCTION"}, spec=spec, cell=cell, selected=selected,
                             down_bytes=0)
    out = D.execute(spec=spec, run_dir=tmp, worker=worker, start_state=start, plans=_plans(w),
                    get_support=lambda p: w.sup[p.key], get_eval_batch=w.get_eval_batch, manifest=w.m,
                    frozen=_frozen(), row_fn=row_fn, say=lambda *_: None, **kw)
    return spec, out


def _pin_on(monkeypatch):
    import v4eval.device_pin as dp
    monkeypatch.setattr(dp, "pin_for_device", lambda dev: PIN)


def test_device_rows_on_the_pinned_class_carry_the_pin(tmp_path, monkeypatch):
    _pin_on(monkeypatch)
    spec, out = _lo(_dworld(), tmp_path / "pin", n_shards=2)
    assert out["status"] == "DONE" and out["device_pin_sha256"] == PIN["self_sha256"]
    res = json.loads((tmp_path / "pin" / "cells" / "LO_P2.result.json").read_text(encoding="utf-8"))
    assert res["eval_config"]["device_pin_sha256"] == PIN["self_sha256"]
    rows = json.loads((tmp_path / "pin" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert rows and all(r["status"] == "MEASURED_PINNED_DEVICE" and r["device_pin_sha256"] == PIN["self_sha256"]
                        for r in rows)
    summ = json.loads((tmp_path / "pin" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert summ["device_pin_sha256"] == PIN["self_sha256"]
    for s in range(2):
        rec = json.loads((tmp_path / "pin" / "shards" / f"shard_{s:04d}.json").read_text(encoding="utf-8"))
        assert rec["eval_device_class"] == "CPU-FP32-DET" and rec["device_pin_sha256"] == PIN["self_sha256"]
        assert rec["code_sha256"] == D.device_code_sha256()


def test_unpinned_device_rows_unchanged(tmp_path):
    spec, out = _lo(_dworld(), tmp_path / "np", n_shards=1)
    rows = json.loads((tmp_path / "np" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert all(r["status"] == "REHEARSAL_NON_FROZEN_DEVICE" and "device_pin_sha256" not in r for r in rows)
    res = json.loads((tmp_path / "np" / "cells" / "LO_P1.result.json").read_text(encoding="utf-8"))
    assert "device_pin_sha256" not in res["eval_config"] and "device_pin_sha256" not in out


def test_shards_of_another_class_pin_or_code_never_merge(tmp_path, monkeypatch):
    w = _dworld()
    with monkeypatch.context() as mp:
        _pin_on(mp)
        _spec, part = _lo(w, tmp_path / "mix", n_shards=2, shard_range=(0, 1))
    assert part["status"] == "PARTIAL"
    with pytest.raises(D.LaunchRefused, match="never merged"):
        _lo(w, tmp_path / "mix", n_shards=2, resume=True)                     # the unpinned process refuses
    spec, done = _lo(w, tmp_path / "ok", n_shards=2)
    assert done["status"] == "DONE"
    with monkeypatch.context() as mp:
        mp.setattr(D, "device_code_sha256", lambda: "another-tree")
        with pytest.raises(D.LaunchRefused, match="never merged"):
            _lo(w, tmp_path / "ok", n_shards=2, resume=True)
    rec_p = tmp_path / "ok" / "shards" / "shard_0001.json"
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    rec_p.write_text(json.dumps(dict(rec, eval_device_class="NVIDIA_A40-FP32-DET")), encoding="utf-8")
    with pytest.raises(D.LaunchRefused, match="never merged"):
        D.finalize(spec=spec, run_dir=tmp_path / "ok", manifest=w.m, frozen=_frozen(), n_shards=2, row_fn=None,
                   evaluator_sha=None, status={"device": "cpu"}, say=lambda *_: None)


# ------------------------------------------------------------------------------------------------ 06E §1 (LO select)
def test_lo_select_on_unseen_val_writes_the_selection(tmp_path):
    w = _dworld()
    spec, out = _lo(w, tmp_path / "sel", role="UNSEEN_VAL", select=True, n_shards=1,
                    eval_manifest_content_sha256="SYN_UNSEEN_VAL_CONTENT_SHA")
    sel = json.loads((tmp_path / "sel" / "SELECTION.json").read_text(encoding="utf-8"))
    order = ["LO_P1", "LO_P2", "LO_P4"]
    assert sel["arm"] == "LO" and sel["selection_role"] == "UNSEEN_VAL" and sel["cell"] == out["selected_cell"]
    assert sel["cell"] == D.select_cell(out["values"], order)
    assert out["values"][sel["cell"]] == max(out["values"].values())
    assert order.index(sel["cell"]) == min(i for i, c in enumerate(order) if out["values"][c] == max(out["values"].values()))
    assert D.select_cell({"LO_P1": 0.5, "LO_P2": 0.5, "LO_P4": 0.4}, order) == "LO_P1"        # tie -> fewer passes
    assert D.select_cell({"LO_P1": 0.4, "LO_P2": 0.5, "LO_P4": 0.5}, order) == "LO_P2"
    assert D.load_selection(tmp_path / "sel" / "SELECTION.json")["cell"] == sel["cell"]
    rows = json.loads((tmp_path / "sel" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert {r["row_status"] for r in rows} == {"DEVICE_SELECTION"}
    with pytest.raises(FileExistsError):                                              # O_EXCL: never rewritten
        _lo(w, tmp_path / "sel", role="UNSEEN_VAL", select=True, n_shards=1, resume=True,
            eval_manifest_content_sha256="SYN_UNSEEN_VAL_CONTENT_SHA")
    for role in ("MAIN", "CALIBRATION", "HEAVY_PANEL"):
        assert_raises(D.LaunchRefused, D.device_spec, "LO", seed=2026, peak_lr=PEAK, role=role, start_digest="x",
                      select=True)


def test_lo_main_rows_mark_unselected_cells_descriptive(tmp_path):
    w = _dworld()
    spec, out = _lo(w, tmp_path / "main", selection={"arm": "LO", "cell": "LO_P2"}, n_shards=1)
    rows = json.loads((tmp_path / "main" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert {r["device_cell"] for r in rows} == {"LO_P1", "LO_P2", "LO_P4"}
    for r in rows:
        if r["device_cell"] == "LO_P2":
            assert r["comparison_status"] == "DOWNSTREAM_RETENTION" and r["device_selection_rule"] == \
                "SELECTED_ON_UNSEEN_VAL" and r["device_selected"] is True
        else:
            assert r["comparison_status"] == "DESCRIPTIVE" and r["row_status"] == "LO_CELL_NOT_SELECTED" and \
                r["device_selection_rule"] == "LO_CELL_NOT_SELECTED" and r["device_selected"] is False


@nc("an unselected LO cell's MAIN rows are downstream-retention evidence")
def test_nc_unselected_lo_cell_is_evidence(tmp_path):
    _spec, _out = _lo(_dworld(), tmp_path / "nc", selection={"arm": "LO", "cell": "LO_P2"}, n_shards=1)
    rows = json.loads((tmp_path / "nc" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert all(r["comparison_status"] == "DOWNSTREAM_RETENTION" for r in rows)
