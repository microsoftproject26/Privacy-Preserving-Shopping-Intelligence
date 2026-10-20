"""The Study A / Study B / events-to-X machinery of v4int.launch_trunc (CPU, a tiny synthetic model on REAL truncated
batches of the frozen v1 bucket 0; the GPU wiring is exercised on the server by the smoke commands).

  * Study A: one locked model through the frozen T3 path (predict_adapter -> evaluate -> rows_from_result), the
    RESULT_ROWS format with the per-h labels; the FULL cell ranks == the production batches' ranks (T41 end to end);
  * Study B: launch_device.execute over a truncated sub-panel with the E11.3 support, the causality cutoff, the
    selected recipe; zero-label users score the start state;
  * the cell list, events-to-X (smallest registered h, no interpolation), the T41 receipt gate and the refusals.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import v4int.launch_trunc as T
from trunckit import K, REAL_DS, assert_raises, nc
from v4int.launch_central import LaunchRefused
from v4trunc.provider import TruncEvalProvider, support_batch

KEY = {"data_protocol_hash": "d", "feature_contract_hash": "f", "split": "VALIDATION", "label_policy_hash": "l",
       "population_hash": "p", "candidate_order_hash": "c", "initialization_block": "i", "topology_hash": "t",
       "objective_id": "o", "precision_id": "fp32", "evaluator_code_sha256": "e", "family": "SASREC"}


def _frozen():
    from v4eval import frozen_config as fc
    fz = fc.load()
    fc.apply_runtime_flags(fz)
    return fz


def _tiny(seed=1):
    from fl.synthetic import make_tiny_adapter
    return make_tiny_adapter(K=K, d=8, tied=False, dropout=0.0, seed=seed)


def _labels(pm, p, h, full):
    return T.trunc_labels(study="A", panel_id=pm.manifest_id, label=pm.label, panel_type=p, h=h, full=full)


# ------------------------------------------------------------------------------------------------ Study A
def test_study_a_cell_rows_format(builder, derived_panel):
    pm = derived_panel
    prov = TruncEvalProvider(builder, pm, "ALL_AVAILABLE", 5)
    res, rows, art = T.study_a_cell(_tiny(), prov, _frozen(), evaluator_sha="e", key_base=KEY,
                                    labels=_labels(pm, "ALL_AVAILABLE", 5, False), arm="C_SMALL", seed=2026,
                                    model_sha256="m", overrides={"trunc_cell": "ALL_AVAILABLE_h5"})
    assert rows and all(r["method"] == "C_SMALL" and r["trunc_query_h"] == 5 and r["trunc_h"] == 5
                        and r["trunc_history"] == "H_h" for r in rows)
    assert {r["target_denominator"] for r in rows} == {"E2E", "RANKABLE"}
    assert rows[0]["comparison_status"] == "WARM_TRUNCATION_STUDY_A"
    assert all(r["row_status"] == "WARM_TRUNCATION_STUDY_A" for r in rows)          # B-F3: never COLD_START_*
    assert rows[0]["trunc_oov_flag_counts"]["n_served"] == prov.n_stream_rows      # B-F7: complete cell counted
    assert "recent_brand_match_share" in rows[0]["trunc_partial_channels"]
    assert int((art.ranks > 0).sum()) == int(prov.manifest.rankable.sum())
    p = T.primary_rows(rows)
    assert len(p) == 1 and 0.0 <= p[0]["value"] <= 1.0


def test_study_a_full_cell_equals_production_ranks(builder, derived_panel):
    """FULL history through the builder == the production USER_HISTORY_L50 batches: identical ranks (T41 end to end)."""
    from v4data.common import VALIDATION_SPLITS
    from v4eval.predict import predict_adapter
    from v4trunc.golden import production_eval_batches
    pm = derived_panel
    fz = _frozen()
    prov = TruncEvalProvider(builder, pm, "ALL_AVAILABLE", 1, full_history=True)
    ad = _tiny()
    a1 = predict_adapter(ad, prov.batches(fz.eval_batch), prov.manifest, config=fz, model_sha256="m")
    prod = (b for _, b in production_eval_batches(prov.manifest.decision_id, splits=VALIDATION_SPLITS, kind="eval",
                                                  state_sha256=builder.state_sha256, ds_root=REAL_DS, buckets=[0],
                                                  batch_size=fz.eval_batch))
    a2 = predict_adapter(ad, prod, prov.manifest, config=fz, model_sha256="m")
    assert np.array_equal(a1.ranks, a2.ranks) and np.array_equal(a1.n_tied, a2.n_tied)


def test_truncation_changes_features_but_not_queries(builder, derived_panel):
    pm = derived_panel
    p50 = TruncEvalProvider(builder, pm, "FIXED_20", 20)
    p0 = TruncEvalProvider(builder, pm, "FIXED_20", 0)
    assert np.array_equal(p50.manifest.decision_id, p0.manifest.decision_id)
    b20, b0 = p50.batch_at(np.arange(4)), p0.batch_at(np.arange(4))
    assert b0["lengths"].tolist() == [1, 1, 1, 1] and b20["lengths"].tolist() == [21, 21, 21, 21]
    assert float(b0["user_context_masks"].abs().sum()) == 0.0              # h = 0: no permitted history


# ------------------------------------------------------------------------------------------------ Study B
def test_study_b_execute_on_truncated_support(builder, derived_panel, tmp_path):
    import v4int.launch_device as D
    from fl.numerics import state_digest
    pm = derived_panel
    prov = TruncEvalProvider(builder, pm, "ALL_AVAILABLE", 5)
    m = prov.manifest
    worker, start_ad = _tiny(1), _tiny(1)
    start = start_ad.broadcast_state(clone=True)
    sel = {"arm": "LO", "cell": "LO_P1", "selection_role": "UNSEEN_VAL", "self_sha256": "s"}
    spec = T.trunc_device_spec("LO", seed=2026, peak_lr=1e-3, panel_id=pm.manifest_id, panel_type="ALL_AVAILABLE",
                               h=5, start_digest=state_digest(start), selection=sel, fused=False)
    hexes = pm.hex_of_key(m.eval_user_key)
    sup = {str(k): np.array([int(r)]) for k, r in zip(hexes.tolist(), prov.event_rows.tolist())}
    plans = D.build_client_plans(m, hexes, row_decision_ts=builder.table.meta["decision_ts_us"][prov.event_rows],
                                 support_pos_by_key=sup, warm=False)
    plans = plans[:24]
    calls = []

    def gs(plan):
        b = support_batch(builder, int(plan.support[0]), 5)
        calls.append(0 if b is None else int(b["target_class"].shape[0]))
        return {"target_class": torch.zeros(0, dtype=torch.long)} if b is None else b
    out = D.execute(spec=spec, run_dir=tmp_path, worker=worker, start_state=start, plans=plans, get_support=gs,
                    get_eval_batch=prov.batch_at, manifest=m, frozen=_frozen(), say=lambda *_: None,
                    smoke_clients=24)
    assert out["status"] == "SMOKE_DONE_NOT_EVIDENCE" and len(calls) == 24 and max(calls) > 0
    recs = [json.loads(p.read_text()) for p in sorted((tmp_path / "shards").glob("shard_*.json"))]
    nosup = sum(1 for r in recs for x in r["records"] if x["no_support"])
    assert nosup == sum(1 for c in calls if c == 0)


def test_study_b_spec_requires_the_unseen_val_selection():
    ok = {"arm": "FT_C", "cell": "FT_C_f0.3_P2", "selection_role": "UNSEEN_VAL", "self_sha256": "s"}
    s = T.trunc_device_spec("FT_C", seed=1, peak_lr=1e-3, panel_id="P", panel_type="FIXED_20", h=5, start_digest="x",
                            selection=ok, fused=None)
    assert [c["cell"] for c in s["cells"]] == ["FT_C_f0.3_P2"] and s["study"] == "B" and s["h"] == 5
    lo = T.trunc_device_spec("LO", seed=1, peak_lr=1e-3, panel_id="P", panel_type="FIXED_20", h=5, start_digest="x",
                             selection=dict(ok, arm="LO", cell="LO_P2"), fused=None)
    assert [c["cell"] for c in lo["cells"]] == ["LO_P1", "LO_P2", "LO_P4"] and lo["selected_cell"] == "LO_P2"
    for sel in (None, dict(ok, selection_role="CALIBRATION"), dict(ok, arm="LO"), dict(ok, cell="FT_C_f0.2_P2")):
        assert_raises(LaunchRefused, T.trunc_device_spec, "FT_C", seed=1, peak_lr=1e-3, panel_id="P",
                      panel_type="FIXED_20", h=5, start_digest="x", selection=sel, fused=None)
    assert_raises(LaunchRefused, T.trunc_device_spec, "FT_FA", seed=1, peak_lr=1e-3, panel_id="P",
                  panel_type="FIXED_20", h=5, start_digest="x", selection=ok, fused=None)


# ------------------------------------------------------------------------------------------------ cells / E2X
def test_study_cells():
    subs = [("FIXED_50", h) for h in (0, 1, 5, 10, 20, 50)] + [("FIXED_20", h) for h in (0, 1, 5, 10, 20)] + \
        [("ALL_AVAILABLE", h) for h in (0, 1, 5, 10, 20, 50)]
    cells = T.study_cells(subs)
    assert len(cells) == 17 + 2 + 6
    assert ("FIXED_50", 50, True) in cells and ("FIXED_20", 20, True) in cells and ("ALL_AVAILABLE", 0, True) in cells
    assert len({T.cell_name(*c) for c in cells}) == len(cells)


def test_events_to_x():
    m = {0: 0.10, 1: 0.12, 5: 0.15, 10: 0.17, 20: 0.18, 50: 0.19}
    assert T.events_to_x(m, 0.18) == {"90": 10, "95": 20}
    assert T.events_to_x(m, 0.19) == {"90": 20, "95": 50}
    assert T.events_to_x(m, 0.30) == {"90": None, "95": None}                 # never reached: not reported
    assert T.events_to_x(m, {0: 0.1, 1: 0.2, 5: 0.2, 10: 0.2, 20: 0.2, 50: 0.2}) == {"90": 0, "95": 0}


@nc("events-to-X must not interpolate between registered budgets")
def test_nc_events_to_x_interpolates():
    m = {5: 0.15, 10: 0.20}
    # REV-ANA-01 B-F5: an interpolating rule returns 8 (not a registered budget); the NC claims a value outside
    # {5, 10} is returned, which the registered (no-interpolation) rule never does
    assert T.events_to_x(m, 0.20)["90"] not in (5, 10)


def _row(p, h, full, v, seed=2026, panel="P", method="C_BIG", study="A"):
    return {"target_denominator": "E2E", "aggregation": "macro", "metric": "mrr", "k": 20, "value": v, "seed": seed,
            "trunc_panel": panel, "trunc_panel_type": p, "trunc_query_h": h, "trunc_full_history": full,
            "method": method, "trunc_study": study}


def test_events_to_x_table():
    rm = [_row("FIXED_20", h, False, v) for h, v in ((0, .1), (1, .12), (5, .16), (10, .185), (20, .195))] + \
        [_row("FIXED_20", 20, True, .2)] + [_row("ALL_AVAILABLE", 5, False, .15), _row("ALL_AVAILABLE", 5, True, .17)]
    rc = [_row("FIXED_20", 20, True, .2), _row("ALL_AVAILABLE", 5, True, .16)]
    t = T.events_to_x_table(rm, rc)
    assert t["FIXED_20"]["events_to_X"] == {"90": 10, "95": 20} and t["FIXED_20"]["M_full_history"] == .2
    assert t["ALL_AVAILABLE"]["events_to_X"] == {"90": 5, "95": None}
    assert_raises(LaunchRefused, T.events_to_x_table, rm, [dict(r, seed=1) for r in rc])


# ------------------------------------------------------------------------------------------------ gate / refusals
def _receipt(tmp_path, **over):
    obj = {"format": T.RECEIPT_FORMAT, "panel": "WARM_TRUNCATION_16384", "status": "PASS",
           "gates": {"golden": True, "deletion": True}, "v4trunc_code_sha256": T.v4trunc_code_sha256()}
    obj.update(over)
    obj["self_sha256"] = T.canon_sha(obj)
    p = tmp_path / "T41_RECEIPT.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_t41_receipt_gate(tmp_path):
    assert T.check_t41_receipt(_receipt(tmp_path), "WARM_TRUNCATION_16384")["self_sha256"]
    for over in ({"status": "FAIL"}, {"gates": {"golden": False}}, {"gates": {}}, {"panel": "COLD_H_16384"},
                 {"v4trunc_code_sha256": "0" * 64}):
        assert_raises(LaunchRefused, T.check_t41_receipt, _receipt(tmp_path, **over), "WARM_TRUNCATION_16384")
    p = _receipt(tmp_path)
    obj = json.loads(p.read_text())
    obj["status"] = "PASS "
    p.write_text(json.dumps(obj))
    assert_raises(LaunchRefused, T.check_t41_receipt, p, "WARM_TRUNCATION_16384")          # self-hash broken
    assert_raises(LaunchRefused, T.check_t41_receipt, tmp_path / "absent.json", "WARM_TRUNCATION_16384")
    adm = _receipt(tmp_path, golden_replay={"admitted_partial_rows": 3})
    assert T.check_t41_receipt(adm, "WARM_TRUNCATION_16384")["admitted_partial_rows"] == 3
    assert_raises(LaunchRefused, T.check_t41_receipt, adm, "WARM_TRUNCATION_16384", exact=True)


def _args(**kw):
    base = dict(study="A", arm="C_BIG", panel=None, seed=2026, gpu=0, run_dir="r", ckpt="c.pt", ckpt_sha256=None,
                base=None, selection=None, t41_receipt="t.json", smoke=True, smoke_queries=64, dry_run=True,
                rows=None, cref_rows=None, allow_shared_gpu=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_preflight_refusals(tmp_path):
    assert T.preflight(_args())["panel"] == "WARM_TRUNCATION"
    assert T.preflight(_args(study="B", arm="LO", ckpt=None, selection="s.json"))["panel"] == "COLD_H"
    assert T.preflight(_args(study="T41", arm=None, ckpt=None, seed=None, gpu=None, t41_receipt=None))
    bad = [dict(study="B", arm="LO", ckpt=None, selection="s.json", panel="WARM_TRUNCATION"),   # B on warm refused
           dict(arm="FT_C"), dict(ckpt=None), dict(t41_receipt=None), dict(gpu=None), dict(seed=None),
           dict(selection="s.json"), dict(study="B", arm="FT_C", ckpt=None, selection="s.json"),
           dict(study="B", arm="LO", ckpt=None, selection=None), dict(study="B", arm="LO", ckpt=None,
                                                                     selection="s.json", base="b.pt"),
           dict(study="T41", arm="C_BIG"), dict(study="E2X", arm=None), dict(smoke_queries=0),
           dict(smoke=False)]
    for kw in bad:
        assert_raises(LaunchRefused, lambda: T.preflight(_args(**kw), gate_path=tmp_path / "absent.json"))


# ------------------------------------------------------------------------------------------------ T41 receipt
def _world(builder, pm, table):
    from v4trunc.panel import check_panel_against_events, eligibility
    pm.record = {"transform_state_self_sha256": builder.state_sha256, "derived": True}
    return {"panel_id": pm.manifest_id, "pm": pm, "table": table, "builder": builder, "ds_root": REAL_DS,
            "buckets": [0], "panel_event_checks": check_panel_against_events(pm, table),
            "eligibility": eligibility(pm)}


def test_t41_receipt_passes_on_real_rows(builder, derived_panel, sample_table, tmp_path):
    obj = T.t41_receipt(_world(builder, derived_panel, sample_table), say=lambda *_: None)
    assert obj["status"] == "PASS" and all(obj["gates"].values()), obj["gates"]
    assert obj["golden_replay"]["n_compared"] > 300 and obj["deletion_invariance"]["n_checked"] > 100
    p = tmp_path / "T41_RECEIPT.json"
    p.write_text(json.dumps(obj, default=str), encoding="utf-8")
    # a bucket-restricted receipt is a SMOKE receipt: it gates a --smoke launch only (REV-ANA-01 B-F1)
    assert T.check_t41_receipt(p, derived_panel.manifest_id, smoke=True)["receipt_is_smoke"] is True
    assert_raises(LaunchRefused, T.check_t41_receipt, p, derived_panel.manifest_id)


@nc("a T41 receipt built with a builder that leaks one extra (future) event must not PASS")
def test_nc_t41_receipt_future_event_builder(derived_panel, sample_table):
    from trunckit import FutureEventBuilder, make_builder
    obj = T.t41_receipt(_world(make_builder(sample_table, FutureEventBuilder), derived_panel, sample_table),
                        say=lambda *_: None)
    assert obj["status"] == "PASS"


# ------------------------------------------------------------------------------------------------ REV-ANA-01 B-F1
def test_t41_receipt_smoke_and_panel_binding(tmp_path):
    """B-F1: a SMOKE receipt (smoke_buckets or label_smoke) gates only a --smoke launch; with the verified panel the
    receipt must carry its content_sha256 and (production) a passing golden replay over exactly its distinct query
    decisions."""
    P = "WARM_TRUNCATION_16384"
    pm = SimpleNamespace(record={"content_sha256": "c" * 64},
                         queries={"query_decision_id": np.array([11, 11, 12, 13, 13], dtype=np.int64)})
    assert T.panel_query_decision_count(pm) == 3
    for smoke in ({"smoke_buckets": [0, 1]}, {"label_smoke": "SMOKE_NOT_EVIDENCE"},
                  {"smoke_buckets": [0], "label_smoke": "SMOKE_NOT_EVIDENCE"}):
        p = _receipt(tmp_path, **smoke)
        assert_raises(LaunchRefused, T.check_t41_receipt, p, P)
        assert T.check_t41_receipt(p, P, smoke=True)["receipt_is_smoke"] is True
    prod = dict(panel_record={"content_sha256": "c" * 64}, golden_replay={"n_compared": 3, "pass": True},
                smoke_buckets=None, label_smoke=None)
    ok = T.check_t41_receipt(_receipt(tmp_path, **prod), P, pm=pm)
    assert ok["receipt_is_smoke"] is False and ok["bound_to_panel"]["panel_query_decisions"] == 3
    assert T.check_t41_receipt(_receipt(tmp_path, **prod), P, pm=pm, smoke=True)["bound_to_panel"]
    for over in ({"panel_record": {"content_sha256": "d" * 64}}, {"panel_record": {}},
                 {"golden_replay": {"n_compared": 2, "pass": True}},          # a partial replay (e.g. 2 buckets)
                 {"golden_replay": {"n_compared": 3, "pass": False}}, {"golden_replay": {"pass": True}}):
        assert_raises(LaunchRefused, T.check_t41_receipt, _receipt(tmp_path, **dict(prod, **over)), P, pm=pm)
    nopm = SimpleNamespace(record={}, queries=pm.queries)
    assert_raises(LaunchRefused, T.check_t41_receipt, _receipt(tmp_path, **prod), P, pm=nopm)
    sm = _receipt(tmp_path, **dict(prod, smoke_buckets=[0], golden_replay={"n_compared": 1, "pass": True}))
    assert T.check_t41_receipt(sm, P, pm=pm, smoke=True)["receipt_is_smoke"] is True    # smoke: content-bound only
    assert_raises(LaunchRefused, T.check_t41_receipt, sm, P, pm=pm)


# ------------------------------------------------------------------------------------------------ REV-ANA-01 B-F3
def test_row_status_follows_the_panel_label_both_panels(builder, sample_table):
    """B-F3 (E11.1a): Study A / Study B rows carry row_status == comparison_status == the study label of the panel:
    WARM_TRUNCATION_STUDY_* on WARM_TRUNCATION, COLD_START_STUDY_* on COLD_H (label COLD_START); never the other one,
    never DOWNSTREAM_*; the non-selected LO pass budgets stay LO_CELL_NOT_SELECTED / DESCRIPTIVE."""
    from v4int.launch_device import LO_NOT_SELECTED
    from v4trunc.panel import LABEL_OF_ROLE, derive_panel
    for role, want in (("WARM_TRUNCATION", "WARM_TRUNCATION_STUDY_"), ("COLD_H", "COLD_START_STUDY_")):
        pm = derive_panel(sample_table, label=LABEL_OF_ROLE[role])
        prov = TruncEvalProvider(builder, pm, "ALL_AVAILABLE", 1)
        prov = T._smoke_provider(prov, 48)
        res, rows, _ = T.study_a_cell(_tiny(), prov, _frozen(), evaluator_sha="e", key_base=KEY,
                                      labels=_labels(pm, "ALL_AVAILABLE", 1, False), arm="C_BIG", seed=2026,
                                      model_sha256="m", overrides={"trunc_cell": "ALL_AVAILABLE_h1"})
        assert rows and all(r["row_status"] == r["comparison_status"] == want + "A" for r in rows), role
        assert rows[0]["trunc_oov_flag_counts"]["n_served"] == prov.n_stream_rows == 48
        sel = {"arm": "LO", "cell": "LO_P2", "selection_role": "UNSEEN_VAL", "self_sha256": "s"}
        spec = T.trunc_device_spec("LO", seed=2026, peak_lr=1e-3, panel_id=pm.manifest_id, panel_type="ALL_AVAILABLE",
                                   h=1, start_digest="x", selection=sel, fused=False)
        lab_b = T.trunc_labels(study="B", panel_id=pm.manifest_id, label=pm.label, panel_type="ALL_AVAILABLE", h=1,
                               full=False)
        cells = {c["cell"]: c for c in spec["cells"]}
        kb = dict(KEY, eval_manifest_id=prov.manifest.manifest_id)
        rb = T.study_b_rows(res, kb, lab_b, spec=spec, cell=cells["LO_P2"], selected=True, down_bytes=0,
                            overrides={"trunc_cell": "ALL_AVAILABLE_h1", "row_status": "X", "comparison_status": "X"})
        assert rb and all(r["row_status"] == r["comparison_status"] == want + "B" and r["trunc_cell"] ==
                          "ALL_AVAILABLE_h1" for r in rb), role
        rn = T.study_b_rows(res, kb, lab_b, spec=spec, cell=cells["LO_P1"], selected=False, down_bytes=0)
        assert rn and all(r["row_status"] == LO_NOT_SELECTED and r["comparison_status"] == "DESCRIPTIVE" for r in rn)


# ------------------------------------------------------------------------------------------------ REV-ANA-01 B-F7
def test_events_to_x_table_cref_arm_and_oov_counts():
    """B-F7: C_ref rows must be Study A rows of C_BIG (a C_SMALL / Study B / unlabelled C_ref row refuses); the
    per-cell OOV-flagged query counts are reported beside M(h), M's FULL value and C_ref."""
    rm = [dict(_row("FIXED_20", h, False, v, method="FA"), trunc_oov_flagged_queries=h)
          for h, v in ((0, .1), (5, .16), (20, .195))] + [dict(_row("FIXED_20", 20, True, .2, method="FA"),
                                                              trunc_oov_flagged_queries=7)]
    rc = [dict(_row("FIXED_20", 20, True, .2), trunc_oov_flagged_queries=9)]
    t = T.events_to_x_table(rm, rc)["FIXED_20"]
    assert t["C_ref_arm"] == "C_BIG" and t["events_to_X"] == {"90": 20, "95": 20}
    assert t["oov_flagged_queries"] == {"M_by_h": {0: 0, 5: 5, 20: 20}, "M_full_history": 7, "C_ref": 9}
    for bad in ([dict(r, method="C_SMALL") for r in rc], [dict(r, trunc_study="B") for r in rc],
                [{k: v for k, v in r.items() if k != "method"} for r in rc], []):
        assert_raises(LaunchRefused, T.events_to_x_table, rm, bad)


# ------------------------------------------------------------------------------------------------ GATE-B B4
def _b_row(h, v, *, cell="LO_P2", selected=True, seed=2026, panel="COLD_H_16384", method="LO", status=None):
    r = dict(_row("ALL_AVAILABLE", h, False, v, seed=seed, panel=panel, method=method, study="B"),
             device_cell=cell, device_selected=selected, row_status=status or "COLD_START_STUDY_B")
    return r


def _study_b_rows():
    from v4int.launch_device import LO_NOT_SELECTED
    return ([_b_row(h, .9, cell="LO_P1", selected=False, status=LO_NOT_SELECTED) for h in (0, 1, 5, 10)]   # first
            + [_b_row(h, v) for h, v in ((0, .05), (1, .10), (5, .15), (10, .185))])


def _cold_cref(hs=(0, 1, 5, 10)):
    return [_row("ALL_AVAILABLE", h, True, .2, panel="COLD_H_16384") for h in hs]


def test_events_to_x_no_c_ref():
    m = {0: 0.10, 5: 0.15, 10: 0.185}
    assert T.events_to_x(m, None) == {"90": T.NO_C_REF, "95": T.NO_C_REF}
    assert T.events_to_x(m, {0: 0.2, 10: 0.2}) == {"90": T.NO_C_REF, "95": T.NO_C_REF}       # h = 5 has no C_ref
    assert T.events_to_x(m, {0: 0.1, 5: None, 10: 0.2}) == {"90": 0, "95": 0}              # hit before the gap
    assert T.events_to_x(m, {0: 0.2, 5: 0.2, 10: 0.2}) == {"90": 10, "95": None}           # NOT reached != NO_C_REF


def check_study_b_table(table_fn=T.events_to_x_table):
    """B4 key property: Study B rows (no FULL cell) produce an events-to-X table: the selected LO cell only (0.9 of the
    non-selected cell never enters), M_full_history None per h, C_ref from the C_BIG Study A FULL rows of the panel."""
    try:
        t = table_fn(_study_b_rows(), _cold_cref())["ALL_AVAILABLE"]
    except Exception as e:                                           # noqa: BLE001  (a crash IS the failure)
        raise AssertionError(f"events_to_x_table refused / crashed on Study B rows: {type(e).__name__}: {e}")
    assert t["M_by_h"] == {0: .05, 1: .10, 5: .15, 10: .185}
    assert t["M_full_history"] == {0: None, 1: None, 5: None, 10: None} and t["study"] == "B"
    assert t["C_ref"] == {0: .2, 1: .2, 5: .2, 10: .2} and t["events_to_X"] == {"90": 10, "95": None}
    assert "C_ref_status" not in t
    return t


def test_events_to_x_table_study_b():
    check_study_b_table()


def test_events_to_x_table_study_b_missing_cref_cell():
    t = T.events_to_x_table(_study_b_rows(), _cold_cref(hs=(0, 1, 10)))["ALL_AVAILABLE"]
    assert t["C_ref"][5] is None and t["C_ref_status"] == T.NO_C_REF
    assert t["events_to_X"] == {"90": T.NO_C_REF, "95": T.NO_C_REF}
    fixed = [dict(_b_row(h, v), trunc_panel_type="FIXED_20") for h, v in ((0, .1), (5, .2))]
    tf = T.events_to_x_table(fixed, _cold_cref())["FIXED_20"]                      # no FIXED_20 FULL C_ref row at all
    assert tf["C_ref"] is None and tf["M_full_history"] is None and tf["events_to_X"]["90"] == T.NO_C_REF


def test_events_to_x_table_study_b_refusals():
    rb = _study_b_rows()
    cases = [rb + [_row("ALL_AVAILABLE", 0, False, .1, panel="COLD_H_16384", method="FA")],           # A + B mixed
             rb + [_b_row(5, .3, cell="LO_P3")],                                                      # 2 selected cells
             rb + [dict(_b_row(10, .2), trunc_full_history=True)]]                                    # a FULL B cell
    for bad in cases:
        assert_raises(LaunchRefused, T.events_to_x_table, bad, _cold_cref())
    assert_raises(LaunchRefused, T.events_to_x_table, rb, [_row("ALL_AVAILABLE", 0, True, .2)])       # other panel
    assert_raises(LaunchRefused, T.events_to_x_table, rb, [dict(r, seed=2027) for r in _cold_cref()])  # other seed


def test_study_a_table_unchanged_when_complete():
    """The Study A output of a complete C_ref is the pre-B4 dict exactly (no new key)."""
    rm = [_row("FIXED_20", h, False, v) for h, v in ((0, .1), (5, .16), (20, .195))] + [_row("FIXED_20", 20, True, .2)]
    t = T.events_to_x_table(rm, [_row("FIXED_20", 20, True, .2)])["FIXED_20"]
    assert set(t) == {"M_by_h", "C_ref", "M_full_history", "events_to_X", "C_ref_arm", "oov_flagged_queries", "rule"}


def test_run_e2x_accepts_study_b_file(tmp_path):
    (tmp_path / "SB").mkdir()
    (tmp_path / "SB" / "RESULT_ROWS.json").write_text(json.dumps({"study": "B", "arm": "LO", "seed": 2026,
                                                                  "rows": _study_b_rows()}), encoding="utf-8")
    (tmp_path / "SA").mkdir()
    (tmp_path / "SA" / "RESULT_ROWS.json").write_text(json.dumps({"study": "A", "arm": "C_BIG", "rows": _cold_cref()}),
                                                      encoding="utf-8")
    args = SimpleNamespace(cref_rows=tmp_path / "SA" / "RESULT_ROWS.json", rows=[tmp_path / "SB" / "RESULT_ROWS.json"],
                           run_dir=tmp_path / "E2X")
    assert T.run_e2x(args, {}, say=lambda *_: None) == 0
    obj = json.loads((tmp_path / "E2X" / "EVENTS_TO_X.json").read_text(encoding="utf-8"))
    assert obj["arms"]["LO"]["ALL_AVAILABLE"]["events_to_X"] == {"90": 10, "95": None}
    bad = tmp_path / "SB" / "BAD.json"
    bad.write_text(json.dumps({"study": "B", "arm": "LO", "rows": [dict(r, trunc_study="A") for r in _study_b_rows()]}),
                   encoding="utf-8")
    assert_raises(LaunchRefused, T.run_e2x, SimpleNamespace(cref_rows=args.cref_rows, rows=[bad],
                                                            run_dir=tmp_path / "E2X2"), {}, say=lambda *_: None)


@nc("a Study B events-to-X table built WITHOUT the selected-cell filter (every LO pass budget enters) must not pass the "
    "Study B key property")
def test_nc_study_b_table_without_cell_filter(monkeypatch):
    monkeypatch.setattr(T, "selected_study_b_rows", lambda rows: list(rows))
    check_study_b_table()


@nc("a missing C_ref cell read as 'not reached' (the pre-B4 rule) must not pass the NO_C_REF property")
def test_nc_missing_cref_reads_not_reached(monkeypatch):
    real = T.events_to_x

    def lenient(values_by_h, cref, xs=T.XS):
        if isinstance(cref, dict):
            cref = {h: (c if c is not None else float("inf")) for h, c in cref.items()}
        return real(values_by_h, float("inf") if cref is None else cref, xs)

    monkeypatch.setattr(T, "events_to_x", lenient)
    t = T.events_to_x_table(_study_b_rows(), _cold_cref(hs=(0, 1, 10)))["ALL_AVAILABLE"]
    assert t["events_to_X"]["90"] == T.NO_C_REF
