"""probe_runner.py (owner: Opus) -- gate_2a() and encoding_check(). Never modifies the code under
test.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import conftest_fo  # noqa: F401
from conftest_fo import tiny_store

import feature_view as fv
import probe_runner as pr


def _probe(delta, ci_lo, s_wd, s_we):
    return {"delta_macro_mrr@20": delta, "bootstrap_macro_mrr@20": {"ci95": [ci_lo, ci_lo + 0.05]},
            "slices": {"weekday": {"delta_macro_mrr@20": s_wd}, "weekend": {"delta_macro_mrr@20": s_we}}}


# --------------------------------------------------------------------- gate_2a
def test_gate_2a_p1_excluded_even_if_strongly_positive():
    probes = {"P1_ALL_SIDE_OFF": _probe(0.5, 0.4, 0.5, 0.5),
              "P2_MAIN_CATEGORY_UNAVAILABLE": _probe(0.01, 0.005, 0.01, 0.01)}
    g = pr.gate_2a(probes)
    assert "P1_ALL_SIDE_OFF" not in g["passing"]
    assert g["C2"] == "P2_MAIN_CATEGORY_UNAVAILABLE"


def test_gate_2a_requires_delta_ge_0_001():
    probes = {"P3_BRAND_UNAVAILABLE": _probe(0.0009, 0.0001, 0.001, 0.001)}   # just under 0.001
    g = pr.gate_2a(probes)
    assert g["passing"] == [] and g["C2"] is None


def test_gate_2a_requires_ci_lower_gt_0():
    probes = {"P3_BRAND_UNAVAILABLE": _probe(0.01, -0.001, 0.01, 0.01)}   # CI lower <= 0
    g = pr.gate_2a(probes)
    assert g["passing"] == [] and g["C2"] is None


def test_gate_2a_requires_both_slices_positive():
    probes = {"P4_CURRENT_PRICE_UNAVAILABLE": _probe(0.01, 0.005, 0.01, -0.001)}   # weekend slice negative
    g = pr.gate_2a(probes)
    assert g["passing"] == [] and g["C2"] is None


def test_gate_2a_negative_deltas_never_candidates():
    probes = {"P5_PREV_GAP_UNAVAILABLE": _probe(-0.05, -0.06, -0.05, -0.05)}
    g = pr.gate_2a(probes)
    assert g["passing"] == [] and g["C2"] is None


def test_gate_2a_picks_the_largest_delta_among_passing():
    probes = {
        "P3_BRAND_UNAVAILABLE": _probe(0.010, 0.005, 0.01, 0.01),
        "P5_PREV_GAP_UNAVAILABLE": _probe(0.030, 0.020, 0.03, 0.03),   # largest
        "P6_ITEM_AGE_UNAVAILABLE": _probe(0.020, 0.010, 0.02, 0.02),
        "P1_ALL_SIDE_OFF": _probe(0.999, 0.9, 0.9, 0.9),               # excluded despite being largest
    }
    g = pr.gate_2a(probes)
    assert set(g["passing"]) == {"P3_BRAND_UNAVAILABLE", "P5_PREV_GAP_UNAVAILABLE", "P6_ITEM_AGE_UNAVAILABLE"}
    assert g["C2"] == "P5_PREV_GAP_UNAVAILABLE"


def test_gate_2a_negative_control_all_failing_gives_no_candidate():
    """Negative control for the 'picks the largest' test: if EVERY probe fails at least one
    condition, C2 must be None (not just 'not the excluded P1')."""
    probes = {
        "P3_BRAND_UNAVAILABLE": _probe(0.030, -0.001, 0.03, 0.03),    # CI fails
        "P5_PREV_GAP_UNAVAILABLE": _probe(0.030, 0.020, -0.01, 0.03),  # weekday slice fails
    }
    g = pr.gate_2a(probes)
    assert g["passing"] == [] and g["C2"] is None


# --------------------------------------------------------------------- encoding_check
def test_encoding_check_reports_unverified_for_a_tiny_random_store():
    store = tiny_store(n_train=8, n_vs=64, K=23, seed=0)
    ds = store.decisions("VALIDATION_SELECT", "eval")
    enc = pr.encoding_check(store, ds)
    assert enc["ok"] is False
    assert enc["unverified"], "a 64-row random synthetic VS split should not clear the 100-row floor for every state"
    assert set(enc["unverified"]) == {k for k, n in enc["counts"].items() if n < pr.ENCODING_MIN_ROWS}


def _consistent_encoding_batch():
    """8 rows (T=1), each row set up to TRIGGER and SATISFY one or more of encoding_check's
    real-pipeline relations, plus one neutral filler row that triggers nothing."""
    R = 8
    N, F, U, M = fv.NUM, fv.FLAG, fv.UCTX, fv.UMASK
    num = torch.zeros(R, 1, len(fv.NUM_NAMES))
    flg = torch.zeros(R, 1, len(fv.FLAG_NAMES))
    uc = torch.zeros(R, len(fv.UCTX_NAMES))
    um = torch.ones(R, len(fv.UMASK_NAMES))   # default "supported"/"has_history" = 1
    item_tok = torch.full((R, 1), 5, dtype=torch.long)     # a generic known item
    brand_tok = torch.full((R, 1), 5, dtype=torch.long)
    main_tok = torch.full((R, 1), 5, dtype=torch.long)
    attention_mask = torch.ones(R, 1, dtype=torch.long)
    lengths = torch.ones(R, dtype=torch.long)

    v_sup = fv.enc_user("current_minus_recent_mean_log_price", 0.0, {"recent_price_supported": 1.0})
    z_run0 = fv.enc_event("current_known_category_run_length", 0.0)
    cu, cm = fv.cold_user_vector()

    # row 0: cold user
    uc[0] = torch.as_tensor(cu)
    um[0] = torch.as_tensor(cm)
    # row 1: price missing, SUPPORTED sentinel
    flg[1, 0, F["price_missing"]] = 1
    flg[1, 0, F["price_zero"]] = 1
    um[1, M["recent_price_supported"]] = 1.0
    uc[1, U["current_minus_recent_mean_log_price"]] = v_sup
    # row 2: price missing, UNSUPPORTED sentinel
    flg[2, 0, F["price_missing"]] = 1
    flg[2, 0, F["price_zero"]] = 1
    um[2, M["recent_price_supported"]] = 0.0
    uc[2, U["current_minus_recent_mean_log_price"]] = 0.0
    # row 3: brand missing + brand-match unsupported
    flg[3, 0, F["brand_missing"]] = 1
    brand_tok[3, 0] = fv.MISSING
    um[3, M["recent_brand_match_supported"]] = 0.0
    uc[3, U["recent_brand_match_share"]] = 0.0
    # row 4: main-category missing + category-match unsupported
    flg[4, 0, F["main_category_missing"]] = 1
    flg[4, 0, F["category_code_missing"]] = 1
    main_tok[4, 0] = fv.MISSING
    num[4, 0, N["current_known_category_run_length"]] = z_run0
    um[4, M["recent_category_match_supported"]] = 0.0
    uc[4, U["recent_category_match_share"]] = 0.0
    # row 5: prev-gap unavailable
    flg[5, 0, F["prev_gap_unavailable"]] = 1
    num[5, 0, N["log1p_prev_raw_view_gap_s"]] = 0.0
    # row 6: item-age unavailable
    flg[6, 0, F["item_age_unavailable"]] = 1
    num[6, 0, N["log1p_item_observed_age_s"]] = 0.0
    # row 7: neutral filler, triggers nothing

    class _FakeDS:
        def __len__(self):
            return R

    class _FakeStore:
        def batch(self, ds, pos):
            pos = np.asarray(pos)
            return {"attention_mask": attention_mask[pos], "lengths": lengths[pos],
                    "event_numeric_features": num[pos], "event_quality_flags": flg[pos],
                    "user_context": uc[pos], "user_context_masks": um[pos],
                    "brand_tokens": brand_tok[pos], "main_category_tokens": main_tok[pos],
                    "item_tokens": item_tok[pos]}

    return _FakeStore(), _FakeDS()


def test_encoding_check_ok_true_when_floor_met_and_no_violations(monkeypatch):
    store, ds = _consistent_encoding_batch()
    monkeypatch.setattr(pr, "ENCODING_MIN_ROWS", 1)
    enc = pr.encoding_check(store, ds)
    assert enc["violations"] == []
    assert enc["unverified"] == []
    assert enc["ok"] is True
    for k, n in enc["counts"].items():
        assert n >= 1, f"fixture never triggered condition {k}"


def test_encoding_check_negative_control_single_violation_gives_ok_false(monkeypatch):
    """Negative control for the 'ok True' test above: break exactly ONE relation (row 1's
    price_zero flag no longer co-occurs with price_missing) and confirm the checker catches it
    even with the floor lowered to 1 (proving 'no violations' really was load-bearing, not just an
    unreachable branch)."""
    store, ds = _consistent_encoding_batch()
    # store.batch returns freshly-sliced tensors each call except our closures capture the same
    # underlying tensors; mutate the flag tensor via a monkeypatched batch that flips it back.
    orig_batch = store.batch

    def broken_batch(ds_, pos):
        b = orig_batch(ds_, pos)
        b = dict(b)
        flg = b["event_quality_flags"].clone()
        # position of row 1 within this chunk (pos is arange(0,8) in one shot -> index 1 is row 1)
        pos_arr = np.asarray(pos)
        rows = np.nonzero(pos_arr == 1)[0]
        if rows.size:
            flg[rows, 0, fv.FLAG["price_zero"]] = 0   # break: price_missing=1 but price_zero=0 now
        b["event_quality_flags"] = flg
        return b

    monkeypatch.setattr(store, "batch", broken_batch)
    monkeypatch.setattr(pr, "ENCODING_MIN_ROWS", 1)
    enc = pr.encoding_check(store, ds)
    assert enc["ok"] is False
    assert any("price_missing_events" in v for v in enc["violations"])
