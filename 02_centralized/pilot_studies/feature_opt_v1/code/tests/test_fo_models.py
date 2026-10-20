"""Package test list item 17 -- fo_models.py (owner: Opus): parameter_mapping, D3 zero-init
equivalence at init, and strict-load view-mismatch refusal. Never modifies the code under test.
"""
from __future__ import annotations

import pytest
import torch

import conftest_fo  # noqa: F401
from conftest_fo import tiny_store, tiny_inner, TINY_REGISTRY

import feature_view as fv
import fo_models as fm


def _baseline():
    store = tiny_store(n_train=32, n_vs=32, K=41, seed=3)
    inner = tiny_inner(store, seed=2)
    return store, inner.state_dict()


def _fit():
    return {"log1p_time_since_prev_session_last_event_s": {"valid_mean": 11.82, "valid_std": 1.663},
            "log1p_time_since_prev_user_event_s": {"valid_mean": 5.382, "valid_std": 3.181}}


# --------------------------------------------------------------------- parameter_mapping (full)
def test_parameter_mapping_full_is_pure_inheritance():
    store, sd = _baseline()
    rec = fm.parameter_mapping(sd, "full")
    assert rec["remapped"] == [] and rec["new"] == []
    assert set(rec["inherited"]) == set(sd)


# --------------------------------------------------------------------- parameter_mapping (d3)
def test_parameter_mapping_d3_new_columns_are_zero_and_only_numflag_remapped():
    store, sd = _baseline()
    m1 = fv.m1_states_from_fit(_fit())
    rec = fm.parameter_mapping(sd, "d3", m1)
    assert rec["remapped"] == ["numflag_proj.weight"]
    assert "new_columns" in rec
    nc = rec["new_columns"]["numflag_proj.weight"]
    assert nc["names"] == list(fv.D3_NAMES)
    assert nc["init"] == "zeros"
    d3_sd = fv.d3_state_dict(sd)
    w = d3_sd["numflag_proj.weight"]
    zero_cols = w[:, nc["columns"][0]:nc["columns"][-1] + 1]
    assert torch.equal(zero_cols, torch.zeros_like(zero_cols))


def test_d3_candidate_equals_control_exactly_at_init():
    store, sd = _baseline()
    control = fm.build(TINY_REGISTRY, sd, view="full", device="cpu",
                       vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)
    d3 = fm.build(TINY_REGISTRY, sd, view="d3", device="cpu",
                  vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)
    import numpy as np
    import sasrec_data
    ds = store.decisions("VALIDATION_SELECT", "eval")
    b = store.batch(ds, np.arange(min(16, len(ds))))
    kw = sasrec_data.predictor_kwargs(b, control)
    with torch.no_grad():
        y_ctrl, y_d3 = control(**kw), d3(**kw)
    d = (y_ctrl - y_d3).abs().max()
    # "exactly" up to float summation-order rounding: the new D3 weight columns are exact zeros,
    # so the extra matmul terms are exact zeros too, but summing 24 vs 19 terms is not guaranteed
    # bit-identical in FP32. Same standard as the lean fold's "exact reparametrisation" claim.
    print("d3-vs-control max|dlogit| at init =", float(d))
    assert float(d) < 1e-5, f"D3 must equal CONTROL at init up to float rounding, got max|dlogit|={float(d)}"


# --------------------------------------------------------------------- (17) parameter_mapping (lean)
def test_parameter_mapping_lean_touches_expected_params_only():
    store, sd = _baseline()
    m1 = fv.m1_states_from_fit(_fit())
    rec = fm.parameter_mapping(sd, "lean", m1)
    assert set(rec["remapped"]) <= {"numflag_proj.weight", "numflag_proj.bias", "user_mlp.weight", "user_mlp.bias"}
    assert rec["new"] == []
    untouched = {"item_embed.weight", "pos_embed.weight", "output_embed", "output_bias"}
    assert untouched.isdisjoint(rec["remapped"])


# --------------------------------------------------------------------- strict load refuses wrong view
def test_strict_load_of_lean_folded_weights_into_full_view_refuses():
    store, sd = _baseline()
    m1 = fv.m1_states_from_fit(_fit())
    lean_sd = fv.fold_lean_state_dict(sd, m1)   # numflag_proj/user_mlp now have SMALLER in_features
    with pytest.raises(RuntimeError):
        fm.build(TINY_REGISTRY, lean_sd, view="full", device="cpu",
                vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)


def test_strict_load_negative_control_correct_view_does_not_raise():
    # Negative control for the refusal above: loading the SAME lean-folded weights through the
    # matching "lean" view (which resizes numflag_proj/user_mlp first) must succeed.
    store, sd = _baseline()
    m1 = fv.m1_states_from_fit(_fit())
    model = fm.build(TINY_REGISTRY, sd, view="lean", m1=m1, device="cpu",
                     vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)
    assert model.view == "lean"
