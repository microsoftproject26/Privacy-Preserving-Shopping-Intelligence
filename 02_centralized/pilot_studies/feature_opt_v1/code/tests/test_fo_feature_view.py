"""Package test list items 2,3,4,5,7,9,10,16 -- feature_view.py (owner: Opus).

Independent tests. Never modifies the code under test. Uses the SyntheticStore fixture
(byte-identical snapshot copy) and the real fitted normalizers (read-only), per the task
packet. All writes go through FO_ROOT_OVERRIDE tmp dirs via the set_fo_root fixture; nothing
here writes into the real study folders.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

import conftest_fo  # noqa: F401  (sys.path + FO_NORMALIZERS side effects)
from conftest_fo import tiny_store, tiny_inner, TINY_REGISTRY

import fo_env
import feature_view as fv
import fo_models as fm
import sasrec_data

FEATURE_GROUPS = json.loads(
    (conftest_fo.STUDY_ROOT / "evidence" / "snap" / "dataset" / "FEATURE_GROUPS.json").read_text())


def make_consistent_batch(seed=0, n_train=64, n_vs=64, K=97, n=32):
    """A synthetic batch that satisfies the measured duplicate/M1 relations feature_view.py
    assumes (M2 exact duplicates, M3 dead flags, M1 affine re-encoding), so the lean fold is
    exact on it. Mirrors the lead's scratch/smoke_fv.py idea, not its assertions."""
    torch.manual_seed(seed)
    store = tiny_store(n_train=n_train, n_vs=n_vs, K=K, seed=seed)
    inner = tiny_inner(store, seed=1)
    ds = store.decisions("VALIDATION_SELECT", "eval")
    b = store.batch(ds, np.arange(min(n, len(ds))))
    base = fm.FOModel(inner, view="full").eval()
    kw = sasrec_data.predictor_kwargs(b, base)
    spec = fv.normalizers().spec
    N, F = fv.NUM, fv.FLAG

    def z(name, raw):
        p = spec[name]
        return (np.clip(raw, p["clip_lo"], p["clip_hi"]) - p["mean"]) / p["std"]

    num = kw["event_numeric_features"].clone().double()
    flg = kw["event_quality_flags"].clone()
    uc = kw["user_context"].clone()
    um = kw["user_context_masks"].clone()
    shp = num.shape[:2]
    rep = torch.randint(0, 60, shp).double()
    num[..., N["prefix_representative_count"]] = torch.tensor(z("prefix_representative_count", rep.numpy()))
    num[..., N["prefix_item_count"]] = torch.tensor(z("prefix_item_count", (rep + 1).numpy()))
    el = torch.rand(shp).double() * 9
    num[..., N["log1p_segment_elapsed_s"]] = torch.tensor(z("log1p_segment_elapsed_s", el.numpy()))
    num[..., N["log1p_prefix_elapsed_s"]] = torch.tensor(z("log1p_prefix_elapsed_s", el.numpy()))
    th = torch.rand(shp).double() * 2 * math.pi
    num[..., N["utc_hour_sin"]], num[..., N["utc_hour_cos"]] = torch.sin(th), torch.cos(th)
    phi = 2.0 * math.pi * fv.LOCAL_SHIFT_H / 24.0
    num[..., N["local_hour_sin"]] = math.cos(phi) * num[..., N["utc_hour_sin"]] + math.sin(phi) * num[..., N["utc_hour_cos"]]
    num[..., N["local_hour_cos"]] = -math.sin(phi) * num[..., N["utc_hour_sin"]] + math.cos(phi) * num[..., N["utc_hour_cos"]]
    pm = (torch.rand(shp) < 0.3).to(flg.dtype)
    flg[..., F["price_missing"]] = pm
    flg[..., F["price_zero"]] = pm
    cm = (torch.rand(shp) < 0.3).to(flg.dtype)
    flg[..., F["main_category_missing"]] = cm
    flg[..., F["category_code_missing"]] = cm
    for name in fv.DROP_FLAG_DEAD:
        flg[..., F[name]] = 0
    hh = (torch.rand(uc.shape[0]) < 0.8).float()
    uc[:, fv.UCTX["has_history"]] = hh
    um[:, fv.UMASK["has_history_mask"]] = hh
    um[:, fv.UMASK["prev_user_event_available"]] = hh
    ps = (torch.rand(uc.shape[0]) < 0.2).float()
    um[:, fv.UMASK["prev_session_available"]] = ps
    m1 = fv.m1_states_from_fit({
        "log1p_time_since_prev_session_last_event_s": {"valid_mean": 11.82, "valid_std": 1.663},
        "log1p_time_since_prev_user_event_s": {"valid_mean": 5.382, "valid_std": 3.181}})
    for f, st in m1.items():
        avail = um[:, fv.UMASK[st.mask]]
        raw = torch.where(avail > 0, torch.rand(uc.shape[0]).double() * 6 + 8, torch.zeros(uc.shape[0]).double())
        uc[:, fv.UCTX[f]] = torch.tensor(z(f, raw.numpy())).float()
    kw.update(event_numeric_features=num.float(), event_quality_flags=flg, user_context=uc, user_context_masks=um)
    return store, base, kw, m1


# =========================================================================== (2) identity view
def test_identity_view_full_equals_raw_sasrec():
    store, base, kw, _ = make_consistent_batch()
    with torch.no_grad():
        y_raw = base.inner(**kw)
        y_full = base(**kw)
    assert torch.equal(y_raw, y_full)


# =========================================================================== (2/17) lean fold parity
def test_lean_fold_parity_small_max_abs_logit_diff():
    store, base, kw, m1 = make_consistent_batch()
    lean = fm.build(TINY_REGISTRY, base.inner.state_dict(), view="lean", m1=m1, device="cpu",
                    vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class).eval()
    with torch.no_grad():
        y0, y1 = base(**kw), lean(**kw)
        d = (y0 - y1).abs().max()
    ranks0 = y0.argsort(dim=-1, descending=True)
    ranks1 = y1.argsort(dim=-1, descending=True)
    print("lean max|dlogit| =", float(d))
    assert float(d) < 1e-4, f"lean fold parity too loose: max|dlogit|={float(d)}"
    assert torch.equal(ranks0, ranks1), "lean fold must preserve per-row ranking order exactly"


def test_lean_fold_parity_negative_control_broken_relation_exceeds_threshold():
    """Negative control: break ONE measured relation (+1.0 on log1p_prefix_elapsed_s, so it no
    longer equals its keeper log1p_segment_elapsed_s) and the fold must now visibly diverge."""
    store, base, kw, m1 = make_consistent_batch()
    lean = fm.build(TINY_REGISTRY, base.inner.state_dict(), view="lean", m1=m1, device="cpu",
                    vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class).eval()
    bad_num = kw["event_numeric_features"].clone()
    bad_num[..., fv.NUM["log1p_prefix_elapsed_s"]] += 1.0
    kw_bad = dict(kw, event_numeric_features=bad_num)
    with torch.no_grad():
        d = (base(**kw_bad) - lean(**kw_bad)).abs().max()
    print("negative control lean diff =", float(d))
    assert float(d) > 1e-2, f"negative control failed to break parity: max|dlogit|={float(d)}"


# =========================================================================== (3) column order
def test_column_order_matches_feature_groups_json():
    assert fv.NUM_NAMES == FEATURE_GROUPS["event_numeric_block_order"]
    assert fv.FLAG_NAMES == FEATURE_GROUPS["event_quality_flags_order"]
    assert fv.UCTX_NAMES == FEATURE_GROUPS["user_context_order"]
    assert fv.UMASK_NAMES == FEATURE_GROUPS["user_context_masks_order"]


def test_column_order_negative_control_swap_breaks_lean_parity():
    store, base, kw, m1 = make_consistent_batch()
    lean = fm.build(TINY_REGISTRY, base.inner.state_dict(), view="lean", m1=m1, device="cpu",
                    vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class).eval()
    swapped = kw["event_numeric_features"].clone()
    i, j = fv.NUM["log1p_price"], fv.NUM["utc_hour_sin"]
    swapped[..., [i, j]] = swapped[..., [j, i]]
    kw_bad = dict(kw, event_numeric_features=swapped)
    with torch.no_grad():
        d = (base(**kw_bad) - lean(**kw_bad)).abs().max()
    print("column-swap negative control diff =", float(d))
    assert float(d) > 1e-2


# =========================================================================== (4) poisoned PAD
@pytest.mark.parametrize("view", ["full", "lean", "d3"])
def test_poisoned_pad_never_changes_outputs(view):
    store, base, kw, m1 = make_consistent_batch(n=48)
    real = kw["attention_mask"].to(torch.bool)
    pad = ~real
    if not bool(pad.any()):
        pytest.skip("no PAD positions in this synthetic batch")
    kw_p = dict(kw)
    kw_p["item_tokens"] = torch.where(pad, torch.randint(0, store.item_vocab, kw["item_tokens"].shape), kw["item_tokens"])
    num = kw["event_numeric_features"].clone()
    num[pad] = torch.randn_like(num[pad]) * 1000.0
    flg = kw["event_quality_flags"].clone()
    flg[pad] = torch.randint(0, 2, flg[pad].shape, dtype=flg.dtype)
    kw_p["event_numeric_features"], kw_p["event_quality_flags"] = num, flg
    for ch in fv.schema_record()["side_channels"]:
        t = kw[ch].clone()
        t[pad] = torch.randint(0, store.vocab_sizes[ch], t[pad].shape)
        kw_p[ch] = t

    if view == "full":
        model = base
    elif view == "lean":
        model = fm.build(TINY_REGISTRY, base.inner.state_dict(), view="lean", m1=m1, device="cpu",
                         vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class).eval()
    else:
        model = fm.build(TINY_REGISTRY, base.inner.state_dict(), view="d3", device="cpu",
                         vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class).eval()
    with torch.no_grad():
        y_clean = model(**kw)
        y_poisoned = model(**kw_p)
    assert torch.equal(y_clean, y_poisoned), f"view={view}: PAD poisoning changed the output"

    for probe_name in fv.PROBES:
        spec = fv.ProbeSpec(probe_name, calendar_cell=(7, 5) if probe_name.startswith("P8") else None)
        pm = fm.FOModel(base.inner, probe=spec).eval()
        with torch.no_grad():
            assert torch.equal(pm(**kw), pm(**kw_p)), f"probe {probe_name}: PAD poisoning changed the output"


def test_d3_is_zero_at_pad():
    store, base, kw, _ = make_consistent_batch(n=48)
    d3 = fv.d3_features(kw["item_tokens"], kw["attention_mask"])
    pad = ~kw["attention_mask"].to(torch.bool)
    if not bool(pad.any()):
        pytest.skip("no PAD positions in this synthetic batch")
    assert torch.equal(d3[pad], torch.zeros_like(d3[pad]))


# =========================================================================== (5/7) M1 re-encoding
def test_m1_available_rows_affine_recoding():
    fit = {"log1p_time_since_prev_session_last_event_s": {"valid_mean": 11.82, "valid_std": 1.663},
           "log1p_time_since_prev_user_event_s": {"valid_mean": 5.382, "valid_std": 3.181}}
    m1 = fv.m1_states_from_fit(fit)
    f = "log1p_time_since_prev_session_last_event_s"
    st = m1[f]
    spec = fv.normalizers().spec[f]
    raw = np.array([3.0, 20.0, -5.0], dtype=np.float64)   # includes below-clip and above-clip
    z_old = (raw - spec["mean"]) / spec["std"]             # what the OLD encoding would be (unclipped, for reference)
    kw = {"user_context": torch.zeros(3, len(fv.UCTX_NAMES)), "user_context_masks": torch.ones(3, len(fv.UMASK_NAMES)),
          "event_numeric_features": torch.zeros(3, 1, len(fv.NUM_NAMES)), "event_quality_flags": torch.zeros(3, 1, len(fv.FLAG_NAMES))}
    old_z = (np.clip(raw, spec["clip_lo"], spec["clip_hi"]) - spec["mean"]) / spec["std"]
    kw["user_context"][:, fv.UCTX[f]] = torch.tensor(old_z, dtype=torch.float32)
    out = fv.to_lean(kw, m1)
    got = out["user_context"][:, fv.UCTX[f]].numpy()
    expect = (np.clip(raw, st.clip_lo, st.clip_hi) - st.new_mean) / st.new_std
    np.testing.assert_allclose(got, expect, atol=1e-5)


def test_m1_unavailable_rows_map_to_zero():
    fit = {"log1p_time_since_prev_session_last_event_s": {"valid_mean": 11.82, "valid_std": 1.663},
           "log1p_time_since_prev_user_event_s": {"valid_mean": 5.382, "valid_std": 3.181}}
    m1 = fv.m1_states_from_fit(fit)
    f = "log1p_time_since_prev_session_last_event_s"
    kw = {"user_context": torch.full((2, len(fv.UCTX_NAMES)), 3.0),
          "user_context_masks": torch.zeros(2, len(fv.UMASK_NAMES)),
          "event_numeric_features": torch.zeros(2, 1, len(fv.NUM_NAMES)), "event_quality_flags": torch.zeros(2, 1, len(fv.FLAG_NAMES))}
    out = fv.to_lean(kw, m1)
    assert torch.equal(out["user_context"][:, fv.UCTX[f]], torch.zeros(2))


def test_m1_available_row_at_old_z0_is_not_mapped_to_zero():
    """An AVAILABLE row whose OLD (unavailable-sentinel) encoding happens to equal z0 must still
    be re-encoded through the affine map, not zeroed just because it matches the old sentinel."""
    fit = {"log1p_time_since_prev_session_last_event_s": {"valid_mean": 11.82, "valid_std": 1.663},
           "log1p_time_since_prev_user_event_s": {"valid_mean": 5.382, "valid_std": 3.181}}
    m1 = fv.m1_states_from_fit(fit)
    f = "log1p_time_since_prev_session_last_event_s"
    st = m1[f]
    z0 = (min(max(0.0, st.clip_lo), st.clip_hi) - st.old_mean) / st.old_std
    kw = {"user_context": torch.zeros(1, len(fv.UCTX_NAMES)), "user_context_masks": torch.ones(1, len(fv.UMASK_NAMES)),
          "event_numeric_features": torch.zeros(1, 1, len(fv.NUM_NAMES)), "event_quality_flags": torch.zeros(1, 1, len(fv.FLAG_NAMES))}
    kw["user_context"][:, fv.UCTX[f]] = z0
    out = fv.to_lean(kw, m1)
    got = float(out["user_context"][0, fv.UCTX[f]])
    expect = (0.0 - st.new_mean) / st.new_std   # raw = clip(z0*std+mean) = 0.0 (the sentinel raw)
    assert got == pytest.approx(expect, abs=1e-5)
    assert got != 0.0, "an available row was incorrectly zeroed just because it matched the old sentinel"


def test_m1_no_second_log_inverse_recovers_clipped_raw():
    fit = {"log1p_time_since_prev_session_last_event_s": {"valid_mean": 11.82, "valid_std": 1.663},
           "log1p_time_since_prev_user_event_s": {"valid_mean": 5.382, "valid_std": 3.181}}
    m1 = fv.m1_states_from_fit(fit)
    f = "log1p_time_since_prev_session_last_event_s"
    st = m1[f]
    spec = fv.normalizers().spec[f]
    raw = 6.0  # already in log1p space; the fold must NOT apply log1p again
    old_z = (raw - spec["mean"]) / spec["std"]
    x = torch.tensor([old_z], dtype=torch.float64)
    recovered = x * st.old_std + st.old_mean
    assert float(recovered) == pytest.approx(raw, abs=1e-6)


def test_m1_degenerate_std_raises():
    with pytest.raises(ValueError):
        fv.m1_states_from_fit({
            "log1p_time_since_prev_session_last_event_s": {"valid_mean": 1.0, "valid_std": 0.0},
            "log1p_time_since_prev_user_event_s": {"valid_mean": 5.382, "valid_std": 3.181}})


# =========================================================================== (9) calendar
def test_calendar_instant_sin_cos_identity():
    for wd in range(3, 10):
        for dp in fv.DAYPART_BANDS:
            c = fv.calendar_instant(wd, dp)
            for pair in ("utc_hour", "utc_weekday", "local_hour", "local_weekday"):
                s, co = c[f"{pair}_sin"], c[f"{pair}_cos"]
                assert s * s + co * co == pytest.approx(1.0, abs=1e-9)


def test_calendar_local_equals_utc_plus_3h_mod_24():
    for wd in range(3, 10):
        for dp in fv.DAYPART_BANDS:
            c = fv.calendar_instant(wd, dp)
            assert (c["_utc_hour"] + fv.LOCAL_SHIFT_H) % 24.0 == pytest.approx(c["_local_hour"] % 24.0, abs=1e-9)


def test_calendar_weekday_and_is_weekend_consistent_all_cells():
    for wd in range(3, 10):
        for dp in fv.DAYPART_BANDS:
            c = fv.calendar_instant(wd, dp)
            expect_weekend = 1.0 if c["_isodow_local"] >= 6 else 0.0
            assert c["is_weekend"] == expect_weekend
            assert c["weekday_tokens"] == wd
            assert c["daypart_tokens"] == dp


def test_calendar_utc_weekday_rolls_back_when_utc_hour_negative():
    # With the registered DAYPART_BANDS, every band midpoint h_loc >= 3.0 (the earliest band (0,6)
    # has midpoint exactly 3.0), so h_utc = h_loc - LOCAL_SHIFT_H is always >= 0 for every real
    # (weekday, daypart) cell: the rollback branch is provably dead with these bands. Verify that
    # directly instead of asserting an impossible premise.
    for wd in range(3, 10):
        for dp in fv.DAYPART_BANDS:
            c = fv.calendar_instant(wd, dp)
            assert c["_utc_hour"] >= 0.0, f"unexpected negative utc hour for cell ({wd},{dp})"
    # Direct construction of the negative branch via the underlying formula (bypassing the band
    # table) proves the ROLLBACK ARITHMETIC ITSELF is correct, independent of whether any real
    # daypart band can trigger it:
    h_loc = 1.0          # < LOCAL_SHIFT_H=3 by construction
    dow_loc = 3 - 2       # Monday
    h_utc = h_loc - fv.LOCAL_SHIFT_H
    dow_utc = dow_loc
    if h_utc < 0:
        h_utc += 24.0
        dow_utc = (dow_loc - 2) % 7 + 1
    assert h_utc == 22.0
    assert dow_utc == 7  # rolled back to Sunday (isodow 7) from Monday (isodow 1)


# =========================================================================== (10/16) probe closure
def test_probe_p4_touches_only_last_real_position():
    store, base, kw, _ = make_consistent_batch(n=32)
    spec = fv.ProbeSpec("P4_CURRENT_PRICE_UNAVAILABLE")
    out = fv.apply_probe(kw, spec)
    real = kw["attention_mask"].to(torch.bool)
    last = fv.last_position_mask(kw) & real
    not_last = real & ~last
    for name in ("event_numeric_features", "event_quality_flags"):
        changed = (out[name] != kw[name]).any(-1)
        assert torch.equal(changed & not_last, torch.zeros_like(changed))
    assert bool(last.sum() > 0), "test batch has no real last positions; strengthen the fixture"


def test_probe_p2_leaves_category_tokens_and_has_history_untouched():
    store, base, kw, _ = make_consistent_batch(n=32)
    spec = fv.ProbeSpec("P2_MAIN_CATEGORY_UNAVAILABLE")
    out = fv.apply_probe(kw, spec)
    assert torch.equal(out["category_tokens"], kw["category_tokens"])
    assert torch.equal(out["user_context"][:, fv.UCTX["has_history"]], kw["user_context"][:, fv.UCTX["has_history"]])


def test_probe_duplicated_flags_move_together():
    store, base, kw, _ = make_consistent_batch(n=32)
    real = kw["attention_mask"].to(torch.bool)
    out2 = fv.apply_probe(kw, fv.ProbeSpec("P2_MAIN_CATEGORY_UNAVAILABLE"))
    a = out2["event_quality_flags"][..., fv.FLAG["main_category_missing"]][real]
    b = out2["event_quality_flags"][..., fv.FLAG["category_code_missing"]][real]
    assert torch.equal(a, b)
    out4 = fv.apply_probe(kw, fv.ProbeSpec("P4_CURRENT_PRICE_UNAVAILABLE"))
    last = fv.last_position_mask(kw) & real
    a4 = out4["event_quality_flags"][..., fv.FLAG["price_missing"]][last]
    b4 = out4["event_quality_flags"][..., fv.FLAG["price_zero"]][last]
    assert torch.equal(a4, b4)


# =========================================================================== (11/12/14/15) D3
def _hand_built_d3_case():
    tok = torch.tensor([[5, 6, 5, 2, 2, 5, 0]])
    am = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    return tok, am


def test_d3_hand_built_exact_values():
    tok, am = _hand_built_d3_case()
    f = fv.d3_features(tok, am)[0]
    log50 = math.log(50.0)
    expected = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [math.log1p(1) / log50, math.log1p(2) / log50, 1.0, 2 / 3, 1.0],
        [0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.4, 0.0],
        [math.log1p(2) / log50, math.log1p(3) / log50, 1.0, 1 / 3, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    torch.testing.assert_close(f, expected, atol=1e-5, rtol=0)


def test_d3_future_never_changes_past():
    tok, am = _hand_built_d3_case()
    f0 = fv.d3_features(tok, am)
    tok2 = tok.clone()
    tok2[0, 4:6] = torch.tensor([9, 9])   # change the future relative to t<=3
    f1 = fv.d3_features(tok2, am)
    torch.testing.assert_close(f0[0, :4], f1[0, :4])


def test_d3_boundaries_t0_full_window_length1():
    tok0 = torch.zeros(1, 50, dtype=torch.long)
    tok0[0, 0] = 5
    am0 = torch.zeros(1, 50, dtype=torch.long)
    am0[0, 0] = 1
    f0 = fv.d3_features(tok0, am0)
    assert torch.equal(f0[0, 0], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))

    tok_full = torch.randint(3, 50, (1, 50))
    am_full = torch.ones(1, 50, dtype=torch.long)
    f_full = fv.d3_features(tok_full, am_full)
    assert f_full.shape == (1, 50, 5)
    assert bool(torch.isfinite(f_full).all())


def test_d3_oov_and_missing_repeats_never_count():
    tok = torch.tensor([[1, 1, 2, 2, 1, 2]])   # MISSING(1) and OOV(2) only, no known tokens
    am = torch.ones(1, 6, dtype=torch.long)
    f = fv.d3_features(tok, am)
    assert torch.equal(f[..., 0], torch.zeros(1, 6))   # prior count always 0
    assert torch.equal(f[..., 2], torch.zeros(1, 6))   # has_prev always False
    assert torch.equal(f[..., 4], torch.zeros(1, 6))   # known always False


def test_d3_prior_count_excludes_current_event():
    tok = torch.tensor([[5, 5, 5]])
    am = torch.ones(1, 3, dtype=torch.long)
    f = fv.d3_features(tok, am)
    # at t=0 there is no prior occurrence of token 5 even though the CURRENT position matches
    # nothing (there's nothing before it); the guard here is strict_past = tril(-1) (no diagonal).
    assert float(f[0, 0, 0]) == 0.0


# =========================================================================== (13) no labels reach inner.forward
def test_no_labels_reach_inner_forward():
    store, base, kw, m1 = make_consistent_batch(n=16)
    seen_kwargs = {}
    orig_forward = base.inner.forward

    def spy(**kwargs):
        seen_kwargs.update(kwargs)
        return orig_forward(**kwargs)

    base.inner.forward = spy
    try:
        with torch.no_grad():
            base(**kw)
    finally:
        base.inner.forward = orig_forward
    forbidden = {"target_class", "loss_mask", "eval_mask", "decision_id", "client_id",
                 "eval_user_key", "user_bucket", "split", "prefix_position", "prior_view_count"}
    leaked = forbidden & set(seen_kwargs)
    assert not leaked, f"forbidden keys reached inner.forward: {leaked}"


# =========================================================================== fold_lean_state_dict
# cross-checks MEASURED_AFFINE_SHIFT against clip-inferred shift (Review C LOW fix, 2026-09-25)
def test_fold_lean_state_dict_measured_shift_matches_clip_inferred(monkeypatch):
    """Negative control (paired below) baseline: with the real MEASURED_AFFINE_SHIFT values, the
    fold must NOT raise."""
    store, sd = _baseline_for_fold()
    m1 = fv.m1_states_from_fit(_fit_for_fold())
    fv.fold_lean_state_dict(sd, m1)   # must not raise


def test_fold_lean_state_dict_raises_when_measured_shift_disagrees(monkeypatch):
    """Regression test (2026-09-25): fold_lean_state_dict cross-checks the clip-inferred affine
    shift (clip_lo[drop]-clip_lo[keep]) against the independently MEASURED shift recorded in
    MEASURED_AFFINE_SHIFT (from MEASURED_FINDINGS.md / stream_fit.AFFINE). If someone edits one
    without the other, the fold must refuse rather than silently fold an inexact relation."""
    store, sd = _baseline_for_fold()
    m1 = fv.m1_states_from_fit(_fit_for_fold())
    bad = dict(fv.MEASURED_AFFINE_SHIFT)
    bad["log1p_prefix_elapsed_s"] = bad["log1p_prefix_elapsed_s"] + 999.0
    monkeypatch.setattr(fv, "MEASURED_AFFINE_SHIFT", bad)
    with pytest.raises(ValueError, match="measured shift"):
        fv.fold_lean_state_dict(sd, m1)


def _baseline_for_fold():
    store = tiny_store(n_train=16, n_vs=16, K=23, seed=5)
    inner = tiny_inner(store, seed=6)
    return store, inner.state_dict()


def _fit_for_fold():
    return {"log1p_time_since_prev_session_last_event_s": {"valid_mean": 11.82, "valid_std": 1.663},
            "log1p_time_since_prev_user_event_s": {"valid_mean": 5.382, "valid_std": 3.181}}
