"""v4fs.importance: grouped permutation importance under the frozen T3 evaluator (CPU, synthetic provider).

  * the binding groups (L3: G1 … G12c; L3S: + G13, SCHEMA-01R1) are checked against the schema by name (a wrong name, an overlap, item_tokens or a missing
    channel is refused);
  * donor maps are deterministic per (permutation seed, group), permutations, and identical for every model;
  * the whole-column swap: every recipient row holds its ONE donor's whole [L] column for every channel of the group
    (value and flag / mask jointly), every other channel and every structural key is unchanged;
  * reproducible: the same model and seeds give the same importances bitwise; other permutation seeds differ;
  * a constant channel gives importance exactly 0 with CI [0, 0] (within noise): a decision-level constant, and a
    per-event channel constant over the whole [L] column under the binding WHOLE_COLUMN swap (or constant at the real
    positions, 0 on PAD, under LENGTH_MATCHED); MEASURED finding: under WHOLE_COLUMN a channel that is constant at
    the real positions but 0 on PAD (the Clean V2 PAD rule) is NOT invariant, because a shorter donor's PAD reaches the
    recipient's real positions (the artifact test below records it);
  * evaluation only: no gradient, no weight change, the module's train flag restored;
  * persistence round trip (FS_USERSTATS.npz) and the two-seed family combination.
Negative control (strict xfail): a cell-wise permutation (each position drawn independently) breaks the window.
"""
from __future__ import annotations

import numpy as np
import torch

from fl.numerics import state_digest
from fskit import L3, L3S, ConstantChannelProvider, assert_raises, eval_provider, model, nc
from v4fs import importance as I


def _stream(ep):
    return list(ep.batches(1024))


def test_group_names_are_checked_against_the_schema():
    G3 = I.groups_for(L3)
    assert list(G3) == [g for g in I.FS_GROUPS if g != "G13"] and list(I.groups_for(L3S)) == list(I.FS_GROUPS)
    I.check_groups(L3, G3)
    I.check_groups(L3S, I.groups_for(L3S))
    assert_raises(I.ImportanceError, I.check_groups, L3, I.FS_GROUPS)           # G13 is not an L3 input
    bad = dict(G3, G1=("category_token",))
    assert_raises(I.ImportanceError, I.check_groups, L3, bad)
    assert_raises(I.ImportanceError, I.check_groups, L3, dict(G3, G2=("brand_tokens", "category_tokens")))
    assert_raises(I.ImportanceError, I.check_groups, L3, dict(G3, G0=("item_tokens",)))
    assert_raises(I.ImportanceError, I.check_groups, L3S, dict(I.FS_GROUPS, G0=("session_offset_tokens",)))
    assert_raises(I.ImportanceError, I.groups_for, L3S, dict(I.FS_GROUPS, GX=("session_mode_share", "nope_x")))
    missing = {k: v for k, v in G3.items() if k != "G9"}
    assert_raises(I.ImportanceError, I.check_groups, L3, missing)
    assert I.check_groups(L3, missing, complete=False)


def test_donor_maps_are_deterministic_permutations():
    lengths = np.array([3, 5, 5, 8, 1, 3, 5, 8, 8, 2])
    a = I.donor_map(lengths, 0, "G5")
    assert np.array_equal(a, I.donor_map(lengths, 0, "G5")) and sorted(a) == list(range(10))
    assert not np.array_equal(a, I.donor_map(lengths, 1, "G5")) or not np.array_equal(a, I.donor_map(lengths, 0, "G6"))
    m = I.donor_map(lengths, 0, "G5", alignment="LENGTH_MATCHED")
    assert np.array_equal(lengths[m], lengths) and sorted(m) == list(range(10))
    d = I.swap_diagnostics(lengths, a)
    assert 0.0 <= d["rows_length_mismatch_share"] <= 1.0 and 0.0 <= d["real_cells_receiving_donor_pad_share"] <= 1.0


def test_whole_column_swap_keeps_every_window_consistent():
    ep = eval_provider()
    b = _stream(ep)[0]
    for g in ("G5", "G12b", "G1", "G4"):
        locs = I.locate(L3, I.FS_GROUPS[g])
        st = I.ChannelStore(locs)
        st.add(b)
        st.finish()
        donor = I.donor_map(st.lengths, 0, g)
        (pb,) = list(I.permuted_batches([b], st, locs, donor))
        I.window_consistency(b, pb, locs, donor, b)
        assert pb["lengths"] is b["lengths"] and torch.equal(pb["attention_mask"], b["attention_mask"])
    assert b["event_numeric_features"].shape[-1] == 15                    # the provider's batch is untouched


def _importance(bm, ep, **kw):
    return I.grouped_permutation_importance(bm.adapter, ep, L3, log=lambda *_: None, **kw)


def test_importance_is_reproducible_and_evaluation_only():
    ep = eval_provider()
    bm = model("GRU")
    bm.module.train()
    before = state_digest(bm.adapter.broadcast_state())
    a = _importance(bm, ep)
    assert bm.module.training, "the train flag must be restored"
    assert state_digest(bm.adapter.broadcast_state()) == before
    assert all(p.grad is None for p in bm.module.parameters()), "importance computed a gradient"
    b = _importance(bm, ep)
    assert [r["group"] for r in a.rows] == list(I.groups_for(L3))
    for x, y in zip(a.rows, b.rows):
        assert x["importance"] == y["importance"] and x["importance_ci95"] == y["importance_ci95"]
        assert [p["donor_map_sha256"] for p in x["per_perm_seed"]] == [p["donor_map_sha256"] for p in y["per_perm_seed"]]
        assert len(x["per_perm_seed"]) == 3 and x["within_noise"] == (x["importance_ci95"][0] <= 0 <= x["importance_ci95"][1])
    assert a.meta["n_evaluations"] == 1 + 14 * 3
    c = _importance(bm, ep, perm_seeds=(7, 8, 9))
    assert any(x["importance"] != z["importance"] for x, z in zip(a.rows, c.rows))
    s = model("SASREC")
    r = _importance(s, ep)
    assert r.baseline["value"] > 0 and len(r.rows) == 14


def test_a_constant_channel_has_importance_exactly_zero():
    groups = {"CONST_EVENT": ("log1p_price",), "CONST_USER": ("recent_mean_log_price",)}
    base = eval_provider()
    whole = ConstantChannelProvider(ConstantChannelProvider(base, "event_numeric_features", 0, 0.75, pad_too=True),
                                    "user_context", 8, -0.5)
    real = ConstantChannelProvider(ConstantChannelProvider(base, "event_numeric_features", 0, 0.75),
                                   "user_context", 8, -0.5)
    for fam in ("GRU", "SASREC"):
        bm = model(fam)
        for ep, alignment in ((whole, "WHOLE_COLUMN"), (real, "LENGTH_MATCHED")):
            r = _importance(bm, ep, groups=groups, complete=False, alignment=alignment)
            for row in r.rows:
                assert row["importance"] == 0.0 and row["importance_ci95"] == [0.0, 0.0], (fam, alignment, row["group"])
                assert row["within_noise"] and all(p["delta"] == 0.0 for p in row["per_perm_seed"])
    # control: the same group on the unmodified provider is not exactly zero for a varying channel
    r = _importance(model("GRU"), base, groups={"G5": I.FS_GROUPS["G5"]}, complete=False)
    assert r.rows[0]["importance"] != 0.0


def test_whole_column_pad_artifact_is_measured():
    """MEASURED (synthetic): constant at the real positions, 0 on PAD: WHOLE_COLUMN moves donor PAD onto real
    positions (nonzero importance for a channel with no information); LENGTH_MATCHED does not."""
    real = ConstantChannelProvider(eval_provider(), "event_numeric_features", 0, 0.75)
    groups = {"CONST_EVENT": ("log1p_price",)}
    bm = model("GRU")
    w = _importance(bm, real, groups=groups, complete=False, alignment="WHOLE_COLUMN").rows[0]
    m = _importance(bm, real, groups=groups, complete=False, alignment="LENGTH_MATCHED").rows[0]
    assert m["importance"] == 0.0
    assert w["importance"] != 0.0 and w["per_perm_seed"][0]["real_cells_receiving_donor_pad_share"] > 0


def test_userstats_round_trip_and_seed_combination(tmp_path):
    ep = eval_provider()
    per = {}
    for seed in (2026, 2027):
        r = _importance(model("GRU", seed=seed), ep)
        p = tmp_path / f"s{seed}.npz"
        I.save_stats(p, r.stats, {"seed": seed})
        st, meta = I.load_stats(p)
        assert meta == {"seed": seed} and sorted(st) == sorted(r.stats)
        assert all(np.array_equal(st[k].S, r.stats[k].S) and np.array_equal(st[k].N, r.stats[k].N) for k in st)
        per[seed] = st
        one = I.combine_seeds({seed: st}, I.groups_for(L3))
        assert all(abs(one[g]["importance"] - x["importance"]) < 1e-15 for g, x in zip(I.groups_for(L3), r.rows))
    fam = I.combine_seeds(per, I.groups_for(L3))
    for g, v in fam.items():
        assert abs(v["importance"] - np.mean([v["per_seed"][s]["importance"] for s in ("2026", "2027")])) < 1e-15
        assert v["within_noise"] == (v["ci95"][0] <= 0.0 <= v["ci95"][1])
    both = I.noise_groups({"GRU": fam, "SASREC": {g: dict(v, within_noise=(g in ("G1", "G8"))) for g, v in fam.items()}})
    assert set(both) <= {"G8"} and both == [g for g in I.FS_GROUPS if g in both]
    everything = {f: {g: dict(v, within_noise=True) for g, v in fam.items()} for f in ("GRU", "SASREC")}
    assert "G1" not in I.noise_groups(everything) and len(I.noise_groups(everything)) == 13   # L3: G2 ... G12c


def test_l3s_g13_swap_and_required_g1():
    """L3S (SCHEMA-01R1): 15 groups; G13's per-event token moves as a whole column and its decision-level values move
    with their mask under ONE donor map; session_offset_tokens / item_tokens never move; G1 is reported descriptively
    and never returned by noise_groups; 14 groups are droppable."""
    ep = eval_provider(session=True)
    b = _stream(ep)[0]
    locs = I.locate(L3S, I.FS_GROUPS["G13"])
    assert [l.per_event for l in locs] == [True, False, False, False]
    st = I.ChannelStore(locs)
    st.add(b)
    st.finish()
    donor = I.donor_map(st.lengths, 0, "G13")
    (pb,) = list(I.permuted_batches([b], st, locs, donor))
    I.window_consistency(b, pb, locs, donor, b)
    d = torch.as_tensor(donor)
    for k in ("current_matches_session_mode", "current_matches_session_mode_available", "session_mode_share"):
        assert torch.equal(pb[k], b[k][d]), k                              # value and mask: the same donor
    assert torch.equal(pb["session_offset_tokens"], b["session_offset_tokens"])
    assert torch.equal(pb["lengths"][d], b["lengths"][d]) and torch.equal(b["lengths"][d], b["lengths"])  # length-matched
    for fam in ("GRU", "SASREC"):
        r = I.grouped_permutation_importance(model(fam, L3S).adapter, ep, L3S, log=lambda *_: None)
        rows = {x["group"]: x for x in r.rows}
        assert list(rows) == list(I.FS_GROUPS) and r.meta["n_evaluations"] == 1 + 15 * 3
        assert rows["G1"]["descriptive_only"] and not rows["G1"]["droppable"]
        assert sum(x["droppable"] for x in r.rows) == 14 and rows["G13"]["droppable"]
    fake = {f: {g: {"importance": 0.0, "ci95": [-1.0, 1.0], "within_noise": True} for g in I.FS_GROUPS}
            for f in ("GRU", "SASREC")}
    ng = I.noise_groups(fake)
    assert "G1" not in ng and len(ng) == 14 and "G13" in ng


@nc("a cell-wise permutation (every position drawn independently) is not a whole-column swap")
def test_nc_cellwise_permutation_breaks_the_window():
    ep = eval_provider()
    b = _stream(ep)[0]
    locs = I.locate(L3, I.FS_GROUPS["G5"])
    donor = I.donor_map(b["lengths"].numpy(), 0, "G5")
    bad = dict(b)
    t = b["event_numeric_features"].clone()
    g = torch.Generator().manual_seed(0)
    for pos in range(t.shape[1]):                                          # a different donor per position
        t[:, pos, 0] = b["event_numeric_features"][torch.randperm(t.shape[0], generator=g), pos, 0]
    bad["event_numeric_features"] = t
    I.window_consistency(b, bad, locs, donor, b)
