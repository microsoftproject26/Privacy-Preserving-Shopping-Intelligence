"""T40 support labels (E11.3) and the panel logic (E11.1a / E11.2; v4trunc.panel).

  * support: e_j in H_h(q), target observed, class >= 0, loss or eval mask, target_ts STRICTLY before decision_ts(q):
    the transition into e_q and the query target are never support; h = 1 may give 0 labels;
  * support batches: one example per label, the window ends at e_j (never e_q or later), loss_mask = the label
    indicator, target_ts < the cutoff, target = e_j's own target;
  * derive_panel (independent §11.2 rules) on real rows: structure checks, panel == events, FIXED queries identical
    across h, eligibility counts; sub-panel evaluation manifests; verify_panel runs the lock / registry checks.
"""
from __future__ import annotations

import numpy as np
import pytest

from trunckit import nc, synthetic_table
from v4trunc.panel import (PanelError, check_panel_against_events, check_panel_structure, eligibility, subpanel_manifest,
                           support_rows)
from v4trunc.provider import TS_KEY, support_batch, support_blocks


# ------------------------------------------------------------------------------------------------ E11.3 synthetic
def _user():
    # e0..e5; each event's target is the next event's time (so e4 -> e5 matures AT the query e5's time)
    return synthetic_table([[
        ("TRAIN", 1000, 0, 0, 1, 3, 3, 5.0, 7, False, True, 1100),
        ("TRAIN", 1100, 0, 0, 2, 3, 3, 5.0, -1, False, False, None),      # no observed target
        ("TRAIN", 1200, 0, 0, 3, 3, 3, 5.0, 9, False, True, 1300),
        ("VALIDATION_SELECT", 5000, 1, 1, 4, 3, 3, 5.0, 4, True, False, 5100),
        ("VALIDATION_SELECT", 5100, 1, 1, 5, 3, 3, 5.0, 6, True, False, 5200),
        ("VALIDATION_SELECT", 5200, 1, 1, 6, 3, 3, 5.0, 8, True, False, 5300)]])


def test_support_rule_synthetic():
    t = _user()
    q = 5
    assert support_rows(t, q, 0).size == 0
    assert support_rows(t, q, 1).size == 0                      # e4 -> e5: matures at decision_ts(q), not before
    assert support_rows(t, q, 2).tolist() == [3]
    assert support_rows(t, q, 5).tolist() == [0, 2, 3]          # e1 has no target; e4 matures at the query
    assert support_rows(t, q, None).tolist() == [0, 2, 3]
    assert np.all(t.meta["target_ts_us"][support_rows(t, q, 5)] < t.meta["decision_ts_us"][q])
    with pytest.raises(PanelError):
        support_rows(t, q, 6)


@nc("a support rule with <= at the query time admits the transition into e_q")
def test_nc_support_includes_transition_into_query():
    t = _user()
    q = 5
    perm = np.arange(q - 1, q)
    bad = perm[t.meta["target_ts_us"][perm] <= t.meta["decision_ts_us"][q]]      # the wrong (non-strict) rule
    assert bad.size == support_rows(t, q, 1).size


# ------------------------------------------------------------------------------------------------ real rows
def test_support_batches_are_causal_and_end_at_their_event(builder, sample_table, derived_panel):
    t = sample_table
    q = derived_panel.queries
    rows = np.flatnonzero((np.asarray(q["n_support_labels"]) > 0))[:60]
    n = 0
    for i in rows.tolist():
        qr = t.row_of(int(q["query_decision_id"][i]))
        h = int(q["h"][i])
        blocks, srows = support_blocks(builder, qr, h)
        bt = support_batch(builder, qr, h)
        assert bt["decision_id"].numpy().tolist() == t.decision_id[srows].tolist()
        assert bt["target_class"].numpy().tolist() == t.meta["target_class"][srows].tolist()
        assert bool(bt["loss_mask"].all())
        assert int(bt[TS_KEY].max()) < int(t.meta["decision_ts_us"][qr])
        for blk, j in zip(blocks, srows.tolist()):
            assert blk.rows[-1] == j and blk.rows[0] >= qr - h and qr not in blk.rows
        n += len(blocks)
    assert n > 50


def test_derived_panel_structure_and_events(sample_table, derived_panel):
    s = check_panel_structure(derived_panel)
    assert s and all(s.values()), s
    c = check_panel_against_events(derived_panel, sample_table)
    assert all(c.values()), c


def test_subpanel_manifests(derived_panel):
    pm = derived_panel
    subs = pm.subpanels()
    assert ("ALL_AVAILABLE", 0) in subs
    fixed_q = {}
    for p, h in subs:
        m, r = subpanel_manifest(pm, p, h)
        assert m.n == r.size and np.all(np.diff(m.decision_id) > 0) and bool(m.eval_mask.all())
        assert np.array_equal(m.rankable, m.target_class >= 0)
        assert np.unique(m.eval_user_key).size == m.n                    # one query per user
        if p != "ALL_AVAILABLE":
            fixed_q.setdefault(p, set()).add(tuple(m.decision_id.tolist()))
        mf, _ = subpanel_manifest(pm, p, h, full_history=True)
        assert np.array_equal(mf.decision_id, m.decision_id) and mf.manifest_id.endswith("::FULL")
    assert all(len(v) == 1 for v in fixed_q.values())                   # FIXED: the same q_u for every h


def test_eligibility_counts(derived_panel):
    e = eligibility(derived_panel)
    q = derived_panel.queries
    for k, v in e["subpanels"].items():
        p, h = k.rsplit("_h", 1)
        m = (q["panel_type"] == p) & (q["h"] == int(h))
        assert v["users"] == int(m.sum()) and v["support_labels"] == int(np.asarray(q["n_support_labels"])[m].sum())
    assert "never used to select" in e["note"]


@nc("a panel whose permitted window is shifted by one event must fail the panel-vs-events proof")
def test_nc_shifted_panel_detected(sample_table, derived_panel):
    import copy
    pm = copy.deepcopy(derived_panel)
    m = np.asarray(pm.queries["h"]) > 0
    pm.queries["permitted_first_decision_id"] = np.where(m, np.asarray(pm.queries["permitted_first_decision_id"]) + 7,
                                                         pm.queries["permitted_first_decision_id"])
    c = check_panel_against_events(pm, sample_table)
    assert all(c.values())


def test_verify_panel_runs_lock_and_registry_checks():
    """Locally the data_v3 panel tables are absent: verify_panel must get through the lock, registry and transform-state
    checks and refuse at the missing table (on the server the same call verifies the real panel)."""
    from dataclasses import replace
    from v4int.verify import V4, DataPaths
    from v4trunc.panel import verify_panel
    reg = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2.json"
    ts = V4 / "data_v2" / "server" / "TRANSFORM_STATE_CLEAN_V2.json"
    if not reg.is_file() or not ts.is_file():
        pytest.skip("registry / transform-state copy not present")
    paths = replace(DataPaths.default(), transform_state_path=ts)
    tab = V4 / "data_v3" / "manifests" / "WARM_TRUNCATION_16384.panel_users.parquet"
    if tab.is_file():
        pm = verify_panel("WARM_TRUNCATION_16384", reg, paths)
        assert pm.label == "WARM_TRUNCATION" and pm.record["checks_passed"] > 30
        return
    with pytest.raises(PanelError, match="missing from disk"):
        verify_panel("WARM_TRUNCATION_16384", reg, paths)
    with pytest.raises(PanelError, match="kind_panel|tables_are_panel"):
        verify_panel("MAIN_VALIDATION_5_100_131072", reg, paths)


# ------------------------------------------------------------------------------------------------ B-F2 re-derivation
def _later_query_panel(sample_table, monkeypatch):
    """A panel whose FIXED_20 q_u is the SECOND qualifying E2E VALIDATION decision (not the earliest) for every user
    that has one: every field and support row is internally consistent with the events, so only the §11.2
    earliest-q_u rule is violated."""
    import v4trunc.panel as P
    orig = P._user_queries

    def later(table, rows):
        qs, e50, e20 = orig(table, rows)
        cand = np.flatnonzero((table.split_code[rows] > 0) & table.meta["eval_mask"][rows])
        c20 = cand[cand >= P.FIXED_MIN["FIXED_20"]]
        if c20.size >= 2:
            qs = [(p, h, int(c20[1]) if p == "FIXED_20" else qi) for p, h, qi in qs]
        return qs, e50, e20
    monkeypatch.setattr(P, "_user_queries", later)
    pm = P.derive_panel(sample_table, label="WARM_TRUNCATION")
    monkeypatch.setattr(P, "_user_queries", orig)
    return pm


def test_panel_rederived_accepts_the_derived_panel(sample_table, derived_panel):
    from v4trunc.panel import check_panel_rederived
    c = check_panel_rederived(derived_panel, sample_table)
    assert c and all(v is True for v in c.values()), c


def test_panel_rederivation_catches_what_the_event_proof_misses(sample_table, derived_panel, monkeypatch):
    """REV-ANA-01 B-F2: a later-than-earliest q_u passes check_panel_against_events (every query / support row
    matches the events) but fails the re-derivation; so do a dropped sub-panel row, a flipped eligibility flag and a
    loaded non-panel user."""
    import copy
    from v4trunc.panel import check_panel_rederived
    later = _later_query_panel(sample_table, monkeypatch)
    moved = int((np.asarray(later.queries["query_decision_id"]) != np.asarray(derived_panel.queries["query_decision_id"])).sum())
    assert moved > 0
    assert all(check_panel_against_events(later, sample_table).values())
    assert check_panel_rederived(later, sample_table)["rederived.query_set_eq_panel"] is False
    drop = copy.deepcopy(derived_panel)
    keep = np.ones(len(drop.queries["h"]), dtype=bool)
    keep[int(np.flatnonzero(np.asarray(drop.queries["panel_type"]) == "FIXED_20")[0])] = False
    drop.queries = {k: np.asarray(v)[keep] for k, v in drop.queries.items()}
    assert check_panel_rederived(drop, sample_table)["rederived.query_set_eq_panel"] is False
    flip = copy.deepcopy(derived_panel)
    flip.users["fixed50_eligible"] = ~np.asarray(flip.users["fixed50_eligible"], dtype=bool)
    assert check_panel_rederived(flip, sample_table)["rederived.fixed_eligibility_eq_panel"] is False
    from trunckit import sample_users
    from v4trunc.panel import derive_panel
    sub = derive_panel(sample_users(sample_table, 40, seed=1), label="WARM_TRUNCATION")    # 40 of the loaded users
    c = check_panel_rederived(sub, sample_table)
    assert c["rederived.loaded_users_are_panel_users"] is False and c["rederived.query_set_eq_panel"] is False


@nc("a registered panel whose q_u is later than the earliest qualifying decision must fail the re-derivation")
def test_nc_later_query_panel_rederived(sample_table, monkeypatch):
    from v4trunc.panel import check_panel_rederived
    assert all(check_panel_rederived(_later_query_panel(sample_table, monkeypatch), sample_table).values())
