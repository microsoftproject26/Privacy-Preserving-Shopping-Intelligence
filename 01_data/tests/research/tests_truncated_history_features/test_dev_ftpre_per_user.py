"""DEV-FTPRE item 2 — the per-user producer of the 06L-C §C10 persistence rule in v4int.launch_trunc (CPU, synthetic
manifests and tiny models only; no real data).

Proves: TRUNC_PER_USER.npz is exactly the PPSI_V4_TRUNC_PER_USER_1 format tools/retention_report.py reads
(load_trunc_per_user; keys = retention_report.trunc_cell_key of the cells' primary rows); each cell's per-user mean
reproduces its macro-E2E MRR@20 row value (Study A through study_a_cell, Study B through launch_device.execute with the
selected device cell); the file is deterministic (byte-identical on a rerun, sha256 recorded); a Study A cell file round
trips (and a changed / unrecorded one is recomputed); with the evidence, retention_report.trunc_curves computes the CIs and
the persistence rule reaches a persistent hit where the points persist, without it the result stays NO_CI.
NCs: the persistence rule without per-user evidence gives a hit; per-user values over the RANKABLE rows reproduce the E2E
row value.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import v4int.launch_trunc as T
from trunckit import assert_raises, nc
from v4int.launch_central import LaunchRefused
import ppsi_paths

TOOLS = ppsi_paths.locate("tools/retention_report.py").parent
K, L, PANEL = 24, 6, "COLD_H_16384"
KEY = {"data_protocol_hash": "d", "feature_contract_hash": "f", "split": "VALIDATION", "label_policy_hash": "l",
       "population_hash": "p", "candidate_order_hash": "c", "initialization_block": "i", "topology_hash": "t",
       "objective_id": "o", "precision_id": "fp32", "evaluator_code_sha256": "e", "family": "SASREC"}
HS = (0, 1, 5, 10, 20)                                            # the registered FIXED_20 budgets


@pytest.fixture(scope="module")
def R():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import retention_report as RR
    RR.bind(ppsi_paths.code_dir("v4train").parent)                # the code folder that holds v4train
    return RR


def _frozen():
    from v4eval import frozen_config as fc
    fz = fc.load()
    fc.apply_runtime_flags(fz)
    return fz


def _tiny(seed):
    from fl.synthetic import make_tiny_adapter
    return make_tiny_adapter(K=K, seed=seed, tied=False)


def _world(n=8, seed=5, per=5):
    """n synthetic users with support examples and a manifest with censored and OOV (E2E, not rankable) rows."""
    from fl.synthetic import client_examples, make_clients
    from v4eval.manifest import build_eval_manifest
    rng = np.random.default_rng(seed)
    clients = make_clients(n, K=K, L=L, seed=seed)
    ev = []
    for u in range(n):
        e = client_examples(per, K, L, rng)
        ev += [(u, {k: v[i] for k, v in e.items()}) for i in range(per)]
    ev = [ev[i] for i in rng.permutation(len(ev))]
    did = np.arange(len(ev), dtype=np.int64) * 3 + 7
    cen = np.zeros(len(ev), dtype=bool)
    cen[::7] = True
    tgt = np.array([int(x["target_class"]) for _, x in ev], dtype=np.int64)
    tgt[cen] = -1
    oov = np.zeros(len(ev), dtype=bool)
    oov[[3, 5, 11]] = ~cen[[3, 5, 11]]
    tgt[oov] = -1
    euk = np.array([u * 10 + 3 for u, _ in ev], dtype=np.int64)
    cols = {"decision_id": did, "eval_user_key": euk, "eval_mask": ~cen, "target_class": tgt, "target_oov": oov,
            "censor_class": np.where(cen, "CEN_NO_NEXT", "NONE").astype(object)}
    m = build_eval_manifest({"catalogue_K": K, "manifest_id": "SYN_TRUNC"}, cols)
    feats = {k: torch.stack([x[k] for _, x in ev]) for k in ("item_tokens", "attention_mask", "lengths")}
    feats["decision_id"] = torch.as_tensor(did)
    feats["target_class"] = torch.as_tensor(tgt)
    stream = np.flatnonzero(m.eval_mask)

    def get_eval_batch(sp):
        return {k: v[torch.as_tensor(stream[np.asarray(sp)])] for k, v in feats.items()}

    def batches(bs):
        for s in range(0, stream.size, bs):
            yield get_eval_batch(np.arange(s, min(stream.size, s + bs)))
    keys = [c.key for c in clients]
    sup = {}
    for c in clients:
        ex = dict(c.examples)
        ex["target_ts"] = torch.arange(ex["target_class"].shape[0], dtype=torch.long)
        sup[c.key] = ex
    row_key = np.array([keys[(k - 3) // 10] for k in euk], dtype=object)
    return SimpleNamespace(m=m, keys=keys, sup=sup, row_key=row_key, get_eval_batch=get_eval_batch, batches=batches)


def _study_a(w, arm, seed_model, ptype, h, full):
    prov = SimpleNamespace(manifest=w.m, batches=w.batches)
    lab = T.trunc_labels(study="A", panel_id=PANEL, label="COLD_H", panel_type=ptype, h=h, full=full)
    res, rows, art = T.study_a_cell(_tiny(seed_model), prov, _frozen(), evaluator_sha="e", key_base=KEY, labels=lab,
                                    arm=arm, seed=2026, model_sha256="m", overrides={"trunc_population_hash": "p"})
    return rows, art


def _series(w):
    """Study A rows + per-user entries: C_BIG (model 1), C_SMALL (the SAME weights: R = 100 %), FA (model 2) on every
    FIXED_20 budget and the FULL cell."""
    rows, cells = [], {}
    for arm, sm in (("C_BIG", 1), ("C_SMALL", 1), ("FA", 2)):
        for h, full in [(h, False) for h in HS] + [(20, True)]:
            r, art = _study_a(w, arm, sm, "FIXED_20", h, full)
            rows += r
            T.add_per_user(cells, T.per_user_entry(r, w.m, art.ranks))
    return rows, cells


@pytest.fixture(scope="module")
def series():
    w = _world()
    rows, cells = _series(w)
    return w, rows, cells


# ------------------------------------------------------------------------------------------------ format / values
def test_per_user_file_is_the_report_format(series, R, tmp_path):
    w, rows, cells = series
    assert len(cells) == 18 and T.TRUNC_PER_USER_FORMAT == R.TRUNC_PER_USER_FORMAT
    rec = T.write_trunc_per_user(tmp_path / "a", cells)
    p = tmp_path / "a" / T.TRUNC_PER_USER_NAME
    assert rec["sha256"] == __import__("hashlib").sha256(p.read_bytes()).hexdigest() and rec["n_cells"] == 18
    pu = R.load_trunc_per_user(p)
    assert sorted(pu) == sorted(cells) == rec["cells"]
    for r in T.primary_rows(rows):
        key = R.trunc_cell_key(r["method"], r["seed"], r["trunc_panel"], r["trunc_panel_type"], r["trunc_query_h"],
                               r["trunc_full_history"])
        assert key == T.trunc_cell_key(r["method"], r["seed"], r["trunc_panel"], r["trunc_panel_type"],
                                       r["trunc_query_h"], r["trunc_full_history"])
        u, m = pu[key]
        assert abs(float(m.mean()) - float(r["value"])) <= R.ROW_VALUE_TOL
        assert np.array_equal(u, np.unique(w.m.eval_user_key[w.m.eval_mask]))
    rec2 = T.write_trunc_per_user(tmp_path / "b", cells)                 # deterministic bytes
    assert rec2["sha256"] == rec["sha256"] and (tmp_path / "b" / T.TRUNC_PER_USER_NAME).read_bytes() == p.read_bytes()


def test_cell_file_round_trip_and_refusals(series, tmp_path):
    w, rows, cells = series
    key = sorted(cells)[0]
    ent = (key, *cells[key])
    p = tmp_path / "c.per_user.npz"
    sha = T.write_cell_per_user(p, ent)
    found, got = T.load_cell_per_user(p, sha)
    assert found and got[0] == key and np.array_equal(got[1], ent[1]) and np.array_equal(got[2], ent[2])
    assert T.load_cell_per_user(p, "0" * 64) == (False, None) and T.load_cell_per_user(p, None) == (False, None)
    sha0 = T.write_cell_per_user(tmp_path / "e.npz", None)
    assert T.load_cell_per_user(tmp_path / "e.npz", sha0) == (True, None)
    r, art = _study_a(w, "FA", 2, "FIXED_20", 1, False)
    bad = [dict(x, value=float(x["value"]) + 1e-6) if x in T.primary_rows(r) else x for x in r]
    assert_raises(LaunchRefused, T.per_user_entry, bad, w.m, art.ranks)          # mean != row value
    assert_raises(LaunchRefused, T.add_per_user, dict(cells), ent)               # the same cell twice
    assert_raises(LaunchRefused, T.cell_per_user, w.m, art.ranks[:-1])
    und = [dict(x, value=None, value_status="UNDEFINED") if x in T.primary_rows(r) else x for x in r]
    assert T.per_user_entry(und, w.m, art.ranks) is None                        # no OK value: no evidence


def test_persistence_rule_uses_the_evidence(series, R, tmp_path):
    w, rows, cells = series
    T.write_trunc_per_user(tmp_path, cells)
    pu = R.load_trunc_per_user(tmp_path / T.TRUNC_PER_USER_NAME)
    curve, ev = R.trunc_curves(rows, None, pu)
    for row in curve:
        if row["arm"] != "C_BIG":
            assert isinstance(row["R_ci95"], list) and len(row["R_ci95"]) == 2, row
            assert isinstance(row["parity_ci95"], list), row
    same = next(e for e in ev if e["arm"] == "C_SMALL")                  # identical weights: R = P = 100 %, CI [100, 100]
    assert same["events_to_90_persistent"] == 0 and same["parity_events_to_95_persistent"] == 0
    _c, ev0 = R.trunc_curves(rows, None, None)
    assert next(e for e in ev0 if e["arm"] == "C_SMALL")["events_to_90_persistent"] == "NO_CI"


def test_study_b_selected_cell_per_user(tmp_path):
    import v4int.launch_device as D
    from fl.numerics import state_digest
    w = _world()
    start = _tiny(2).broadcast_state(clone=True)
    for arm, sel_cell in (("FT_C", "FT_C_f0.3_P1"), ("LO", "LO_P2")):
        sel = {"arm": arm, "cell": sel_cell, "selection_role": "UNSEEN_VAL", "self_sha256": "s"}
        st = start if arm != "LO" else _tiny(1).broadcast_state(clone=True)
        spec = T.trunc_device_spec(arm, seed=2026, peak_lr=1e-2, panel_id=PANEL, panel_type="ALL_AVAILABLE", h=5,
                                   start_digest=state_digest(st), selection=sel, fused=False)
        plans = D.build_client_plans(w.m, w.row_key, row_decision_ts=np.full(w.m.n, 10_000, dtype=np.int64),
                                     support_pos_by_key={k: np.arange(3) for k in w.keys}, warm=False)
        lab = T.trunc_labels(study="B", panel_id=PANEL, label="COLD_H", panel_type="ALL_AVAILABLE", h=5, full=False)

        def row_fn(res, cell, selected, _spec=spec):
            return T.study_b_rows(res, dict(KEY, eval_manifest_id="SYN_TRUNC"), lab, spec=_spec, cell=cell,
                                  selected=selected, down_bytes=0)
        d = tmp_path / arm
        out = D.execute(spec=spec, run_dir=d, worker=_tiny(3), start_state=st, plans=plans,
                        get_support=lambda q: w.sup[q.key], get_eval_batch=w.get_eval_batch, manifest=w.m,
                        frozen=_frozen(), row_fn=row_fn, say=lambda *_: None, n_shards=2)
        assert out["status"] == "DONE"
        rows = json.loads((d / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
        key, u, m = T.study_b_cell_per_user(d, spec, w.m, rows)
        prim = [r for r in T.primary_rows(rows) if r["device_cell"] == sel_cell]
        assert len(prim) == 1 and key == T.trunc_cell_key(arm, 2026, PANEL, "ALL_AVAILABLE", 5, False)
        assert abs(float(m.mean()) - float(prim[0]["value"])) <= T.PER_USER_TOL
        assert float(m.mean()) == out["values"][sel_cell]


# ------------------------------------------------------------------------------------------------ negative controls
@nc("the persistence rule without per-user evidence reaches a persistent hit")
def test_nc_persistence_without_evidence(series, R):
    _w, rows, _cells = series
    _c, ev = R.trunc_curves(rows, None, {})                                      # MUTANT: no per-user evidence
    assert next(e for e in ev if e["arm"] == "C_SMALL")["events_to_90_persistent"] == 0


@nc("per-user values over the RANKABLE rows (not the E2E users) reproduce the macro-E2E row value")
def test_nc_rankable_per_user(series):
    from v4eval.metrics import credits_from_ranks, per_user
    w, _rows, _cells = series
    r, art = _study_a(w, "FA", 2, "FIXED_20", 5, False)
    credit = credits_from_ranks(art.ranks, (20,))["mrr@20"]
    _k, s, c = per_user(credit, w.m.eval_mask & w.m.rankable, w.m.eval_user_key)     # MUTANT: RANKABLE mask
    v = float(T.primary_rows(r)[0]["value"])
    assert abs(float((s / c).mean()) - v) <= T.PER_USER_TOL
