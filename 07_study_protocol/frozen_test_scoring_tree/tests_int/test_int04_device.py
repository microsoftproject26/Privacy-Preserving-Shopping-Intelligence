"""INT-04 — v4int.launch_device (LO / FT_C / FT_FA; CPU, tiny synthetic model, clients and evaluation manifest; the
real-row wiring is test_int04_real_clients.py).

  * the registered specs (LO cells P1/P2/P4 s_efe "end"; FT grid 3 x 2; FT applies the ONE selected cell) and refusals
    (select on MAIN / a panel, FT without selection, COLD_H / WARM_TRUNCATION, foreign selection, preflight);
  * per-client assembly == predict_adapter when every client has the start model (no support);
  * LO cell == a direct single-client fl.local_only run; determinism (two runs bitwise); resume by shard;
  * FT select on CALIBRATION -> SELECTION.json -> apply; causality refusal; rows format; PF personal_fn hook.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import v4int.launch_device as D
from intkit import assert_raises, nc
from test_launcher import fake_runner, gate

from fl.finetune import CausalityError
from fl.numerics import state_digest
from fl.synthetic import client_examples, make_clients, make_tiny_adapter

K, L, PEAK = 24, 6, 1e-2
CUT = 10_000


def _frozen():
    from v4eval import frozen_config as fc
    fz = fc.load()
    fc.apply_runtime_flags(fz)
    return fz


def _world(n=6, seed=5, rows_per_user=5):
    """n synthetic users: support examples (target_ts < CUT) and an evaluation manifest with censored rows."""
    from v4eval.manifest import build_eval_manifest
    rng = np.random.default_rng(seed)
    clients = make_clients(n, K=K, L=L, seed=seed)
    ev = []
    for u in range(n):
        e = client_examples(rows_per_user, K, L, rng)
        for i in range(rows_per_user):
            ev.append((u, {k: v[i] for k, v in e.items()}))
    order = rng.permutation(len(ev))
    ev = [ev[i] for i in order]
    did = np.arange(len(ev), dtype=np.int64) * 3 + 7
    cen = np.zeros(len(ev), dtype=bool)
    cen[::7] = True
    tgt = np.array([int(x["target_class"]) for _, x in ev], dtype=np.int64)
    tgt[cen] = -1
    oov = np.zeros(len(ev), dtype=bool)
    oov[3] = not cen[3]
    tgt[oov] = -1
    euk = np.array([u for u, _ in ev], dtype=np.int64)
    cols = {"decision_id": did, "eval_user_key": euk, "eval_mask": ~cen, "target_class": tgt, "target_oov": oov,
            "censor_class": np.where(cen, "CEN_NO_NEXT", "NONE").astype(object)}
    m = build_eval_manifest({"catalogue_K": K, "manifest_id": "SYN_DEVICE"}, cols)
    feats = {k: torch.stack([x[k] for _, x in ev]) for k in ("item_tokens", "attention_mask", "lengths")}
    feats["decision_id"] = torch.as_tensor(did)
    feats["target_class"] = torch.as_tensor(tgt)
    stream = np.flatnonzero(m.eval_mask)

    def get_eval_batch(sp):
        rows = stream[np.asarray(sp)]
        return {k: v[torch.as_tensor(rows)] for k, v in feats.items()}

    def full_batches(bs):
        for s in range(0, stream.size, bs):
            yield get_eval_batch(np.arange(s, min(stream.size, s + bs)))

    keys = [c.key for c in clients]
    sup = {}
    for c in clients:
        ex = dict(c.examples)
        ex[D.TS_KEY] = torch.arange(ex["target_class"].shape[0], dtype=torch.long)
        sup[c.key] = ex
    row_key = np.array([keys[u] for u in euk], dtype=object)
    dts = np.full(m.n, CUT, dtype=np.int64)
    return SimpleNamespace(m=m, keys=keys, sup=sup, row_key=row_key, dts=dts, get_eval_batch=get_eval_batch,
                           full_batches=full_batches, stream=stream)


def _plans(w, with_support=True, warm=False):
    pos = {k: np.arange(3) for k in w.keys} if with_support else None
    return D.build_client_plans(w.m, w.row_key, row_decision_ts=w.dts, support_pos_by_key=pos, warm=warm,
                                train_end_us=CUT + 5)


def _run(w, tmp, arm="LO", plans=None, role="MAIN", select=False, selection=None, **kw):
    worker = make_tiny_adapter(K=K, seed=1, tied=False)
    start = make_tiny_adapter(K=K, seed=2 if arm != "LO" else 1, tied=False).broadcast_state(clone=True)
    spec = D.device_spec(arm, seed=2026, peak_lr=PEAK, role=role, start_digest=state_digest(start), select=select,
                         selection=selection)
    out = D.execute(spec=spec, run_dir=tmp, worker=worker, start_state=start,
                    plans=_plans(w) if plans is None else plans, get_support=lambda p: w.sup[p.key],
                    get_eval_batch=w.get_eval_batch, manifest=w.m, frozen=_frozen(), say=lambda *_: None, **kw)
    return spec, out, start


# ------------------------------------------------------------------------------------------------ spec / refusals
def test_spec_cells_and_refusals():
    lo = D.device_spec("LO", seed=2026, peak_lr=PEAK, role="MAIN", start_digest="x")
    assert [c["cell"] for c in lo["cells"]] == ["LO_P1", "LO_P2", "LO_P4"]
    assert lo["knobs"]["lo_lr_shape"] == "s_efe" and lo["knobs"]["exposure_point"] == "end"
    g = D.device_spec("FT_C", seed=2026, peak_lr=PEAK, role="UNSEEN_VAL", start_digest="x", select=True)
    assert len(g["cells"]) == 6 and g["cells"][0]["cell"] == "FT_C_f0.1_P1" and g["selection_substitute"]
    ap = D.device_spec("FT_C", seed=2027, peak_lr=PEAK, role="MAIN", start_digest="x",
                       selection={"arm": "FT_C", "cell": "FT_C_f0.3_P2"})
    assert [c["cell"] for c in ap["cells"]] == ["FT_C_f0.3_P2"]
    for kw in (dict(arm="LO", role="MAIN", select=True), dict(arm="FT_C", role="HEAVY_PANEL", select=True),
               dict(arm="FT_C", role="CALIBRATION", select=True),          # superseded substitute: refused
               dict(arm="FT_C", role="MAIN"), dict(arm="LO", role="COLD_H"), dict(arm="LO", role="WARM_TRUNCATION"),
               dict(arm="FT_FA", role="MAIN", selection={"arm": "FT_C", "cell": "FT_C_f0.3_P2"}),
               dict(arm="FT_C", role="MAIN", selection={"arm": "FT_C", "cell": "FT_C_f0.2_P2"})):
        a = kw.pop("arm")
        assert_raises(D.LaunchRefused, lambda: D.device_spec(a, seed=1, peak_lr=PEAK, start_digest="x", **kw))


def _args(**kw):
    base = dict(arm="LO", family="SASREC", seed=2026, gpu=0, history_policy="USER_HISTORY_L50", role="MAIN",
                select=False, selection=None, base=None, base_kind=None, n_shards=4, shards=None, smoke=True,
                smoke_clients=4, dry_run=False, allow_shared_gpu=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_preflight_refusals(tmp_path):
    ok = D.preflight(_args(), runner=fake_runner(), environ={})
    assert ok["g1_gate"]["required"] is False and "gpu_ready" in ok["gpu_check"]
    for kw in (dict(arm="FT_C", selection="s.json"), dict(arm="FT_C", base="b.pt", base_kind="fl", selection="s"),
               dict(base="b.pt"), dict(role="MAIN", select=True), dict(shards="3:1"), dict(family="GRU"),
               dict(smoke=False), dict(smoke_clients=0)):
        assert_raises(D.LaunchRefused, lambda: D.preflight(_args(**kw), runner=fake_runner(), environ={},
                                                           gate_path=tmp_path / "absent.json"))
    D.preflight(_args(arm="FT_FA", base="b.pt", base_kind="fl", selection="s.json", smoke=False),
                runner=fake_runner(), environ={}, gate_path=gate(tmp_path))
    shared = D.preflight(_args(allow_shared_gpu=True), runner=fake_runner(mem={"0": 5000.0, "1": 0.0}), environ={})
    assert shared["gpu_check"]["shared_gpu"]["allowed"] is True


# ------------------------------------------------------------------------------------------------ assembly identity
def test_no_support_assembly_equals_predict_adapter(tmp_path):
    from v4eval.evaluate import evaluate
    from v4eval.predict import predict_adapter
    w = _world()
    spec, out, start = _run(w, tmp_path / "a", plans=_plans(w, with_support=False), n_shards=3)
    assert out["status"] == "DONE" and out["n_no_support"] == len(w.keys)
    ref = make_tiny_adapter(K=K, seed=1, tied=False)
    ref.load_state_(start)
    fz = _frozen()
    art = predict_adapter(ref, w.full_batches(fz.eval_batch), w.m, config=fz, model_sha256="ref")
    res = evaluate(w.m, art, config=fz)
    for c in ("LO_P1", "LO_P2", "LO_P4"):
        got = np.load(tmp_path / "a" / "cells" / f"{c}.ranks_int32.npy")
        assert np.array_equal(got, art.ranks.astype(np.int32))
        assert out["values"][c] == float(res["E2E"]["macro"]["mrr@20"])


def test_lo_cell_equals_direct_single_client_run(tmp_path):
    from fl.client import ClientData
    from fl.local_only import run_local_only
    w = _world()
    spec, out, start = _run(w, tmp_path / "b", n_shards=2)
    plan = _plans(w)[0]
    fz = _frozen()
    ref = make_tiny_adapter(K=K, seed=1, tied=False)
    rec = D.make_runners(spec, ref, start)[2][1].recipe                  # the P4 recipe
    batches = [w.get_eval_batch(sp) for sp, _ in D._chunks(plan, fz.eval_batch)]
    rch = [r for _, r in D._chunks(plan, fz.eval_batch)]
    r = run_local_only(ref, start, ClientData(plan.key, w.sup[plan.key]), rec,
                       eval_hook=lambda a, b: D.rank_client(a, batches, rch, w.m, fz))
    got = np.load(tmp_path / "b" / "cells" / "LO_P4.ranks_int32.npy")
    assert np.array_equal(got[plan.rows], r.evals[4][0].astype(np.int32))
    js = json.loads((tmp_path / "b" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert js["bytes"]["up"] == 0 and js["bytes"]["down_after_bootstrap"] == 0


def test_determinism_and_resume_by_shard(tmp_path):
    w = _world()
    _, o1, _ = _run(w, tmp_path / "r1", n_shards=3)
    _, o2, _ = _run(w, tmp_path / "r2", n_shards=3)
    assert o1["values"] == o2["values"] and o1["shard_set_sha256"] == o2["shard_set_sha256"]
    _, p, _ = _run(w, tmp_path / "r3", n_shards=3, shard_range=(0, 1))
    assert p["status"] == "PARTIAL" and p["missing_shards"] == [1, 2]
    assert_raises(D.LaunchRefused, lambda: _run(w, tmp_path / "r3", n_shards=3))           # needs --resume
    _, o3, _ = _run(w, tmp_path / "r3", n_shards=3, resume=True)
    assert o3["status"] == "DONE" and o3["values"] == o1["values"] and o3["shards_this_leg"] == 2
    for a, b in zip(sorted((tmp_path / "r1" / "cells").glob("*.npy")), sorted((tmp_path / "r3" / "cells").glob("*.npy"))):
        assert a.read_bytes() == b.read_bytes()


def test_ft_select_on_unseen_val_then_apply(tmp_path):
    w = _world()
    spec, out, start = _run(w, tmp_path / "sel", arm="FT_C", role="UNSEEN_VAL", select=True, n_shards=2,
                            eval_manifest_content_sha256="SYN_DEVICE_CONTENT_SHA")
    sel = D.load_selection(tmp_path / "sel" / "SELECTION.json")
    assert sel["cell"] == D.select_cell(out["values"], [c["cell"] for c in spec["cells"]])
    assert sel["selection_role"] == "UNSEEN_VAL" and "UNSEEN_VAL" in sel["substitute"]
    assert sel["eval_manifest_content_sha256"] == "SYN_DEVICE_CONTENT_SHA"
    assert_raises(Exception, lambda: _run(w, tmp_path / "sel", arm="FT_C", role="UNSEEN_VAL", select=True,
                                          n_shards=2, resume=True,
                                          eval_manifest_content_sha256="SYN_DEVICE_CONTENT_SHA"))  # O_EXCL
    spec2, out2, _ = _run(w, tmp_path / "app", arm="FT_C", role="MAIN", selection=sel, n_shards=2)
    assert [c["cell"] for c in spec2["cells"]] == [sel["cell"]]
    assert out2["values"][sel["cell"]] == out["values"][sel["cell"]]           # nested trajectories: same value
    bad = dict(sel, selection_role="MAIN")
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    assert_raises(D.LaunchRefused, lambda: D.load_selection(p))


def test_warm_roles_unchanged_by_unseen_val_selection():
    """item 3: UNSEEN_VAL becoming the selection role must not make it warm (TRAIN_END applies to MAIN/CALIBRATION
    only; UNSEEN_VAL's cutoff stays the eval rows' own earliest decision_ts, per amendment 06A support = TRAIN period
    UNEW_VAL_SUPPORT_32768)."""
    # INT-07: MAIN_TEST is warm too (cutoff min(earliest RETRO_TEST decision_ts, TRAIN_END)); no other TEST role is
    assert D.WARM_ROLES == ("MAIN", "CALIBRATION", "MAIN_TEST") and D.SELECTION_ROLE not in D.WARM_ROLES
    w = _world()
    w.dts[:] = CUT + 1000                                   # after TRAIN_END = CUT + 5
    warm = _plans(w, warm=True)[0].cutoff                   # MAIN / CALIBRATION: capped at TRAIN_END
    cold = _plans(w, warm=False)[0].cutoff                  # UNSEEN_VAL: the row's own decision_ts, uncapped
    assert warm == CUT + 5 and cold == CUT + 1000


def test_select_cell_tie_break():
    assert D.select_cell({"a": 0.2, "b": 0.2, "c": 0.1}, ["a", "b", "c"]) == "a"
    assert D.select_cell({"a": 0.1, "b": 0.3, "c": 0.3}, ["a", "b", "c"]) == "b"


# ------------------------------------------------------------------------------------------------ causality
def test_causality_cutoff_and_refusal(tmp_path):
    w = _world()
    w.dts[:] = CUT
    first = w.row_key == w.keys[0]
    w.dts[np.flatnonzero(first)[0]] = 0                    # client 0's earliest query at ts 0: every label is future
    plans = _plans(w)
    assert plans[0].cutoff == 0 and all(p.cutoff == CUT for p in plans[1:])
    assert _plans(w, warm=True)[1].cutoff == CUT             # min(CUT, TRAIN_END = CUT + 5)
    for arm, sel, role in (("LO", None, "MAIN"), ("FT_C", None, "UNSEEN_VAL")):
        assert_raises(CausalityError, lambda: _run(w, tmp_path / arm, arm=arm, plans=plans, role=role,
                                                   select=arm != "LO", selection=sel, n_shards=1,
                                                   eval_manifest_content_sha256="SYN_DEVICE_CONTENT_SHA"))
    assert D.support_causal_ok(np.array([0, 1]), 2) and not D.support_causal_ok(np.array([0, 2]), 2)


# ------------------------------------------------------------------------------------------------ rows / PF hook
def test_device_rows_format(tmp_path):
    from v4eval.rows import COMPARISON_KEY
    w = _world()
    spec, out, _ = _run(w, tmp_path / "rows", n_shards=1)
    res = json.loads((tmp_path / "rows" / "cells" / "LO_P2.result.json").read_text(encoding="utf-8"))
    kb = {f: "x" for f in COMPARISON_KEY}
    kb.update(aggregation="macro", target_denominator="E2E")
    rows = D.device_rows(res, kb, {"run_label": "PRODUCTION"}, spec=spec, cell=spec["cells"][1], selected=False,
                         down_bytes=0)
    assert rows and {r["method"] for r in rows} == {"LO"} and {r["up_bytes"] for r in rows} == {0}
    assert {r["device_cell"] for r in rows} == {"LO_P2"} and {r["comparison_status"] for r in rows} == \
        {"DOWNSTREAM_RETENTION"} and {r["eval_stream"] for r in rows} == {"E2E_ROWS"}
    prim = [r for r in rows if r["metric"] == "mrr" and r["k"] == 20 and r["aggregation"] == "macro"
            and r["target_denominator"] == "E2E"]
    assert len(prim) == 1 and prim[0]["value"] == out["values"]["LO_P2"]


def test_pf_personal_fn(tmp_path):
    from fl.client import PersonalState
    from fl.personal import PersonalStore
    from v4eval.predict import predict_adapter
    w = _world()
    ad = make_tiny_adapter(K=K, seed=1, tied=False)
    store = PersonalStore(ad.query_dim)
    hex_by_euk = np.array(w.keys, dtype=object)
    fz = _frozen()
    fn = D.pf_personal_fn(store, w.m, hex_by_euk, torch.device("cpu"))
    base = predict_adapter(ad, w.full_batches(fz.eval_batch), w.m, config=fz, model_sha256="b")
    zero = predict_adapter(ad, w.full_batches(fz.eval_batch), w.m, config=fz, model_sha256="z", personal_fn=fn)
    assert np.array_equal(base.ranks, zero.ranks)                          # an empty store is p = 0
    store.commit(w.keys[0], PersonalState(torch.full((ad.query_dim,), 3.0), None, 1, 1))
    b = next(w.full_batches(fz.eval_batch))
    P = fn(b)
    own = torch.as_tensor(w.row_key[np.searchsorted(w.m.decision_id, b["decision_id"].numpy())] == w.keys[0])
    assert torch.all(P[own] == 3.0) and torch.all(P[~own] == 0.0)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a support row at the cutoff (a future label) passes the causality property")
def test_nc_future_support_is_causal():
    assert D.support_causal_ok(np.array([0, 5, 10]), 10)


@nc("selecting the FT cell on MAIN is allowed")
def test_nc_select_on_main():
    assert D.selection_role_ok("MAIN")


@nc("--select on --role CALIBRATION (the superseded INT-04 substitute) is still accepted (amendment 06A §1)")
def test_nc_select_on_calibration_refused():
    try:
        D.check_role("CALIBRATION", select=True, arm="FT_C", selection=None)
        refused = False
    except D.LaunchRefused:
        refused = True
    assert not refused


@nc("a SELECTION.json recorded on another role (not UNSEEN_VAL) applies on MAIN / a panel")
def test_nc_apply_selection_from_other_role_refused(tmp_path):
    obj = {"format": D.SELECTION_FORMAT, "arm": "FT_C", "seed": 2026, "selection_role": "CALIBRATION",
           "eval_manifest_content_sha256": "X", "cell": "FT_C_f0.1_P1",
           "cells": {"cell": "FT_C_f0.1_P1", "passes": 1, "lr_frac": 0.1},
           "values_primary": {"FT_C_f0.1_P1": 0.1}, "primary": "E2E/macro/mrr@20",
           "rule": "max; ties -> canonical order", "spec_sha256": "x", "shard_set_sha256": "y",
           "substitute": D.SELECTION_SUBSTITUTE}
    obj["self_sha256"] = D._canon(obj)
    p = tmp_path / "other_role.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    try:
        D.load_selection(p)
        refused = False
    except D.LaunchRefused:
        refused = True
    assert not refused


@nc("per-client LO weights give the same ranks as the shared start model (the per-client models are not used)")
def test_nc_shared_model_equals_per_client(tmp_path):
    w = _world()
    _, _, _ = _run(w, tmp_path / "s", plans=_plans(w, with_support=False), n_shards=1)
    _, _, _ = _run(w, tmp_path / "p", n_shards=1)
    a = np.load(tmp_path / "s" / "cells" / "LO_P4.ranks_int32.npy")
    b = np.load(tmp_path / "p" / "cells" / "LO_P4.ranks_int32.npy")
    assert np.array_equal(a, b)


@nc("LO budgets are nested: a 4-pass s_efe trajectory evaluated after pass 1 equals the 1-pass recipe")
def test_nc_lo_budgets_nested():
    from fl.client import ClientData
    from fl.local_only import LORecipe, run_local_only
    w = _world()
    start = make_tiny_adapter(K=K, seed=1, tied=False).broadcast_state(clone=True)
    spec = D.device_spec("LO", seed=2026, peak_lr=PEAK, role="MAIN", start_digest=state_digest(start))
    ad = make_tiny_adapter(K=K, seed=1, tied=False)
    runners = D.make_runners(spec, ad, start)
    r4 = runners[2][1].recipe
    nested = LORecipe(**dict(r4.__dict__, eval_at=(1, 4)))
    cd = ClientData(w.keys[0], w.sup[w.keys[0]])
    d_after_1 = run_local_only(ad, start, cd, nested, eval_hook=lambda a, b: state_digest(a.extract_shared()))
    d_one = run_local_only(ad, start, cd, runners[0][1].recipe)
    assert d_after_1.evals[1] == d_one.final_digest


@nc("a shard written under another spec (other seed) is accepted on resume")
def test_nc_resume_other_spec(tmp_path):
    w = _world()
    spec, _, _ = _run(w, tmp_path / "o", n_shards=1)
    rec = json.loads((tmp_path / "o" / "shards" / "shard_0000.json").read_text(encoding="utf-8"))
    assert rec["spec_sha256"] == D.spec_digest(dict(spec, seed=2027))


@nc("plans that score one row for two clients cover the manifest exactly once")
def test_nc_plans_double_cover():
    w = _world()
    plans = _plans(w)
    plans[1].rows = np.concatenate([plans[1].rows, plans[0].rows[:1]])
    try:
        D.check_plans_cover(w.m, plans)
    except D.LaunchRefused:
        raise AssertionError("double cover detected")


def test_launch_fl_pf_uses_the_personal_evaluator(tmp_path):
    """INT-04 item 3: launch_fl.execute scores PF through evaluate_pf(mark, store) (shared + p_u), not at p = 0."""
    import v4int.launch_fl as F
    from test_int03_launch_fl import _tiny_spec, _world as fl_world
    w = fl_world()
    seen = []

    def evaluate_pf(mark, store):
        seen.append((mark, len(store)))
        return w["evaluate"](mark)
    out = F.execute(spec=_tiny_spec("PF"), run_id="pf", run_dir=tmp_path / "pf", say=lambda *_: None,
                    evaluate_pf=evaluate_pf, **w)
    assert out["status"] == "DONE_MAIN" and seen and seen[-1][0] == 6.0 and seen[-1][1] > 0
    seen.clear()
    F.execute(spec=_tiny_spec("FA"), run_id="fa", run_dir=tmp_path / "fa", say=lambda *_: None,
              evaluate_pf=evaluate_pf, **fl_world())
    assert not seen                                                      # FA has no store: the shared evaluator
