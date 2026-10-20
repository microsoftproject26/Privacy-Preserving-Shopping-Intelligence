"""FL-14 (PREG1_AMENDMENT_06L A1 / A2 / A4 as read by 06L_D D1-D4) — CPU, synthetic configuration records and stage files
(no run dirs, no data) plus the tiny FL world of test_fl05_launch for A4.

Proves:
  D1  S2 REQUIRED = 06J §1 + FedAvgM m {1, 2} (deduped when m* in {1, 2}); the live m* = 8 set with every edge false;
      FedAvgM m 1 raw best -> FedAvgM m 0.5 REQUIRED (EdgeRunsMissing naming the launch flags); FedAdam eta 0.01 best ->
      eta {0.003, 0.001} then tau; edge order eta -> A1 -> tau; tau evaluated over incumbent + required + the complete
      condition-true edges (the m 0.5 run can stop the tau edge); m* = 0.5 needs no A1 edge group.
  D2  S3 REQUIRED = 06J §2 on the S2 winner + the two A2 cells (deduped when the S2 winner is FedAdam eta 0.01 tau 1e-3)
      for S2 winner FedAvg / FedAvgM m 2 / FedAdam eta 0.01 / FedAdam eta 0.1; the SGD edge reads only the 06J cells;
      tie order incumbent -> AdamW -> smaller SGD lr -> FedAvg family before FedAdam.
  D3  S4 on an A2 S3 winner; launch_fl.load_fl_tuning round-trips every stage file above and refuses 06J-only files.
  D4  FA_1024 writes CLIP_LOG.jsonl (round, n_clipped, n_deltas, clipped_fraction; no norms), resume gives the same file,
      the trajectory digest is unchanged; 06L_E E1: DP (z > 0) records nothing, even if fl.dp releases a count
      (REV-FL-14 M1); a line lost in the checkpoint / append crash window is flagged complete False (REV-FL-14 m1).
NCs (strict xfail on AssertionError): the 06J-only S2 set accepted; the A1 edge dropped; A2 cells triggering the SGD
edge; FA_1024 without a clip log; a z > 0 count logged; a lost line reported complete.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import v4int.launch_fl as F
import v4int.lr_screen as L
from intkit import assert_raises, nc
from test_fl05_launch import run_tiny, tiny_spec
from test_fl10_fedadam_launch import PRE_FL10_TRAJ, _tiny

T = F.tuning_config
LAB = F.tuning_label
M_STAR = 8.0                                               # the live S1b winner (contracts/fl/LR_SELECTION_EXT.json)
CHAIN = ["S1b", "S2", "S3", "S4"]


def cfg(value, incumbent=False, excluded=False, **kw):
    t = T(**kw)
    return dict(t, label=LAB(t), value=value, incumbent=incumbent, excluded=excluded)


def pt(value, t, incumbent=False):
    return dict(t, label=LAB(t), value=value, incumbent=incumbent, excluded=False)


def _accepted(fn, *a, exc=L.SelectionRefused, **kw):
    try:
        fn(*a, **kw)
        return True
    except exc:
        return False


def req_labels(stage, inc_t, m_star=M_STAR):
    return [LAB(t) for t in L.stage_required(stage, inc_t, m_star)]


# ------------------------------------------------------------------------------------------------ S2 builders
ADAM01_1, ADAM01_M, ADAM1_1, ADAM1_M = "fedadam|eta=0.01|m=1", "fedadam|eta=0.01|m=8", "fedadam|eta=0.1|m=1", \
    "fedadam|eta=0.1|m=8"
AVGM_M, AVGM_1, AVGM_2, AVGM_05 = "fedavgm|m=8", "fedavgm|m=1", "fedavgm|m=2", "fedavgm|m=0.5"


def s2_live(vals=None):
    """The live S2 set (incumbent FedAvg x8 + the 7 launched runs; values by label); defaults: every edge false."""
    v = {"inc": 0.160, ADAM01_1: 0.150, ADAM01_M: 0.152, ADAM1_1: 0.155, ADAM1_M: 0.158, AVGM_M: 0.140,
         AVGM_1: 0.130, AVGM_2: 0.135}
    v.update(vals or {})
    out = [cfg(v["inc"], True, lr_mult=M_STAR)]
    for lr, m in F.fedadam_grid(M_STAR):
        out.append(cfg(v[f"fedadam|eta={F.num_tag(lr)}|m={F.num_tag(m)}"], optimizer="fedadam", server_lr=lr, lr_mult=m))
    for m in (M_STAR, 1.0, 2.0):
        out.append(cfg(v[f"fedavgm|m={F.num_tag(m)}"], optimizer="fedavgm", lr_mult=m))
    return out


INC2 = LAB(T(lr_mult=M_STAR))
REQ2 = req_labels("S2", T(lr_mult=M_STAR))


def eta_edge(v1=0.170, v2=0.168):
    return [cfg(v1, optimizer="fedadam", server_lr=0.003, lr_mult=M_STAR),
            cfg(v2, optimizer="fedadam", server_lr=0.001, lr_mult=M_STAR)]


def tau_run(v, lr, m):
    return cfg(v, optimizer="fedadam", server_lr=lr, server_tau=1e-5, lr_mult=m)


# ================================================================================================ D1 (S2)
def test_s2_required_sets():
    assert REQ2 == [ADAM01_1, ADAM01_M, ADAM1_1, ADAM1_M, AVGM_M, AVGM_1, AVGM_2]
    assert req_labels("S2", T(), 1.0) == ["fedadam|eta=0.01|m=1", "fedadam|eta=0.01|m=4", "fedadam|eta=0.1|m=1",
                                          "fedadam|eta=0.1|m=4", AVGM_1, AVGM_2]           # m* = 1: one m 1 run
    assert req_labels("S2", T(lr_mult=2.0), 2.0)[4:] == [AVGM_2, AVGM_1]                  # m* = 2: one m 2 run
    assert req_labels("S2", T(lr_mult=0.5), 0.5)[4:] == [AVGM_05, AVGM_1, AVGM_2]
    assert L.MIN_GAIN == 0.002 and L.TIE == 0.0005                                          # registered values kept


def test_s2_live_set_all_edges_false():
    c = s2_live()
    assert L.edge_groups("S2", c, INC2, REQ2) == ([], None)
    assert L.check_stage_runs("S2", c, INC2, REQ2) == []
    d = L.stage_decision("S2", c)
    assert d["winner"] == T(lr_mult=M_STAR) and d["no_measurable_gain"]
    for extra in (cfg(0.1, optimizer="fedavgm", lr_mult=0.5), *eta_edge(), tau_run(0.1, 0.1, M_STAR)):
        with pytest.raises(L.SelectionRefused, match="not allowed"):                     # no condition holds
            L.check_stage_runs("S2", c + [extra], INC2, REQ2)
    assert_raises(L.SelectionRefused, L.check_stage_runs, "S2", [x for x in c if x["label"] != AVGM_1], INC2, REQ2)


def test_s2_a1_fedavgm_m1_best_requires_m05():
    c = s2_live({AVGM_1: 0.170})
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S2", c, INC2, REQ2)
    assert [t["label"] for t in ei.value.needed] == [AVGM_05]
    assert "launch_fl --role CALIBRATION --server-opt fedavgm --lr-mult 0.5" in str(ei.value)
    full = c + [cfg(0.150, optimizer="fedavgm", lr_mult=0.5)]
    edges = L.check_stage_runs("S2", full, INC2, REQ2)
    assert [e["labels"] for e in edges] == [[AVGM_05]] and "06L A1" in edges[0]["rule"]
    assert L.stage_decision("S2", full)["winner"] == T(optimizer="fedavgm", lr_mult=1.0)
    # raw values, as in 06J: m 1 raw best but below the incumbent still requires m 0.5
    low = s2_live({AVGM_1: 0.145})
    assert [t["label"] for t in pytest.raises(L.EdgeRunsMissing, L.check_stage_runs, "S2", low, INC2, REQ2)
            .value.needed] == [AVGM_05]
    # an excluded (non-finite) m 1 run is not the raw best
    exc = [dict(x, excluded=True, value=None) if x["label"] == AVGM_1 else x for x in c]
    assert L.check_stage_runs("S2", exc, INC2, REQ2) == []


def test_s2_a1_edge_waits_then_tau_over_the_pool():
    c = s2_live({AVGM_1: 0.170, ADAM1_M: 0.190})       # FedAdam wins; A1 condition true
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S2", c, INC2, REQ2)
    assert [t["label"] for t in ei.value.needed] == [AVGM_05] and "then the 06J §1 tau edge" in str(ei.value)
    c05 = c + [cfg(0.150, optimizer="fedavgm", lr_mult=0.5)]
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S2", c05, INC2, REQ2)
    assert [t["label"] for t in ei.value.needed] == ["fedadam|eta=0.1|tau=1e-05|m=8"]
    edges = L.check_stage_runs("S2", c05 + [tau_run(0.180, 0.1, M_STAR)], INC2, REQ2)
    assert [e["labels"] for e in edges] == [[AVGM_05], ["fedadam|eta=0.1|tau=1e-05|m=8"]]


def test_s2_eta_edge_then_tau():
    c = s2_live({ADAM01_M: 0.175})
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S2", c, INC2, REQ2)
    assert [t["label"] for t in ei.value.needed] == ["fedadam|eta=0.003|m=8", "fedadam|eta=0.001|m=8"]
    assert "tau edge condition" in str(ei.value)
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S2", c + eta_edge(), INC2, REQ2)
    assert [t["label"] for t in ei.value.needed] == ["fedadam|eta=0.01|tau=1e-05|m=8"]
    full = c + eta_edge() + [tau_run(0.171, 0.01, M_STAR)]
    assert [e["labels"] for e in L.check_stage_runs("S2", full, INC2, REQ2)] == [
        ["fedadam|eta=0.003|m=8", "fedadam|eta=0.001|m=8"], ["fedadam|eta=0.01|tau=1e-05|m=8"]]


def test_s2_edge_order_and_the_m05_run_in_the_tau_pool():
    c = s2_live({ADAM01_M: 0.175, AVGM_1: 0.172})
    with pytest.raises(L.EdgeRunsMissing) as ei:                                 # both condition-true groups named
        L.check_stage_runs("S2", c, INC2, REQ2)
    assert [t["label"] for t in ei.value.needed] == ["fedadam|eta=0.003|m=8", "fedadam|eta=0.001|m=8", AVGM_05]
    with pytest.raises(L.EdgeRunsMissing) as ei:                                 # eta done: A1 still missing
        L.check_stage_runs("S2", c + eta_edge(), INC2, REQ2)
    assert [t["label"] for t in ei.value.needed] == [AVGM_05] and "then the 06J §1 tau edge" in str(ei.value)
    hi = c + eta_edge() + [cfg(0.180, optimizer="fedavgm", lr_mult=0.5)]       # m 0.5 wins: no tau edge
    assert [e["labels"] for e in L.check_stage_runs("S2", hi, INC2, REQ2)] == [
        ["fedadam|eta=0.003|m=8", "fedadam|eta=0.001|m=8"], [AVGM_05]]
    lo = c + eta_edge() + [cfg(0.150, optimizer="fedavgm", lr_mult=0.5)]       # FedAdam eta 0.01 wins: tau edge
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S2", lo, INC2, REQ2)
    assert [t["label"] for t in ei.value.needed] == ["fedadam|eta=0.01|tau=1e-05|m=8"]
    edges = L.check_stage_runs("S2", lo + [tau_run(0.150, 0.01, M_STAR)], INC2, REQ2)
    assert [e["rule"][:6] for e in edges] == ["06J §1", "06L A1", "06J §1"] and "tau" in edges[2]["rule"]


def test_s2_m_star_half_needs_no_a1_group():
    inc = T(lr_mult=0.5)
    req = req_labels("S2", inc, 0.5)
    c = [cfg(0.1, True, lr_mult=0.5)] + [cfg(0.09 if lr == 0.01 else 0.092, optimizer="fedadam", server_lr=lr,
                                             lr_mult=m) for lr, m in F.fedadam_grid(0.5)]      # no eta edge
    c += [cfg(0.08, optimizer="fedavgm", lr_mult=0.5), cfg(0.095, optimizer="fedavgm", lr_mult=1.0),
          cfg(0.07, optimizer="fedavgm", lr_mult=2.0)]
    assert L.check_stage_runs("S2", c, LAB(inc), req) == []


# ================================================================================================ D2 (S3)
A2_01, A2_1 = "fedadam|eta=0.01|m=1|shape=fl_const|client=sgd_m|clr=0.1", \
    "fedadam|eta=0.01|m=1|shape=fl_const|client=sgd_m|clr=1"
W_AVG, W_AVGM2 = T(lr_mult=M_STAR), T(optimizer="fedavgm", lr_mult=2.0)
W_ADAM01, W_ADAM1 = T(optimizer="fedadam", server_lr=0.01, lr_mult=M_STAR), T(optimizer="fedadam", server_lr=0.1,
                                                                                lr_mult=1.0)


def s3_set(w2, inc=0.10, adamw=0.09, sgd=(0.09, 0.095, 0.09), a2=(0.05, 0.05)):   # SGD 0.1 best: no edge
    """Incumbent (S2 winner w2) + the S3 REQUIRED cells with values; an A2 cell equal to a 06J cell takes the 06J value."""
    req = L.stage_required("S3", w2, M_STAR)
    vals = {LAB(req[0]): adamw, **{LAB(t): v for t, v in zip(req[1:4], sgd)}}
    for t, v in zip(L.a2_cells(), a2):
        vals.setdefault(LAB(t), v)
    return [pt(inc, w2, True)] + [pt(vals[LAB(t)], t) for t in req]


def test_s3_required_sets_per_s2_winner():
    for w2, n in ((W_AVG, 6), (W_AVGM2, 6), (W_ADAM01, 4), (T(optimizer="fedadam", server_lr=0.01, lr_mult=1.0), 4),
                  (W_ADAM1, 6), (T(optimizer="fedadam", server_lr=0.01, server_tau=1e-5, lr_mult=M_STAR), 6)):
        req = req_labels("S3", w2)
        assert len(req) == len(set(req)) == n, (LAB(w2), req)
        assert A2_01 in req and A2_1 in req
        assert req[0] == LAB(dict(w2, lr_shape="fl_const"))                    # 06J cells first
    assert req_labels("S3", W_AVG)[4:] == [A2_01, A2_1]
    assert req_labels("S3", W_ADAM01)[2:] == [A2_01, A2_1]                      # the A2 cells ARE 06J cells here
    for w2 in (W_AVG, W_AVGM2, W_ADAM01, W_ADAM1):
        c = s3_set(w2)
        assert L.check_stage_runs("S3", c, LAB(w2), req_labels("S3", w2)) == []
        assert_raises(L.SelectionRefused, L.check_stage_runs, "S3", [x for x in c if x["label"] != A2_1], LAB(w2),
                      req_labels("S3", w2))


def test_s3_sgd_edge_reads_only_the_06j_cells():
    req = req_labels("S3", W_AVG)
    c = s3_set(W_AVG, sgd=(0.10, 0.12, 0.11), a2=(0.05, 0.20))                # A2 lr 1.0 best of all SGD runs
    assert L.edge_groups("S3", c, LAB(W_AVG), req) == ([], None)
    assert L.check_stage_runs("S3", c, LAB(W_AVG), req) == []
    sgd3 = pt(0.1, T(optimizer="fedadam", server_lr=0.01, lr_shape="fl_const", client_opt="sgd_m", client_lr=3.0))
    avg3 = pt(0.1, T(lr_shape="fl_const", client_opt="sgd_m", client_lr=3.0))
    for extra in (sgd3, avg3):
        with pytest.raises(L.SelectionRefused, match="not allowed"):
            L.check_stage_runs("S3", c + [extra], LAB(W_AVG), req)
    c = s3_set(W_AVG, sgd=(0.10, 0.11, 0.13), a2=(0.20, 0.05))                # 06J lr 1.0 best of the 06J cells
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S3", c, LAB(W_AVG), req)
    assert [t["label"] for t in ei.value.needed] == ["fedavg|m=1|shape=fl_const|client=sgd_m|clr=3",
                                                     "fedavg|m=1|shape=fl_const|client=sgd_m|clr=10"]
    c = s3_set(W_AVG, sgd=(0.13, 0.11, 0.10))
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S3", c, LAB(W_AVG), req)
    assert [t["label"] for t in ei.value.needed] == ["fedavg|m=1|shape=fl_const|client=sgd_m|clr=0.003"]
    # dedupe path: S2 winner FedAdam eta 0.01 -> the A2 / 06J lr 1.0 cell drives the edge on FedAdam eta 0.01
    reqd = req_labels("S3", W_ADAM01)
    c = s3_set(W_ADAM01, sgd=(0.10, 0.11, 0.13))
    with pytest.raises(L.EdgeRunsMissing) as ei:
        L.check_stage_runs("S3", c, LAB(W_ADAM01), reqd)
    assert [t["label"] for t in ei.value.needed] == [A2_1.replace("clr=1", "clr=3"), A2_1.replace("clr=1", "clr=10")]


def test_s3_tie_order_and_a2_winner():
    w2 = W_AVG
    c = s3_set(w2, sgd=(0.09, 0.13, 0.09), a2=(0.13, 0.05))                   # 06J SGD 0.1 == A2 SGD 0.1
    d = L.stage_decision("S3", c)
    assert d["tie_band_set"] == ["fedavg|m=1|shape=fl_const|client=sgd_m|clr=0.1", A2_01]   # FedAvg family first
    assert d["winner"]["optimizer"] == "fedavg"
    c = s3_set(w2, sgd=(0.09, 0.09, 0.13), a2=(0.13, 0.05))                   # smaller SGD lr before the family
    assert L.stage_decision("S3", c)["winner"] == L.a2_cells()[0]
    c = s3_set(w2, adamw=0.13, sgd=(0.09, 0.13, 0.09), a2=(0.13, 0.05))       # AdamW before SGD
    assert L.stage_decision("S3", c)["winner"]["client_opt"] == "adamw_fresh"
    c = s3_set(w2, sgd=(0.09, 0.09, 0.09), a2=(0.05, 0.20))                   # A2 strictly best -> the S3 winner
    assert L.stage_decision("S3", c)["winner"] == L.a2_cells()[1]
    c = s3_set(w2, sgd=(0.09, 0.09, 0.09), a2=(0.05, 0.1015))                 # 06J §4(ii): gain < 0.002
    assert L.stage_decision("S3", c)["winner"] == w2
    assert "FedAvg family before FedAdam" in L.TIE_ORDER_TEXT["S3"] and "06L A2" in L.RULE_TEXT_STAGE["S3"]


# ================================================================================================ D3 (files, S4)
def stage_file(tmp_path: Path, name: str, stage: str, configs, m_star=M_STAR) -> Path:
    """A synthetic stage file with every derived field computed by the lr_screen functions (as select-joint writes)."""
    inc = next(c for c in configs if c["incumbent"])
    fin = [c for c in configs if not c["excluded"]]
    d = L.stage_decision(stage, fin)
    req = [LAB(t) for t in L.stage_required(stage, T(**{k: inc[k] for k in F.TUNING_KEYS}), m_star)]
    obj = {"format": F.FL_TUNING_FORMAT, "stage": stage, "configs": list(configs), "incumbent": inc["label"],
           "required": req, "edges": L.check_stage_runs(stage, configs, inc["label"], req), "m_star": m_star,
           "winner": dict(d["winner"], label=LAB(d["winner"])), "tie_set": d["tie_set"],
           "values": {c["label"]: c["value"] for c in fin},
           "stage_history": [{"stage": s} for s in CHAIN[:CHAIN.index(stage)]]}
    obj.update({k: d[k] for k in ("gain", "best_candidate", "no_measurable_gain", "tie_band_set", "min_gain")})
    obj["self_sha256"] = F._canon(obj)
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p


def resign(tmp_path, src: Path, name: str, **changes) -> Path:
    obj = json.loads(src.read_text(encoding="utf-8"))
    obj.update(changes)
    obj["self_sha256"] = F._canon(obj)
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_load_fl_tuning_round_trips_s2_files(tmp_path):
    cases = {"all_false": s2_live(),
             "a1": s2_live({AVGM_1: 0.170}) + [cfg(0.150, optimizer="fedavgm", lr_mult=0.5)],
             "eta_tau": s2_live({ADAM01_M: 0.175}) + eta_edge() + [tau_run(0.171, 0.01, M_STAR)],
             "eta_a1": s2_live({ADAM01_M: 0.175, AVGM_1: 0.172}) + eta_edge()
             + [cfg(0.180, optimizer="fedavgm", lr_mult=0.5)]}
    for name, c in cases.items():
        p = stage_file(tmp_path, name, "S2", c)
        obj = F.load_fl_tuning(p)
        assert obj["required"] == REQ2 and obj["stage"] == "S2"
    a1 = json.loads((tmp_path / "a1.json").read_text(encoding="utf-8"))
    assert a1["edges"][0]["labels"] == [AVGM_05] and a1["winner"]["label"] == AVGM_1
    # refused: the 06J-only file (required without FedAvgM m 1 / 2), a dropped A1 edge, a dropped required A1 cell
    old = [c for c in s2_live() if c["label"] not in (AVGM_1, AVGM_2)]
    p = resign(tmp_path, tmp_path / "all_false.json", "old06j", configs=old, required=REQ2[:5],
               values={c["label"]: c["value"] for c in old})
    with pytest.raises(F.LaunchRefused, match="required"):
        F.load_fl_tuning(p)
    no_edge = [c for c in a1["configs"] if c["label"] != AVGM_05]
    p = resign(tmp_path, tmp_path / "a1.json", "a1_dropped", configs=no_edge, edges=[],
               values={c["label"]: c["value"] for c in no_edge})
    with pytest.raises(F.LaunchRefused, match="edge"):
        F.load_fl_tuning(p)
    p = resign(tmp_path, tmp_path / "a1.json", "a1_no_group", edges=[])
    assert_raises(F.LaunchRefused, F.load_fl_tuning, p)


@pytest.mark.parametrize("w2", [W_AVG, W_AVGM2, W_ADAM01, W_ADAM1], ids=["fedavg", "fedavgm_m2", "fedadam_0.01",
                                                                         "fedadam_0.1"])
def test_load_fl_tuning_round_trips_s3_and_s4_on_an_a2_winner(tmp_path, w2):
    dedupe = w2 == W_ADAM01                                # the A2 cells are 06J cells: they take the 06J values
    c = s3_set(w2, sgd=(0.09, 0.20, 0.09) if dedupe else (0.09, 0.095, 0.09), a2=(0.20, 0.05))   # A2 lr 0.1 wins
    p3 = stage_file(tmp_path, "s3", "S3", c)
    obj = F.load_fl_tuning(p3)
    assert obj["winner"]["label"] == A2_01 and obj["required"] == req_labels("S3", w2) and obj["edges"] == []
    assert len(obj["configs"]) == 1 + (4 if dedupe else 6)
    if not dedupe:                                         # refused: the 06J-only S3 file (no A2 cells)
        old = [x for x in c if x["label"] not in (A2_01, A2_1)]
        bad = resign(tmp_path, p3, "s3_old", configs=old, required=obj["required"][:4])
        assert_raises(F.LaunchRefused, F.load_fl_tuning, bad)
    # S4 on the A2 winner (06J §3 unchanged)
    a2 = L.a2_cells()[0]
    req4 = L.stage_required("S4", a2, M_STAR)
    assert [LAB(t) for t in req4] == [A2_01 + "|passes=1", A2_01 + "|passes=4", A2_01 + "|batch=8"]
    c4 = [pt(0.20, a2, True), pt(0.21, req4[0]), pt(0.19, req4[1]), pt(0.18, req4[2])]
    p4 = stage_file(tmp_path, "s4", "S4", c4)
    o4 = F.load_fl_tuning(p4)
    assert o4["winner"]["label"] == A2_01 + "|passes=1" and o4["edges"] == []
    assert F.poisson_tuning("DP", T(**{k: o4["winner"][k] for k in F.TUNING_KEYS}))[0] == a2   # DL 219 fields


# ================================================================================================ D4 (A4)
def _clip_lines(d: Path):
    return [json.loads(x) for x in (d / F.CLIP_LOG_FILE).read_text(encoding="utf-8").splitlines()]


def test_clip_round_record_units():
    base = {"round": 3, "dp": {"clip_norm": 0.5, "noise_multiplier": 0.0, "n_survived": 5, "n_noop": 1,
                               "n_clipped": 2, "denominator": 1024}}
    assert F.clip_round_record(base) == {"round": 3, "n_clipped": 2, "n_deltas": 4, "clipped_fraction": 0.5}
    zero = {"round": 1, "dp": dict(base["dp"], n_survived=1, n_noop=1, n_clipped=0)}
    assert F.clip_round_record(zero)["clipped_fraction"] is None
    dp = {"round": 4, "dp": dict(base["dp"], noise_multiplier=0.56)}   # 06L_E E1: z > 0 records nothing, even
    assert F.clip_round_record(dp) is None                              # with a released count
    for out in ({"round": 0, "dp": None}, {"round": 0},
                {"round": 0, "dp": {"clip_norm": None, "noise_multiplier": 0.0, "norms": [1.0]}},   # S_CAL
                {"round": 0, "dp": {k: v for k, v in dp["dp"].items() if k != "n_clipped"}}):   # z > 0 as fl.dp gives it
        assert F.clip_round_record(out) is None


def test_fa1024_clip_log_resume_and_digest(tmp_path):
    a = run_tiny(tmp_path, _tiny("FA_1024"), "a")
    assert a["status"] == "DONE_MAIN" and a["summary"]["server_digest"] == PRE_FL10_TRAJ["FA_1024"]   # numerics kept
    lines = _clip_lines(tmp_path / "a")
    assert lines and [r["round"] for r in lines] == sorted({r["round"] for r in lines})
    assert all(set(r) == {"round", "n_clipped", "n_deltas", "clipped_fraction"} for r in lines)
    assert all(0 <= r["n_clipped"] <= r["n_deltas"] and r["clipped_fraction"] == r["n_clipped"] / r["n_deltas"]
               for r in lines if r["n_deltas"])
    cl = a["summary"]["clip_log"]
    assert cl["rounds_logged"] == len(lines) and cl["file"] == F.CLIP_LOG_FILE
    summ = json.loads((tmp_path / "a" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert summ["clip_log"]["sha256"] == F._file_sha(tmp_path / "a" / F.CLIP_LOG_FILE)
    part = run_tiny(tmp_path, _tiny("FA_1024"), "p", interrupt_at=5)
    assert part["status"] == "INTERRUPTED"
    with open(tmp_path / "p" / F.CLIP_LOG_FILE, "a", encoding="utf-8") as f:     # a stale line past the checkpoint
        f.write(json.dumps({"round": 6, "n_clipped": 99, "n_deltas": 99, "clipped_fraction": 1.0}) + "\n")
    res = run_tiny(tmp_path, _tiny("FA_1024"), "p", resume=True)
    assert res["summary"]["server_digest"] == a["summary"]["server_digest"]
    assert (tmp_path / "p" / F.CLIP_LOG_FILE).read_bytes() == (tmp_path / "a" / F.CLIP_LOG_FILE).read_bytes()


def _release_count_at_z_gt_0(monkeypatch):
    """A future fl.dp that releases n_clipped for z > 0 too (the numerics are untouched)."""
    import fl.pool
    import fl.server
    real, released = fl.server.finalize_dp, []

    def leaky(agg, *a, **kw):
        new, info = real(agg, *a, **kw)
        if info.get("clip_norm") is not None and "n_clipped" not in info:
            info["n_clipped"] = sum(1 for x in agg.norms if x[3] < 1.0)
            released.append(info["n_clipped"])
        return new, info
    monkeypatch.setattr(fl.server, "finalize_dp", leaky)
    monkeypatch.setattr(fl.pool, "finalize_dp", leaky)
    return released


def test_dp_records_nothing_even_if_fl_dp_releases_a_count(tmp_path, monkeypatch):
    """06L_E E1 (REV-FL-14 M1): a z > 0 run writes no CLIP_LOG and no clip_log record whatever fl.dp returns."""
    released = _release_count_at_z_gt_0(monkeypatch)
    out = run_tiny(tmp_path, _tiny("DP"), "dp")
    assert released                                            # the count really reached the launcher
    assert out["status"] == "DONE_MAIN" and out["summary"]["server_digest"] == PRE_FL10_TRAJ["DP"]
    assert not (tmp_path / "dp" / F.CLIP_LOG_FILE).exists() and "clip_log" not in out["summary"]
    assert "06L_E" in F.CLIP_LOG_AUTHORITY and not F.clip_log_applies(_tiny("DP"))
    assert F.clip_log_applies(_tiny("FA_1024")) and not F.clip_log_applies(_tiny("S_CAL"))


def test_clip_log_completeness_against_the_participation_log(tmp_path):
    """REV-FL-14 m1: a line lost between the in-step checkpoint and the append (crash) is flagged in RUN_SUMMARY."""
    a = run_tiny(tmp_path, _tiny("FA_1024"), "a")
    cl = a["summary"]["clip_log"]
    assert cl["complete"] is True and cl["n_missing"] == 0 and cl["rounds_expected"] == cl["rounds_logged"]
    part = run_tiny(tmp_path, _tiny("FA_1024"), "p", interrupt_at=5)
    assert part["status"] == "INTERRUPTED"
    lines = _clip_lines(tmp_path / "p")
    lost = lines[-1]["round"]                                 # the crash window: checkpointed, line not appended
    (tmp_path / "p" / F.CLIP_LOG_FILE).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines[:-1]),
                                                   encoding="utf-8", newline="\n")
    said = []
    res = F.execute(spec=_tiny("FA_1024"), run_id="p", run_dir=tmp_path / "p", say=said.append, resume=True,
                    **__import__("test_fl05_launch").fl05_world())
    assert res["summary"]["server_digest"] == a["summary"]["server_digest"]
    c2 = res["summary"]["clip_log"]
    assert c2["complete"] is False and c2["missing_rounds"] == [lost] and c2["n_missing"] == 1
    assert json.loads((tmp_path / "p" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))["clip_log"]["complete"] is False
    assert any("WARNING" in m and F.CLIP_LOG_FILE in m for m in said)
    # the summary function alone: missing / unexpected rounds
    d = F.clip_log_summary(tmp_path / "a", [r["round"] for r in _clip_lines(tmp_path / "a")] + [999])
    assert d["complete"] is False and d["missing_rounds"] == [999] and d["unexpected_rounds"] == []
    assert F.clip_log_summary(tmp_path / "none", None) is None
    e = F.clip_log_summary(tmp_path / "none", [0, 1])                   # expected but nothing logged
    assert e["complete"] is False and e["rounds_logged"] == 0 and e["missing_rounds"] == [0, 1]


def test_dp_and_fa_write_no_clip_log(tmp_path):
    for arm in ("DP", "FA"):
        out = run_tiny(tmp_path, _tiny(arm), arm)
        assert out["status"] == "DONE_MAIN" and out["summary"]["server_digest"] == PRE_FL10_TRAJ[arm]
        assert not (tmp_path / arm / F.CLIP_LOG_FILE).exists() and "clip_log" not in out["summary"]


# ================================================================================================ negative controls
@nc("the 06J-only S2 run set (no FedAvgM m 1 / 2) is accepted")
def test_nc_06j_only_s2_set_accepted():
    old = [c for c in s2_live() if c["label"] not in (AVGM_1, AVGM_2)]
    assert _accepted(L.check_stage_runs, "S2", old, INC2, REQ2), "refused (as it must be)"


@nc("FedAvgM m 1 as the raw-best FedAvgM cell does not require the m 0.5 run")
def test_nc_a1_edge_dropped():
    assert _accepted(L.check_stage_runs, "S2", s2_live({AVGM_1: 0.170}), INC2, REQ2), "refused (as it must be)"


@nc("an A2 cell (lr 1.0 best of all SGD runs) triggers the 06J SGD edge")
def test_nc_a2_triggers_sgd_edge():
    c = s3_set(W_AVG, sgd=(0.10, 0.12, 0.11), a2=(0.05, 0.20))
    assert L.edge_groups("S3", c, LAB(W_AVG), req_labels("S3", W_AVG))[0]


@nc("a z > 0 round with a released clipped count is logged (the 06L_E z == 0 gate removed)")
def test_nc_z_gt_0_count_logged():
    dp = {"round": 4, "dp": {"clip_norm": 0.5, "noise_multiplier": 0.5642, "n_survived": 5, "n_noop": 0,
                             "n_clipped": 2}}
    assert F.clip_round_record(dp) is not None


@nc("a lost CLIP_LOG line is reported complete")
def test_nc_lost_clip_line_reported_complete(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / F.CLIP_LOG_FILE).write_text(json.dumps({"round": 0, "n_clipped": 1, "n_deltas": 2,
                                                              "clipped_fraction": 0.5}) + "\n", encoding="utf-8")
    assert F.clip_log_summary(tmp_path / "r", [0, 1])["complete"] is True


@nc("an FA_1024 run writes no per-round clipped fraction")
def test_nc_fa1024_without_clip_log(tmp_path):
    run_tiny(tmp_path, _tiny("FA_1024"), "a")
    assert not (tmp_path / "a" / F.CLIP_LOG_FILE).exists()
