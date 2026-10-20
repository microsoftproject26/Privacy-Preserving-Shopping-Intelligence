"""P2-06 — the Phase-2 CALIB selection tool v4int.p2_select (contracts/PREG2.md REGISTERED §4.2, §5.1-§5.9), after
REV-P2-06. CPU only; synthetic run dirs (JSON files only, no training, no data, no server). NO MONKEYPATCH anywhere:
every P2 run dir is built from the REAL launch_fl spec builders (p2_arm_spec, fl_rows, warm_row_labels, p2_row_labels)
and read by the REAL reader (p2_select.read_p2_run, which re-implements lr_screen.read_run's checks R1-R12 with
check_p2_spec / p2_row_labels); S8's Phase-1 incumbent through the unedited lr_screen.read_run; C1 runs through the
P2-05 verifier central_ft.read_c1_run.

Proves: the registered cell sets; labels / recipe keys identical to launch_fl's; the tie band 0.001 inclusive in exact
arithmetic (boundary at exactly 0.001 and 0.001 +- 1e-12) and the §5.6 preference order; the anchor flag strict < 0.001
(boundaries); the §5.7 step table WITHOUT a corner run, the single extension (missing -> refused naming the runs;
extension wins -> EDGE_UNRESOLVED; otherwise EDGE_RESOLVED with notes); futility strict < 0.001 on both conditions
(boundaries); the §5.9 single strict drop order and a cap plan that always passes check_dropped; non-finite selection
values excluded and recorded, non-finite diagnostics (D1-ext NaN, a NaN mark) flagged without blocking the file;
write-once self-hashed files that re-derive on load (tampered value / flags refused); refusal of a Phase-1 label, a MAIN
role, a MAIN / TEST manifest, a missing required run, a hand-edited P2 spec; the Phase-1 incumbent only with a full P2-T3
receipt and an identical cfg / theta_pre; C1 per variant; FINAL with drops bound to the cap plan and under futility.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import v4int.central_ft as CF
import v4int.launch_fl as F
import v4int.lr_screen as L
import v4int.p2_select as P
from intkit import nc

CALIB = P.CALIB_EVAL_MANIFEST
PEAK = 0.000711
DEV = {"eval_device_class": "CPU-FP32-DET", "device_pin_sha256": "d" * 64}
IDENT = {"kind": "PRETRAIN_FOR_FL", "run_id": "PRETRAIN_FOR_FL_SASREC_s2026", "run_label": "PRETRAIN_FOR_FL",
         "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN", "fl_init_kind": "endpoint_6.0", "checkpoint_sha256": "a" * 64,
         "checkpoint_bytes": 123, "state_digest": "b" * 64, "model_variant": "SASREC_D64_B2", "family": "SASREC",
         "seed": 2026}
S8_BYTES, S8_NUMEL = 1_055_296, 263_822


def R(so, eta, c, **kw):
    return P.p2_recipe(server_opt=so, server_lr=eta, client_lr=c, **kw)


def raw(recipe):
    return {k: recipe[k] for k in P.RECIPE_KEYS}


# ------------------------------------------------------------------------------------------------ synthetic runs
def p2_spec(recipe, *, init=IDENT, role="CALIBRATION"):
    return F.p2_arm_spec("P2_W", recipe=raw(recipe), seed=2026, peak_lr=PEAK, role=role, init_from=dict(init))


def write_run(root: Path, name: str, spec: dict, value, *, bytes_up=S8_BYTES, numel=S8_NUMEL, eval_id=CALIB,
              device=DEV, best_u=8, d1ext=None, marks=None) -> Path:
    """A finished run dir exactly as launch_fl.execute writes it (ARM_SPEC / RUN_SUMMARY / RESULT_ROWS fields read by the
    selection), the PRACTICAL_BEST row built by the REAL launch_fl.fl_rows with the run's own tuning labels."""
    rd = Path(root) / name
    rd.mkdir(parents=True)
    dig = F.spec_digest(spec)
    fin = isinstance(value, float) and math.isfinite(value)
    evals = [[u, u / 2, (value if u == best_u else value - 0.01) if fin else value] for u in range(1, 13)]
    for u, v in (marks or {}).items():
        evals[u - 1][2] = v
    if F.is_p2_arm(spec["arm"]):
        tl = dict(F.warm_row_labels(spec), **F.p2_row_labels(spec))
    else:
        tl = dict(F.row_labels(F.spec_tuning(spec)), **F.warm_row_labels(spec), **F.poisson_row_labels(spec))
    base = {"selection_rule": "PRACTICAL_BEST", "aggregation": "macro", "target_denominator": "E2E", "metric": "mrr",
            "k": 20, "value": value, "value_status": "OK" if fin else "NON_FINITE", "eval_manifest_id": eval_id,
            "split": "VALIDATION", "run_label": "HARNESS"}
    row = F.fl_rows([base], arm=spec["arm"], run_id=name, efe_raw=best_u / 2, mark=best_u / 2, digest=dig,
                    role=spec["role"], lr_mult=1.0, tuning_labels=tl)[0]
    summ = {"status": "DONE_MAIN", "arm": spec["arm"], "arm_spec": spec, "arm_spec_sha256": dig, "evals": evals,
            "best_half_efe": best_u, "bytes": {"per_visit": {"upload_bytes": bytes_up, "shared_numel": numel}},
            "update_norm_log": d1ext}
    if F.is_p2_arm(spec["arm"]):
        summ["p2"] = {"label": spec["p2_label"], "recipe": dict(spec["p2_recipe"]), "pre_server": None,
                      "post_run": None, "authority": F.P2_AUTHORITY}
    (rd / "RUN_SUMMARY.json").write_text(json.dumps(summ), encoding="utf-8")
    (rd / "ARM_SPEC.json").write_text(json.dumps({"arm": spec["arm"], "run_id": name, "arm_spec_sha256": dig,
                                                  "arm_spec": spec, "device": device, "code_sha256": "0" * 64}),
                                      encoding="utf-8")
    (rd / "RESULT_ROWS.json").write_text(json.dumps({"run_id": name, "rows": [row]}), encoding="utf-8")
    return rd


def write_p2(root, recipe, value, **kw) -> Path:
    name = P.p2_label(recipe).replace("|", "_").replace("=", "")
    sk = {k: kw.pop(k) for k in ("init", "role") if k in kw}
    return write_run(root, name, p2_spec(recipe, **sk), value, **kw)


def phase1_spec(init=IDENT, **over):
    kw = dict(seed=2026, peak_lr=PEAK, role="CALIBRATION", server_opt="fedadam", server_lr=0.01, lr_shape="fl_const",
              client_opt="sgd_m", client_lr=0.1, local_passes=1, init_from=dict(init), warm_server_lr_factor=0.1)
    kw.update(over)
    kw = {k: v for k, v in kw.items() if v is not None}
    return F.arm_spec("FA_WARM_FROZEN", **kw)


def write_phase1_incumbent(root, value, name="S8Tc_phase1", **over) -> Path:
    return write_run(root, name, phase1_spec(**over), value)


def receipt_for(run_dir: Path, path: Path, **over) -> Path:
    obj = {"test": "P2-T3", "verdict": "BIT_IDENTICAL", "rounds": 64, "s8tc_run_id": Path(run_dir).name,
           "result_rows_sha256": L._sha(Path(run_dir) / "RESULT_ROWS.json"),
           "run_summary_sha256": L._sha(Path(run_dir) / "RUN_SUMMARY.json")}
    obj.update(over)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def write_anchor(root, value, eval_id=CALIB) -> Path:
    od = Path(root) / "ANCHOR"
    od.mkdir(parents=True)
    row = {"method": "P2_ANCHOR_PRE_CALIB", "aggregation": "macro", "target_denominator": "E2E", "metric": "mrr",
           "k": 20, "value": value, "value_status": "OK", "eval_manifest_id": eval_id, "split": "VALIDATION"}
    (od / "RESULT_ROWS.json").write_text(json.dumps({"role": "CALIBRATION", "rows": [row]}), encoding="utf-8")
    (od / "REEVAL_RECORD.json").write_text("{}", encoding="utf-8")
    return od


def write_c1(root, variant, lr, value, *, mark=3.0, eval_id=CALIB, role="CALIBRATION", numel=20_700_000) -> Path:
    """A finished C1 run in the layout central_ft.read_c1_run verifies (RUN_START identity first)."""
    rid = CF.c1_run_id(variant, "CALIBRATION", lr, 2026)
    rd = Path(root) / rid
    rd.mkdir(parents=True)
    arm = CF.C1_ARMS[variant]
    fin = isinstance(value, float) and math.isfinite(value)
    start = {"run_id": rid, "seed": 2026, "eval_provider": {"manifest_id": eval_id},
             "p2_c1": {"identity": {"arm": arm, "variant": variant, "role": role, "peak_lr": lr,
                                    "frozen_item_tables": variant == "FROZEN"}, "trained_numel": numel}}
    summ = {"status": "DONE_MAIN", "checkpoints": {}, "p2_c1": {"arm": arm},
            "evals": [{"mark": 0.5 * i} for i in range(1, 13)],
            "practical_best": {"mark": mark, "value": value if fin else None, "efe": mark}}
    row = {"selection_rule": "PRACTICAL_BEST", "aggregation": "macro", "target_denominator": "E2E", "metric": "mrr",
           "k": 20, "value": value, "eval_manifest_id": eval_id, "method": arm, "row_status": "P2_CONTROL",
           "c_matched_eligible": False, "efe": mark}
    (rd / "RUN_START.json").write_text(json.dumps(start), encoding="utf-8")
    (rd / "RUN_SUMMARY.json").write_text(json.dumps(summ), encoding="utf-8")
    (rd / "RESULT_ROWS.json").write_text(json.dumps({"run_id": rid, "rows": [row]}), encoding="utf-8")
    return rd


WA_WIN = R("fedadam", 1e-4, 0.1)                                    # WA-ext 1e-4: interior -> NOT_ON_EDGE
INC = R("fedadam", 1e-3, 0.1)


def wa_values(top=0.1716, win=0.1710, rest=0.1690):
    """label -> value: the incumbent is the raw best; the WA-ext 1e-4 cell is 0.0006 below it (in the 0.001 band) and
    wins on the smaller server step; every other cell is out of the band."""
    v = {lab: rest - 0.0001 * (i % 5) for i, lab in enumerate(sorted(P.wa_cells()))}
    v[P.p2_label(INC)], v[P.p2_label(WA_WIN)] = top, win
    return v


def make_wa(root, values=None, *, nan=(), incumbent_phase1=False, d1ext=None) -> list:
    vals = values or wa_values()
    dirs = []
    for lab, (_st, r) in P.wa_cells().items():
        if incumbent_phase1 and r == INC:
            continue
        dirs.append(write_p2(root, r, float("nan") if lab in nan else vals[lab], d1ext=(d1ext or {}).get(lab)))
    return dirs


def c1_values(base=0.1690, full_best=0.1705, frozen_edge=0.1708):
    v = {}
    for var in P.C1_LABELS:
        for i, lr in enumerate(P.C1_PEAK_LRS):
            v[(var, lr)] = base - 0.0001 * i
    v[("FULL", 1e-4)], v[("FULL", 3e-5)] = full_best, full_best - 0.0005        # tie -> the smaller LR 3e-5
    v[("FROZEN", 3e-4)] = frozen_edge                                             # on the edge -> 1e-3 required
    return v


def make_c1(root, values=None, *, nan=()) -> list:
    vals = values or c1_values()
    return [write_c1(root, var, lr, float("nan") if (var, lr) in nan else vals[(var, lr)])
            for var in P.C1_LABELS for lr in P.C1_PEAK_LRS]


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ registered sets
def test_registered_cell_sets_and_launch_fl_alignment():
    wa = P.wa_cells()
    sets = [s for s, _r in wa.values()]
    assert len(wa) == 21 and sets.count("WA_CORE") == 18 and sets.count("WA_EXT") == 2 and sets.count("INCUMBENT") == 1
    assert sorted(wa) == sorted(F.p2_label(r) for r in F.p2_wa_recipes())          # = launch_fl's 21 WA recipes
    assert P.RECIPE_KEYS == F.P2_RECIPE_KEYS and P.P2_ARMS == F.P2_ARMS
    edges = [e["recipe"] for _l, (_s, r) in wa.items() for e in P.edge_extension("WA", r)]
    for r in [r for _s, r in wa.values()] + edges + [r for _s, r in P.final_cells(WA_WIN).values()]:
        assert P.p2_label(r) == F.p2_label(r)
    for r in [r for _s, r in wa.values()] + edges:                                  # every WA / edge cell launchable
        assert F.check_p2_recipe(raw(r)) == r
    fc = P.final_cells(WA_WIN)
    assert sorted(s for s, _ in fc.values()) == ["A_STAR", "WB1", "WB2", "WB3", "WC1", "WC2", "WC3", "WC4", "WC5"]
    d = P.final_cells(WA_WIN, P.check_dropped(["DP1", "SECAGG_MAIN", "DPFTRL_S2027", "WC3/WC4", "WB2"]))
    assert sorted(s for s, _ in d.values()) == ["A_STAR", "WB1", "WB3", "WC1", "WC2", "WC5"]
    assert list(P.final_cells(WA_WIN, futility_skip=True)) == [P.p2_label(WA_WIN)]
    assert len(P.c1_cells()) == 8
    assert (P.TIE_CALIB, P.FUTILITY_MIN, P.ANCHOR_MIN_GAIN, P.CAP_GPU_H) == (0.001, 0.001, 0.001, 250.0)
    with pytest.raises(P.P2SelectionRefused):
        P.p2_recipe(server_opt="fedavg", server_lr=0.3, client_lr=0.01, upload="delta_rows")
    with pytest.raises(P.P2SelectionRefused):
        P.p2_recipe(server_opt="fedprox", server_lr=0.3, client_lr=0.01)


# ------------------------------------------------------------------------------------------------ §5.6 decision
def C(recipe, value, *, bytes_up=S8_BYTES, numel=S8_NUMEL, excluded=False, set_name="WA_CORE"):
    return {"label": P.recipe_label(recipe), "recipe": recipe, "value": None if excluded else value,
            "excluded": excluded, "bytes_per_visit": bytes_up,
            "trained_params": [numel, 1 if recipe.get("personal_head") else 0], "set": set_name}


def test_effective_server_step_is_exact():
    avgm, avg = R("fedavgm", 0.03, 0.01), R("fedavg", 0.3, 0.01)
    assert 0.03 / (1 - 0.9) != 0.3                              # the float trap the exact rational avoids
    assert P.server_step(avgm) == P.server_step(avg) == P.Fraction(3, 10)
    assert P.server_step(R("fedavgm", 0.1, 0.01)) == 1
    lora = R("fedadam", 1e-4, 0.01, lora_rank=4, lora_lr_factor=10)
    assert P.client_lr_applied(lora) == P.Fraction(1, 10)
    d = P.p2_decide("WA", [C(avg, 0.1700), C(avgm, 0.1700), C(R("fedavg", 0.3, 0.003), 0.1699)], 0.1600)
    assert d["winner"] == "fedavg|eta=0.3|c=0.003"
    d = P.p2_decide("WA", [C(avg, 0.1700), C(avgm, 0.1700)], 0.1600)
    assert d["winner"] == "fedavgm|eta=0.03|c=0.01"                     # (3) and (4) equal -> (5) the label ('m' < '|')


def test_tie_preference_order():
    a = R("fedadam", 1e-4, 0.01)
    big = C(R("fedadam", 1e-4, 0.01, items="unfrozen", item_lr_mult=0.1), 0.1716, bytes_up=82_800_000,
            numel=20_700_000)
    small = C(R("fedadam", 1e-4, 0.01, lora_rank=16, lora_lr_factor=1), 0.1710, bytes_up=400_000, numel=100_000)
    d = P.p2_decide("FINAL", [big, small], 0.16)
    assert d["winner"] == small["label"] and d["best_value_label"] == big["label"]      # (1) bytes
    pf = C(R("fedadam", 1e-4, 0.01, personal_head=True), 0.1716)
    assert P.p2_decide("FINAL", [C(a, 0.1710), pf], 0.16)["winner"] == P.p2_label(a)     # (2) trained params
    d = P.p2_decide("WA", [C(R("fedadam", 3e-4, 0.01), 0.1716), C(a, 0.1707)], 0.16)
    assert d["winner"] == P.p2_label(a)                                                   # (3) server step
    d = P.p2_decide("WA", [C(R("fedadam", 1e-4, 0.03), 0.1716), C(R("fedadam", 1e-4, 0.003), 0.1707)], 0.16)
    assert d["winner"] == "fedadam|eta=0.0001|c=0.003"                                     # (4) client LR


def test_tie_band_boundary_exact():
    hi, lo = R("fedadam", 3e-4, 0.01), R("fedadam", 1e-4, 0.01)
    assert 0.1716 - 0.1706 > 0.001                                       # float would NOT tie an exact 0.001 gap
    for v, tied in ((0.1706, True), (0.1706 + 1e-12, True), (0.1706 - 1e-12, False), (0.1705, False)):
        d = P.p2_decide("WA", [C(hi, 0.1716), C(lo, v)], 0.16)
        assert (d["winner"] == P.p2_label(lo)) is tied and (len(d["tie_set"]) == 2) is tied, v
    assert 0.1720 - 0.1710 < 0.001                                       # ... and float rounds this one the other way
    assert len(P.p2_decide("WA", [C(hi, 0.1720), C(lo, 0.1710)], 0.16)["tie_set"]) == 2


def test_anchor_flag_threshold_0_001_and_boundary():
    a = R("fedadam", 1e-4, 0.01)
    assert P.p2_decide("WA", [C(a, 0.1706)], 0.1700)["no_measurable_calib_gain"]           # 0.0006 < 0.001: flagged
    assert P.p2_decide("WA", [C(a, 0.1709)], 0.1700)["no_measurable_calib_gain"]
    for anc, flagged in ((0.1706, False), (0.1706 + 1e-12, True), (0.1706 - 1e-12, False)):
        d = P.p2_decide("WA", [C(a, 0.1716)], anc)
        assert d["no_measurable_calib_gain"] is flagged and d["winner"] == P.p2_label(a), anc   # carried regardless
    assert P.p2_decide("WA", [C(a, 0.1716)], 0.1706)["calib_gain"] == 0.001


def test_nonfinite_selection_value_rules():
    a = R("fedadam", 1e-4, 0.01)
    d = P.p2_decide("WA", [C(a, 0.1704), C(R("fedadam", 3e-4, 0.01), None, excluded=True)], 0.17)
    assert d["excluded"] == ["fedadam|eta=0.0003|c=0.01"] and d["n_ranked"] == 1
    with pytest.raises(P.P2SelectionRefused):
        P.p2_decide("WA", [C(a, float("nan"))], 0.17)
    with pytest.raises(P.P2SelectionRefused):
        P.p2_decide("WA", [C(a, None, excluded=True)], 0.17)
    with pytest.raises(P.P2SelectionRefused):
        P.p2_decide("WA", [C(a, 0.17)], float("nan"))
    with pytest.raises(P.P2SelectionRefused):
        P.p2_decide("C1", [C(P.c1_recipe("FULL", 1e-5), 0.17), C(P.c1_recipe("FROZEN", 1e-5), 0.17)], 0.16)


# ------------------------------------------------------------------------------------------------ §5.7 edges
def _labels(cells):
    return sorted(e["label"] for e in cells)


def test_edge_step_table_no_corner():
    E = P.edge_extension
    assert _labels(E("WA", R("fedavg", 1.0, 0.03))) == ["fedavg|eta=1|c=0.1", "fedavg|eta=2|c=0.03"]   # no corner
    assert _labels(E("WA", R("fedavgm", 0.03, 0.01))) == ["fedavgm|eta=0.01|c=0.01"]
    assert _labels(E("WA", R("fedavgm", 0.1, 0.003))) == ["fedavgm|eta=0.1|c=0.001", "fedavgm|eta=0.3|c=0.003"]
    assert _labels(E("WA", R("fedadam", 3e-4, 0.03))) == ["fedadam|eta=0.0003|c=0.1", "fedadam|eta=0.001|c=0.03"]
    assert _labels(E("WA", R("fedadam", 1e-4, 0.01))) == ["fedadam|eta=3e-05|c=0.01"]
    assert _labels(E("WA", R("fedadam", 1e-5, 0.1))) == ["fedadam|eta=3e-06|c=0.1"]
    assert E("WA", WA_WIN) == [] and E("WA", INC) == [] and E("WA", R("fedadam", 3e-5, 0.01)) == []
    assert max(len(E("WA", r)) for _s, r in P.wa_cells().values()) == 2 <= P.MAX_EXTENSION_RUNS
    wb2 = R("fedadam", 1e-4, 0.1, items="unfrozen", item_lr_mult=0.01)
    assert _labels(E("FINAL", wb2)) == ["fedadam|eta=0.0001|c=0.1|items=unfrozen|ilr=0.001"]
    assert E("FINAL", R("fedadam", 1e-4, 0.1, items="unfrozen", item_lr_mult=0.1)) == []
    assert E("FINAL", R("fedadam", 1e-4, 0.1, items="unfrozen", item_lr_mult=0.1, upload="delta_rows")) == []
    assert _labels(E("FINAL", R("fedadam", 1e-4, 0.1, lora_rank=16, lora_lr_factor=10))) == [
        "fedadam|eta=0.0001|c=0.1|lora=r64|lrx=10"]
    assert _labels(E("FINAL", R("fedadam", 1e-4, 0.1, lora_rank=4, lora_lr_factor=1))) == [
        "fedadam|eta=0.0001|c=0.1|lora=r1|lrx=1"]
    assert E("FINAL", WA_WIN) == [] and E("FINAL", R("fedadam", 1e-4, 0.1, personal_head=True)) == []
    assert _labels(E("C1", P.c1_recipe("FULL", 1e-5))) == ["P2_C1_FT_CENTRAL_FULL|lr=3e-06"]
    assert _labels(E("C1", P.c1_recipe("FROZEN", 3e-4))) == ["P2_C1_FT_CENTRAL_FROZEN|lr=0.001"]
    assert E("C1", P.c1_recipe("FULL", 3e-5)) == []


def _wa_base(values):
    return [C(r, values[lab], set_name=st) for lab, (st, r) in P.wa_cells().items()]


def test_single_extension_statuses_and_notes():
    v = wa_values(top=0.1690, win=0.1690)
    w0 = R("fedavgm", 0.03, 0.01)
    v[P.p2_label(w0)] = 0.1712
    base = _wa_base(v)
    ext = R("fedavgm", 0.01, 0.01)
    with pytest.raises(P.P2EdgeRunsMissing) as e:
        P.stage_outcome("WA", base, [], 0.165)
    assert [n["label"] for n in e.value.needed] == [P.p2_label(ext)]
    out = P.stage_outcome("WA", base, [C(ext, 0.1700, set_name="WA_EDGE")], 0.165)        # extension loses
    assert out["decision"]["winner"] == P.p2_label(w0) and out["edge"]["status"] == "EDGE_RESOLVED"
    assert out["edge"]["notes"] == []
    out = P.stage_outcome("WA", base, [C(ext, 0.1725, set_name="WA_EDGE")], 0.165)        # extension wins
    assert out["decision"]["winner"] == P.p2_label(ext) and out["edge"]["status"] == "EDGE_UNRESOLVED"
    out = P.stage_outcome("WA", base, [C(ext, None, excluded=True, set_name="WA_EDGE")], 0.165)
    assert out["edge"]["status"] == "EDGE_RESOLVED" and "non-finite" in out["edge"]["notes"][0]   # literal §5.7
    with pytest.raises(P.P2SelectionRefused):                                             # unregistered extra run
        P.stage_outcome("WA", base, [C(ext, 0.17, set_name="WA_EDGE"), C(R("fedavgm", 0.3, 0.01), 0.17)], 0.165)
    out = P.stage_outcome("WA", _wa_base(wa_values()), [], 0.165)
    assert out["decision"]["winner"] == P.p2_label(WA_WIN) and out["edge"]["status"] == "NOT_ON_EDGE"
    with pytest.raises(P.P2SelectionRefused):
        P.stage_outcome("WA", _wa_base(wa_values()), [C(R("fedadam", 3e-6, 0.1), 0.16)], 0.165)


def test_extension_existing_cell_and_moved_winner_note():
    v = wa_values(top=0.1690, win=0.1690)
    w0 = R("fedadam", 1e-4, 0.03)                                   # client step 0.03 -> 0.1 = the WA-ext cell
    v[P.p2_label(w0)] = 0.1712
    with pytest.raises(P.P2EdgeRunsMissing) as e:
        P.stage_outcome("WA", _wa_base(v), [], 0.165)
    assert [n["label"] for n in e.value.needed] == ["fedadam|eta=3e-05|c=0.03"]
    out = P.stage_outcome("WA", _wa_base(v), [C(R("fedadam", 3e-5, 0.03), 0.1700)], 0.165)
    assert out["edge"]["existing_cells"] == [P.p2_label(WA_WIN)] and out["edge"]["status"] == "EDGE_RESOLVED"
    v = wa_values(top=0.1680, win=0.1680)                          # A0 < B < E: d0 -> A0, union -> B (base)
    a0, b = R("fedadam", 3e-4, 0.01), R("fedadam", 3e-4, 0.03)
    v[P.p2_label(a0)], v[P.p2_label(b)] = 0.1700, 0.1706
    base = _wa_base(v)
    assert P.p2_decide("WA", base, 0.165)["winner"] == P.p2_label(a0)
    out = P.stage_outcome("WA", base, [C(R("fedadam", 1e-3, 0.01), 0.1715)], 0.165)
    assert out["decision"]["winner"] == P.p2_label(b) and out["edge"]["status"] == "EDGE_RESOLVED"
    assert any("moved" in n for n in out["edge"]["notes"])


def test_final_edge_cells_pure():
    """FINAL WB2 / LoRA edges (WB3 / LoRA runs cannot be launched until P2-03 / P2-04 land: pure records here)."""
    a = WA_WIN
    cells = P.final_cells(a)
    vals = {"WB1": 0.1700, "WB2": 0.1730, "WB3": 0.1705, "WC1": 0.1702, "WC2": 0.1701, "WC3": 0.1699,
            "WC4": 0.1698, "WC5": 0.1709, "A_STAR": 0.1710}
    by = {"WB1": (82_800_000, 20_700_000), "WB2": (82_800_000, 20_700_000), "WB3": (2_700_000, 20_700_000),
          "WC1": (100_000, 25_000), "WC2": (400_000, 100_000), "WC3": (100_000, 25_000), "WC4": (400_000, 100_000),
          "WC5": (S8_BYTES, S8_NUMEL), "A_STAR": (S8_BYTES, S8_NUMEL)}
    base = [C(r, vals[s], bytes_up=by[s][0], numel=by[s][1], set_name=s) for _l, (s, r) in cells.items()]
    ext = R("fedadam", 1e-4, 0.1, items="unfrozen", item_lr_mult=0.001)
    with pytest.raises(P.P2EdgeRunsMissing):
        P.stage_outcome("FINAL", base, [], 0.165)
    out = P.stage_outcome("FINAL", base, [C(ext, 0.1715, bytes_up=82_800_000, numel=20_700_000)], 0.165)
    assert out["decision"]["winner"].endswith("ilr=0.01") and out["edge"]["status"] == "EDGE_RESOLVED"
    out = P.stage_outcome("FINAL", base, [C(ext, 0.1722, bytes_up=82_800_000, numel=20_700_000)], 0.165)
    assert out["edge"]["status"] == "EDGE_UNRESOLVED"               # tied -> label 'ilr=0.001' < 'ilr=0.01' (Q2)


# ------------------------------------------------------------------------------------------------ §5.8 / §5.9
def test_futility_strict_0_001_boundaries():
    assert P.futility(0.1709, 0.1709, 0.1700)["decision"] == "FUTILITY_SKIP_WB_WC"
    assert P.futility(0.1711, 0.1700, 0.1700)["decision"] == "CONTINUE_WB_WC"
    assert P.futility(0.1700, 0.1711, 0.1700)["decision"] == "CONTINUE_WB_WC"
    for anc, skip in ((0.1706, False), (0.1706 + 1e-12, True), (0.1706 - 1e-12, False)):
        assert P.futility(0.1716, 0.1716, anc)["skip_wb_wc"] is skip, anc
    assert P.futility(0.1716, 0.1700, 0.1706)["skip_wb_wc"] is False           # C1 gain exactly 0.001
    r = P.futility(0.1690, 0.1690, 0.1700)
    assert r["skip_wb_wc"] and r["disclosure"].startswith("D-j") and r["threshold"] == 0.001
    with pytest.raises(P.P2SelectionRefused):
        P.futility(float("nan"), 0.17, 0.17)


SAV = {u: 5.0 for u in P.DROP_ORDER}


def test_drop_order_single_strict_order_and_cap_plan():
    assert P.check_dropped([]) == [] and P.check_dropped(["DP1"]) == ["DP1"]
    assert P.check_dropped(list(reversed(P.DROP_ORDER))) == list(P.DROP_ORDER)
    for bad in (["SECAGG_MAIN"], ["DPFTRL_S2027"], ["DP1", "DPFTRL_S2027"], ["WC3/WC4"],
                ["DP1", "SECAGG_MAIN", "DPFTRL_S2027", "WB2"], ["C2"], ["WB1"], ["DP1", "DP1"]):
        with pytest.raises(P.P2SelectionRefused):
            P.check_dropped(bad)
    p = P.cap_drop_plan(260.0, SAV)
    assert p["dropped"] == ["DP1", "SECAGG_MAIN"] and p["within_cap"] and p["projected_after"] == 250.0
    for proj in (190.0, 251.0, 284.0, 300.0, 1000.0):
        plan = P.cap_drop_plan(proj, dict(SAV, DP1=0.0))
        assert P.check_dropped(plan["dropped"]) == plan["dropped"]                 # always a prefix
    p = P.cap_drop_plan(300.0, SAV)
    assert p["dropped"] == list(P.DROP_ORDER) and p["status"] == "CAP_EXCEEDED_PRIMARY_PATH"
    assert P.cap_drop_plan(190.0, SAV)["dropped"] == []
    with pytest.raises(P.P2SelectionRefused):
        P.cap_drop_plan(300.0, {"DP1": 7.0})                                        # every unit's savings required


# ------------------------------------------------------------------------------------------------ files (real readers)
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("p2sel")
    anchor = write_anchor(root, 0.1680)
    wa = make_wa(root / "wa")
    c1 = make_c1(root / "c1")
    c1_edge = write_c1(root / "c1", "FROZEN", 1e-3, 0.1700)
    files = root / "contracts_fl"
    rec_c1 = P.write_selection("C1", files / "P2_C1_SELECTION.json", runs=c1 + [c1_edge], anchor=anchor)
    rec_wa = P.write_selection("WA", files / "P2_WARM_SELECTION_A.json", runs=wa, anchor=anchor)
    return {"root": root, "anchor": anchor, "wa": wa, "c1": c1, "c1_edge": c1_edge, "files": files,
            "rec_c1": rec_c1, "rec_wa": rec_wa}


def test_real_p2_run_read_without_monkeypatch(world):
    c = P.read_p2_run(world["wa"][0])
    assert c["source"] == "P2_RUN" and c["eval_manifest_id"] == CALIB and c["value"] is not None
    assert c["_spec"] == p2_spec(c["recipe"]) and c["_common"]["init_from"] == IDENT
    assert c["_common"]["cfg.solver.passes"] == 1 and c["nonfinite_fields"] == []
    with pytest.raises(L.SelectionRefused):                                         # the reason this reader exists
        L.read_run(world["wa"][0], arms=P.P2_ARMS)


def test_wa_file_written_once_verified_and_tamper_refused(world, tmp_path):
    rec = world["rec_wa"]
    obj = J(rec["path"])
    assert obj["format"] == "PPSI_V4_P2_WARM_SELECTION_A_1" and obj["self_sha256"] == F._canon(obj)
    assert obj["winner"] == P.p2_label(WA_WIN) and rec["winner"] == obj["winner"]
    assert obj["tie_set"] == [P.p2_label(WA_WIN), P.p2_label(INC)] and obj["best_value_label"] == P.p2_label(INC)
    assert obj["edge"]["status"] == "NOT_ON_EDGE" and obj["flags"] == [] and obj["anchor_min_gain"] == 0.001
    assert len(obj["candidates"]) == 21 and obj["common"]["init_from"] == IDENT
    assert obj["tool"]["p2_select_sha256"] == L._sha(Path(P.__file__))
    assert P.PREG2_SHA256 in obj["authority"]
    assert P.load_p2_selection(rec["path"], "WA")["self_sha256"] == obj["self_sha256"]
    with pytest.raises(P.P2SelectionRefused):                                      # O_EXCL
        P.write_selection("WA", Path(rec["path"]), runs=world["wa"], anchor=world["anchor"])
    bad = tmp_path / "tampered.json"
    t = dict(obj)                                                                  # tampered value, re-signed
    t["candidates"] = [dict(c, value=0.1730) if c["label"] == P.p2_label(INC) else c for c in obj["candidates"]]
    t["self_sha256"] = F._canon(t)
    bad.write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(P.P2SelectionRefused):
        P.load_p2_selection(bad, "WA")
    t = dict(obj, flags=[])                                                        # flags re-derived
    t["flags"] = ["NO_MEASURABLE_CALIB_GAIN"]
    t["self_sha256"] = F._canon(t)
    bad.write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(P.P2SelectionRefused):
        P.load_p2_selection(bad, "WA")
    t = dict(obj, winner="fedavg|eta=1|c=0.01")                                    # self-sha no longer verifies
    bad.write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(P.P2SelectionRefused):
        P.load_p2_selection(bad, "WA")


def test_wa_refusals(world, tmp_path):
    a = world["anchor"]
    out = tmp_path / "A.json"
    ph1 = write_phase1_incumbent(tmp_path / "ph1", 0.1716)
    with pytest.raises(P.P2SelectionRefused):                                      # Phase-1 label among --runs
        P.write_selection("WA", out, runs=world["wa"] + [ph1], anchor=a)
    with pytest.raises(P.P2SelectionRefused):
        P.read_p2_run(ph1)
    with pytest.raises(P.P2SelectionRefused):                                      # a required run missing
        P.write_selection("WA", out, runs=world["wa"][1:], anchor=a)
    for eid in ("MAIN_VALIDATION_5_100_131072", "MAIN_TEST_5_100_131072"):         # MAIN / TEST manifest refused
        with pytest.raises(P.P2SelectionRefused):
            P.read_p2_run(write_p2(tmp_path / eid, WA_WIN, 0.17, eval_id=eid))
    m = write_run(tmp_path / "role", "main_role", p2_spec(WA_WIN, role="MAIN"), 0.17)
    with pytest.raises(P.P2SelectionRefused):                                      # MAIN role refused unread
        P.read_p2_run(m)
    hand = p2_spec(WA_WIN)
    hand["cfg"]["solver"]["clip"] = 2.0                                           # hand-edited spec
    with pytest.raises(P.P2SelectionRefused):
        P.read_p2_run(write_run(tmp_path / "hand", "hand", hand, 0.17))
    with pytest.raises(P.P2SelectionRefused):                                      # anchor on a MAIN manifest
        P.read_anchor(write_anchor(tmp_path / "anc", 0.168, eval_id="MAIN_VALIDATION_5_100_131072"))
    assert not out.exists()


def test_diverged_run_excluded_and_nan_diagnostics_never_block(tmp_path):
    """REV M3: a NaN-valued run whose D1-ext summary carries NaN is excluded and recorded; a finite run with a NaN in
    its D1-ext and at a non-best mark is flagged and ranked; the file is written and verifies."""
    a = write_anchor(tmp_path, 0.1680)
    nan_lab, flag_lab = "fedavgm|eta=0.1|c=0.03", "fedavg|eta=0.3|c=0.01"
    d1 = {nan_lab: {"file": "UPDATE_NORM_LOG.jsonl", "last_total_T": float("nan"), "complete": True},
          flag_lab: {"file": "UPDATE_NORM_LOG.jsonl", "last_total_T": float("inf"), "complete": True}}
    dirs = make_wa(tmp_path / "wa", nan=(nan_lab,), d1ext=d1)
    fl = next(d for d in dirs if d.name == flag_lab.replace("|", "_").replace("=", ""))
    s = J(fl / "RUN_SUMMARY.json")
    s["evals"][11][2] = float("nan")                                               # a NaN at mark 6.0 (best = 4.0)
    (fl / "RUN_SUMMARY.json").write_text(json.dumps(s), encoding="utf-8")
    rec = P.write_selection("WA", tmp_path / "A.json", runs=dirs, anchor=a)
    obj = P.load_p2_selection(rec["path"], "WA")
    assert [x["label"] for x in obj["excluded"]] == [nan_lab] and nan_lab not in obj["values"]
    exc = next(c for c in obj["candidates"] if c["label"] == nan_lab)
    assert exc["excluded"] and exc["value"] is None and exc["d1ext"]["last_total_T"] == "NaN"
    flg = next(c for c in obj["candidates"] if c["label"] == flag_lab)
    assert not flg["excluded"] and flg["value"] is not None and flg["d1ext"]["last_total_T"] == "Infinity"
    assert flg["nonfinite_fields"] == ["d1ext.last_total_T", "evals[11]"]
    assert obj["nonfinite_diagnostics"] == sorted([nan_lab, flag_lab]) and obj["winner"] == P.p2_label(WA_WIN)


def test_wa_edge_required_cli_exit_codes_and_no_value_printed(tmp_path):
    a = write_anchor(tmp_path, 0.1680)
    v = wa_values(top=0.1690, win=0.1690)
    w0 = R("fedavg", 1.0, 0.01)
    v[P.p2_label(w0)] = 0.1712
    dirs = make_wa(tmp_path / "wa", v)
    out = tmp_path / "A.json"
    with pytest.raises(P.P2EdgeRunsMissing) as e:
        P.write_selection("WA", out, runs=dirs, anchor=a)
    assert [n["label"] for n in e.value.needed] == ["fedavg|eta=2|c=0.01"]
    assert P.main(["select", "--stage", "WA", "--runs", *map(str, dirs), "--anchor", str(a), "--out", str(out)]) == 3
    ext = write_p2(tmp_path / "ext", R("fedavg", 2.0, 0.01), 0.1725)
    said = []
    assert P.main(["select", "--stage", "WA", "--runs", *map(str, dirs + [ext]), "--anchor", str(a),
                   "--out", str(out)], say=said.append) == 0
    printed = json.loads(said[0])
    assert printed["winner"] == "fedavg|eta=2|c=0.01" and printed["flags"] == ["EDGE_UNRESOLVED"]
    assert not any(k in said[0] for k in ("0.1725", "0.1712"))
    obj = P.load_p2_selection(out, "WA")
    assert obj["edge"]["status"] == "EDGE_UNRESOLVED" and obj["edge"]["winner_before_extension"] == P.p2_label(w0)
    assert next(c for c in obj["candidates"] if c["label"] == "fedavg|eta=2|c=0.01")["set"] == "WA_EDGE"
    assert obj["wa_best"] == 0.1725
    assert P.main(["select", "--stage", "WA", "--runs", *map(str, dirs + [ext]), "--anchor", str(a),
                   "--out", str(out)]) == 2                                        # write-once via the CLI too


def test_phase1_incumbent_receipt_cfg_and_theta_pre(tmp_path):
    a = write_anchor(tmp_path, 0.1680)
    dirs = make_wa(tmp_path / "wa", incumbent_phase1=True)
    ph1 = write_phase1_incumbent(tmp_path / "ph1", 0.1716)
    good = receipt_for(ph1, tmp_path / "rc.json")
    with pytest.raises(P.P2SelectionRefused):                                      # no receipt
        P.write_selection("WA", tmp_path / "A0.json", runs=dirs, anchor=a, incumbent_phase1=ph1)
    for over in ({"verdict": "DIFFERS"}, {"rounds": 32}, {"s8tc_run_id": "other"}, {"result_rows_sha256": "0" * 64}):
        bad = receipt_for(ph1, tmp_path / f"rc_{sorted(over)[0]}.json", **over)
        with pytest.raises(P.P2SelectionRefused):
            P.write_selection("WA", tmp_path / "A0.json", runs=dirs, anchor=a, incumbent_phase1=ph1, p2t3_receipt=bad)
    (tmp_path / "rc_empty.json").write_text("{}", encoding="utf-8")
    with pytest.raises(P.P2SelectionRefused):
        P.write_selection("WA", tmp_path / "A0.json", runs=dirs, anchor=a, incumbent_phase1=ph1,
                          p2t3_receipt=tmp_path / "rc_empty.json")
    with pytest.raises(P.P2SelectionRefused):                                      # incumbent missing altogether
        P.write_selection("WA", tmp_path / "A0.json", runs=dirs, anchor=a)
    rec = P.write_selection("WA", tmp_path / "A.json", runs=dirs, anchor=a, incumbent_phase1=ph1, p2t3_receipt=good)
    obj = P.load_p2_selection(rec["path"], "WA")
    assert obj["incumbent"]["source"] == "PHASE1_INCUMBENT" and obj["incumbent"]["p2t3_receipt"]["sha256"] == L._sha(good)
    assert obj["winner"] == P.p2_label(WA_WIN) and obj["incumbent"]["init_from"] == IDENT
    for name, over in (("theta", {"init": dict(IDENT, checkpoint_sha256="f" * 64)}),   # another theta_pre
                       ("passes2", {"local_passes": None})):                         # cfg differs (passes 2)
        ph = write_phase1_incumbent(tmp_path / name, 0.1716, name=f"S8_{name}", **over)
        with pytest.raises(P.P2SelectionRefused):
            P.write_selection("WA", tmp_path / f"A_{name}.json", runs=dirs, anchor=a, incumbent_phase1=ph,
                              p2t3_receipt=receipt_for(ph, tmp_path / f"rc_{name}.json", s8tc_run_id=f"S8_{name}"))
    ph = write_phase1_incumbent(tmp_path / "sf1", 0.1716, name="S8_sf1", warm_server_lr_factor=None)   # not the cell
    with pytest.raises(P.P2SelectionRefused):
        P.write_selection("WA", tmp_path / "A_sf1.json", runs=dirs, anchor=a, incumbent_phase1=ph,
                          p2t3_receipt=receipt_for(ph, tmp_path / "rc_sf1.json", s8tc_run_id="S8_sf1"))


def test_c1_file_per_variant_edges_and_refusals(world, tmp_path):
    obj = P.load_p2_selection(world["rec_c1"]["path"], "C1")
    full, frozen = obj["variants"]["FULL"], obj["variants"]["FROZEN"]
    assert full["winner"] == "P2_C1_FT_CENTRAL_FULL|lr=3e-05" and full["edge"]["status"] == "NOT_ON_EDGE"
    assert full["tie_set"] == ["P2_C1_FT_CENTRAL_FULL|lr=3e-05", "P2_C1_FT_CENTRAL_FULL|lr=0.0001"]
    assert frozen["winner"] == "P2_C1_FT_CENTRAL_FROZEN|lr=0.0003" and frozen["edge"]["status"] == "EDGE_RESOLVED"
    assert full["winner_peak_lr"] == 3e-5 and frozen["winner_peak_lr"] == 3e-4 and obj["c1_best"] == 0.1708
    a = world["anchor"]
    with pytest.raises(P.P2EdgeRunsMissing):                                       # FROZEN 3e-4 edge run absent
        P.selection_c1(world["c1"], a)
    o = P.selection_c1(make_c1(tmp_path / "n", nan=(("FULL", 1e-5),)) + [write_c1(tmp_path / "n", "FROZEN", 1e-3,
                                                                                    0.1720)], a)
    assert [x["label"] for x in o["excluded"]] == ["P2_C1_FT_CENTRAL_FULL|lr=1e-05"]
    assert o["variants"]["FROZEN"]["edge"]["status"] == "EDGE_UNRESOLVED"          # the C1 extension wins
    with pytest.raises(P.P2SelectionRefused):                                      # mark 0 never a candidate
        P.read_c1_run(write_c1(tmp_path / "m0", "FULL", 1e-5, 0.17, mark=0.0))
    for kw in ({"eval_id": "MAIN_VALIDATION_5_100_131072"}, {"eval_id": "MAIN_TEST_5_100_131072"}, {"role": "MAIN"}):
        with pytest.raises(P.P2SelectionRefused):
            P.read_c1_run(write_c1(tmp_path / f"x{sorted(kw.values())[0]}", "FULL", 1e-5, 0.17, **kw))


# ------------------------------------------------------------------------------------------------ FINAL
ALL_EXPLORATION = ["DP1", "SECAGG_MAIN", "DPFTRL_S2027", "WC3/WC4", "WB2", "WC1/WC2", "WB3"]


def write_cap_plan(path, projected, savings=None) -> Path:
    path.write_text(json.dumps({"format": P.CAP_PLAN_FORMAT, "projected_gpu_h": projected,
                                "savings": savings or SAV}), encoding="utf-8")
    return path


def _a_star_dir(dirs, recipe):
    return next(d for d in dirs if J(Path(d) / "ARM_SPEC.json")["arm_spec"]["p2_recipe"] == raw(recipe))


def test_final_with_drops_bound_to_the_cap_plan(world, tmp_path):
    """WB3 (delta_rows) and WC1-WC4 (LoRA) are not launchable before P2-03 / P2-04: the real-run FINAL here drops every
    exploration cell through a cap plan whose projection requires exactly those drops."""
    wb1 = write_p2(tmp_path / "f", R("fedadam", 1e-4, 0.1, items="unfrozen", item_lr_mult=0.1), 0.1730,
                   bytes_up=82_800_000, numel=20_700_000)
    wc5 = write_p2(tmp_path / "f", R("fedadam", 1e-4, 0.1, personal_head=True), 0.1712)
    runs = [_a_star_dir(world["wa"], WA_WIN), wb1, wc5]
    kw = dict(anchor=world["anchor"], wa_selection=world["rec_wa"]["path"], c1_selection=world["rec_c1"]["path"])
    assert P.futility_from_files(kw["wa_selection"], kw["c1_selection"])["decision"] == "CONTINUE_WB_WC"
    with pytest.raises(P.P2SelectionRefused):                                      # WB2 / WB3 / WC1-4 missing
        P.write_selection("FINAL", tmp_path / "F0.json", runs=runs, **kw)
    with pytest.raises(P.P2SelectionRefused):                                      # drops without a cap plan
        P.write_selection("FINAL", tmp_path / "F0.json", runs=runs, dropped=ALL_EXPLORATION, **kw)
    small = write_cap_plan(tmp_path / "cap_small.json", 260.0)                     # needs only DP1 + SECAGG
    with pytest.raises(P.P2SelectionRefused):
        P.write_selection("FINAL", tmp_path / "F0.json", runs=runs, dropped=ALL_EXPLORATION, cap_plan=small, **kw)
    cap = write_cap_plan(tmp_path / "cap.json", 284.0)                             # 284 - 7 x 5 = 249: C2 kept
    with pytest.raises(P.P2SelectionRefused):                                      # A* run absent
        P.write_selection("FINAL", tmp_path / "F0.json", runs=runs[1:], dropped=ALL_EXPLORATION, cap_plan=cap, **kw)
    rec = P.write_selection("FINAL", tmp_path / "F.json", runs=runs, dropped=ALL_EXPLORATION, cap_plan=cap, **kw)
    obj = P.load_p2_selection(rec["path"], "FINAL")
    assert obj["winner"] == "fedadam|eta=0.0001|c=0.1|items=unfrozen|ilr=0.1" and obj["winner_set"] == "WB1"
    assert obj["edge"]["status"] == "NOT_ON_EDGE" and obj["flags"] == []
    assert obj["dropped"]["units"] == ALL_EXPLORATION and obj["dropped"]["cap_plan"]["projected_after"] == 249.0
    assert len(obj["required"]) == 3 and obj["common"]["init_from"] == IDENT
    assert obj["format"] == "PPSI_V4_P2_WARM_SELECTION_FINAL_1"
    t = dict(obj)                                                                  # tampered drop record
    t["dropped"] = dict(obj["dropped"], units=["WC3/WC4"])
    t["self_sha256"] = F._canon(t)
    (tmp_path / "Ft.json").write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(P.P2SelectionRefused):
        P.load_p2_selection(tmp_path / "Ft.json", "FINAL")


def test_final_futility_skip_and_no_gain_flag(tmp_path):
    a = write_anchor(tmp_path, 0.1712)                              # WA best 0.1716 / C1 best 0.1715: gains < 0.001
    wa = make_wa(tmp_path / "wa")
    c1 = make_c1(tmp_path / "c1", c1_values(base=0.1700, full_best=0.1715, frozen_edge=0.1714))
    c1e = write_c1(tmp_path / "c1", "FROZEN", 1e-3, 0.1700)
    fc = P.write_selection("C1", tmp_path / "C1.json", runs=c1 + [c1e], anchor=a)
    fa = P.write_selection("WA", tmp_path / "A.json", runs=wa, anchor=a)
    assert "NO_MEASURABLE_CALIB_GAIN" in fa["flags"]
    said = []
    assert P.main(["futility", "--wa-selection", fa["path"], "--c1-selection", fc["path"]], say=said.append) == 0
    assert json.loads(said[0])["decision"] == "FUTILITY_SKIP_WB_WC"
    a_dir = _a_star_dir(wa, WA_WIN)
    wb1 = write_p2(tmp_path / "wb", R("fedadam", 1e-4, 0.1, items="unfrozen", item_lr_mult=0.1), 0.18,
                   bytes_up=82_800_000, numel=20_700_000)
    kw = dict(anchor=a, wa_selection=fa["path"], c1_selection=fc["path"])
    with pytest.raises(P.P2SelectionRefused):                                      # WB / WC skipped: refused
        P.write_selection("FINAL", tmp_path / "F0.json", runs=[a_dir, wb1], **kw)
    rec = P.write_selection("FINAL", tmp_path / "F.json", runs=[a_dir], **kw)
    obj = P.load_p2_selection(rec["path"], "FINAL")
    assert obj["winner"] == P.p2_label(WA_WIN) and obj["futility"]["skip_wb_wc"]
    assert obj["flags"] == ["NO_MEASURABLE_CALIB_GAIN", "FUTILITY_SKIP_WB_WC"]
    assert obj["futility"]["disclosure"].startswith("D-j")
    other = write_anchor(tmp_path / "other", 0.1690)
    with pytest.raises(P.P2SelectionRefused):
        P.write_selection("FINAL", tmp_path / "F1.json", runs=[a_dir], anchor=other, wa_selection=fa["path"],
                          c1_selection=fc["path"])


def test_final_with_phase1_incumbent_as_a_star_under_futility(tmp_path):
    a = write_anchor(tmp_path, 0.1725)                              # incumbent 0.1730 - 0.1725 < 0.001
    dirs = make_wa(tmp_path / "wa", wa_values(top=0.1730, win=0.1710), incumbent_phase1=True)
    ph1 = write_phase1_incumbent(tmp_path / "ph1", 0.1730)
    rc = receipt_for(ph1, tmp_path / "rc.json")
    fa = P.write_selection("WA", tmp_path / "A.json", runs=dirs, anchor=a, incumbent_phase1=ph1, p2t3_receipt=rc)
    wa = J(fa["path"])
    assert wa["winner"] == P.p2_label(INC) and wa["winner_source"] == "PHASE1_INCUMBENT"
    c1 = make_c1(tmp_path / "c1", c1_values(base=0.1700, full_best=0.1715, frozen_edge=0.1714))
    fc = P.write_selection("C1", tmp_path / "C1.json", runs=c1 + [write_c1(tmp_path / "c1", "FROZEN", 1e-3, 0.17)],
                           anchor=a)
    kw = dict(anchor=a, wa_selection=fa["path"], c1_selection=fc["path"])
    with pytest.raises(P.P2SelectionRefused):                                      # A* must come via the flag
        P.write_selection("FINAL", tmp_path / "F0.json", runs=[dirs[0]], **kw)
    said = []                                                                      # CLI: no P2 run under futility
    assert P.main(["select", "--stage", "FINAL", "--anchor", str(a), "--wa-selection", fa["path"], "--c1-selection",
                   fc["path"], "--incumbent-phase1", str(ph1), "--p2t3-receipt", str(rc), "--out",
                   str(tmp_path / "F.json")], say=said.append) == 0
    rec = P.load_p2_selection(tmp_path / "F.json", "FINAL")
    assert rec["winner"] == P.p2_label(INC) and rec["winner_source"] == "PHASE1_INCUMBENT"
    assert rec["common"]["init_from"] == IDENT and "FUTILITY_SKIP_WB_WC" in rec["flags"]


# ------------------------------------------------------------------------------------------------ delta fixes N1 / N2 / C1
def _edit_row(rd: Path, edit) -> Path:
    rr = J(rd / "RESULT_ROWS.json")
    edit(rr["rows"][0])
    (rd / "RESULT_ROWS.json").write_text(json.dumps(rr), encoding="utf-8")
    return rd


def test_n1_extra_phase1_row_labels_refused(tmp_path):
    src = Path(L.__file__).read_text(encoding="utf-8")
    assert all(f'"{k}"' in src for k in P.READ_RUN_LABEL_KEYS)                    # = read_run's listed label set
    assert P.read_p2_run(write_p2(tmp_path / "ok", WA_WIN, 0.17))["value"] == 0.17
    for k, v in (("fl_server_tau", 1e-3), ("fl_warm_lr_factor", 1.0), ("fl_local_batch", 16),
                 ("fl_warm_server_lr_factor", 0.1), ("p2_item_lr_mult", 0.1)):
        rd = _edit_row(write_p2(tmp_path / k, WA_WIN, 0.17), lambda r, k=k, v=v: r.__setitem__(k, v))
        with pytest.raises(P.P2SelectionRefused, match="labels not in the spec"):
            P.read_p2_run(rd)


def test_n2_missing_or_null_schedule_mark_refused(tmp_path):
    for name, edit in (("missing", lambda r: r.pop("schedule_mark")), ("null", lambda r: r.__setitem__("schedule_mark", None)),
                       ("wrong", lambda r: r.__setitem__("schedule_mark", 1.0))):
        rd = _edit_row(write_p2(tmp_path / name, WA_WIN, 0.17), edit)
        with pytest.raises(P.P2SelectionRefused, match="best mark"):
            P.read_p2_run(rd)


def write_c1_diverged(root, variant, lr, *, error="TrainingDiverged: non-finite loss or gradient in the window ending "
                      "at update 4096", role="CALIBRATION", summary=None) -> Path:
    """A C1 run as v4train.trainer leaves it after TrainingDiverged: RUN_START + STATUS.json FAILED, no RUN_SUMMARY."""
    rd = write_c1(root, variant, lr, 0.17, role=role)
    (rd / "RUN_SUMMARY.json").unlink()
    (rd / "RESULT_ROWS.json").unlink()
    if summary is not None:
        (rd / "RUN_SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
    (rd / "STATUS.json").write_text(json.dumps({"run_id": rd.name, "status": "FAILED", "error": error}),
                                    encoding="utf-8")
    return rd


def test_c1_diverged_run_excluded_and_recorded(world, tmp_path):
    vals = c1_values()
    runs = [write_c1(tmp_path / "c1", var, lr, vals[(var, lr)]) for var in P.C1_LABELS for lr in P.C1_PEAK_LRS
            if (var, lr) != ("FULL", 1e-5)]
    div = write_c1_diverged(tmp_path / "c1", "FULL", 1e-5)
    with pytest.raises(CF.CentralFTRefused):                                       # P2-05's verifier refuses it
        CF.read_c1_run(div)
    c = P.read_c1_run(div)
    assert c["excluded"] and c["value"] is None and c["reason"].startswith("diverged (STATUS FAILED: TrainingDiverged")
    assert c["c1_nonfinite_path"]["kind"] == "DIVERGED" and c["label"] == "P2_C1_FT_CENTRAL_FULL|lr=1e-05"
    rec = P.write_selection("C1", tmp_path / "C1.json", runs=runs + [div, write_c1(tmp_path / "c1", "FROZEN", 1e-3,
                                                                                   0.1700)], anchor=world["anchor"])
    obj = P.load_p2_selection(rec["path"], "C1")
    assert [x["label"] for x in obj["excluded"]] == ["P2_C1_FT_CENTRAL_FULL|lr=1e-05"]
    assert obj["excluded"][0]["reason"].startswith("diverged")
    assert obj["variants"]["FULL"]["winner"] == "P2_C1_FT_CENTRAL_FULL|lr=3e-05"
    nf = write_c1_diverged(tmp_path / "nf", "FULL", 1e-5, error="", summary={       # a DONE_MAIN run, NaN PB value
        "status": "DONE_MAIN", "checkpoints": {}, "p2_c1": {"arm": CF.C1_ARMS["FULL"]},
        "evals": [{"mark": 0.5 * i} for i in range(1, 13)], "practical_best": {"mark": 3.0, "value": float("nan")}})
    assert P.read_c1_run(nf)["c1_nonfinite_path"]["kind"] == "NON_FINITE"
    for name, kw in (("disk", {"error": "DiskGuardError: free 2 GiB"}), ("main", {"role": "MAIN"})):
        with pytest.raises(P.P2SelectionRefused):                                  # not a divergence / not CALIB
            P.read_c1_run(write_c1_diverged(tmp_path / name, "FULL", 1e-5, **kw))


@nc("delta N1: a one-directional R9 (expected keys only) accepts a row with an extra Phase-1 label")
def test_nc_r9_one_directional_accepts_extra_label():
    spec = p2_spec(WA_WIN)
    want = dict(F.warm_row_labels(spec), **F.p2_row_labels(spec))
    row = dict(want, fl_server_tau=1e-3)
    mutant_ok = all(row.get(k) == v for k, v in want.items())                    # MUTANT: no set comparison
    assert not mutant_ok


@nc("delta N2: abs(NaN - x) > tol is False, so a NaN-coerced missing schedule_mark passes")
def test_nc_nan_schedule_mark_passes_tolerance_check():
    sm = P._value_or_nan(None)
    assert abs(sm - 4.0) > 1e-12                                                   # MUTANT: no finiteness check


@nc("delta C1: central_ft.read_c1_run alone cannot record a diverged C1 run as excluded")
def test_nc_central_ft_alone_records_diverged_c1(tmp_path):
    rd = write_c1_diverged(tmp_path, "FULL", 1e-5)
    try:
        r = CF.read_c1_run(rd)                                                     # MUTANT: no p2_select fallback
    except CF.CentralFTRefused:
        r = None
    assert r is not None and r["finite"] is False


# ------------------------------------------------------------------------------------------------ negative control
@nc("plain argmax ignores the §5.6 tie band / preference order")
def test_nc_argmax_ignores_tie_preference():
    a, b = C(R("fedadam", 3e-4, 0.01), 0.1716), C(R("fedadam", 1e-4, 0.01), 0.1710)
    assert max([a, b], key=lambda c: c["value"])["label"] == P.p2_decide("WA", [a, b], 0.16)["winner"]
