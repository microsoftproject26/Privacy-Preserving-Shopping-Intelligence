"""FIX-DELTA-03 — the fixes of company/reviews/REV-DELTA-03_OPUS_REVIEW.md on the analysis side (synthetic blocks only):

  M1  PRE / C_FULL identity: the rows must be v4int.reeval rows of that registered central source (central_source = the
      arm, source_train_manifest_id PRETRAIN_TRAIN_5_100_NONMAIN / FULL_TRAIN_5_100_TIEBREAK) on every role, PRE never
      carries method "C", a central-source row never enters under another arm; the index requires the REEVAL_RECORD's
      central_source for PRE / C_FULL (and refuses it on any other arm).
  M2  FA_WARM / FA_WARM_FROZEN / DP8_FROZEN need RUN_SUMMARY arm_spec.init_from = the theta_pre identity, and no other FL
      arm may carry one (report and index; rows labelled fl_init likewise).
  M3  the Holm band after the first non-rejection is at alpha / (m - r): band and decision agree.
  m3  the BASELINE_CPU rows share one key among themselves.
  m4-m7  claim vocabulary, K20c, the K11b dependency note, the disclosures.
  C10 (m8)  matched-h parity M(h) / C_BIG(h) and the persistence rule (next registered h + CI lower bound >= X) beside
      the first-hit rule, with CIs from the PPSI_V4_TRUNC_PER_USER_1 per-user evidence.
(M5 — TEST refuses every contrasts format but v3 — is in test_ana03.)
Negative controls (test_nc_*, strict xfail on AssertionError): the positive check against the pre-fix code.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ana_block_index as BI
import anakit
import layoutkit
import retention_report as R
from test_ana04 import add_fl
from test_retention_report import _set_rows_field, check_refuses, nc, result, run

PB, CB, M = "PRACTICAL_BEST", "CONTROLLED_BUDGET", "MAIN"


def _refused(fn, needle: str) -> None:
    try:
        fn()
    except (R.AnaRefused, BI.IndexRefused) as e:
        assert needle in str(e), str(e)
        return
    raise AssertionError(f"not refused (expected {needle!r})")


# ================================================================================================ M1: report
def _entry(arm, role=M):
    return R.Entry(arm, 2026, PB, role, None, None, {}, None, None, False, {})


def _row(arm, role=M, **over):
    r = {"method": arm, "train_manifest_id": anakit.TRAIN_OF.get(arm, "MAIN_TRAIN_5_100_131072") if role == M else
         "UNEW_HOLDOUT_SUPPORT_32768"}
    r.update(anakit.SOURCE_LABELS.get(arm, {}))
    r.update(over)
    return {k: v for k, v in r.items() if v is not None}


def check_source_identity() -> None:
    """M1 key property (R.check_arm_identity looked up late, so a mutant can replace it)."""
    good = [("PRE", M, _row("PRE", train_manifest_id="MAIN_TRAIN_5_100_131072")),        # PRE on MAIN: the role key
            ("PRE", "UNSEEN_HOLDOUT", _row("PRE", "UNSEEN_HOLDOUT")),
            ("C_FULL", M, _row("C_FULL")), ("C_FULL", "SPARSE_PANEL", _row("C_FULL", "SPARSE_PANEL")),
            ("C_SMALL", M, _row("C_SMALL"))]
    for arm, role, r in good:
        R.check_arm_identity(_entry(arm, role), r)
    no_src = {"central_source": None, "source_train_manifest_id": None}
    bad = [  # the review's case: C_SMALL endpoint rows (method C, train MAIN_TRAIN) in a PRE-named entry
           ("PRE", M, _row("PRE", method="C", train_manifest_id="MAIN_TRAIN_5_100_131072", **no_src), "ARM_IDENTITY"),
           ("PRE", M, _row("PRE", **no_src), "central_source"),                     # the pre-B1 reeval (no source check)
           ("PRE", M, _row("PRE", method="C"), "method 'C'"),
           ("PRE", M, _row("PRE", central_source="C_FULL"), "central_source"),
           ("PRE", "HEAVY_PANEL", _row("PRE", "HEAVY_PANEL", source_train_manifest_id="MAIN_TRAIN_5_100_131072"),
            "source_train_manifest_id"),
           ("C_FULL", M, _row("C_FULL", **no_src), "central_source"),
           ("C_FULL", "SPARSE_PANEL", _row("C_FULL", "SPARSE_PANEL", source_train_manifest_id="MAIN_TRAIN_5_100_131072"),
            "FULL_TRAIN_5_100_TIEBREAK"),
           ("C_SMALL", "UNSEEN_HOLDOUT", _row("C_SMALL", "UNSEEN_HOLDOUT", central_source="PRE"), "central source"),
           ("C_BIG", M, _row("C_BIG", central_source="C_FULL"), "central source")]
    for arm, role, r, needle in bad:
        _refused(lambda: R.check_arm_identity(_entry(arm, role), r), needle)


def test_pre_and_c_full_rows_need_their_registered_source():
    check_source_identity()


def _pre_block(tmp):
    blk = anakit.build(tmp)
    anakit.add_warm_start(blk)
    return blk


def test_c_small_rows_relabelled_pre_refused_by_the_report(tmp_path):
    """The review's negative control, end to end: a C_SMALL endpoint row set (method C, train MAIN_TRAIN, no source
    labels) under a PRE entry, and a pre-B1 PRE reeval (method PRE, no central_source)."""
    for i, fields in enumerate(({"method": "C", "central_source": None, "source_train_manifest_id": None},
                                {"central_source": None, "source_train_manifest_id": None})):
        b = _pre_block(tmp_path / f"b{i}")
        for f, v in fields.items():
            _set_rows_field(b, "MAIN/PRE_s2026/RESULT_ROWS.json", f, v)
        check_refuses(b, tmp_path / f"b{i}", "ARM_IDENTITY")


def _old_check_arm_identity(entry, r):                                  # MUTANT: the pre-FIX-DELTA-03 A-F7 check
    tm = r.get("train_manifest_id")
    if entry.role == "MAIN" and entry.arm in R.TRAIN_MANIFEST_OF and tm is not None and tm != R.TRAIN_MANIFEST_OF[entry.arm]:
        raise R.AnaRefused(f"{entry.eid}: ARM_IDENTITY — train_manifest_id {tm!r}")
    if entry.role == "MAIN" and entry.arm == "PRE" and tm is not None and tm not in R.PRE_TRAIN_MANIFESTS:
        raise R.AnaRefused(f"{entry.eid}: ARM_IDENTITY — PRE train_manifest_id {tm!r}")
    meth = r.get("method")
    ok = ({entry.arm, R.FL_METHOD.get(entry.arm)} | set(R.FL_METHOD_ALT.get(entry.arm, ()))
          | ({"C"} if entry.arm in R.CENTRAL_ARMS else set()))
    if meth is not None and meth not in ok:
        raise R.AnaRefused(f"{entry.eid}: ARM_IDENTITY — method {meth!r}")


@nc("the pre-FIX-DELTA-03 check_arm_identity (PRE accepts MAIN_TRAIN and method C, no central_source check) must not "
    "pass the PRE / C_FULL source-identity property")
def test_nc_pre_identity_without_the_source_check(monkeypatch):
    monkeypatch.setattr(R, "check_arm_identity", _old_check_arm_identity)
    check_source_identity()


# ================================================================================================ M1: index
def _index(lay, out) -> tuple:
    lines = []
    rc = BI.main(lay["args"] + ["--out", str(out)], say=lines.append)
    return rc, "\n".join(lines)


def _edit_record(d: Path, fn) -> dict:
    p = d / "REEVAL_RECORD.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    orig = json.loads(json.dumps(rec))
    fn(rec)
    anakit.jdump(p, rec)
    return orig


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fd03_layout")
    b = anakit.build(tmp / "a")
    lay = layoutkit.build_layout(tmp / "lay", b)
    src = b.root / "MAIN" / "C_SMALL_s2026"                            # a PRE reeval dir (index-level identity only)
    layoutkit.write_reeval_dir(lay["R"] / "REEVAL_A40" / "PRE_s2026_CB", src / "RESULT_ROWS.json",
                               src / "ranks_CONTROLLED_BUDGET.npy", {
        "format": "PPSI_V4_REEVAL_RECORD_1", "arm_label": "PRE", "seed": 2026, "role": "MAIN",
        "selection_rule": CB, "checkpoint_id": "PRE_s2026:endpoint_6.0", "checkpoint_sha256": "9" * 64,
        "device_class": anakit.DEV, "device_pin_sha256": anakit.PIN, "eval_manifest_id": anakit.ROLES["MAIN"][0],
        "central_source": layoutkit.central_source_record("PRE")}, "MAIN")
    return b, lay, tmp


def check_index_source_records(layout, tag: str) -> None:
    b, lay, tmp = layout
    rc, text = _index(lay, tmp / f"ok{tag}")
    assert rc == 0, text
    keys = {(e["arm"], e["seed"], e["rule"], e["role"])
            for e in json.loads((tmp / f"ok{tag}" / R.INDEX_NAME).read_text(encoding="utf-8"))["entries"]}
    assert ("PRE", 2026, CB, M) in keys and ("C_FULL", 2026, PB, M) in keys
    ree = lay["R"] / "REEVAL_A40"
    cases = [("PRE_s2026_CB", lambda r: r.pop("central_source"), "central_source.source = PRE"),   # the pre-B1 reeval
             ("PRE_s2026_CB", lambda r: r.__setitem__("central_source", layoutkit.central_source_record("C_FULL")),
              "central_source.source = PRE"),
             ("PRE_s2026_CB", lambda r: r["central_source"]["pair"].__setitem__("train", "MAIN_TRAIN_5_100_131072"),
              "PRETRAIN_TRAIN_5_100_NONMAIN"),
             ("C_FULL_s2026_PB", lambda r: r.pop("central_source"), "central_source.source = C_FULL"),
             ("C_BIG_s2026_PB", lambda r: r.__setitem__("central_source", layoutkit.central_source_record("C_FULL")),
              "carries a central_source")]
    for i, (name, fn, needle) in enumerate(cases):
        orig = _edit_record(ree / name, fn)
        try:
            rc, text = _index(lay, tmp / f"bad{i}{tag}")
        finally:
            anakit.jdump(ree / name / "REEVAL_RECORD.json", orig)
        assert rc == BI.REFUSED and needle in text, (i, text)


def test_index_requires_the_reeval_central_source(layout):
    check_index_source_records(layout, "p")


@nc("an index builder without the REEVAL_RECORD central_source check (the ANA-04 reeval()) must not pass it")
def test_nc_index_without_the_central_source_check(layout, monkeypatch):
    monkeypatch.setattr(BI, "check_central_source", lambda arm, rec, where: None)
    check_index_source_records(layout, "nc")


# ================================================================================================ M2: warm-start identity
def _drop_init(summ):
    summ = json.loads(json.dumps(summ))
    summ["arm_spec"].pop("init_from", None)
    return summ


def check_warm_identity(tmp) -> None:
    b = anakit.build(tmp / "a")                                        # FA_WARM whose run has no init_from
    add_fl(b, "FA_WARM", patch=_drop_init)
    check_refuses(b, tmp / "a", "FA_WARM starts from theta_pre")
    b = anakit.build(tmp / "b")                                        # init_from trained on MAIN users
    add_fl(b, "FA_WARM", patch=lambda s: dict(s, arm_spec=dict(s["arm_spec"], init_from=dict(
        anakit.INIT_FROM, train_manifest="MAIN_TRAIN_5_100_131072", run_id="C_SMALL_s2026"))))
    check_refuses(b, tmp / "b", "not the theta_pre identity")
    b = anakit.build(tmp / "c")                                        # a cold-start arm carrying init_from
    for seed in (2026, 2027):
        p = b.root / "MAIN" / f"FL_FA_s{seed}" / "RUN_SUMMARY.json"
        summ = json.loads(p.read_text(encoding="utf-8"))
        summ["arm_spec"]["init_from"] = dict(anakit.INIT_FROM)
        anakit.jdump(p, summ)
    check_refuses(b, tmp / "c", "carries arm_spec.init_from")
    b = anakit.build(tmp / "d")                                        # rows labelled with a warm start under FA
    _set_rows_field(b, "MAIN/FL_FA_s2026/RESULT_ROWS.json", "fl_init", "PRETRAIN_FOR_FL")
    _set_rows_field(b, "MAIN/FL_FA_s2026/RESULT_ROWS.json", "fl_init_run_id", anakit.INIT_FROM["run_id"])
    check_refuses(b, tmp / "d", "warm-started")


def test_warm_arms_need_theta_pre_and_cold_arms_refuse_it(tmp_path):
    check_warm_identity(tmp_path)


def test_init_from_problem_known_answers():
    assert R.init_from_problem(dict(anakit.INIT_FROM)) is None
    for over in ({"kind": "C"}, {"run_label": "PRODUCTION"}, {"train_manifest": "FULL_TRAIN_5_100_TIEBREAK"},
                 {"fl_init_kind": "best"}, {"run_id": "C_FULL_02B_SASREC_s2026"}, {"checkpoint_sha256": "x"}):
        assert R.init_from_problem(dict(anakit.INIT_FROM, **over)), over
    assert R.init_from_problem(None) and R.WARM_INIT_ARMS == ("FA_WARM", "FA_WARM_FROZEN", "DP8_FROZEN")


@nc("a report without the warm-start identity rule (the directory name alone makes FA_WARM) must not pass it")
def test_nc_warm_identity_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_warm_init", lambda arm, summary, label: None)
    monkeypatch.setattr(R, "check_arm_identity", _old_check_arm_identity)
    check_warm_identity(tmp_path)


def check_index_warm(layout, tag: str) -> None:
    _b, lay, tmp = layout
    fl = lay["R"] / "FL"
    made = []
    try:
        for seed in (2026, 2027):                                      # a cold-start FA copied into FA_WARM_s<S>/
            shutil.copytree(fl / f"FA_s{seed}", fl / f"FA_WARM_s{seed}")
            made.append(fl / f"FA_WARM_s{seed}")
        rc, text = _index(lay, tmp / f"w1{tag}")
        assert rc == BI.REFUSED and "init_from" in text, text
        for d in made:                                                 # with the theta_pre identity: admitted
            p = d / "RUN_SUMMARY.json"
            summ = json.loads(p.read_text(encoding="utf-8"))
            summ["arm_spec"]["init_from"] = dict(anakit.INIT_FROM)
            anakit.jdump(p, summ)
            lp = d / "LAUNCH_0.json"                                   # ANA-05: the 2-D S7 warm-selection binding
            anakit.jdump(lp, dict(json.loads(lp.read_text(encoding="utf-8")),
                                  warm_selection=anakit.warm_binding("FA_WARM")))
        rc, text = _index(lay, tmp / f"w2{tag}")
        assert rc == 0, text
        p = fl / "FA_s2026" / "RUN_SUMMARY.json"                       # a cold-start FA carrying init_from
        orig = p.read_text(encoding="utf-8")
        summ = json.loads(orig)
        summ["arm_spec"]["init_from"] = dict(anakit.INIT_FROM)
        anakit.jdump(p, summ)
        try:
            rc, text = _index(lay, tmp / f"w3{tag}")
        finally:
            p.write_text(orig, encoding="utf-8")
        assert rc == BI.REFUSED and "carries arm_spec.init_from" in text, text
    finally:
        for d in made:
            shutil.rmtree(d, ignore_errors=True)


def test_index_warm_start_identity(layout):
    check_index_warm(layout, "p")


@nc("the ANA-04 index (FA_WARM by directory name, no init_from rule) must not pass the warm-start identity property")
def test_nc_index_warm_by_directory_name(layout, monkeypatch):
    monkeypatch.setattr(R, "WARM_INIT_ARMS", ())
    check_index_warm(layout, "nc")


# ================================================================================================ M3: Holm band
def _holm_world(ps, m=26):
    """A fake engine / confirmatory list with delta contrasts whose normal p-values are `ps` (SE 1)."""
    from statistics import NormalDist
    nd = NormalDist()
    eng, conf = SimpleNamespace(boot={}), []
    for i, p in enumerate(ps):
        kid = R.V3_IDS[i]
        z = nd.inv_cdf(1.0 - p / 2.0)
        a = np.sqrt(999.0 / 1000.0)
        eng.boot[(kid, PB)] = np.array([z - a, z + a] * 500)          # sd (ddof 1) = 1, mean = z
        conf.append({"id": kid, "confirmatory_rule": PB, "kind": "delta", "by_rule": {PB: {"point": float(z),
                                                                                           "ci95": [0.0, 1.0]}}})
    return eng, conf, {"n_contrasts": m}


def check_holm_band_agrees_with_the_decision(fn_name="holm_band") -> None:
    ps = [1e-6, 0.01] + list(np.linspace(0.011, 0.02, 23))            # rank 25: p = 0.02 <= 0.05 / 2 after a failure
    eng, conf, contrasts = _holm_world(ps)
    getattr(R, fn_name)(eng, conf, contrasts)
    hs = sorted((c["by_rule"][PB]["holm"] for c in conf), key=lambda h: h["holm_rank"])
    assert [h["holm_reject"] for h in hs] == [True] + [False] * 24
    for h in hs:
        lo, hi = h["holm_band"]
        assert (not (lo <= h["reference"] <= hi)) == h["holm_reject"], h
        assert h["bonferroni_band"][0] <= lo and hi <= h["bonferroni_band"][1]
    assert hs[-1]["p_normal_two_sided"] <= hs[-1]["holm_level"]      # the review's example is in the world
    assert all(h["holm_band_level"] == pytest.approx(0.05 / 25) for h in hs[1:])


def test_holm_band_agrees_with_the_decision():
    check_holm_band_agrees_with_the_decision()


def _old_holm_band(eng, conf, contrasts, alpha=0.05):                    # MUTANT: every rank at its own level
    from statistics import NormalDist
    nd = NormalDist()
    m = int(contrasts["n_contrasts"])
    comp = []
    for rec in conf:
        b = rec["by_rule"][rec["confirmatory_rule"]]
        boot = eng.boot[(rec["id"], rec["confirmatory_rule"])]
        se = float(np.std(boot, ddof=1))
        z = (b["point"] - 0.0) / se
        comp.append((2.0 * (1.0 - nd.cdf(abs(z))), R.order_key(rec["id"]), b, se))
    comp.sort(key=lambda x: (x[0], x[1]))
    running, stop = 0.0, False
    zb = nd.inv_cdf(1.0 - alpha / (2.0 * m))
    for j, (p, _o, b, se) in enumerate(comp, start=1):
        level = alpha / (m - j + 1)
        running = max(running, min(1.0, (m - j + 1) * p))
        reject = (not stop) and p <= level
        stop = stop or not reject
        zq = nd.inv_cdf(1.0 - level / 2.0)
        b["holm"] = {"holm_rank": j, "holm_level": level, "holm_band_level": level, "p_holm": running,
                     "p_normal_two_sided": p, "holm_reject": bool(reject), "reference": 0.0,
                     "holm_band": [b["point"] - zq * se, b["point"] + zq * se],
                     "bonferroni_band": [b["point"] - zb * se, b["point"] + zb * se]}


@nc("the ANA-04 Holm band (each rank at its own level after the first non-rejection) must not pass band = decision")
def test_nc_holm_band_disagrees_with_the_decision(monkeypatch):
    monkeypatch.setattr(R, "holm_band", _old_holm_band)
    check_holm_band_agrees_with_the_decision()


# ================================================================================================ m3: BASELINE_CPU key
def check_baseline_rows_agree(tmp) -> None:
    b = anakit.build(tmp / "a")
    anakit.add_baselines_ranked(b, arms=("LAST",))
    anakit.add_baselines_ranked(b, arms=("RULE",), extra_rows={"precision_id": "ANOTHER_PRECISION"})
    check_refuses(b, tmp / "a", "do not share one key")
    _refused(lambda: R.baseline_key_consistency([], {f: None for f in R.KEY_FIELDS}), "carries no value")


def test_baseline_rows_share_one_key(tmp_path):
    check_baseline_rows_agree(tmp_path)


@nc("the ANA-04 BASELINE_CPU check (each row against the main block only; nothing when the main key is empty) must not "
    "pass the one-key property")
def test_nc_baseline_rows_checked_against_main_only(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "baseline_key_consistency", lambda entries, main_key: None)
    check_baseline_rows_agree(tmp_path)


# ================================================================================================ m4-m7
def test_claim_vocabulary_k20c_k11b_note_and_disclosures(tmp_path):
    reg = R.claim_register(R.load_contrasts(anakit.V3_FILE))
    text = json.dumps(reg, ensure_ascii=False)
    assert "no detectable difference" in text and "not resolved" not in text                          # m4 (§7.6)
    k20c = " ".join(R.CLAIM_FORBIDDEN["K20c"])
    assert "unless" not in k20c and "unconditional" in k20c                                             # m5
    assert "K11b" in R.DEPENDENCY_NOTES and "new evidence" in R.DEPENDENCY_NOTES["K11b"]                # m6
    assert "except K11b" in R.multiplicity_line(list(R.V3_IDS))
    assert "except K11b" not in R.multiplicity_line([f"K{i}" for i in range(1, 20)])
    joined = " ".join(R.DISCLOSURES_ANA04)                                                              # m7
    assert "z = 0 twin = S8" in joined and "first mark" in joined and "TRAIN decisions" in joined
    assert any("persistent hit" in g for g in R.CLAIM_GENERAL_RULES)                                   # C10


# ================================================================================================ C10: persistence
N_USERS = 40
PANEL, PT = "COLD_H_16384", "ALL_AVAILABLE"
KEY = {"eval_device_class": anakit.DEV, "split": "VALIDATION", "evaluator_code_sha256": anakit.EVSHA}
C_FULL_V = 0.2
C_BIG_H = dict(zip(R.TRUNC_H, (0.5, 0.9, 0.9, 1.0, 1.0, 1.0)))   # C_BIG(h) / C_ref
ARMS_C10 = {"C_SMALL": (dict(zip(R.TRUNC_H, (0.5, 0.92, 0.80, 0.97, 0.98, 0.99))), 0.005),   # tight users
            "FA": (dict(zip(R.TRUNC_H, (0.5, 0.6, 0.7, 0.91, 0.92, 0.93))), 0.9)}           # very noisy users


def _c10_world(drop=(), panel_type=PT):
    """DATA-07-shaped primary rows + per-user evidence: C_ref = C_BIG FULL (constant users 0.2), C_BIG H rows, and two
    arms whose per-user values are ratio x 0.2 x (1 +- noise) (point = the ratio exactly)."""
    users = np.arange(1000, 1000 + N_USERS, dtype=np.int64)
    sign = np.where(np.arange(N_USERS) % 2 == 0, 1.0, -1.0)
    rows, pu = [], {}

    def add(arm, h, full, m):
        ck = R.trunc_cell_key(arm, 2026, PANEL, panel_type, h, full)
        pu[ck] = (users, m)
        rows.append(dict(KEY, method=arm, seed=2026, metric="mrr", k=20, aggregation="macro", target_denominator="E2E",
                         value=float(m.mean()), value_status="OK", trunc_panel=PANEL, trunc_panel_type=panel_type,
                         trunc_query_h=h, trunc_h=None if full else h, trunc_full_history=full, trunc_study="A"))
    for h in R.TRUNC_H:
        add("C_BIG", h, True, np.full(N_USERS, C_FULL_V))
        add("C_BIG", h, False, np.full(N_USERS, C_FULL_V * C_BIG_H[h]))
        for arm, (ratio, noise) in ARMS_C10.items():
            if (arm, h) not in drop:
                add(arm, h, False, C_FULL_V * ratio[h] * (1.0 + noise * sign))
    return rows, pu


def _direct_lo(m, c) -> float:
    """An independent paired user bootstrap of 100 M / C (v4eval.bootstrap.make_W over the sorted users)."""
    from v4eval.bootstrap import make_W
    W = make_W(m.shape[0])
    return float(np.percentile(100.0 * m[W].mean(axis=1) / c[W].mean(axis=1), 2.5))


def _direct_rule(pts, x, grid):
    """The registered persistence rule, written out independently: pts = {h: (point ratio, CI lower %)}."""
    for i, h in enumerate(grid[:-1]):
        nh = grid[i + 1]
        (p1, l1), (p2, l2) = pts[h], pts[nh]
        if p1 >= x and p2 >= x and l1 >= 100 * x and l2 >= 100 * x:
            return h
    return "NOT_REACHED"


def check_c10(curves=None) -> None:
    curves = curves or R.trunc_curves
    rows, pu = _c10_world()
    curve, ev = curves(rows, KEY, pu)
    by = {e["arm"]: e for e in ev}
    # the first-hit rule is unchanged; the persistence rule differs from it
    assert by["C_SMALL"]["events_to_90"] == 1 and by["C_SMALL"]["events_to_90_persistent"] == 10
    assert by["C_SMALL"]["events_to_95"] == 10 and by["C_SMALL"]["events_to_95_persistent"] == 10
    assert by["FA"]["events_to_90"] == 10 and by["FA"]["events_to_90_persistent"] == "NOT_REACHED"   # CI < 90
    # matched-h parity P(h) = M(h) / C_BIG(h)
    cs = {t["h"]: t for t in curve if t["arm"] == "C_SMALL"}
    for h in R.TRUNC_H:
        assert cs[h]["parity_matched_h"] == pytest.approx(100.0 * ARMS_C10["C_SMALL"][0][h] / C_BIG_H[h])
        assert cs[h]["C_BIG_h"] == pytest.approx(C_FULL_V * C_BIG_H[h])
    assert by["C_SMALL"]["parity_events_to_90"] == 0 and by["C_BIG"]["parity_events_to_90"] == "SELF_REFERENCE"
    # every persistent result = the independent rule on independent bootstrap lower bounds
    for arm in ARMS_C10:
        ratio = ARMS_C10[arm][0]
        m = {h: pu[R.trunc_cell_key(arm, 2026, PANEL, PT, h, False)][1] for h in R.TRUNC_H}
        c_full = np.full(N_USERS, C_FULL_V)
        r_pts = {h: (ratio[h], _direct_lo(m[h], c_full)) for h in R.TRUNC_H}
        p_pts = {h: (ratio[h] / C_BIG_H[h], _direct_lo(m[h], np.full(N_USERS, C_FULL_V * C_BIG_H[h])))
                 for h in R.TRUNC_H}
        for x in R.X_LEVELS:
            tag = int(round(100 * x))
            assert by[arm][f"events_to_{tag}_persistent"] == _direct_rule(r_pts, x, R.TRUNC_H), (arm, x)
            assert by[arm][f"parity_events_to_{tag}_persistent"] == _direct_rule(p_pts, x, R.TRUNC_H), (arm, x)
        for h in R.TRUNC_H:
            t = next(t for t in curve if t["arm"] == arm and t["h"] == h)
            assert t["R_ci95"][0] == pytest.approx(r_pts[h][1], rel=1e-12)
    # statuses: no per-user evidence, a missing next h, the largest h of a FIXED_20 sub-panel
    _c, ev0 = curves(rows, KEY, None)
    assert {e["arm"]: e["events_to_90_persistent"] for e in ev0}["C_SMALL"] == "NO_CI"
    assert {e["arm"]: e["events_to_90"] for e in ev0}["C_SMALL"] == 1
    rows2, pu2 = _c10_world(drop=(("C_SMALL", 20),))
    _c, ev2 = curves(rows2, KEY, pu2)
    assert {e["arm"]: e["events_to_90_persistent"] for e in ev2}["C_SMALL"] == "NO_NEXT_H"
    rows3, pu3 = _c10_world(drop=tuple(("C_SMALL", h) for h in (50,)), panel_type="FIXED_20")
    rows3 = [r for r in rows3 if r["trunc_query_h"] <= 20 and not (r["trunc_full_history"] and r["trunc_query_h"] != 20)]
    for r in rows3:
        if r["method"] == "C_SMALL":
            r["value"] = 0.1 if r["trunc_query_h"] < 20 else 0.2          # only the largest h reaches X (points only)
    _c, ev3 = curves(rows3, KEY, None)
    e3 = {e["arm"]: e for e in ev3}["C_SMALL"]
    assert e3["events_to_90"] == 20 and e3["events_to_90_persistent"] == "NOT_REACHED"


def test_c10_parity_and_persistence_known_answers():
    check_c10()


def test_c10_refuses_bad_per_user_evidence():
    rows, pu = _c10_world()
    k = R.trunc_cell_key("C_SMALL", 2026, PANEL, PT, 10, False)
    bad = dict(pu, **{k: (pu[k][0], pu[k][1] + 0.01)})
    _refused(lambda: R.trunc_curves(rows, KEY, bad), "does not reproduce the row value")
    u = pu[k][0].copy()
    u[-1] += 100
    bad = dict(pu, **{k: (u, pu[k][1])})
    _refused(lambda: R.trunc_curves(rows, KEY, bad), "not on the same users")


def test_c10_registered_h_grids_are_v4trunc_rules():
    from v4trunc import rules as TR                                     # read-only import (v4trunc unchanged)
    assert R.TRUNC_H_BY_TYPE == {"FIXED_50": tuple(TR.FIXED_BUDGETS["FIXED_50"]),
                                 "FIXED_20": tuple(TR.FIXED_BUDGETS["FIXED_20"]), "ALL_AVAILABLE": tuple(TR.BUDGETS)}


@nc("a persistence rule that is the first-hit rule (no next-h, no CI condition) must not pass the C10 property")
def test_nc_persistence_is_first_hit(monkeypatch):
    monkeypatch.setattr(R, "_persistent_hit_or_status",
                        lambda seq, x, ptype: R._first_hit_or_status([(h, m, c, None) for h, m, c, _lo in seq], x))
    check_c10()


@nc("a persistence rule on the points only (the CI lower bound ignored) must not pass the C10 property")
def test_nc_persistence_ignores_the_ci(monkeypatch):
    real = R._persistent_hit_or_status
    monkeypatch.setattr(R, "_persistent_hit_or_status",
                        lambda seq, x, ptype: real([(h, m, c, 1e9) for h, m, c, _lo in seq], x, ptype))
    check_c10()


def _write_per_user(path: Path, pu: dict) -> None:
    arrs = {"format": np.array(R.TRUNC_PER_USER_FORMAT)}
    for k, (u, m) in pu.items():
        arrs[f"u__{k}"], arrs[f"m__{k}"] = u, m
    np.savez(path, **arrs)


def test_c10_through_the_report_and_the_index(tmp_path):
    """The per-user evidence enters as records.TRUNC_ROWS[i].per_user (report) and as TRUNC_PER_USER.npz beside a
    --trunc dir's RESULT_ROWS (index); the result carries the persistent keys and the report the C10 table."""
    b = anakit.build(tmp_path / "a")
    rows, pu = _c10_world()
    anakit.jdump(b.root / "records" / "TRUNC.json", {"rows": rows})
    _write_per_user(b.root / "records" / "TRUNC_PER_USER.npz", pu)
    idx = b.index()
    idx["records"]["TRUNC_ROWS"] = [{"rows": "records/TRUNC.json", "rule": PB, "per_user": "records/TRUNC_PER_USER.npz"}]
    b.write_index(idx)
    rc, text = run(b, tmp_path / "out")
    assert rc == 0, text
    res = result(tmp_path / "out")
    by = {e["arm"]: e for e in res["curves"]["trunc_events_to_x"]}
    assert by["C_SMALL"]["events_to_90"] == 1 and by["C_SMALL"]["events_to_90_persistent"] == 10
    assert by["FA"]["events_to_90_persistent"] == "NOT_REACHED" and "persistence rule" in res["curves"][
        "trunc_persistence_rule"]
    assert res["records"]["TRUNC_ROWS"]["files"][0]["per_user_cells"] == len(pu)
    rep = (tmp_path / "out" / R.REPORT_NAME).read_text(encoding="utf-8")
    assert "06L-C §C10 beside the registered first-hit rule" in rep and "P: persistent 90%" in rep
    lay = layoutkit.build_layout(tmp_path / "lay", b)                    # the index copies the per-user file
    d = lay["R"] / "TRUNC" / "SA_C_SMALL_s2026"
    shutil.copyfile(b.root / "records" / "TRUNC_PER_USER.npz", d / BI.TRUNC_PER_USER_NAME)
    anakit.jdump(d / "STATUS.json", {"status": "DONE", "per_user": {"file": BI.TRUNC_PER_USER_NAME, "sha256":   # M4
                                                                    BI.sha_file(d / BI.TRUNC_PER_USER_NAME)}})
    rc, text = _index(lay, tmp_path / "idx")
    assert rc == 0, text
    item = json.loads((tmp_path / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))["records"]["TRUNC_ROWS"][0]
    assert item["per_user"] == f"TRUNC/SA_C_SMALL_s2026/{BI.TRUNC_PER_USER_NAME}"
    assert (tmp_path / "idx" / item["per_user"]).read_bytes() == (d / BI.TRUNC_PER_USER_NAME).read_bytes()
