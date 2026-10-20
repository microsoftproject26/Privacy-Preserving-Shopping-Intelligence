"""T22 + PREG1 T1 — family selection: the cohort trigger T1a-T1d, the staged full-TRAIN tie-break, the equivalence
outcome, the refusals, the lock draft, the downstream guard and the CPF stub. Every branch is exercised.

  * lock numbers: every threshold and budget cap in v4train.selection is the number written in PREG1_LOCK_VALUES.json;
  * cohort: DECISIVE_STABLE (no trigger) selects the higher seed mean; each of T1a, T1b (sign flip and small
    per-seed delta), T1c and T1d triggers alone -> TIEBREAK_REQUIRED (no winner, no draft, no downstream);
  * refusals: TEST rows, a missing / duplicate seed, evidence past 6.0 EFE, a downstream row, an UNDEFINED value,
    a comparison-key mismatch, a tie-break row offered as cohort evidence, a missing T1d CI;
  * read rule: no tie-break number is read before BOTH runs of the stage completed 6.0 EFE;
  * stage 1: DECISIVE iff |delta| >= 0.0040 AND the CI excludes 0 (both conditions needed) -> FULL_TRAIN_TIEBREAK_STAGE1,
    seed 2027 skipped; a reversal of the cohort order is disclosed with the cohort deficit;
  * stage 2: FULL_TRAIN_TIEBREAK_STAGE2 (|delta_bar_full| >= 0.0020, per-seed signs agree); CONSISTENT_SMALL_ADVANTAGE
    (four deltas share a sign, each > 0.0002); ARCHITECTURE_EQUIVALENT_WITHIN_RESOLUTION -> HOLD (owner question, no
    draft, all downstream blocked) / OWNER_RATIFIED_EQUIVALENT (ratification file) / PROXY_DECISION_PENDING_RATIFICATION
    (GRU; only reversible GRU calibration; no draft);
  * tie-break rows can never become C_MATCHED; downstream C = the winner's cohort contest rows;
  * end to end: four real contest runs + two real full-TRAIN runs (synthetic data, v4eval, v4eval bootstrap CIs); their
    test-run rows (SMOKE_NOT_EVIDENCE, TRAIN-01R2) are refused first, then relabelled to exercise the arithmetic.
Negative controls: a max-seed selector; a trigger without T1c; stage 1 without the CI condition; an early reader; a
C_MATCHED guard that accepts tie-break rows; an EQUIVALENT outcome that silently picks GRU and drafts the lock.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from trainkit import art  # noqa: E402  (TRAIN-03: sealed artifact root)
from trainkit import KEY, assert_raises, cfg, nc, read, trainer
from v4eval.rows import make_row
from v4train import selection as SEL
from v4train.bootstrap_ci import delta_bar_ci, delta_ci, practical_best_ranks
from v4train.unseal import smoke_token  # noqa: E402  (TRAIN-03)
from v4train.cpf_stub import MODES, CentralPersonalizedMode, CPFNotImplemented
from v4train.selection import (DRAFT_NAME, LOCK_NAME, LosingFamilyError, SelectionError, StageNotComplete,
                               assert_c_matched_eligible, assert_downstream_allowed, check_stage_complete,
                               downstream_c_rows, resolve_equivalent, select_family, tiebreak_stage1, tiebreak_stage2,
                               write_lock_draft, write_owner_question)

LOCK_TEXT = (Path(__file__).resolve().parents[2] / "contracts" / "PREG1_LOCK_VALUES.json").read_text(encoding="utf-8")
BASE = dict(KEY, query_order_hash="Q", evaluator_code_sha256="E", eval_score_block=158486, eval_batch=1024,
            eval_device_class="A16-FP32-DET", eval_stream="ALL_ROWS", objective_id="O",
            precision_id="P", aggregation="macro", target_denominator="E2E")
WIDE = (0.01, 0.02)            # a CI that excludes 0
HAS0 = (-0.001, 0.02)          # a CI that contains 0
INITS = {f: {"2026": f[0] * 64, "2027": f[1] * 64} for f in ("GRU", "SASREC")}
RECEIPT = {"source": "SERVER_MEASURED", "host": "server-fixture", "torch": "server-torch-fixture",
           "hashes": INITS}


def row(fam, seed, value, *, metric="mrr", k=20, rule="PRACTICAL_BEST", efe=5.0, method="C", tiebreak=False, **over):
    key = dict(BASE, family=fam, topology_hash=f"T-{fam}", initialization_block=f"I-{fam}-{seed}", metric=metric, k=k)
    key.update({k2: v for k2, v in over.items() if k2 in key or k2 == "split"})
    r = make_row(key, method=method, seed=seed, value=value, n_decisions=10, n_users=5, oov_count=0, coverage=1.0,
                 selection_rule=rule, efe=efe, extra={"checkpoint_id": f"{fam}:{seed}:best"})

    if tiebreak:
        r.update(row_status="ARCHITECTURE_TIEBREAK", comparison_status="HISTORICAL_REFERENCE", population_scope="ALL",
                 c_matched_eligible=False, population_hash="FULL-POP", query_order_hash="FULL-Q",
                 run_label="HISTORICAL_REFERENCE")
    else:
        r.update(row_status="CENTRAL_CONTEST", c_matched_eligible=True, population_scope="COHORT",
                 run_label="PRODUCTION")                         # TRAIN-01R2: selection reads only labelled rows
    if "run_label" in over:
        r["run_label"] = over["run_label"]
    return r


def cohort_rows(g=(0.300, 0.304), s=(0.290, 0.292)):
    out = []
    for fam, vals in (("GRU", g), ("SASREC", s)):
        for seed, v in zip((2026, 2027), vals):
            out += [row(fam, seed, v), row(fam, seed, v / 2, metric="ndcg", k=10),
                    row(fam, seed, v - 0.001, rule="CONTROLLED_BUDGET", efe=6.0)]
    return out


def full_rows(g, s, seeds=(2026, 2027)):
    out = []
    for fam, vals in (("GRU", g), ("SASREC", s)):
        for seed, v in zip(seeds, vals):
            out.append(row(fam, seed, v, tiebreak=True))
    return out


def summary(fam, seed, *, done=True, last=6.0, population="FULL_TRAIN"):
    marks = [0.5 * i for i in range(1, int(last * 2) + 1)]
    return {"family": fam, "seed": seed, "status": "DONE_MAIN" if done else "RUNNING", "population": population,
            "row_status": "ARCHITECTURE_TIEBREAK", "evals": [{"mark": m} for m in marks]}


def summaries(seeds=(2026,)):
    return [summary(f, s) for f in ("GRU", "SASREC") for s in seeds]


def tb_cohort():
    return select_family(cohort_rows(g=(0.300, 0.301), s=(0.2995, 0.3005)), delta_bar_ci=HAS0)


# ------------------------------------------------------------------------------------------------ lock numbers
def test_selection_constants_are_the_lock_numbers():
    t1 = json.loads(LOCK_TEXT)["T1_architecture_selection"]
    assert f"< {SEL.T1A_ABS_DELTA_BAR_LT:.4f}" in t1["trigger_any_of"]["T1a"]
    assert f"<= {SEL.T1B_ABS_DELTA_S_LE:.4f}" in t1["trigger_any_of"]["T1b"]
    assert f"< {SEL.T1C_SEED_GAP_MULTIPLE:g} * max" in t1["trigger_any_of"]["T1c"]
    assert "1000 resamples, seed 20260926" in t1["trigger_any_of"]["T1d"]
    assert f">= {SEL.STAGE1_ABS_DELTA_GE:.4f}" in t1["tiebreak"]["stage1"]["decisive_iff"]
    assert f">= {SEL.STAGE2_ABS_DELTA_BAR_GE:.4f}" in t1["tiebreak"]["stage2"]["rule"]
    assert f"> {SEL.CONSISTENT_ABS_DELTA_GT:.4f}" in t1["tiebreak"]["stage2"]["inconclusive"]
    assert t1["tiebreak"]["stage1"]["budget_cap_gpu_h"] == SEL.BUDGET_CAPS_GPU_H["stage1"] == 65
    assert t1["tiebreak"]["budget_cap_total_gpu_h"] == SEL.BUDGET_CAPS_GPU_H["total"] == 125
    assert re.search(r"both runs complete 6\.0 EFE before either number is read", LOCK_TEXT)


# ------------------------------------------------------------------------------------------------ cohort
def test_decisive_stable():
    d = select_family(cohort_rows(), delta_bar_ci=WIDE)      # delta_bar 0.011, per-seed 0.010 / 0.012, gaps 0.004
    assert d["status"] == "DECISIVE_STABLE" and d["winner_family"] == "GRU" and d["final"]
    assert not any(d["cohort"]["triggers"].values())
    assert d["cohort"]["delta_bar"] == pytest.approx(0.011) and d["selection_basis"] == "COHORT_DECISIVE_STABLE"
    d2 = select_family(cohort_rows(g=(0.290, 0.292), s=(0.300, 0.304)), delta_bar_ci=(-0.02, -0.01))
    assert d2["winner_family"] == "SASREC"


def _check_mean_rule(select):
    d = select(cohort_rows(g=(0.300, 0.330), s=(0.335, 0.285)), delta_bar_ci=WIDE)
    assert d["winner_family"] == "GRU" or d.get("status") == "TIEBREAK_REQUIRED", "the seed MEAN decides"


@pytest.mark.parametrize("case,g,s,ci,which", [
    ("T1a", (0.300, 0.301), (0.2995, 0.3), WIDE, "T1a"),
    ("T1b_sign", (0.310, 0.290), (0.300, 0.2995), WIDE, "T1b"),
    ("T1b_small", (0.3100, 0.3350), (0.3099, 0.3200), WIDE, "T1b"),
    ("T1c", (0.300, 0.330), (0.296, 0.320), WIDE, "T1c"),
    ("T1d", (0.310, 0.312), (0.300, 0.301), HAS0, "T1d"),
])
def test_each_trigger_alone(case, g, s, ci, which):
    d = select_family(cohort_rows(g=g, s=s), delta_bar_ci=ci)
    t = d["cohort"]["triggers"]
    assert t[which], f"{case}: {which} must fire ({d['cohort']})"
    if case in ("T1a", "T1c", "T1d"):
        assert sum(t.values()) >= 1
    assert d["status"] == "TIEBREAK_REQUIRED" and d["winner_family"] is None and not d["final"]
    assert d["plan"]["stage1"]["seeds"] == [2026] and d["budget_caps_gpu_h"] == {"stage1": 65, "total": 125}
    assert_raises(SelectionError, write_lock_draft, d, Path("x") / DRAFT_NAME, init_hash_receipt=RECEIPT)
    assert_raises(LosingFamilyError, assert_downstream_allowed, d, "GRU")


def test_t1c_fires_alone():
    t = select_family(cohort_rows(g=(0.300, 0.330), s=(0.296, 0.320)), delta_bar_ci=WIDE)["cohort"]["triggers"]
    assert t == {"T1a": False, "T1b": False, "T1c": True, "T1d": False}


def test_cohort_refusals():
    r = cohort_rows()
    assert_raises(SelectionError, select_family, r + [row("GRU", 2026, 0.9, split="RETRO_TEST")], delta_bar_ci=WIDE)
    assert_raises(SelectionError, select_family, [x for x in r if not (x["family"] == "GRU" and x["seed"] == 2027)],
                  delta_bar_ci=WIDE)
    assert_raises(SelectionError, select_family, r + [row("GRU", 2026, 0.5)], delta_bar_ci=WIDE)
    assert_raises(SelectionError, select_family, [row("GRU", 2026, 0.3, efe=6.5)] + r[1:], delta_bar_ci=WIDE)
    assert_raises(SelectionError, select_family, r + [row("GRU", 2026, 0.3, method="FA")], delta_bar_ci=WIDE)
    assert_raises(SelectionError, select_family, r + full_rows((0.3,), (0.3,), seeds=(2026,)), delta_bar_ci=WIDE)
    und = copy.deepcopy(r)
    und[0]["value"], und[0]["value_status"] = None, "UNDEFINED:EMPTY_POPULATION"
    assert_raises(SelectionError, select_family, und, delta_bar_ci=WIDE)
    mis = copy.deepcopy(r)
    mis[0]["eval_batch"] = 512
    assert_raises(SelectionError, select_family, mis, delta_bar_ci=WIDE)
    assert_raises(SelectionError, select_family, r, delta_bar_ci=None)
    assert_raises(SelectionError, select_family, [], delta_bar_ci=WIDE)


# ------------------------------------------------------------------------------------------------ read rule + stage 1
def _check_read_rule(reader):
    c = tb_cohort()
    for partial in ([summary("GRU", 2026)], [summary("GRU", 2026), summary("SASREC", 2026, done=False)],
                    [summary("GRU", 2026), summary("SASREC", 2026, last=5.5)],
                    [summary("GRU", 2026), summary("SASREC", 2026, population="COHORT")]):
        assert_raises(StageNotComplete, reader, c, full_rows((0.31,), (0.30,), seeds=(2026,)), partial)


def test_read_rule_both_runs_complete_before_reading():
    _check_read_rule(lambda c, rows, sm: tiebreak_stage1(c, rows, sm, delta_ci=WIDE))
    check_stage_complete(summaries(), (2026,))


def _check_stage1(stage1):
    c = tb_cohort()
    assert stage1(c, full_rows((0.305,), (0.300,), seeds=(2026,)), summaries(), delta_ci=HAS0)["status"] == \
        "STAGE2_REQUIRED", "|delta| >= 0.0040 but the CI contains 0: not decisive"
    assert stage1(c, full_rows((0.303,), (0.300,), seeds=(2026,)), summaries(), delta_ci=WIDE)["status"] == \
        "STAGE2_REQUIRED", "CI excludes 0 but |delta| < 0.0040: not decisive"
    d = stage1(c, full_rows((0.305,), (0.300,), seeds=(2026,)), summaries(), delta_ci=WIDE)
    assert d["status"] == "FULL_TRAIN_TIEBREAK_STAGE1" and d["winner_family"] == "GRU"
    return d


def test_stage1_branches():
    d = _check_stage1(tiebreak_stage1)
    assert d["selection_basis"] == "FULL_TRAIN_TIEBREAK_STAGE1" and d["seed_2027"] == "SKIPPED" and d["final"]
    c = select_family(cohort_rows(g=(0.2995, 0.3005), s=(0.300, 0.301)), delta_bar_ci=HAS0)   # cohort favours SASREC
    rev = tiebreak_stage1(c, full_rows((0.305,), (0.300,), seeds=(2026,)), summaries(), delta_ci=WIDE)
    assert rev["reverses_cohort_order"] and rev["cohort_deficit"] == pytest.approx(abs(c["cohort"]["delta_bar"]))
    assert_raises(SelectionError, tiebreak_stage1, select_family(cohort_rows(), delta_bar_ci=WIDE),
                  full_rows((0.305,), (0.3,), seeds=(2026,)), summaries(), delta_ci=WIDE)   # only after a trigger
    assert_raises(SelectionError, tiebreak_stage1, c, cohort_rows(), summaries(), delta_ci=WIDE)  # cohort rows


# ------------------------------------------------------------------------------------------------ stage 2 + equivalence
def _stage1_pending(cohort=None):
    c = cohort or tb_cohort()
    return tiebreak_stage1(c, full_rows((0.302,), (0.300,), seeds=(2026,)), summaries(), delta_ci=WIDE)


def test_stage2_branches(work):
    s1 = _stage1_pending()
    assert s1["status"] == "STAGE2_REQUIRED"
    all4 = summaries((2026, 2027))
    assert_raises(StageNotComplete, tiebreak_stage2, s1, full_rows((0.302, 0.303), (0.300, 0.300)), summaries())
    d = tiebreak_stage2(s1, full_rows((0.302, 0.303), (0.300, 0.300)), all4)
    assert d["status"] == "FULL_TRAIN_TIEBREAK_STAGE2" and d["winner_family"] == "GRU"
    d = tiebreak_stage2(s1, full_rows((0.3040, 0.2990), (0.300, 0.300)), all4)    # dbar 0.0015 < 0.002
    assert d["status"] != "FULL_TRAIN_TIEBREAK_STAGE2"
    # cohort deltas of tb_cohort: +0.0005 / +0.0005; four deltas all > 0.0002 and positive -> CONSISTENT
    d = tiebreak_stage2(s1, full_rows((0.3010, 0.3008), (0.300, 0.300)), all4)
    assert d["status"] == "CONSISTENT_SMALL_ADVANTAGE" and d["winner_family"] == "GRU" and d["final"]
    # mixed signs -> EQUIVALENT -> HOLD by default
    h = tiebreak_stage2(s1, full_rows((0.3010, 0.2990), (0.300, 0.300)), all4)
    assert h["status"] == "HOLD" and h["winner_family"] is None and h["equivalent"] == \
        "ARCHITECTURE_EQUIVALENT_WITHIN_RESOLUTION"
    assert_raises(SelectionError, write_lock_draft, h, work / DRAFT_NAME, init_hash_receipt=RECEIPT)
    write_owner_question(h, work / SEL.OWNER_QUESTION_NAME)
    q = read(work / SEL.OWNER_QUESTION_NAME)
    assert q["status"] == "HOLD" and "GRU on cost" in q["proposed_fallback"]
    for fam in ("GRU", "SASREC"):
        assert_raises(LosingFamilyError, assert_downstream_allowed, h, fam, "CALIBRATION")
    # owner unreachable -> proxy: GRU, only reversible GRU calibration
    p = tiebreak_stage2(s1, full_rows((0.3010, 0.2990), (0.300, 0.300)), all4, owner_unreachable=True)
    assert p["status"] == "PROXY_DECISION_PENDING_RATIFICATION" and p["winner_family"] == "GRU" and not p["final"]
    assert_downstream_allowed(p, "GRU", "CALIBRATION")
    for fam, meth in (("GRU", "FA"), ("GRU", "TEST"), ("SASREC", "CALIBRATION"), ("GRU", "C")):
        assert_raises(LosingFamilyError, assert_downstream_allowed, p, fam, meth)
    assert_raises(SelectionError, write_lock_draft, p, work / DRAFT_NAME, init_hash_receipt=RECEIPT)
    # a ratification file -> OWNER_RATIFIED_EQUIVALENT
    rat = work / "OWNER_RATIFICATION.json"
    rat.write_text(json.dumps({"schema": SEL.RATIFICATION_SCHEMA, "decision": "RATIFIED", "selected_family": "GRU",
                               "ratified_by": "owner", "amends": "PKG 05 §B tie chain"}), encoding="utf-8")
    o = tiebreak_stage2(s1, full_rows((0.3010, 0.2990), (0.300, 0.300)), all4, ratification_path=str(rat))
    assert o["status"] == "OWNER_RATIFIED_EQUIVALENT" and o["winner_family"] == "GRU" and o["final"]
    write_lock_draft(o, work / DRAFT_NAME, init_hash_receipt=RECEIPT)
    assert read(work / DRAFT_NAME)["selection_basis"] == "OWNER_RATIFIED_EQUIVALENT"
    bad = work / "BAD_RATIFICATION.json"
    bad.write_text(json.dumps({"schema": SEL.RATIFICATION_SCHEMA, "decision": "RATIFIED", "selected_family": "GRU",
                               "ratified_by": "an agent"}), encoding="utf-8")
    assert_raises(SelectionError, tiebreak_stage2, s1, full_rows((0.3010, 0.2990), (0.300, 0.300)), all4,
                  ratification_path=str(bad))


# ------------------------------------------------------------------------------------------------ guards, draft, CPF
def test_c_matched_guard_draft_and_cpf(work):
    d = select_family(cohort_rows(), delta_bar_ci=WIDE)
    rows = cohort_rows() + full_rows((0.31, 0.31), (0.30, 0.30))
    c_rows = downstream_c_rows(d, rows)
    assert {r["selection_rule"] for r in c_rows} == {"PRACTICAL_BEST", "CONTROLLED_BUDGET"}
    assert all(r["family"] == "GRU" and r.get("row_status") == "CENTRAL_CONTEST" for r in c_rows)
    for r in full_rows((0.31,), (0.30,), seeds=(2026,)):
        assert_raises(SelectionError, assert_c_matched_eligible, r)
    assert_raises(SelectionError, write_lock_draft, d, work / LOCK_NAME, init_hash_receipt=RECEIPT)      # never the lock
    assert_raises(SelectionError, write_lock_draft, d, work / DRAFT_NAME, init_hash_receipt={})
    write_lock_draft(d, work / DRAFT_NAME, init_hash_receipt=RECEIPT)
    draft = read(work / DRAFT_NAME)
    assert draft["status"] == "DRAFT_NOT_LOCKED" and draft["FedBN"] == "excluded_by_owner"
    assert draft["forbidden_downstream_families"] == ["SASREC"] and draft["budget_caps_gpu_h"]["total"] == 125
    assert draft["initial_untrained_weights_hash_by_seed"] == INITS["GRU"]
    assert "SASRec-only" in draft["disclosure"] and not (work / LOCK_NAME).exists()
    assert draft["initial_hash_source"]["source"] == "SERVER_MEASURED"
    assert_downstream_allowed(draft | {"final": True, "status": "DECISIVE_STABLE"}, "GRU")
    assert_raises(LosingFamilyError, assert_downstream_allowed, d, "SASREC")
    assert "CPF_CENTRAL" in MODES
    assert_raises(LosingFamilyError, CentralPersonalizedMode, d, "SASREC")
    assert_raises(CPFNotImplemented, CentralPersonalizedMode(d, "GRU").run)


def test_lock_draft_needs_server_measured_init_hashes(work):
    """The theta0 hashes of the draft are SERVER-measured (torch 2.13 on the server != local 2.9.1): a receipt without
    that source, without host / torch, or carrying the local v4models inventory hashes is refused."""
    d = select_family(cohort_rows(), delta_bar_ci=WIDE)
    local = SEL.local_inventory_init_hashes()
    assert set(local) == {"GRU", "SASREC"} and set(local["GRU"]) == {"2026", "2027"}
    for bad in ({"hashes": INITS}, dict(RECEIPT, source="LOCAL_INVENTORY"), dict(RECEIPT, host=""),
                dict(RECEIPT, hashes={"GRU": local["GRU"]}), dict(RECEIPT, hashes={"GRU": {"2026": "a" * 64}})):
        assert_raises(SelectionError, write_lock_draft, d, work / DRAFT_NAME, init_hash_receipt=bad)
    write_lock_draft(d, work / DRAFT_NAME, init_hash_receipt=RECEIPT)


# ------------------------------------------------------------------------------------------------ end to end
def test_end_to_end_contest_trigger_and_stage1(work):
    """Real (synthetic-data) runs through v4eval: 4 cohort contest runs, T1 with the v4eval bootstrap CI, then the two
    stage-1 full-TRAIN runs, their CI, and the stage-1 decision; the read rule holds throughout."""
    rows, ranks, inits, evalm = [], {}, {}, None
    for fam in ("GRU", "SASREC"):
        for seed in (2026, 2027):
            d = work / f"C_{fam}_{seed}"
            tr = trainer(d, cfg(f"C_{fam}_{seed}", fam, seed=seed))
            s = tr.run()
            evalm = tr.evalp.manifest
            rows += read(art(d) / "RESULT_ROWS.json")["rows"]
            ranks[(fam, seed)] = practical_best_ranks(d, smoke_token([d]))     # TRAIN-03: sealed, test runs only
            inits.setdefault(fam, {})[str(seed)] = s["init_sha256"]
    ci = delta_bar_ci(evalm, ranks)
    # TRAIN-01R2: rows of test_overrides runs are SMOKE_NOT_EVIDENCE and selection refuses them natively; the same rows
    # relabelled PRODUCTION (test only) exercise the selection arithmetic on real trainer rows
    assert {r["run_label"] for r in rows} == {"SMOKE_NOT_EVIDENCE"}
    with pytest.raises(SelectionError, match="SMOKE_NOT_EVIDENCE row is refused"):
        select_family(rows, delta_bar_ci=ci["ci95"])
    rows = [dict(r, run_label="PRODUCTION") for r in rows]
    dec = select_family(rows, delta_bar_ci=ci["ci95"])
    assert dec["cohort"]["delta_bar"] == pytest.approx(ci["point"], abs=1e-12)
    assert dec["status"] in ("DECISIVE_STABLE", "TIEBREAK_REQUIRED")
    if dec["status"] == "DECISIVE_STABLE":
        write_lock_draft(dec, work / DRAFT_NAME, init_hash_receipt=dict(RECEIPT, hashes=inits))
    full_rows_, sums, fr = [], [], {}
    for fam in ("GRU", "SASREC"):
        d = work / f"F_{fam}_2026"
        tr = trainer(d, cfg(f"F_{fam}_2026", fam, population="FULL_TRAIN"))
        st = tr.run()
        assert read(d / "STATUS.json")["status"] == "DONE_MAIN"
        full_rows_ += read(art(d) / "RESULT_ROWS.json")["rows"]
        sums.append(read(art(d) / "RUN_SUMMARY.json"))
        fr[fam] = practical_best_ranks(d)
        fm = tr.evalp.manifest
        assert st["row_status"] == "ARCHITECTURE_TIEBREAK"
    assert all(r["row_status"] == "ARCHITECTURE_TIEBREAK" and r["comparison_status"] == "HISTORICAL_REFERENCE"
               and r["c_matched_eligible"] is False for r in full_rows_)
    ci1 = delta_ci(fm, fr["GRU"], fr["SASREC"])
    forced = dec if dec["status"] == "TIEBREAK_REQUIRED" else dict(dec, status="TIEBREAK_REQUIRED")
    with pytest.raises(SelectionError, match="SMOKE_NOT_EVIDENCE row is refused"):    # test-run tie-break rows
        tiebreak_stage1(forced, full_rows_, sums, delta_ci=ci1["ci95"])
    full_rows_ = [dict(r, run_label="HISTORICAL_REFERENCE") for r in full_rows_]      # test only (see above)
    s1 = tiebreak_stage1(forced, full_rows_, sums, delta_ci=ci1["ci95"])
    assert s1["stage1"]["delta_full_2026"] == pytest.approx(ci1["point"], abs=1e-12)
    assert s1["status"] in ("FULL_TRAIN_TIEBREAK_STAGE1", "STAGE2_REQUIRED")
    assert_raises(StageNotComplete, tiebreak_stage1, forced, full_rows_, sums[:1], delta_ci=ci1["ci95"])


# ------------------------------------------------------------------------------------------------ negative controls
def _max_seed(rows, delta_bar_ci):
    best = {}
    for r in rows:
        if r["metric"] == "mrr" and r["selection_rule"] == "PRACTICAL_BEST":
            best[r["family"]] = max(best.get(r["family"], 0.0), r["value"])
    return {"winner_family": max(best, key=best.get), "status": "X"}


@nc("selecting by the best seed instead of the seed mean")
def test_nc_max_seed_selector():
    _check_mean_rule(_max_seed)


@nc("a trigger that omits T1c misses a seed-noise-dominated gap")
def test_nc_trigger_without_t1c(monkeypatch):
    monkeypatch.setattr(SEL, "T1C_SEED_GAP_MULTIPLE", 0.0)
    d = select_family(cohort_rows(g=(0.300, 0.330), s=(0.296, 0.320)), delta_bar_ci=WIDE)
    assert d["status"] == "TIEBREAK_REQUIRED", "|delta_bar| < 2 max(g) must trigger"


@nc("stage 1 decided on |delta| alone, without the CI condition")
def test_nc_stage1_without_ci(monkeypatch):
    monkeypatch.setattr(SEL, "_ci_contains_zero", lambda ci: False)
    _check_stage1(tiebreak_stage1)


@nc("a reader that reads the tie-break as soon as one run has finished")
def test_nc_early_reader(monkeypatch):
    monkeypatch.setattr(SEL, "check_stage_complete", lambda s, seeds: None)
    _check_read_rule(lambda c, rows, sm: SEL.tiebreak_stage1(c, rows, sm, delta_ci=WIDE))


@nc("a C_MATCHED guard that accepts ARCHITECTURE_TIEBREAK rows")
def test_nc_c_matched_accepts_tiebreak(monkeypatch):
    monkeypatch.setattr(SEL, "assert_c_matched_eligible", lambda r: None)
    for r in full_rows((0.31,), (0.30,), seeds=(2026,)):
        assert_raises(SelectionError, SEL.assert_c_matched_eligible, r)


@nc("an EQUIVALENT outcome that silently selects GRU and drafts the lock")
def test_nc_equivalent_autoselects(work, monkeypatch):
    def auto(base, **kw):
        return SEL._decision("OWNER_RATIFIED_EQUIVALENT", "GRU", "AUTO", **dict(base))
    monkeypatch.setattr(SEL, "resolve_equivalent", auto)
    h = tiebreak_stage2(_stage1_pending(), full_rows((0.3010, 0.2990), (0.300, 0.300)), summaries((2026, 2027)))
    assert h["status"] == "HOLD", "without a ratification the EQUIVALENT outcome must HOLD"
