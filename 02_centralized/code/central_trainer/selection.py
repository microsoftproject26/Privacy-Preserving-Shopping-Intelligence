"""v4train.selection — the registered family selection (T22) with the PREG1 T1 trigger and staged full-TRAIN tie-break.

Binding source: contracts/PREG1_LOCK_VALUES.json T1_architecture_selection (Fable 5.1, frozen before any central
result). All functions are pure over result rows / run summaries, except the writers.

Cohort evidence (`cohort_evidence`): per family and seed (2026, 2027), the PRACTICAL_BEST client-macro MRR@20 (E2E) of
method C on M02, [0.5, 6.0] EFE. Refused anywhere in the input: TEST / RETRO_TEST rows, downstream (non-C) rows,
ARCHITECTURE_TIEBREAK rows (never cohort evidence), a missing / duplicate seed, an UNDEFINED value, comparison keys that
differ across families outside topology_hash / initialization_block.
Trigger (`trigger_report`), with delta_bar = mean_GRU - mean_SASRec, delta_s = GRU_s - SASRec_s, g_f = within-family
|MRR_2026 - MRR_2027|:
  T1a |delta_bar| < 0.0020;  T1b sign(delta_2026) != sign(delta_2027) or any |delta_s| <= 0.0002;
  T1c |delta_bar| < 2 * max(g_GRU, g_SASRec);  T1d the paired user bootstrap 95% CI of delta_bar (1000 resamples,
  seed 20260926, shared W; v4train.bootstrap_ci) contains 0.
  Any -> TIEBREAK_REQUIRED (no winner). None -> DECISIVE_STABLE: the higher cohort mean. (An exact tie is always T1a,
  so the old NDCG / latency / lexical chain of PKG 05 §B is never reached; the EQUIVALENT outcome below replaces it,
  pending the owner's ratification.)
Tie-break (full-TRAIN comparison union, N = 40,180,608; the same recipe; rows ARCHITECTURE_TIEBREAK):
  read rule: both runs of a stage complete 6.0 EFE before either number is read (`check_stage_complete`, called
             before any tie-break row is opened);
  stage 1 (seed 2026): DECISIVE iff |delta_full_2026| >= 0.0040 AND its paired bootstrap 95% CI excludes 0 ->
             selection_basis FULL_TRAIN_TIEBREAK_STAGE1 (seed 2027 skipped); else STAGE2_REQUIRED;
  stage 2 (seed 2027): winner = the higher full-data seed mean iff |delta_bar_full| >= 0.0020 AND both full per-seed
             deltas share its sign (FULL_TRAIN_TIEBREAK_STAGE2); else CONSISTENT_SMALL_ADVANTAGE if all four per-seed
             deltas (2 cohort + 2 full) share one sign with |delta| > 0.0002; else
             ARCHITECTURE_EQUIVALENT_WITHIN_RESOLUTION -> `resolve_equivalent`: an owner ratification file ->
             OWNER_RATIFIED_EQUIVALENT; else, owner unreachable -> PROXY_DECISION_PENDING_RATIFICATION (GRU on cost;
             only reversible GRU calibration may begin; no TEST, no ALL_METHODS_LOCK, no ARCHITECTURE_LOCK); else HOLD
             (no ARCHITECTURE_LOCK draft, no downstream run, one owner question).
  budget caps (metadata): 65 GPU-h stage 1, 125 GPU-h total.
Downstream C = the selected family's COHORT contest rows (PRACTICAL_BEST, CONTROLLED_BUDGET) reused verbatim;
ARCHITECTURE_TIEBREAK rows can never become C_MATCHED (`assert_c_matched_eligible`) nor seed a downstream run.
`write_lock_draft` writes ARCHITECTURE_LOCK.DRAFT.json for a final decision only; it never writes the lock itself.
Run labels (TRAIN-01R2; G1 residual R3), refused natively, with no relabelling needed: every row must carry a
`run_label` in recipe.RUN_LABELS. The cohort contest, the calibration choice and C_MATCHED accept only PRODUCTION rows;
the full-TRAIN tie-break accepts only HISTORICAL_REFERENCE rows (its own rows, ARCHITECTURE_TIEBREAK). SMOKE_NOT_EVIDENCE
and FEATURE_SEARCH_SCREEN rows, and unlabelled rows, are refused everywhere with a SelectionError naming the label.
Calibration: `calibration_choice` picks the peak from the two arms' 2.0-EFE rows (PRODUCTION only); a run consumes a
CALIBRATION_<FAMILY>.json only through `accept_calibration_record` (PRODUCTION record, same family and data version).
TRAIN-02: HP_SEARCH (the P2 TUNING rows, endpoint and CKPT_AVG) is a run label no consumer here accepts, so every
function above refuses such a row natively; the P2 mechanical selection over HP_SEARCH values lives in
v4train.p2_driver and never produces contest evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from .checkpoint import read_json, write_json_atomic
from .recipe import RUN_LABELS
from .schedule import CALIBRATION_END

SEEDS = (2026, 2027)
FAMILIES = ("GRU", "SASREC")                   # delta = GRU - SASREC throughout (the lock's delta_bar)
TEST_SPLITS = ("RETRO_TEST", "TEST", "LEGACY_TEST")
SAME_ACROSS_FAMILIES = ("data_protocol_hash", "feature_contract_hash", "split", "label_policy_hash", "population_hash",
                        "query_order_hash", "candidate_order_hash", "objective_id", "precision_id",
                        "evaluator_code_sha256", "eval_score_block", "eval_batch", "eval_device_class", "eval_stream",
                        "data_version", "manifest_registry_sha256", "schema_id", "schema_self_sha256")
# which run labels each consumer accepts (TRAIN-01R2, G1 R3); anything else, or no label, is refused
EVIDENCE_LABELS = {"COHORT": ("PRODUCTION",), "TIEBREAK": ("HISTORICAL_REFERENCE",), "CALIBRATION": ("PRODUCTION",),
                   "C_MATCHED": ("PRODUCTION",)}
CALIBRATION_TIE_TOL = 1e-8
LOCK_NAME = "ARCHITECTURE_LOCK.json"
DRAFT_NAME = "ARCHITECTURE_LOCK.DRAFT.json"
OWNER_QUESTION_NAME = "OWNER_QUESTION.json"
RATIFICATION_SCHEMA = "V4_OWNER_RATIFICATION_1"
# PREG1_LOCK_VALUES T1 (quoted; a test checks each number against the lock file text)
T1A_ABS_DELTA_BAR_LT = 0.0020
T1B_ABS_DELTA_S_LE = 0.0002
T1C_SEED_GAP_MULTIPLE = 2.0
STAGE1_ABS_DELTA_GE = 0.0040
STAGE2_ABS_DELTA_BAR_GE = 0.0020
CONSISTENT_ABS_DELTA_GT = 0.0002
BUDGET_CAPS_GPU_H = {"stage1": 65, "total": 125}
FINAL = ("DECISIVE_STABLE", "FULL_TRAIN_TIEBREAK_STAGE1", "FULL_TRAIN_TIEBREAK_STAGE2", "CONSISTENT_SMALL_ADVANTAGE",
         "OWNER_RATIFIED_EQUIVALENT")
DISCLOSURE = ("the 0.000869 full-data seed gap is SASRec-only (v1); GRU full-data seed spread under the common "
              "schedule is unmeasured (PREG1 T1 tiebreak.disclosure)")
TIEBREAK_PLAN = {"population": "full TRAIN comparison union: 2,694,998 clients, 40,180,608 examples, N_EFE = 40,180,608",
                 "train_manifest": "M_FULL_TRAIN", "eval_manifest": "full warm VALIDATION (642,228 users, 8,430,391 "
                 "rows; registered and hashed before launch); M02 as a descriptive subset",
                 "recipe": "identical to the cohort contest: cohort-calibrated peak LR, common schedule, "
                           "exposure_point=end, 6.0 EFE, stopping rule C3, no continuation, no new calibration",
                 "stage1": {"seeds": [2026], "budget_cap_gpu_h": 65,
                            "read_rule": "both runs complete 6.0 EFE before either number is read"},
                 "stage2": {"seeds": [2027]}, "budget_cap_total_gpu_h": 125,
                 "checkpoints": "strictly-best + endpoint_6.0 weights only; deleted after ARCHITECTURE_LOCK",
                 "row_status": "ARCHITECTURE_TIEBREAK, population ALL, comparison_status HISTORICAL_REFERENCE"}


class SelectionError(ValueError):
    pass


class StageNotComplete(SelectionError):
    pass


class NeedsFable(SelectionError):
    """Both MAIN candidates of a family were excluded."""
    pass


class LosingFamilyError(PermissionError):
    pass


def _is_metric(r: Mapping, metric: str, k: int) -> bool:
    return (r.get("metric") == metric and int(r.get("k", -1)) == k and r.get("aggregation") == "macro"
            and r.get("target_denominator") == "E2E")


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def rows_sha256(rows: Iterable[Mapping]) -> str:
    return hashlib.sha256(json.dumps(list(rows), sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def refuse_run_labels(rows: Iterable[Mapping], stage: str) -> None:
    """Refuse any row whose run_label the `stage` (COHORT | TIEBREAK | CALIBRATION | C_MATCHED) does not accept."""
    allowed = EVIDENCE_LABELS[stage]
    for r in rows:
        lab = r.get("run_label")
        if lab is None:
            raise SelectionError(f"{stage}: an unlabelled row (no run_label) is refused; rows must come from a "
                                 f"TRAIN-01R2 trainer (labels {RUN_LABELS})")
        if lab not in RUN_LABELS:
            raise SelectionError(f"{stage}: unknown run_label {lab!r} (labels {RUN_LABELS})")
        if lab not in allowed:
            raise SelectionError(f"{stage}: a {lab} row is refused; only {' / '.join(allowed)} rows are evidence here "
                                 "(SMOKE_NOT_EVIDENCE, FEATURE_SEARCH_SCREEN and HP_SEARCH rows never are)")


def _refuse_common(rows: list) -> None:
    if not rows:
        raise SelectionError("no result rows")
    for r in rows:
        if str(r.get("split", "")).upper() in TEST_SPLITS:
            raise SelectionError("TEST rows refused: family selection is VALIDATION-only (PKG 05 §D; E10.1)")
    for r in rows:
        if r.get("method") != "C":
            raise SelectionError(f"downstream row (method {r.get('method')!r}) refused: nothing downstream may run "
                                 "before the architecture lock")
        if r.get("split") != "VALIDATION":
            raise SelectionError(f"row split {r.get('split')!r} refused (VALIDATION only)")
        if r.get("selection_rule") == "PRACTICAL_BEST_EXTENDED":             # TRAIN-03 (ruling cond. 4; E12.1.7)
            raise SelectionError("a PRACTICAL_BEST_EXTENDED (continuation) row is never T1 selection evidence")


def _evidence(rows: list, metric: str, k: int, *, tiebreak: bool, seeds: Sequence[int] = SEEDS) -> dict:
    ev = {}
    for r in rows:
        is_tb = r.get("row_status") == "ARCHITECTURE_TIEBREAK"
        if is_tb != tiebreak:
            if not tiebreak and is_tb:
                raise SelectionError("an ARCHITECTURE_TIEBREAK row is never cohort evidence")
            continue
        if r.get("selection_rule") != "PRACTICAL_BEST" or not _is_metric(r, metric, k):
            continue                                  # CONTROLLED_BUDGET / PRACTICAL_BEST_EXTENDED: never evidence
        efe = r.get("efe")
        if efe is None or not (0.5 - 1e-9 <= float(efe) <= 6.0 + 1e-9):
            raise SelectionError(f"PRACTICAL_BEST evidence must come from [0.5, 6.0] EFE, got {efe}")
        fam = str(r.get("family", "")).upper()
        if fam not in FAMILIES:
            raise SelectionError(f"unknown family {r.get('family')!r}")
        if r.get("value") is None or r.get("value_status") != "OK":
            raise SelectionError(f"evidence value missing or UNDEFINED ({fam}, seed {r.get('seed')})")
        key = (fam, int(r["seed"]))
        if key in ev:
            raise SelectionError(f"duplicate evidence row for {key}")
        ev[key] = r
    for fam in FAMILIES:
        got = sorted(s for f, s in ev if f == fam)
        if got != sorted(seeds):
            raise SelectionError(f"{fam}: evidence needs seeds {tuple(seeds)} exactly once each, got {got}")
    ref = None
    for r in ev.values():
        kk = {f: r.get(f) for f in SAME_ACROSS_FAMILIES}
        if ref is None:
            ref = kk
        elif kk != ref:
            diff = sorted(f for f in SAME_ACROSS_FAMILIES if kk[f] != ref[f])
            raise SelectionError(f"evidence rows differ in the comparison key {diff} (HISTORICAL_REFERENCE only)")
    return ev


def cohort_evidence(rows: Iterable[Mapping]) -> dict:
    rows = [dict(r) for r in rows]
    _refuse_common(rows)
    refuse_run_labels(rows, "COHORT")
    return {"primary": _evidence(rows, "mrr", 20, tiebreak=False), "ndcg10": _evidence(rows, "ndcg", 10, tiebreak=False),
            "rows_sha256": rows_sha256(rows)}


def _ci_contains_zero(ci) -> bool:
    lo, hi = float(ci[0]), float(ci[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        raise SelectionError(f"invalid confidence interval {ci}")
    return lo <= 0.0 <= hi


def trigger_report(ev: dict, delta_bar_ci) -> dict:
    """The PREG1 T1 trigger on the cohort evidence (values of (family, seed) -> PRACTICAL_BEST MRR@20)."""
    if delta_bar_ci is None:
        raise SelectionError("T1d needs the paired user bootstrap 95% CI of delta_bar (v4train.bootstrap_ci)")
    v = {k: float(r["value"]) for k, r in ev.items()}
    mean = {f: sum(v[(f, s)] for s in SEEDS) / len(SEEDS) for f in FAMILIES}
    delta_bar = mean["GRU"] - mean["SASREC"]
    delta_s = {s: v[("GRU", s)] - v[("SASREC", s)] for s in SEEDS}
    gap = {f: abs(v[(f, 2026)] - v[(f, 2027)]) for f in FAMILIES}
    t = {"T1a": abs(delta_bar) < T1A_ABS_DELTA_BAR_LT,
         "T1b": _sign(delta_s[2026]) != _sign(delta_s[2027]) or any(abs(d) <= T1B_ABS_DELTA_S_LE
                                                                    for d in delta_s.values()),
         "T1c": abs(delta_bar) < T1C_SEED_GAP_MULTIPLE * max(gap.values()),
         "T1d": _ci_contains_zero(delta_bar_ci)}
    return {"triggers": t, "triggered": any(t.values()), "family_mean": mean, "delta_bar": delta_bar,
            "delta_s": {str(s): d for s, d in delta_s.items()}, "seed_gap": gap,
            "delta_bar_ci95": [float(delta_bar_ci[0]), float(delta_bar_ci[1])], "per_seed": {
                f: {str(s): v[(f, s)] for s in SEEDS} for f in FAMILIES}}


def _winner_of(delta: float) -> str:
    if delta == 0:
        raise SelectionError("a zero delta has no winner")
    return "GRU" if delta > 0 else "SASREC"


def _decision(status: str, winner: Optional[str], basis: str, **extra) -> dict:
    loser = None if winner is None else [f for f in FAMILIES if f != winner][0]
    return {"status": status, "winner_family": winner, "losing_family": loser, "selection_basis": basis,
            "final": status in FINAL, "forbidden_downstream_families": ([loser] if loser else list(FAMILIES)),
            "selected_on": "VALIDATION", "metric": "client_macro_mrr@20", "selection_rule": "PRACTICAL_BEST",
            "seeds": list(SEEDS), "budget_caps_gpu_h": dict(BUDGET_CAPS_GPU_H), "disclosure": DISCLOSURE, **extra}


def select_family(rows: Iterable[Mapping], *, delta_bar_ci) -> dict:
    """Cohort decision: DECISIVE_STABLE (winner) or TIEBREAK_REQUIRED (no winner; run stage 1)."""
    ev = cohort_evidence(rows)
    rep = trigger_report(ev["primary"], delta_bar_ci)
    common = {"cohort": rep, "evidence_rows_sha256": rows_sha256(sorted(ev["primary"].values(),
                                                                         key=lambda r: (r["family"], r["seed"]))),
              "input_rows_sha256": ev["rows_sha256"],
              "comparison_key": {f: next(iter(ev["primary"].values())).get(f) for f in SAME_ACROSS_FAMILIES},
              "evidence_checkpoints": {f"{f}:{s}": ev["primary"][(f, s)].get("checkpoint_id")
                                       for f, s in sorted(ev["primary"])},
              "ndcg10_family_mean": {f: sum(float(ev["ndcg10"][(f, s)]["value"]) for s in SEEDS) / 2
                                     for f in FAMILIES}}
    if rep["triggered"]:
        return _decision("TIEBREAK_REQUIRED", None, "PENDING_FULL_TRAIN_TIEBREAK", plan=TIEBREAK_PLAN, **common)
    return _decision("DECISIVE_STABLE", _winner_of(rep["delta_bar"]), "COHORT_DECISIVE_STABLE", **common)


def within_family_candidates(rows: Iterable[Mapping], recipes: Mapping, *, unseal_token,
                             diagnostic_rows=None) -> dict:
    """Amendment 05: compare two 6.0 endpoint means within each family."""
    from .unseal import UnsealToken, _ISSUER
    if (not isinstance(unseal_token, UnsealToken) or unseal_token.issuer is not _ISSUER
            or unseal_token.record.get("smoke") or not unseal_token.record.get("parameter_counts")):
        raise SelectionError("G2 needs an eight-run unseal token with RUN_START parameter counts")
    parameter_counts = unseal_token.record["parameter_counts"]

    def decide(source):
        source = [dict(r) for r in source]
        _refuse_common(source)
        refuse_run_labels(source, "COHORT")
        out = {}
        for fam in FAMILIES:
            entries = {rid: r for rid, r in recipes.items() if r["family"] == fam}
            if not 1 <= len(entries) <= 2 or f"02B_{fam}" not in entries:
                raise SelectionError(f"{fam}: expected C1 and at most one C2 recipe")
            values, excluded = {}, {}
            for rid, entry in entries.items():
                chosen = [r for r in source if r.get("family") == fam and r.get("recipe_id") == rid
                          and r.get("selection_rule") == "CONTROLLED_BUDGET" and _is_metric(r, "mrr", 20)
                          and r.get("efe") is not None and abs(float(r["efe"]) - 6.0) <= 1e-9
                          and r.get("checkpoint_variant", "ENDPOINT") == "ENDPOINT"]
                if len(chosen) != 2 or {int(r.get("seed", -1)) for r in chosen} != set(SEEDS):
                    raise SelectionError(f"{rid}: exactly two 6.0 endpoint rows are required")
                vals = [float(r["value"]) if r.get("value") is not None else float("nan") for r in chosen]
                loss_ok = all(all(math.isfinite(float(r[k])) for k in ("loss", "train_loss") if r.get(k) is not None)
                              for r in chosen)
                if not loss_ok or any(not math.isfinite(v) for v in vals) or any(r.get("value_status") != "OK" for r in chosen):
                    excluded[rid] = "NON_FINITE_OR_UNDEFINED"
                values[rid] = vals
            finite = [v for vals in values.values() for v in vals if math.isfinite(v)]
            best = max(finite) if finite else float("nan")
            for rid, vals in values.items():
                if rid not in excluded and any(v < 0.5 * best for v in vals):
                    excluded[rid] = "BELOW_HALF_FAMILY_BEST"
            valid = [rid for rid in entries if rid not in excluded]
            if not valid:
                raise NeedsFable(f"{fam}: both candidates excluded; needs Fable")
            means = {rid: sum(values[rid]) / 2 for rid in valid}
            high = max(means.values())
            ties = [rid for rid in valid if high - means[rid] <= 0.0005 + 1e-12]
            if any(rid not in parameter_counts for rid in entries):
                raise SelectionError(f"{fam}: parameter counts are required for every candidate")
            winner = min(ties, key=lambda rid: (int(parameter_counts[rid]), float(entries[rid]["peak_lr"]), rid))
            out[fam] = {"selected_recipe_id": winner, "means": means, "excluded": excluded,
                        "labels": {rid: "SELECTED" if rid == winner else "CANDIDATE_NOT_SELECTED" for rid in entries}}
        return out

    primary_rows = list(rows)
    _require_primary_manifest(primary_rows)
    primary = decide(primary_rows)
    if diagnostic_rows is not None:
        try:
            diagnostic = decide(diagnostic_rows)
        except (NeedsFable, SelectionError) as e:
            for fam in FAMILIES:
                primary[fam]["diagnostic_error"] = str(e)
                primary[fam]["diagnostic_flip"] = None
        else:
            for fam in FAMILIES:
                primary[fam]["diagnostic_selected_recipe_id"] = diagnostic[fam]["selected_recipe_id"]
                primary[fam]["diagnostic_flip"] = (diagnostic[fam]["selected_recipe_id"] !=
                                                   primary[fam]["selected_recipe_id"])
    return primary


def _require_primary_manifest(rows: Iterable[Mapping]) -> None:
    if any(r.get("eval_manifest_id") != "MAIN_VALIDATION_5_100_131072" for r in rows):
        raise SelectionError("G2 primary rows must all name the MAIN_VALIDATION eval manifest id")


def select_g2_candidates(rows: Iterable[Mapping], recipes: Mapping, *, unseal_token,
                         delta_bar_ci, diagnostic_rows=None) -> dict:
    """Within-family endpoint choice, then the existing T1 on the selected recipes."""
    rows = list(rows)
    within = within_family_candidates(rows, recipes, unseal_token=unseal_token,
                                      diagnostic_rows=diagnostic_rows)
    selected = {v["selected_recipe_id"] for v in within.values()}
    t1_rows = [r for r in rows if r.get("recipe_id") in selected and r.get("selection_rule") == "PRACTICAL_BEST"]
    return {"within_family": within, "between_families": select_family(t1_rows, delta_bar_ci=delta_bar_ci)}


# ------------------------------------------------------------------------------------------------ the tie-break
def check_stage_complete(summaries: Iterable[Mapping], seeds: Sequence[int]) -> None:
    """Read rule: every (family, seed) run of the stage finished 6.0 EFE before any number is read."""
    got = {}
    for s in summaries:
        got[(str(s.get("family", "")).upper(), int(s.get("seed", -1)))] = s
    for fam in FAMILIES:
        for seed in seeds:
            s = got.get((fam, seed))
            if s is None:
                raise StageNotComplete(f"{fam} seed {seed}: no run summary — both runs must complete 6.0 EFE first")
            marks = [e.get("mark") for e in s.get("evals", [])]
            if (s.get("status") not in ("DONE_MAIN",) or s.get("population") != "FULL_TRAIN" or not marks
                    or abs(float(marks[-1]) - 6.0) > 1e-9 or s.get("row_status") != "ARCHITECTURE_TIEBREAK"):
                raise StageNotComplete(f"{fam} seed {seed}: the full-TRAIN run has not completed 6.0 EFE "
                                       f"(status {s.get('status')!r}); no tie-break number may be read")


def _full_deltas(full_rows, seeds) -> tuple:
    rows = [dict(r) for r in full_rows]
    _refuse_common(rows)
    refuse_run_labels(rows, "TIEBREAK")
    for r in rows:
        if r.get("row_status") != "ARCHITECTURE_TIEBREAK" or r.get("comparison_status") != "HISTORICAL_REFERENCE":
            raise SelectionError("tie-break evidence must be ARCHITECTURE_TIEBREAK / HISTORICAL_REFERENCE rows")
    ev = _evidence([r for r in rows if int(r.get("seed", -1)) in seeds], "mrr", 20, tiebreak=True, seeds=seeds)
    d = {s: float(ev[("GRU", s)]["value"]) - float(ev[("SASREC", s)]["value"]) for s in seeds}
    return d, ev


def tiebreak_stage1(cohort: Mapping, full_rows: Iterable[Mapping], summaries: Iterable[Mapping], *,
                    delta_ci) -> dict:
    if cohort.get("status") != "TIEBREAK_REQUIRED":
        raise SelectionError("stage 1 runs only when the cohort decision is TIEBREAK_REQUIRED")
    summaries = list(summaries)
    check_stage_complete(summaries, (2026,))            # BEFORE any tie-break number is read
    d, ev = _full_deltas(full_rows, (2026,))
    delta = d[2026]
    excludes = not _ci_contains_zero(delta_ci)
    stage = {"delta_full_2026": delta, "ci95": [float(delta_ci[0]), float(delta_ci[1])], "ci_excludes_0": excludes,
             "threshold": STAGE1_ABS_DELTA_GE, "abs_delta_ge_threshold": abs(delta) >= STAGE1_ABS_DELTA_GE}
    base = {"cohort": cohort["cohort"], "stage1": stage, "plan": TIEBREAK_PLAN,
            "comparison_key": cohort.get("comparison_key"), "input_rows_sha256": cohort.get("input_rows_sha256")}
    if abs(delta) >= STAGE1_ABS_DELTA_GE and excludes:
        w = _winner_of(delta)
        rev = _sign(cohort["cohort"]["delta_bar"]) not in (0, _sign(delta))
        return _decision("FULL_TRAIN_TIEBREAK_STAGE1", w, "FULL_TRAIN_TIEBREAK_STAGE1", seed_2027="SKIPPED",
                         reverses_cohort_order=rev, cohort_deficit=(abs(cohort["cohort"]["delta_bar"]) if rev else 0.0),
                         **base)
    return _decision("STAGE2_REQUIRED", None, "PENDING_FULL_TRAIN_TIEBREAK_STAGE2", **base)


def tiebreak_stage2(stage1: Mapping, full_rows: Iterable[Mapping], summaries: Iterable[Mapping], *,
                    ratification_path: Optional[str] = None, owner_unreachable: bool = False) -> dict:
    if stage1.get("status") != "STAGE2_REQUIRED":
        raise SelectionError("stage 2 runs only when stage 1 was not decisive")
    summaries = list(summaries)
    check_stage_complete(summaries, SEEDS)              # all four full-TRAIN runs, before any number is read
    d, ev = _full_deltas(full_rows, SEEDS)
    dbar = sum(d.values()) / len(d)
    coh = {int(s): float(v) for s, v in stage1["cohort"]["delta_s"].items()}
    stage = {"delta_full": {str(s): v for s, v in d.items()}, "delta_bar_full": dbar,
             "threshold": STAGE2_ABS_DELTA_BAR_GE, "per_seed_share_sign": all(_sign(x) == _sign(dbar) for x in d.values())}
    base = {"cohort": stage1["cohort"], "stage1": stage1["stage1"], "stage2": stage, "plan": TIEBREAK_PLAN,
            "comparison_key": stage1.get("comparison_key"), "input_rows_sha256": stage1.get("input_rows_sha256")}
    if abs(dbar) >= STAGE2_ABS_DELTA_BAR_GE and stage["per_seed_share_sign"] and _sign(dbar) != 0:
        w = _winner_of(dbar)
        rev = _sign(stage1["cohort"]["delta_bar"]) not in (0, _sign(dbar))
        return _decision("FULL_TRAIN_TIEBREAK_STAGE2", w, "FULL_TRAIN_TIEBREAK_STAGE2", reverses_cohort_order=rev,
                         cohort_deficit=(abs(stage1["cohort"]["delta_bar"]) if rev else 0.0), **base)
    four = [coh[2026], coh[2027], d[2026], d[2027]]
    signs = {_sign(x) for x in four}
    if len(signs) == 1 and 0 not in signs and all(abs(x) > CONSISTENT_ABS_DELTA_GT for x in four):
        w = "GRU" if signs == {1} else "SASREC"
        return _decision("CONSISTENT_SMALL_ADVANTAGE", w, "CONSISTENT_SMALL_ADVANTAGE", four_deltas=four, **base)
    return resolve_equivalent(dict(base, four_deltas=four), ratification_path=ratification_path,
                              owner_unreachable=owner_unreachable)


def resolve_equivalent(base: Mapping, *, ratification_path: Optional[str] = None, owner_unreachable: bool = False) -> dict:
    """ARCHITECTURE_EQUIVALENT_WITHIN_RESOLUTION: ratified -> the owner's family; unreachable -> GRU proxy; else HOLD."""
    extra = dict(base, equivalent="ARCHITECTURE_EQUIVALENT_WITHIN_RESOLUTION",
                 proposed_fallback="select GRU on cost (downstream core 88 vs 263 GPU-h); never on export feasibility",
                 amendment_status="OWNER_RATIFICATION_PENDING (amends PKG 05 §B tie chain)")
    if ratification_path is not None and Path(ratification_path).is_file():
        rat = read_json(Path(ratification_path))
        fam = str(rat.get("selected_family", "")).upper()
        if rat.get("schema") != RATIFICATION_SCHEMA or rat.get("decision") != "RATIFIED" or fam not in FAMILIES \
                or rat.get("ratified_by") != "owner":
            raise SelectionError("the ratification file is not an owner ratification (schema / decision / family)")
        return _decision("OWNER_RATIFIED_EQUIVALENT", fam, "OWNER_RATIFIED_EQUIVALENT", ratification=rat, **extra)
    if owner_unreachable:
        d = _decision("PROXY_DECISION_PENDING_RATIFICATION", "GRU", "OWNER_PROXY_GRU_ON_COST", **extra)
        d.update(permissions={"GRU": ["CALIBRATION"]}, prohibited=["TEST", "ALL_METHODS_LOCK", "ARCHITECTURE_LOCK",
                                                                   "any SASREC run", "any non-calibration GRU run"],
                 forbidden_downstream_families=["SASREC"])
        return d
    d = _decision("HOLD", None, "HOLD_OWNER_QUESTION", **extra)
    d["owner_question"] = ("The two families are equivalent within resolution on the cohort and the full-TRAIN "
                           "tie-break. Do you ratify the amendment 'EQUIVALENT -> GRU on cost' (PKG 05 §B), or select "
                           "another family?")
    return d


# ------------------------------------------------------------------------------------------------ calibration choice
def choose_peak(values_at_end: Mapping[float, float], executed_multiplier: float = 1.0) -> float:
    """The multiplier with the highest value at 2.0 EFE; an exact tie (<= 1e-8) keeps the executed peak."""
    if len(values_at_end) != 2:
        raise SelectionError("calibration compares exactly two peaks")
    (m1, v1), (m2, v2) = sorted(values_at_end.items(), reverse=True)
    if abs(v1 - v2) <= CALIBRATION_TIE_TOL:
        return executed_multiplier
    return m1 if v1 > v2 else m2


def calibration_choice(arm_rows: Mapping[float, Iterable[Mapping]], executed_peak_lr: float = 1e-3) -> dict:
    """The calibration choice from each arm's 2.0-EFE result rows (multiplier -> rows): PRODUCTION rows only, the
    client-macro MRR@20 (E2E) of method C at 2.0 EFE, one per arm, the same data version / registry / schema."""
    values, keys = {}, {}
    for mult, rows in arm_rows.items():
        rows = [dict(r) for r in rows]
        if not rows:
            raise SelectionError(f"calibration arm x{mult:g}: no rows")
        refuse_run_labels(rows, "CALIBRATION")
        prim = [r for r in rows if _is_metric(r, "mrr", 20) and r.get("method") == "C"]
        if len(prim) != 1:
            raise SelectionError(f"calibration arm x{mult:g}: expected one MRR@20 macro E2E row, got {len(prim)}")
        r = prim[0]
        if r.get("value") is None or r.get("value_status") != "OK":
            raise SelectionError(f"calibration arm x{mult:g}: value missing or UNDEFINED")
        if r.get("efe") is None or abs(float(r["efe"]) - CALIBRATION_END) > 1e-9:
            raise SelectionError(f"calibration arm x{mult:g}: the choice is made at {CALIBRATION_END} EFE, "
                                 f"not {r.get('efe')}")
        values[float(mult)] = float(r["value"])
        keys[float(mult)] = {f: r.get(f) for f in ("data_version", "manifest_registry_sha256", "schema_self_sha256",
                                                   "family", "seed", "split")}
    if len({json.dumps(k, sort_keys=True) for k in keys.values()}) != 1:
        raise SelectionError(f"calibration arms differ in data version / registry / schema / family: {keys}")
    chosen = choose_peak(values)
    return {"chosen_multiplier": chosen, "chosen_peak_lr": chosen * executed_peak_lr, "values_at_2.0": values,
            "run_label": "PRODUCTION", "data_version": next(iter(keys.values()))["data_version"]}


def accept_calibration_record(record: Mapping, *, family: str, data_version: str) -> float:
    """The calibrated peak a PRODUCTION-grade run may use: the record must be a PRODUCTION calibration of the same
    family and data version (a SMOKE / screen / historical calibration is never used by a non-test run)."""
    lab = record.get("run_label")
    if lab != "PRODUCTION":
        raise SelectionError(f"calibration record with run_label {lab!r} refused: only a PRODUCTION calibration "
                             "sets the peak of a non-test run")
    if str(record.get("family", "")).upper() != str(family).upper():
        raise SelectionError("calibration record is for another family")
    if record.get("data_version") != data_version:
        raise SelectionError(f"calibration record data_version {record.get('data_version')!r} != the run's "
                             f"{data_version!r}")
    if record.get("evidence") is not True:
        raise SelectionError("the calibration record is not marked as evidence")
    return float(record["chosen_peak_lr"])


# ------------------------------------------------------------------------------------------------ writers and guards
def local_inventory_init_hashes() -> dict:
    """The LOCAL (torch 2.9.1+cpu) theta0 hashes of v4models/inventory — never valid ARCHITECTURE_LOCK input."""
    import v4models
    inv = Path(v4models.__file__).resolve().parent / "inventory"
    out = {}
    for fam in FAMILIES:
        p = inv / f"{fam}_CLEAN_V2_MANIFEST_INVENTORY.json"
        if p.is_file():
            rec = read_json(p)
            out[fam] = {str(s): v["init_sha256"] for s, v in rec.get("init_state", {}).items()}
    return out


def _check_init_receipt(receipt: Mapping, winner: str) -> dict:
    """The theta0 hashes must be SERVER-measured (the server torch build differs from the local 2.9.1+cpu)."""
    if not isinstance(receipt, Mapping) or receipt.get("source") != "SERVER_MEASURED":
        raise SelectionError("init hashes must come from a SERVER_MEASURED receipt, never the local inventory")
    for f in ("host", "torch", "hashes"):
        if not receipt.get(f):
            raise SelectionError(f"the init-hash receipt lacks {f!r}")
    hashes = receipt["hashes"].get(winner, {})
    if set(map(str, hashes)) != {str(s) for s in SEEDS}:
        raise SelectionError(f"server-measured initial-state hashes for {winner} seeds {SEEDS} are required")
    local = local_inventory_init_hashes().get(winner, {})
    if any(local.get(str(s)) == hashes[str(s)] for s in SEEDS):
        raise SelectionError("the receipt carries the LOCAL inventory theta0 hashes (torch build differs on the "
                             "server): re-measure on the server")
    return {str(s): hashes[str(s)] for s in SEEDS}


def write_lock_draft(decision: Mapping, out_path, *, init_hash_receipt: Mapping,
                     extra: Optional[Mapping] = None) -> str:
    p = Path(out_path)
    if p.name == LOCK_NAME or p.name != DRAFT_NAME:
        raise SelectionError(f"only the draft ({DRAFT_NAME}) may be written; the lock is Fable's G2 act")
    if decision.get("status") not in FINAL:
        raise SelectionError(f"no ARCHITECTURE_LOCK draft for status {decision.get('status')!r} (HOLD / proxy / pending)")
    w = decision["winner_family"]
    init = _check_init_receipt(init_hash_receipt, w)
    draft = {"status": "DRAFT_NOT_LOCKED", "requires": "Fable G2 verdict; server re-measured theta0 hashes",
             "winner_family": w, "selection_basis": decision["selection_basis"], "decision_status": decision["status"],
             "selected_on": "VALIDATION", "metric": "client_macro_mrr@20", "selection_rule": "PRACTICAL_BEST",
             "trigger": decision.get("cohort"), "stage1": decision.get("stage1"), "stage2": decision.get("stage2"),
             "reverses_cohort_order": decision.get("reverses_cohort_order", False),
             "cohort_deficit": decision.get("cohort_deficit", 0.0), "seeds": list(SEEDS), "representative_seed": 2026,
             "family_seed_metrics": decision["cohort"]["per_seed"], "family_mean_metric": decision["cohort"]["family_mean"],
             "initial_untrained_weights_hash_by_seed": init,
             "initial_hash_source": {"source": "SERVER_MEASURED", "host": init_hash_receipt["host"],
                                     "torch": init_hash_receipt["torch"]},
             "forbidden_downstream_families": list(decision["forbidden_downstream_families"]),
             "allowed_subsequent_method_codes": ["C", "FA", "FP", "PF", "CPF", "LO"],
             "downstream_C": "the selected family's cohort contest runs (PRACTICAL_BEST and CONTROLLED_BUDGET rows), "
                             "reused verbatim; ARCHITECTURE_TIEBREAK rows are never C_MATCHED",
             "budget_caps_gpu_h": dict(BUDGET_CAPS_GPU_H), "disclosure": DISCLOSURE,
             "fedbn_allowed": False, "FedBN": "excluded_by_owner",
             "input_rows_sha256": decision.get("input_rows_sha256"), "comparison_key": decision.get("comparison_key"),
             "note": "Winner family is locked; its trained weights are NOT default initialization for FL/LO."}
    draft.update(dict(extra or {}))
    return write_json_atomic(p, draft)


def write_owner_question(decision: Mapping, out_path) -> str:
    p = Path(out_path)
    if decision.get("status") != "HOLD":
        raise SelectionError("an owner question is written only for a HOLD")
    return write_json_atomic(p, {"status": "HOLD", "question": decision["owner_question"],
                                 "proposed_fallback": decision["proposed_fallback"],
                                 "amendment_status": decision["amendment_status"], "cohort": decision["cohort"],
                                 "stage1": decision.get("stage1"), "stage2": decision.get("stage2"),
                                 "ratification_schema": RATIFICATION_SCHEMA,
                                 "blocked": ["ARCHITECTURE_LOCK", "every downstream run of either family", "TEST"]})


def assert_downstream_allowed(decision: Mapping, family: str, method: str = "FA") -> None:
    """Refuse any downstream use of a family that is not the selected one (and anything under HOLD / pending)."""
    fam = str(family).upper()
    st = decision.get("status")
    if st == "PROXY_DECISION_PENDING_RATIFICATION":
        if method in (decision.get("permissions", {}).get(fam) or []):
            return
        raise LosingFamilyError(f"{fam} {method} refused under PROXY_DECISION_PENDING_RATIFICATION (only reversible "
                                "GRU calibration may begin; no TEST, no ALL_METHODS_LOCK)")
    if st not in FINAL or decision.get("winner_family") not in FAMILIES:
        raise LosingFamilyError(f"no final selection (status {st!r}): downstream work is blocked")
    if fam != decision["winner_family"] or fam in decision.get("forbidden_downstream_families", []):
        raise LosingFamilyError(f"downstream use of {fam} refused: the selected family is {decision['winner_family']}")


def assert_c_matched_eligible(row: Mapping) -> None:
    """A row may serve as C / C_MATCHED only if it is a cohort contest row (never a tie-break row)."""
    if (row.get("row_status") == "ARCHITECTURE_TIEBREAK" or row.get("comparison_status") == "HISTORICAL_REFERENCE"
            or row.get("population_scope") == "ALL" or row.get("c_matched_eligible") is False):
        raise SelectionError("ARCHITECTURE_TIEBREAK / HISTORICAL_REFERENCE rows can never become C_MATCHED")
    refuse_run_labels([row], "C_MATCHED")


def downstream_c_rows(decision: Mapping, rows: Iterable[Mapping]) -> list:
    """The selected family's cohort contest rows (PRACTICAL_BEST, CONTROLLED_BUDGET), reused verbatim as C."""
    assert_downstream_allowed(decision, decision.get("winner_family") or "", "C")
    out = []
    for r in rows:
        if str(r.get("family", "")).upper() == decision["winner_family"] and r.get("row_status") == "CENTRAL_CONTEST" \
                and r.get("selection_rule") in ("PRACTICAL_BEST", "CONTROLLED_BUDGET") and r.get("method") == "C":
            assert_c_matched_eligible(r)                  # tie-break rows are never CENTRAL_CONTEST rows
            out.append(dict(r))
    if not out:
        raise SelectionError("no cohort contest rows of the selected family")
    return out


__all__ = ["SelectionError", "StageNotComplete", "NeedsFable", "LosingFamilyError", "cohort_evidence", "trigger_report",
           "select_family", "check_stage_complete", "tiebreak_stage1", "tiebreak_stage2", "resolve_equivalent",
           "write_lock_draft", "write_owner_question", "assert_downstream_allowed", "assert_c_matched_eligible",
           "downstream_c_rows", "rows_sha256", "SEEDS", "FAMILIES", "FINAL", "LOCK_NAME", "DRAFT_NAME",
           "OWNER_QUESTION_NAME", "RATIFICATION_SCHEMA", "T1A_ABS_DELTA_BAR_LT", "T1B_ABS_DELTA_S_LE",
           "T1C_SEED_GAP_MULTIPLE", "STAGE1_ABS_DELTA_GE", "STAGE2_ABS_DELTA_BAR_GE", "CONSISTENT_ABS_DELTA_GT",
           "BUDGET_CAPS_GPU_H", "TIEBREAK_PLAN", "EVIDENCE_LABELS", "refuse_run_labels", "choose_peak",
           "calibration_choice", "accept_calibration_record", "CALIBRATION_TIE_TOL",
           "within_family_candidates", "select_g2_candidates"]
