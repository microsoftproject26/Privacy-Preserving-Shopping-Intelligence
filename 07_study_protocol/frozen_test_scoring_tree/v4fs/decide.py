"""v4fs.decide — the FS drop rule and acceptance rule (binding: Fable R2 D3 amendments 4-5 + R4;
POPULATION_5_100_SCIENTIFIC_SYNTHESIS §D3 items 5-7).

Statistic: the CONTROLLED_BUDGET 6.0-EFE endpoint client-macro MRR@20 (E2E) of each run, per training seed
2026 / 2027, on the FS VALIDATION users; every contrast uses the ONE shared W of that user set (v4eval.bootstrap:
1,000 resamples, seed 20260926).

Drop rule: a group is dropped iff its importance CI includes 0 for BOTH families (v4fs.importance.noise_groups over
combine_seeds). S* = L3 minus the dropped groups (v4fs.schemas.derive; id "L3_NEAR_DUPLICATE-minus-<groups>").

Relative drop of a candidate versus L0 (per family): the SEED-MEAN relative drop, the ratio bootstrapped directly
(E7.5, E7.6): point = 100 (C̄ - M̄) / C̄ with C̄ = mean_s C_s (L0) and M̄ = mean_s M_s (candidate); per resample b,
100 (C̄_b - M̄_b) / C̄_b with C̄_b = mean_s C_{s,b}; CI = the 95% percentile interval; a zero C̄_b makes the CI
UNDEFINED. Per-seed ratios (seed s vs seed s) and the mean of the per-seed points are reported too (secondary).

Acceptance: ACCEPT iff, for the WORSE family (the larger seed-mean relative-drop point), point <= 1.0 % AND the 95 %
upper bound <= 2.0 %. Schemas (SCHEMA-01R1, DECISION_LOG 47): the reference "L0" below is L0S_CLEAN_V2_SESSION and the candidate base "L3"
is L3S_NEAR_DUPLICATE_SESSION (both + session_offset_tokens); G1 is REQUIRED and never dropped (noise_groups).
Search (the stop rule: the first accepted schema is locked):
  1. no group dropped -> S* is L3: LOCK L3 if L3 passes, else LOCK L0;
  2. TRAIN S* (both families, both seeds); ACCEPT -> LOCK S*;
  3. fail -> restore the most important dropped group (the largest seed-mean importance for the family that failed
     the rule, i.e. the worse family of S*; ties: the other family's importance, then the group order G1 … G12c)
     and TRAIN it once (if only one group was dropped, the restored schema IS L3: no retrain); ACCEPT -> LOCK it;
  4. fail -> LOCK L3 if L3 passes the same rule, else LOCK L0.
Outcome discipline: the decision record carries relative drops, CIs and importances only (no absolute MRR).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from v4eval.bootstrap import UserStats, ci95, make_plan, point, resampled
from v4eval.values import ZERO_DENOMINATOR_IN_RESAMPLES, Undefined, to_jsonable

RUN_LABEL = "FEATURE_SEARCH_SCREEN"
STATISTIC = "macro_e2e"
POINT_MAX_PERCENT = 1.0
UB_MAX_PERCENT = 2.0
L0_ID = "L0S_CLEAN_V2_SESSION"           # SCHEMA-01R1 (DECISION_LOG 47): the reference is L0S
L3_ID = "L3S_NEAR_DUPLICATE_SESSION"     # the candidate base is L3S; S* = L3S minus the dropped groups
FAMILIES = ("GRU", "SASREC")


class DecisionError(RuntimeError):
    pass


def sstar_id(dropped: Iterable[str], base_id: str = L3_ID) -> str:
    """The derived-schema id v4fs.schemas.derive gives L3 minus `dropped` (L3 itself when nothing is dropped)."""
    d = sorted(set(dropped))
    return base_id if not d else f"{base_id}-minus-{'+'.join(d)}"


def _plan_for(*stats: UserStats):
    users = stats[0].user_keys
    for s in stats:
        if not np.array_equal(s.user_keys, users):
            raise DecisionError("the runs cover different user sets (one W per user set: pairing is impossible)")
    return make_plan(users)


def relative_drop(ref_by_seed: Mapping[int, UserStats], cand_by_seed: Mapping[int, UserStats],
                  plan=None) -> dict:
    """The seed-mean relative drop (percent) of a candidate versus the reference (L0), bootstrapped directly."""
    seeds = sorted(ref_by_seed)
    if not seeds or sorted(cand_by_seed) != seeds:
        raise DecisionError(f"seeds differ: reference {sorted(ref_by_seed)} vs candidate {sorted(cand_by_seed)}")
    plan = plan or _plan_for(*[ref_by_seed[s] for s in seeds], *[cand_by_seed[s] for s in seeds])
    C = {s: point(ref_by_seed[s], STATISTIC) for s in seeds}
    M = {s: point(cand_by_seed[s], STATISTIC) for s in seeds}
    rC = {s: resampled(plan, ref_by_seed[s], STATISTIC) for s in seeds}
    rM = {s: resampled(plan, cand_by_seed[s], STATISTIC) for s in seeds}
    cbar, mbar = float(np.mean([C[s] for s in seeds])), float(np.mean([M[s] for s in seeds]))
    if cbar == 0.0:
        raise DecisionError("the reference seed-mean value is 0: the relative drop is undefined")
    cb = np.mean([rC[s] for s in seeds], axis=0)
    mb = np.mean([rM[s] for s in seeds], axis=0)
    out = {"point": 100.0 * (cbar - mbar) / cbar, "seeds": seeds, "W_sha256": plan.W_sha256,
           "n_users": plan.n_users, "statistic": STATISTIC, "definition": "100 (mean_s C_s - mean_s M_s) / mean_s C_s"}
    zero = int(np.count_nonzero(cb == 0))
    if zero:
        out["ci95"] = Undefined(ZERO_DENOMINATOR_IN_RESAMPLES)
        out["ub"] = None
    else:
        out["ci95"] = ci95(100.0 * (cb - mb) / cb)
        out["ub"] = out["ci95"][1]
    out["n_resamples_zero_denominator"] = zero
    per = {}
    for s in seeds:
        d = {"point": 100.0 * (C[s] - M[s]) / C[s] if C[s] else None}
        z = int(np.count_nonzero(rC[s] == 0))
        d["ci95"] = Undefined(ZERO_DENOMINATOR_IN_RESAMPLES) if z else ci95(100.0 * (rC[s] - rM[s]) / rC[s])
        per[str(s)] = d
    out["per_seed"] = per
    vals = [per[str(s)]["point"] for s in seeds if per[str(s)]["point"] is not None]
    out["mean_of_per_seed_points"] = float(np.mean(vals)) if vals else None
    return out


def passes(drop: Mapping) -> bool:
    ub = drop.get("ub")
    return ub is not None and drop["point"] <= POINT_MAX_PERCENT and ub <= UB_MAX_PERCENT


def accept(drops_by_family: Mapping[str, Mapping]) -> dict:
    """The rule on the WORSE family (the larger seed-mean relative-drop point; ties: the larger upper bound)."""
    fams = sorted(drops_by_family)
    if sorted(fams) != sorted(FAMILIES):
        raise DecisionError(f"the acceptance rule needs both families {FAMILIES}, got {fams}")

    def badness(f):
        d = drops_by_family[f]
        return (d["point"], d["ub"] if d.get("ub") is not None else float("inf"))
    worse = max(fams, key=badness)
    per = {f: {"point": drops_by_family[f]["point"], "ub": drops_by_family[f].get("ub"),
               "pass": passes(drops_by_family[f])} for f in fams}
    ok = per[worse]["pass"]
    return {"worse_family": worse, "accept": bool(ok), "per_family": per,
            "both_pass": all(v["pass"] for v in per.values()),
            "rule": f"worse family: seed-mean relative drop point <= {POINT_MAX_PERCENT}% AND 95% UB <= "
                    f"{UB_MAX_PERCENT}%"}


def most_important_dropped(dropped: Sequence[str], importance_by_family: Mapping[str, Mapping[str, dict]],
                           worse_family: str, order: Sequence[str] = ()) -> str:
    if not dropped:
        raise DecisionError("no dropped group to restore")
    other = [f for f in importance_by_family if f != worse_family]
    rank = {g: i for i, g in enumerate(order)}

    def key(g):
        w = importance_by_family[worse_family][g]["importance"]
        o = max((importance_by_family[f][g]["importance"] for f in other), default=0.0)
        return (w, o, -rank.get(g, len(rank)))
    return max(dropped, key=key)


def search(*, dropped: Sequence[str], importance_by_family: Optional[Mapping] = None,
           results: Mapping[str, Mapping] = (), order: Sequence[str] = ()) -> dict:
    """The next step of the FS search from what is known. `results` maps schema_id -> accept() output (vs L0).
    Returns {"action": "LOCK" | "TRAIN", "schema_id", "reason", "trace"}."""
    results = dict(results or {})
    trace = []
    if L3_ID not in results:
        return {"action": "TRAIN", "schema_id": L3_ID, "reason": "L0 and L3 are trained first", "trace": trace}
    l3_ok = results[L3_ID]["accept"]
    fallback = {"action": "LOCK", "schema_id": L3_ID if l3_ok else L0_ID,
                "reason": "L3 passes the rule" if l3_ok else "L3 fails the rule: L0"}
    from v4fs.importance import required_groups
    bad = sorted(set(dropped) & set(required_groups()))
    if bad:
        raise DecisionError(f"groups {bad} hold REQUIRED channels (DECISION_LOG 47): they are never dropped")
    dset = set(dropped)
    dropped = [g for g in order if g in dset] + sorted(g for g in dset if g not in set(order))
    if not dropped:
        trace.append("no group is within noise for both families: S* = L3")
        return dict(fallback, trace=trace)
    s1 = sstar_id(dropped)
    if s1 not in results:
        return {"action": "TRAIN", "schema_id": s1, "dropped_groups": dropped, "reason": "S* = L3 minus the groups "
                "within noise for BOTH families", "trace": trace}
    if results[s1]["accept"]:
        trace.append(f"{s1} accepted")
        return {"action": "LOCK", "schema_id": s1, "dropped_groups": dropped, "reason": "S* accepted",
                "trace": trace}
    worse = results[s1]["worse_family"]
    if importance_by_family is None:
        raise DecisionError("the fallback needs the importances")
    restored = most_important_dropped(dropped, importance_by_family, worse, order)
    rest = [g for g in dropped if g != restored]
    s2 = sstar_id(rest)
    trace.append(f"{s1} failed (worse family {worse}); restore {restored} -> {s2}")
    if s2 == L3_ID:
        trace.append("restoring the only dropped group gives L3 itself")
        return dict(fallback, restored_group=restored, trace=trace)
    if s2 not in results:
        return {"action": "TRAIN", "schema_id": s2, "dropped_groups": rest, "restored_group": restored,
                "reason": "the single fallback retrain", "trace": trace}
    if results[s2]["accept"]:
        trace.append(f"{s2} accepted")
        return {"action": "LOCK", "schema_id": s2, "dropped_groups": rest, "restored_group": restored,
                "reason": "the fallback schema accepted", "trace": trace}
    trace.append(f"{s2} failed")
    return dict(fallback, restored_group=restored, trace=trace)


# ------------------------------------------------------------------------------------------------ CLI over run dirs
def load_runs(run_dirs: Iterable[Path]) -> dict:
    """{(schema_id, family, seed): {"summary": FS_SUMMARY, "stats": {...}}} from FS run directories."""
    from v4fs.importance import load_stats
    out = {}
    for d in run_dirs:
        d = Path(d)
        s = json.loads((d / "FS_SUMMARY.json").read_text(encoding="utf-8"))
        if s.get("run_label") != RUN_LABEL:
            raise DecisionError(f"{d}: not a FEATURE_SEARCH_SCREEN run")
        stats, meta = load_stats(d / s["userstats_file"])
        k = (s["schema"]["schema_id"], s["family"], int(s["seed"]))
        if k in out:
            raise DecisionError(f"two runs for {k}")
        out[k] = {"summary": s, "stats": stats, "meta": meta, "dir": str(d)}
    return out


def decide_runs(runs: Mapping) -> dict:
    from v4fs.importance import FS_GROUPS, PERM_SEEDS, combine_seeds, noise_groups
    by = {}
    for (sid, fam, seed), r in runs.items():
        by.setdefault(sid, {}).setdefault(fam, {})[seed] = r
    if L0_ID not in by or L3_ID not in by:
        return {"run_label": RUN_LABEL, "action": "TRAIN", "reason": "L0 and L3 runs are missing"}
    imp = {}
    for fam in FAMILIES:
        l3 = by[L3_ID].get(fam, {})
        if sorted(l3) != [2026, 2027]:
            raise DecisionError(f"L3 {fam}: seeds {sorted(l3)} (both 2026 and 2027 are needed)")
        present = [g for g in FS_GROUPS if all(f"{g}|{p}" in l3[s]["stats"] for s in l3 for p in PERM_SEEDS)]
        imp[fam] = combine_seeds({s: l3[s]["stats"] for s in l3}, present, PERM_SEEDS)
    dropped = noise_groups(imp)
    results = {}
    drops = {}
    for sid, fams in by.items():
        if sid == L0_ID:
            continue
        if sorted(fams) != sorted(FAMILIES) or any(sorted(fams[f]) != [2026, 2027] for f in fams):
            continue
        dd = {}
        for fam in FAMILIES:
            ref = {s: by[L0_ID][fam][s]["stats"]["endpoint"] for s in (2026, 2027)}
            cand = {s: fams[fam][s]["stats"]["endpoint"] for s in (2026, 2027)}
            dd[fam] = relative_drop(ref, cand)
        drops[sid] = dd
        results[sid] = accept(dd)
    step = search(dropped=dropped, importance_by_family=imp, results=results, order=list(FS_GROUPS))
    return to_jsonable({"run_label": RUN_LABEL, "statistic": "CONTROLLED_BUDGET 6.0 client-macro MRR@20 (E2E)",
                        "importance": imp, "dropped_groups": dropped, "relative_drops": drops,
                        "acceptance": results, "next": step,
                        "note": "relative drops, CIs and importances only; absolute MRR stays in the run receipts"})


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = decide_runs(load_runs(a.runs))
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(res.get("next"), indent=1))
    return 0


if __name__ == "__main__":
    import sys
    _CODE = Path(__file__).resolve().parents[1]
    if str(_CODE) not in sys.path:
        sys.path.insert(0, str(_CODE))
    import ppsi_env  # noqa: F401
    raise SystemExit(main())
