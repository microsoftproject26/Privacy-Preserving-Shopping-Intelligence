"""v4bench.selection — the registered decisions of amendment 07 that sit between the runs and the LOCK.

  trackb_select   R5 / O1: the Track-B winner of ONE arm on one dataset, from its finished grid-cell runs (seed 2026):
                  best inner-validation NDCG@10 at PRACTICAL_BEST; ties within 0.0005 -> the Track-A cell, then the
                  smaller budget, the smaller max_len, the larger dropout.
  decide_warm     B2 (N2-type rule): FA_WARM* = S7 (FA_WARM) or S8 (FA_WARM_FROZEN) by the higher mean-over-seeds
                  PRACTICAL_BEST inner-validation NDCG@10 (ALL users); ties within 0.0005 -> S8.
  seed3_check     R6: a third seed (2028) for Track-A C_FULL and FA_WARM* when the two seeds' PRACTICAL_BEST inner-validation
                  NDCG@10 differ by more than 0.002.
  tree_digest     R9: the code-tree digest (the r6 files v4bench imports: fl, v4models, v4train, v4sup, snap_v1, ppsi_env).
All decisions are write-once JSON files under runs_bench/<dataset>/ and are listed (sha256) in LOCK.json.
BM-03: every decision first checks that its runs share ONE device class (refused before any value is computed) and records it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .common import (CODE, BenchRefused, canon_sha256, check_dataset, read_json, sha256_file, utc, write_exclusive)

TIE = 0.0005
SEED_GAP = 0.002
TRACK_A_CELL = (6, 50, 0.1)                       # (budget EFE, max_len, dropout)
REQUIRED_SEEDS = (2026, 2027)
TREE_DIRS = ("fl", "v4models", "v4train", "v4sup", "snap_v1")
TREE_FILES = ("ppsi_env.py",)


def tree_digest() -> str:
    rows = []
    for d in TREE_DIRS:
        for p in sorted((CODE / d).rglob("*")):
            if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts:
                rows.append((p.relative_to(CODE).as_posix(), sha256_file(p)))
    for f in TREE_FILES:
        rows.append((f, sha256_file(CODE / f)))
    return canon_sha256(rows)


def result_of(runs_root, dataset: str, run: str) -> dict:
    p = Path(runs_root) / check_dataset(dataset) / run / "RESULT.json"
    if not p.is_file():
        raise BenchRefused(f"{run}: no RESULT.json")
    res = read_json(p)
    if res.get("status") != "DONE_MAIN":
        raise BenchRefused(f"{run}: status {res.get('status')!r}")
    return res


def same_device(runs_root, dataset: str, runs: Sequence[str], what: str) -> str:
    """BM-03: the one device class of `runs` (RESULT.json; absent = CPU-FP32); a mix is refused before any value is read."""
    from .accel import device_class_of, one_class
    return one_class({r: device_class_of(result_of(runs_root, dataset, r)) for r in runs}, what)


def pb_value(res: Mapping) -> float:
    """PRACTICAL_BEST inner-validation NDCG@10 of the run's selection population (PRE: band; every other arm: all)."""
    pop = res["spec"].get("select_population", "ALL") if res.get("kind") == "central" else "ALL"
    return float(res["validation"]["PRACTICAL_BEST"][pop]["ndcg@10"])


def cell_of(res: Mapping) -> tuple:
    sp = res["spec"]
    return (int(sp.get("budget_efe") or 6), int(sp.get("max_len") or 50), float((sp.get("dropout") or [0.1])[0]))


def trackb_pick(cells: Mapping[str, tuple], values: Mapping[str, float]) -> str:
    best = max(values.values())
    near = [r for r, v in values.items() if best - v <= TIE + 1e-15]
    return sorted(near, key=lambda r: (cells[r] != TRACK_A_CELL, cells[r][0], cells[r][1], -cells[r][2], r))[0]


def trackb_select(runs_root, dataset: str, arm: str, runs: Sequence[str]) -> dict:
    dclass = same_device(runs_root, dataset, runs, "trackb-select")
    cells, values = {}, {}
    for r in runs:
        res = result_of(runs_root, dataset, r)
        if res["arm"] != arm or int(res["spec"].get("seed", res["spec"].get("cfg", {}).get("seed", -1))) != 2026:
            raise BenchRefused(f"{r}: Track-B grid cells are {arm} runs at seed 2026")
        cells[r], values[r] = cell_of(res), pb_value(res)
    if TRACK_A_CELL not in cells.values():
        raise BenchRefused("the grid must contain the Track-A cell (6 EFE, max_len 50, dropout 0.1) for the tie rule")
    win = trackb_pick(cells, values)
    obj = {"format": "BM02_TRACKB_WINNER_1", "dataset": dataset, "arm": arm, "winner_run": win,
           "winner_cell": list(cells[win]), "values": values, "cells": {r: list(c) for r, c in cells.items()},
           "rule": "max PB inner-val NDCG@10; ties <= 0.0005 -> Track-A cell, smaller budget, smaller max_len, larger dropout",
           "next": f"run the winning cell with seed 2027 ({arm})", "utc": utc(), "device_class": dclass}
    write_exclusive(Path(runs_root) / dataset / "trackb" / f"WINNER_{arm}.json", obj)
    return obj


def assert_track_a(res: Mapping, name: str) -> None:
    if cell_of(res) != TRACK_A_CELL:
        raise BenchRefused(f"{name}: not the Track-A cell {TRACK_A_CELL} (cell {cell_of(res)}); Track A and B are never mixed")


def decide_warm(runs_root, dataset: str, s7_runs: Sequence[str], s8_runs: Sequence[str]) -> dict:
    dclass = same_device(runs_root, dataset, list(s7_runs) + list(s8_runs), "decide-warm")

    def mean(runs, arm):
        vs = []
        for r in runs:
            res = result_of(runs_root, dataset, r)
            if res["arm"] != arm:
                raise BenchRefused(f"{r} is not an {arm} run")
            assert_track_a(res, r)
            vs.append(pb_value(res))
        if sorted(int(result_of(runs_root, dataset, r)["spec"]["cfg"]["seed"]) for r in runs) != list(REQUIRED_SEEDS):
            raise BenchRefused(f"{arm}: the decision needs exactly the seeds {REQUIRED_SEEDS} (Track A, 6 EFE)")
        return sum(vs) / len(vs), vs
    m7, v7 = mean(s7_runs, "FA_WARM")
    m8, v8 = mean(s8_runs, "FA_WARM_FROZEN")
    chosen = "FA_WARM" if m7 > m8 + TIE else "FA_WARM_FROZEN"
    obj = {"format": "BM02_WARM_DECISION_1", "dataset": dataset, "FA_WARM_STAR": chosen, "S7_mean": m7, "S8_mean": m8,
           "S7_values": v7, "S8_values": v8, "s7_runs": sorted(s7_runs), "s8_runs": sorted(s8_runs),
           "rule": "S7 only if mean(S7) > mean(S8) + 0.0005, else S8 (ties -> S8)", "utc": utc(),
           "device_class": dclass}
    write_exclusive(Path(runs_root) / dataset / "FA_WARM_DECISION.json", obj)
    return obj


def check_dp_variant(runs_root, dataset: str, arm: str) -> None:
    """B9 (4): a warm DP-chain arm (S_CAL / FA_1024 / DP8) runs on the variant FA_WARM* decided in Track A:
    S7 -> S_CAL_WARM / FA_1024_WARM / DP_WARM (DP8_WARM), S8 -> S_CAL_WARM_FROZEN / ... / DP8_FROZEN."""
    from . import recipe
    wv = recipe.warm_variant(arm)
    if not recipe.is_poisson_arm(arm) or wv is None:
        return
    p = Path(runs_root) / check_dataset(dataset) / "FA_WARM_DECISION.json"
    if not p.is_file():
        raise BenchRefused(f"{arm}: {p} is missing - the DP chain follows the FA_WARM* decision (B9)")
    star = read_json(p)["FA_WARM_STAR"]
    if wv != star:
        raise BenchRefused(f"{arm} is a {wv} arm but FA_WARM* = {star} on {dataset} (B9: the DP row is bound to it)")


def seed3_check(runs_root, dataset: str, runs_by_arm: Mapping[str, Sequence[str]]) -> dict:
    """runs_by_arm: {"C_FULL": [s2026 run, s2027 run], "<FA_WARM*>": [...]} (Track A)."""
    dec = read_json(Path(runs_root) / dataset / "FA_WARM_DECISION.json")
    if set(runs_by_arm) != {"C_FULL", dec["FA_WARM_STAR"]}:
        raise BenchRefused(f"seed-3 check covers C_FULL and {dec['FA_WARM_STAR']} only")
    from .accel import device_class_of, one_class
    dclass = one_class({"FA_WARM_DECISION.json": device_class_of(dec),
                        "runs": same_device(runs_root, dataset, [r for rs in runs_by_arm.values() for r in rs], "seed3")},
                       "seed3")
    gaps, need = {}, []
    for arm, runs in runs_by_arm.items():
        for r in runs:
            assert_track_a(result_of(runs_root, dataset, r), r)
        vals = {int(result_of(runs_root, dataset, r)["spec"].get("seed") or result_of(runs_root, dataset, r)["spec"]["cfg"]["seed"]):
                pb_value(result_of(runs_root, dataset, r)) for r in runs}
        if sorted(vals) != list(REQUIRED_SEEDS):
            raise BenchRefused(f"{arm}: needs one run per seed {REQUIRED_SEEDS}")
        gaps[arm] = abs(vals[2026] - vals[2027])
        if gaps[arm] > SEED_GAP:
            need.append(arm)
    obj = {"format": "BM02_SEED3_DECISION_1", "dataset": dataset, "gaps": gaps, "threshold": SEED_GAP,
           "third_seed_2028_needed_for": sorted(need), "utc": utc(), "device_class": dclass}
    write_exclusive(Path(runs_root) / dataset / "SEED3_DECISION.json", obj)
    return obj
