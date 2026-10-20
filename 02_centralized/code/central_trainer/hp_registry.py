"""v4train.hp_registry — the P2 HP registry (TRAIN-02 item 1; TRAIN-02R1: PREG1_AMENDMENT_02R2, which supersedes the
LR / dropout assignment of 02R1 §2; everything else in 02R1 stands).

  python -B -m v4train.hp_registry --write contracts/hp_registry/HP_REGISTRY_P2R2.json   # O_EXCL: never overwrites
  python -B -m v4train.hp_registry --verify contracts/hp_registry/HP_REGISTRY_P2R2.json  # exit 0 / 1 (R5, server)

Design (02R2; registered values, not a procedure; no random draw anywhere):
  * LR grid: stratum midpoints on log10 over [log10 3e-4, log10 3e-3] at full float precision — GRU 6 strata per size
    (w = 1/6 decade), SASRec 4 strata per size (w = 1/4 decade): peak_lr(k) = 10 ** (lo + (k + 0.5) w);
  * cells per size level, IDENTICAL for every size level of a family (02R2 §2):
      GRU    S0 -> strata (0, 3, 5) with p (0.1, 0.2, 0.3); S1 -> strata (1, 2, 4) with p (0.3, 0.2, 0.1);
             dropout sites (input p, readout p / 2); 2 sizes x 6 = 12 trials;
      SASREC S0 -> strata (0, 3) with p (0.1, 0.3);    S1 -> strata (1, 2) with p (0.2, 0.2); single p;
             3 sizes x 4 = 12 trials; family p counts 0.1: 3, 0.2: 6, 0.3: 3 (02R2 §3);
  * trial ids name size, schedule and stratum: P2_<VARIANT>_<SCHEDULE>_s<stratum> (e.g. P2_GRU_D256_S0_s3,
    P2_SASREC_D384_B2_S1_s2); canonical order: size, then schedule (S0, S1), then the §2 order; each trial carries the
    sha256 of its canonical JSON; there is no "control" trial;
  * seeds: stage 1 = 2026 for every trial; stage 2 = 2027 for the best trial of each size level (the seed is not part of
    a trial, so a stage-2 re-run uses the same hp_id).
The file is self-hashed (canonical JSON rule of v4train.registry / v4models.derived). `load_hp_registry` verifies the
format (a superseded registry, e.g. the 02R1 HP_REGISTRY_P2.json, is refused: it is kept on disk as the record and never
launched — any trial under it would be HP_SEARCH_VOID), the self-hash, every trial's sha256, and that the design
sections equal a fresh `registry_obj()` build (file and code cannot drift). `verify_file` (the --verify CLI) recomputes
the grid midpoints and the cells and reports every difference with exit 1; the registered file stays authoritative.
RunConfig.validate (v4train.recipe) accepts a peak LR, dropout, schedule and model variant that differ from the
registered recipe ONLY when they equal the trial named by `hp_id`.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping

_HERE = Path(__file__).resolve().parent
V4 = _HERE.parents[1]
REGISTRY_PATH = V4 / "contracts" / "hp_registry" / "HP_REGISTRY_P2R2.json"
STALE_REGISTRIES = {"PPSI_V4_HP_REGISTRY_P2_1": "02R1 HP_REGISTRY_P2.json, superseded by PREG1_AMENDMENT_02R2"}
FORMAT = "PPSI_V4_HP_REGISTRY_P2R2_1"
SELF_KEY = "self_sha256"
FAMILIES = ("GRU", "SASREC")
LR_BOUNDS = (3e-4, 3e-3)
N_STRATA = {"GRU": 6, "SASREC": 4}
SCHEDULE_IDS = ("S0", "S1")
VARIANT_LEVELS = {"GRU": ("GRU_D128", "GRU_D256"), "SASREC": ("SASREC_D256_B2", "SASREC_D256_B3", "SASREC_D384_B2")}
CELLS = {"GRU": {"S0": ((0, 0.1), (3, 0.2), (5, 0.3)), "S1": ((1, 0.3), (2, 0.2), (4, 0.1))},     # (stratum, p)
         "SASREC": {"S0": ((0, 0.1), (3, 0.3)), "S1": ((1, 0.2), (2, 0.2))}}
SASREC_P_COUNTS = {0.1: 3, 0.2: 6, 0.3: 3}
TRIALS_PER_FAMILY = 12
SEEDS = {"stage1": 2026, "stage2": 2027}
AUTHORITY = ("contracts/PREG1_AMENDMENT_02R2.md §1-§4 (registered 2026-09-27; supersedes the LR / dropout assignment of "
             "02R1 §2); contracts/PREG1_AMENDMENT_02R1.md (the rest); TRAIN-02 item 1, TRAIN-02R1")
DESIGN_KEYS = ("families", "lr_grid", "cells", "schedules", "seeds", "fixed")


class HPRegistryError(ValueError):
    pass


def canonical_sha256(obj: Mapping, exclude=(SELF_KEY,)) -> str:
    body = {k: v for k, v in obj.items() if k not in exclude}
    enc = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(enc.encode("utf-8")).hexdigest()


def lr_grid() -> dict:
    """The registered LR values: per family, the stratum midpoints on log10 over [log10 3e-4, log10 3e-3]."""
    lo, hi = math.log10(LR_BOUNDS[0]), math.log10(LR_BOUNDS[1])
    out = {"rule": "peak_lr(k) = 10 ** (log10(3e-4) + (k + 0.5) * (log10(3e-3) - log10(3e-4)) / n_strata); "
                   "GRU n_strata 6, SASREC 4 (02R2 §1)", "bounds": list(LR_BOUNDS)}
    for fam, n in N_STRATA.items():
        w = (hi - lo) / n
        out[fam] = {"n_strata": n, "strata": [{"stratum": k, "log10_lo": lo + k * w, "log10_hi": lo + (k + 1) * w,
                                               "log10_mid": lo + (k + 0.5) * w,
                                               "peak_lr": float(10.0 ** (lo + (k + 0.5) * w))} for k in range(n)]}
    out["sha256"] = canonical_sha256(out, exclude=("sha256",))
    return out


def cells() -> dict:
    body = {"rule": "02R2 §2: the same (schedule, stratum, p) cells for every size level of a family",
            **{fam: [{"schedule": s, "lr_stratum": k, "dropout_p": p} for s in SCHEDULE_IDS for k, p in CELLS[fam][s]]
               for fam in FAMILIES}}
    body["sha256"] = canonical_sha256(body, exclude=("sha256",))
    return body


def _dropout_sites(family: str, p: float) -> list:
    return [p, p / 2] if family == "GRU" else [p]


def family_trials(family: str, grid: dict) -> list:
    trials, i = [], 0
    for v in VARIANT_LEVELS[family]:
        for s in SCHEDULE_IDS:
            for k, p in CELLS[family][s]:
                i += 1
                st = grid[family]["strata"][k]
                t = {"hp_id": f"P2_{v}_{s}_s{k}", "family": family, "index": i,
                     "cell": {"model_variant": v, "schedule": s, "lr_stratum": k, "dropout_p": p},
                     "model_variant": v, "schedule": s, "dropout_p": p, "dropout": _dropout_sites(family, p),
                     "dropout_sites": ["input", "readout"] if family == "GRU" else ["all"],
                     "lr_stratum": k, "log10_peak_lr": st["log10_mid"], "peak_lr": st["peak_lr"]}
                t["sha256"] = canonical_sha256(t, exclude=("sha256",))
                trials.append(t)
    return trials


def cell_multiset(trials, variant: str) -> str:
    """The canonical (LR, schedule, dropout) multiset of one size level (02R2 §4: byte-identical across sizes)."""
    rows = sorted([[t["peak_lr"], t["schedule"], t["dropout"]] for t in trials if t["model_variant"] == variant],
                  key=lambda r: json.dumps(r))
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def _check_design(fams: Mapping) -> None:
    for fam, trials in fams.items():
        if len(trials) != TRIALS_PER_FAMILY or len({t["hp_id"] for t in trials}) != TRIALS_PER_FAMILY:
            raise HPRegistryError(f"{fam}: {TRIALS_PER_FAMILY} trials with unique ids required")
        sets = {v: cell_multiset(trials, v) for v in VARIANT_LEVELS[fam]}
        if len(set(sets.values())) != 1:
            raise HPRegistryError(f"{fam}: the cell multiset differs across size levels")
        ps = sorted({t["dropout_p"] for t in trials})
        for v in VARIANT_LEVELS[fam]:
            for p in ps:
                if not any(t["model_variant"] == v and t["dropout_p"] == p for t in trials):
                    raise HPRegistryError(f"{fam}: pair ({v}, {p}) missing")
    counts = {p: sum(t["dropout_p"] == p for t in fams["SASREC"]) for p in SASREC_P_COUNTS}
    if counts != SASREC_P_COUNTS:
        raise HPRegistryError(f"SASREC p counts {counts} != 02R2 §3 {SASREC_P_COUNTS}")


def registry_obj() -> dict:
    from .schedule import SCHEDULES
    grid = lr_grid()
    fams = {f: family_trials(f, grid) for f in FAMILIES}
    _check_design(fams)
    obj = {"format": FORMAT, "task": "TRAIN-02 item 1 / TRAIN-02R1 (02R2 re-issue)", "authority": AUTHORITY,
           "supersedes": {"file": "contracts/hp_registry/HP_REGISTRY_P2.json", "format": "PPSI_V4_HP_REGISTRY_P2_1",
                          "self_sha256": "4b8a8cd4db22334314c5c4df62b0aa6ab15377c78bc33c5628b83ea4d3a7e7c2",
                          "status": "kept as the record; never launched (HP_SEARCH_VOID if it were)"},
           "families": fams, "lr_grid": grid, "cells": cells(),
           "schedules": {k: [list(p) for p in v] for k, v in SCHEDULES.items()},
           "seeds": dict(SEEDS),
           "fixed": {"optimizer": "AdamW (fused)", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-5,
                     "weight_decay_scope": "ndim>=2", "grad_clip": 1.0, "effective_batch": 256,
                     "label_smoothing": 0.0, "efe": 6.0, "eval_marks": [2.0, 4.0, 6.0], "endpoint": 6.0,
                     "snapshot_marks": [5.0, 5.5, 6.0], "determinism_mode": "STRICT",
                     "train_manifest_role": "CALIBRATION (CALIB_TRAIN_5_100_32768)",
                     "eval_manifest_role": "CALIBRATION (CALIB_VALIDATION_5_100_32768); MAIN_VALIDATION never read",
                     "run_role": "TUNING", "run_label": "HP_SEARCH"},
           "disclosure": "02R2 §5: within a size each schedule sees half of the LR strata, so the schedule and dropout "
                         "effects remain partly confounded with LR; size is not",
           "selection": "02R1 §3 (v4train.p2_driver): stage 2 = the best 6.0 value of each size level at seed 2027; "
                        "tuned = the higher 2-seed mean at 6.0; tie |delta| <= 0.0005 -> fewer parameters, then the "
                        "lower LR; exclusions: a non-finite loss or endpoint metric, or a 6.0 value < 0.5 x the family "
                        "best; a size level without a valid 6.0 run is a Fable trigger; CKPT_AVG carried iff > +0.0005 "
                        "on the 2-seed mean"}
    obj[SELF_KEY] = canonical_sha256(obj)
    return obj


def write_registry(path=REGISTRY_PATH) -> dict:
    from v4models.variants import write_exclusive
    obj = registry_obj()
    text = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    return {"path": str(path), "file_sha256": write_exclusive(Path(path), text), "self_sha256": obj[SELF_KEY],
            "lr_grid_sha256": obj["lr_grid"]["sha256"], "cells_sha256": obj["cells"]["sha256"]}


def _read(p: Path) -> tuple:
    if not p.is_file():
        raise HPRegistryError(f"HP registry {p} does not exist")
    data = p.read_bytes()
    try:
        obj = json.loads(data.decode("utf-8"),
                         parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except ValueError as e:
        raise HPRegistryError(f"HP registry {p} is not strict JSON ({e})") from None
    if not isinstance(obj, dict):
        raise HPRegistryError(f"HP registry {p} is not a JSON object")
    return obj, data


def load_hp_registry(path=REGISTRY_PATH) -> dict:
    """Read and verify: strict JSON, the current format (a superseded registry is refused), self-hash, each trial's
    sha256, the design equal to the code's."""
    p = Path(path)
    obj, data = _read(p)
    if obj.get("format") in STALE_REGISTRIES:
        raise HPRegistryError(f"HP registry {p} is a superseded registry ({STALE_REGISTRIES[obj['format']]}); it is "
                              "never launched")
    if obj.get("format") != FORMAT:
        raise HPRegistryError(f"HP registry {p} is not a {FORMAT} object")
    if obj.get(SELF_KEY) != canonical_sha256(obj):
        raise HPRegistryError(f"HP registry {p}: self-hash mismatch (stated {obj.get(SELF_KEY)!r}, recomputed "
                              f"{canonical_sha256(obj)})")
    for fam, trials in (obj.get("families") or {}).items():
        for t in trials:
            if t.get("sha256") != canonical_sha256(t, exclude=("sha256",)):
                raise HPRegistryError(f"HP registry {p}: trial {t.get('hp_id')} sha256 mismatch")
    fresh = registry_obj()
    bad = [k for k in DESIGN_KEYS if obj.get(k) != fresh[k]]
    if bad:
        raise HPRegistryError(f"HP registry {p}: the sections {bad} differ from the registered design (code)")
    return {"path": str(p), "file_sha256": hashlib.sha256(data).hexdigest(), "self_sha256": obj[SELF_KEY],
            "families": obj["families"], "lr_grid_sha256": obj["lr_grid"]["sha256"], "raw": obj}


def verify_file(path=REGISTRY_PATH) -> dict:
    """F4 (R5 on the server): recompute the grid midpoints and the cells and compare with the file, field by field.
    Never modifies anything; `ok` is True only when the file loads and every recomputed value equals the file's."""
    p = Path(path)
    out = {"path": str(p), "python": sys.version.split()[0], "problems": []}
    try:
        obj, data = _read(p)
    except HPRegistryError as e:
        out["problems"].append(str(e))
        out["ok"] = False
        return out
    out["file_sha256"] = hashlib.sha256(data).hexdigest()
    fresh = registry_obj()
    for fam in FAMILIES:
        got = [s.get("peak_lr") for s in ((obj.get("lr_grid") or {}).get(fam) or {}).get("strata", [])]
        want = [s["peak_lr"] for s in fresh["lr_grid"][fam]["strata"]]
        if got != want:
            out["problems"].append(f"{fam}: LR grid {got} != recomputed {want}")
        gt = [(t.get("hp_id"), t.get("model_variant"), t.get("schedule"), t.get("lr_stratum"), t.get("dropout"),
               t.get("peak_lr")) for t in (obj.get("families") or {}).get(fam, [])]
        wt = [(t["hp_id"], t["model_variant"], t["schedule"], t["lr_stratum"], t["dropout"], t["peak_lr"])
              for t in fresh["families"][fam]]
        if gt != wt:
            out["problems"].append(f"{fam}: trials differ from the recomputed cells")
    try:
        r = load_hp_registry(p)
        out.update(self_sha256=r["self_sha256"], lr_grid_sha256=r["lr_grid_sha256"])
    except HPRegistryError as e:
        out["problems"].append(str(e))
    out["recomputed_self_sha256"] = fresh[SELF_KEY]
    out["ok"] = not out["problems"]
    return out


def trial(reg: Mapping, hp_id: str) -> dict:
    for trials in reg["families"].values():
        for t in trials:
            if t["hp_id"] == hp_id:
                if t["sha256"] != canonical_sha256(t, exclude=("sha256",)):
                    raise HPRegistryError(f"trial {hp_id}: sha256 mismatch")
                return copy.deepcopy(t)
    raise HPRegistryError(f"hp_id {hp_id!r} is not in the HP registry {reg.get('path')}")


def trial_record(reg: Mapping, t: Mapping) -> dict:
    """What a run records about its trial (RUN_START, rows): ids and hashes."""
    return {"hp_id": t["hp_id"], "hp_sha256": t["sha256"], "hp_registry_path": reg["path"],
            "hp_registry_file_sha256": reg["file_sha256"], "hp_registry_self_sha256": reg["self_sha256"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write / verify the P2 HP registry (TRAIN-02 item 1; 02R2)")
    ap.add_argument("--write", default=None)
    ap.add_argument("--verify", default=None)
    args = ap.parse_args(argv)
    rc = 0
    if args.write:
        print(json.dumps(write_registry(Path(args.write)), indent=1))
    if args.verify:
        res = verify_file(args.verify)
        print(json.dumps(res, indent=1))
        rc = 0 if res["ok"] else 1
    return rc


__all__ = ["REGISTRY_PATH", "FORMAT", "STALE_REGISTRIES", "LR_BOUNDS", "N_STRATA", "CELLS", "SCHEDULE_IDS",
           "VARIANT_LEVELS", "SEEDS", "SASREC_P_COUNTS", "HPRegistryError", "canonical_sha256", "lr_grid", "cells",
           "family_trials", "cell_multiset", "registry_obj", "write_registry", "load_hp_registry", "verify_file",
           "trial", "trial_record"]

if __name__ == "__main__":
    _code = str(_HERE.parent)
    if _code not in sys.path:
        sys.path.insert(0, _code)
    raise SystemExit(main())
