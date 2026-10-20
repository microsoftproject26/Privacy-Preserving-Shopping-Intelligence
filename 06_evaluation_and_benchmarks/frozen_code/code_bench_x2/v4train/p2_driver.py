"""v4train.p2_driver — the P2 search driver of one family (TRAIN-02 item 6; PREG1_AMENDMENT_02R1 §2-§3).

  python -B code/v4train/p2_driver.py --family GRU --stage 1 --gpu 0 --out-dir <P2 dir> [--dry-run]
  python -B code/v4train/p2_driver.py --family GRU --stage 2 --gpu 0 --out-dir <P2 dir> [--dry-run]
  (--rung is an alias of --stage; --history-policy defaults to the locked USER_HISTORY_L50)

Stage 1: the family's 12 registered trials of contracts/hp_registry/HP_REGISTRY_P2R2.json (02R2; the superseded
HP_REGISTRY_P2.json is refused), seed 2026, one after the other. After both stage-2 files: `register-02b --p2-dir <dir>`
writes contracts/hp_registry/TUNED_RECIPES_02B.json (v4train.tuned_recipes; TRAIN-02R1 F6).
Stage 2: the best trial (6.0 value) of EACH model-variant level, re-run at seed 2027 (GRU 2 runs, SASRec 3 runs), read
from P2_<FAMILY>_STAGE1.json (its self-hash verified). Every run goes through v4int/launch_central.py as its own process:
    python -B code/v4int/launch_central.py --family F --seed S --phase CALIBRATION --role TUNING --hp-id <id> --gpu G
           --run-dir <out>/<F>/stage<k>/<id>_s<S> --history-policy USER_HISTORY_L50 [--hp-registry ..] [--theta0-variants ..]
(the launcher's own preflight — G1 gate, GPU ownership, disk — and the trainer's refusals apply to every run). A run
directory that already holds a DONE_TUNING summary is not launched again; one with a RUN_START but no summary is
resumed (--resume). After the stage: P2_<FAMILY>_STAGE<k>.json (new file; an existing one is refused) with every run's
values (6.0 endpoint, CKPT_AVG, the 2.0 / 4.0 diagnostics, status, exit code) and the mechanical selection:
  * exclusions: a run with a missing / failed summary or a non-finite endpoint, or an endpoint < 0.5 x the family best
    (the best finite endpoint over all P2 runs of the family so far), is excluded;
  * stage 1 -> stage 2 plan: per model-variant level the valid trial with the highest endpoint (an exact tie: the lower
    LR, then the hp_id); a level with no valid run is a FABLE_TRIGGER (no stage 2);
  * stage 2: a stage-2 config is valid iff both its seed-2026 (stage 1) and seed-2027 runs are valid; tuned = the higher
    2-seed mean; a tie |delta| <= 0.0005 to the best mean -> fewer parameters (shared numel), then the lower LR, then
    the hp_id; a variant level without a valid 2-seed config is a FABLE_TRIGGER;
  * CKPT_AVG is carried to P4 iff its 2-seed mean beats the endpoint 2-seed mean of the same config by > 0.0005.
  (EPS = 1e-12 guards the decimal boundaries of 0.0005 against float representation: tie iff delta <= 0.0005 + EPS,
  carry iff gain > 0.0005 + EPS.)
The rows stay HP_SEARCH (refused by v4train.selection); nothing here is contest evidence. The pure functions below
(run_values, exclusions, stage1_plan, stage2_select, ckpt_avg_carry, choose_with_tie) are unit-tested with fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

# The environment BEFORE any v4train / snapshot import: importing v4train (ppsi_env -> snap_v1/reuse/train.py) sets
# CUDA_VISIBLE_DEVICES="0" by default in THIS process, which a child launcher would inherit and refuse (--gpu 1).
_ENV0 = dict(os.environ)
_CVD0 = os.environ.get("CUDA_VISIBLE_DEVICES")
if "v4train" in sys.modules:
    _launch = getattr(sys.modules["v4train"], "LAUNCH_CUDA_VISIBLE_DEVICES", None)
    if _launch != "UNKNOWN_PPSI_ENV_ALREADY_IMPORTED":
        _CVD0 = _launch
_HERE = Path(__file__).resolve().parent
CODE = _HERE.parent
V4 = CODE.parent
LAUNCHER = CODE / "v4int" / "launch_central.py"
TIE_ABS = 0.0005
CARRY_ABS = 0.0005
EXCLUDE_FRACTION = 0.5
EPS = 1e-12
SEEDS = {1: 2026, 2: 2027}
FORMAT = "PPSI_V4_P2_STAGE_1"


class P2Error(RuntimeError):
    pass


def _canon_sha(obj: Mapping, exclude=("self_sha256",)) -> str:
    body = {k: v for k, v in obj.items() if k not in exclude}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ------------------------------------------------------------------------------------------------ pure functions
def run_values(summary: Optional[Mapping], exit_code: Optional[int] = 0) -> dict:
    """The values of one TUNING run from its RUN_SUMMARY (None: no summary)."""
    if not summary or summary.get("status") != "DONE_TUNING":
        return {"status": (summary or {}).get("status", "NO_SUMMARY"), "endpoint": None, "ckpt_avg": None,
                "finite": False, "shared_numel": (summary or {}).get("shared_numel"), "exit_code": exit_code,
                "values_at_marks": (summary or {}).get("values_at_marks")}
    end = (summary.get("endpoint") or {}).get("value")
    ck = (summary.get("ckpt_avg") or {}).get("value")
    ck = ck if _finite(ck) else None               # F1: a CKPT_AVG_NON_FINITE step never excludes a valid endpoint
    return {"status": "DONE_TUNING", "endpoint": end, "ckpt_avg": ck, "finite": _finite(end),
            "ckpt_avg_outcome": (summary.get("ckpt_avg") or {}).get("outcome"),
            "shared_numel": summary.get("shared_numel"), "exit_code": exit_code,
            "values_at_marks": summary.get("values_at_marks"), "peak_lr": summary.get("peak_lr"),
            "model_variant": summary.get("model_variant"), "seed": summary.get("seed")}


def family_best(values: Sequence[Mapping]) -> Optional[float]:
    fin = [float(v["endpoint"]) for v in values if v.get("finite")]
    return max(fin) if fin else None


def exclusions(values: Mapping[str, Mapping], best: Optional[float]) -> dict:
    """{run key: reason or None}: non-finite / missing, or an endpoint < 0.5 x the family best."""
    out = {}
    for k, v in values.items():
        if not v.get("finite"):
            out[k] = f"EXCLUDED_NON_FINITE_OR_FAILED ({v.get('status')})"
        elif best is not None and float(v["endpoint"]) < EXCLUDE_FRACTION * best:
            out[k] = f"EXCLUDED_BELOW_HALF_FAMILY_BEST ({v['endpoint']} < 0.5 x {best})"
        else:
            out[k] = None
    return out


def stage1_plan(trials: Sequence[Mapping], results: Mapping[str, Mapping], variant_levels: Sequence[str]) -> dict:
    """Stage 1 -> stage 2: the best valid trial (endpoint) of each model-variant level."""
    best = family_best(list(results.values()))
    exc = exclusions(results, best)
    by_id = {t["hp_id"]: t for t in trials}
    plan, trig = {}, []
    for lvl in variant_levels:
        cand = [h for h in results if by_id[h]["model_variant"] == lvl and exc[h] is None]
        if not cand:
            plan[lvl] = None
            trig.append(f"model-variant level {lvl} has no valid 6.0 run (02R1 §3: Fable trigger)")
            continue
        plan[lvl] = sorted(cand, key=lambda h: (-float(results[h]["endpoint"]), float(by_id[h]["peak_lr"]), h))[0]
    return {"family_best": best, "excluded": exc, "stage2_plan": plan,
            "status": "FABLE_TRIGGER" if trig else "OK", "triggers": trig}


def choose_with_tie(cands: Sequence[Mapping]) -> dict:
    """cands: [{hp_id, mean, n_params, peak_lr}] -> the best mean; |delta| <= 0.0005 to it -> fewer parameters, then
    the lower LR, then the hp_id."""
    if not cands:
        raise P2Error("no valid candidate")
    top = max(float(c["mean"]) for c in cands)
    ties = [c for c in cands if top - float(c["mean"]) <= TIE_ABS + EPS]
    win = sorted(ties, key=lambda c: (int(c["n_params"]), float(c["peak_lr"]), c["hp_id"]))[0]
    return {"tuned": win["hp_id"], "best_mean": top, "tie_set": sorted(c["hp_id"] for c in ties),
            "tie_rule_applied": len(ties) > 1}


def ckpt_avg_carry(end_vals: Sequence, ck_vals: Sequence) -> dict:
    if len(end_vals) != 2 or len(ck_vals) != 2 or not all(_finite(x) for x in list(end_vals) + list(ck_vals)):
        return {"carry": False, "reason": "CKPT_AVG or endpoint value missing / non-finite", "gain": None}
    gain = (float(ck_vals[0]) + float(ck_vals[1])) / 2 - (float(end_vals[0]) + float(end_vals[1])) / 2
    return {"carry": gain > CARRY_ABS + EPS, "gain": gain, "endpoint_mean": (end_vals[0] + end_vals[1]) / 2,
            "ckpt_avg_mean": (ck_vals[0] + ck_vals[1]) / 2, "rule": "carry iff 2-seed CKPT_AVG mean - endpoint mean "
                                                                     "> 0.0005"}


def stage2_select(trials: Sequence[Mapping], plan: Mapping[str, Optional[str]], s1: Mapping[str, Mapping],
                  s2: Mapping[str, Mapping], variant_levels: Sequence[str]) -> dict:
    """s1: stage-1 values by hp_id (seed 2026); s2: stage-2 values by hp_id (seed 2027)."""
    by_id = {t["hp_id"]: t for t in trials}
    allv = {f"{h}_s2026": v for h, v in s1.items()}
    allv.update({f"{h}_s2027": v for h, v in s2.items()})
    best = family_best(list(allv.values()))
    exc = exclusions(allv, best)
    configs, cands, trig = {}, [], []
    for lvl in variant_levels:
        h = plan.get(lvl)
        if h is None:
            trig.append(f"model-variant level {lvl} has no stage-2 config")
            continue
        a, b = f"{h}_s2026", f"{h}_s2027"
        valid = a in exc and b in exc and exc[a] is None and exc[b] is None
        c = {"hp_id": h, "model_variant": lvl, "peak_lr": by_id[h]["peak_lr"], "valid": valid,
             "endpoint": {"2026": s1.get(h, {}).get("endpoint"), "2027": s2.get(h, {}).get("endpoint")},
             "ckpt_avg": {"2026": s1.get(h, {}).get("ckpt_avg"), "2027": s2.get(h, {}).get("ckpt_avg")},
             "n_params": s1.get(h, {}).get("shared_numel") or s2.get(h, {}).get("shared_numel"),
             "exclusion": {"2026": exc.get(a, "NO_RUN"), "2027": exc.get(b, "NO_RUN")}}
        if valid:
            c["mean"] = (float(c["endpoint"]["2026"]) + float(c["endpoint"]["2027"])) / 2
            c["ckpt_avg_rule"] = ckpt_avg_carry([c["endpoint"]["2026"], c["endpoint"]["2027"]],
                                                [c["ckpt_avg"]["2026"], c["ckpt_avg"]["2027"]])
            cands.append(c)
        else:
            trig.append(f"model-variant level {lvl} has no valid 2-seed 6.0 result (02R1 §3: Fable trigger)")
        configs[lvl] = c
    out = {"family_best": best, "excluded": exc, "configs": configs, "triggers": trig,
           "status": "FABLE_TRIGGER" if trig else "OK"}
    if cands:
        ch = choose_with_tie(cands)
        win = [c for c in cands if c["hp_id"] == ch["tuned"]][0]
        out.update(choice=ch, tuned={"hp_id": win["hp_id"], "model_variant": win["model_variant"],
                                     "peak_lr": win["peak_lr"], "schedule": by_id[win["hp_id"]]["schedule"],
                                     "dropout": by_id[win["hp_id"]]["dropout"], "mean_6.0": win["mean"]},
                   ckpt_avg_carried_to_p4=win["ckpt_avg_rule"]["carry"], ckpt_avg_rule=win["ckpt_avg_rule"])
    return out


# ------------------------------------------------------------------------------------------------ the driver
def launch_cmd(family: str, hp_id: str, seed: int, gpu: int, run_dir: Path, *, history_policy: str,
               hp_registry: Optional[Path] = None, theta0_variants: Optional[Path] = None, resume: bool = False,
               python: str = sys.executable) -> list:
    cmd = [python, "-B", str(LAUNCHER), "--family", family, "--seed", str(seed), "--phase", "CALIBRATION",
           "--role", "TUNING", "--hp-id", hp_id, "--gpu", str(int(gpu)), "--run-dir", str(run_dir),
           "--history-policy", history_policy]
    if hp_registry:
        cmd += ["--hp-registry", str(hp_registry)]
    if theta0_variants:
        cmd += ["--theta0-variants", str(theta0_variants)]
    if resume:
        cmd.append("--resume")
    return cmd


def child_env() -> dict:
    """The launcher's environment: this process's environment before any v4train import (so the snapshot's
    CUDA_VISIBLE_DEVICES default never leaks into the child), with PYTHONDONTWRITEBYTECODE=1."""
    env = dict(_ENV0, PYTHONDONTWRITEBYTECODE="1")
    if _CVD0 is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = _CVD0
    return env


def _read(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write_new(path: Path, obj: dict) -> str:
    from v4models.variants import write_exclusive
    obj = dict(obj)
    obj["self_sha256"] = _canon_sha(obj)
    return write_exclusive(Path(path), json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")


def load_stage(path: Path) -> dict:
    if not Path(path).is_file():
        raise P2Error(f"{path} does not exist (run the previous stage first)")
    obj = _read(path)
    if obj.get("format") != FORMAT or obj.get("self_sha256") != _canon_sha(obj):
        raise P2Error(f"{path}: not a verified {FORMAT} file (self-hash)")
    return obj


def run_stage(args, runner=subprocess.run, say=print) -> int:
    sys.path.insert(0, str(CODE))
    from v4train.hp_registry import REGISTRY_PATH, VARIANT_LEVELS, load_hp_registry
    fam, stage = args.family, int(args.stage)
    hp_path = Path(args.hp_registry) if args.hp_registry else REGISTRY_PATH
    reg = load_hp_registry(hp_path)
    trials = reg["families"][fam]
    levels = VARIANT_LEVELS[fam]
    out_dir = Path(args.out_dir)
    stage_file = out_dir / f"P2_{fam}_STAGE{stage}.json"
    if stage_file.exists():
        raise P2Error(f"{stage_file} exists: a finished stage is never re-run or rewritten")
    s1_obj = None
    if stage == 1:
        todo = [t["hp_id"] for t in trials]
    else:
        s1_obj = load_stage(out_dir / f"P2_{fam}_STAGE1.json")
        if s1_obj["selection"]["status"] != "OK":
            raise P2Error("stage 1 ended in a FABLE_TRIGGER: no stage 2")
        if s1_obj["hp_registry"]["self_sha256"] != reg["self_sha256"]:
            raise P2Error("the HP registry changed since stage 1")
        todo = [s1_obj["selection"]["stage2_plan"][lvl] for lvl in levels]
    seed = SEEDS[stage]
    runs, results = [], {}
    for h in todo:
        rd = out_dir / fam / f"stage{stage}" / f"{h}_s{seed}"
        done = (rd / "RUN_SUMMARY.json").exists() and _read(rd / "RUN_SUMMARY.json").get("status") == "DONE_TUNING"
        resume = (rd / "RUN_START.json").exists() and not done
        cmd = launch_cmd(fam, h, seed, args.gpu, rd, history_policy=args.history_policy,
                         hp_registry=args.hp_registry, theta0_variants=args.theta0_variants, resume=resume)
        rec = {"hp_id": h, "seed": seed, "run_dir": str(rd), "cmd": cmd, "already_done": done}
        if args.dry_run:
            say(" ".join(f'"{c}"' if " " in c else c for c in cmd))
            runs.append(rec)
            continue
        if not done:
            t0 = time.time()
            say(f"[p2] {fam} stage {stage}: {h} seed {seed} ({'resume' if resume else 'fresh'})")
            p = runner(cmd, cwd=str(CODE), env=child_env())
            rec.update(exit_code=int(p.returncode), wall_s=round(time.time() - t0, 1))
        else:
            rec["exit_code"] = 0
        summ = _read(rd / "RUN_SUMMARY.json") if (rd / "RUN_SUMMARY.json").exists() else None
        results[h] = run_values(summ, rec["exit_code"])
        runs.append(rec)
    if args.dry_run:
        return 0
    if stage == 1:
        sel = stage1_plan(trials, results, levels)
    else:
        s1 = {h: v for h, v in s1_obj["values"].items()}
        sel = stage2_select(trials, s1_obj["selection"]["stage2_plan"], s1, results, levels)
    obj = {"format": FORMAT, "task": "TRAIN-02 item 6 (P2 driver)", "family": fam, "stage": stage, "seed": seed,
           "hp_registry": {"path": str(hp_path), "file_sha256": reg["file_sha256"], "self_sha256": reg["self_sha256"]},
           "stage1_file": None if s1_obj is None else {"self_sha256": s1_obj["self_sha256"]},
           "runs": runs, "values": results, "selection": sel, "run_label": "HP_SEARCH (never selection evidence)",
           "rules": {"tie_abs": TIE_ABS, "carry_abs": CARRY_ABS, "exclude_fraction": EXCLUDE_FRACTION, "eps": EPS},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    sha = _write_new(stage_file, obj)
    say(f"[p2] {stage_file.name} written (sha256 {sha}); status {sel['status']}")
    return 0 if sel["status"] == "OK" else 3


def register_02b(p2_dir: Path, out: Optional[Path] = None, hp_registry: Optional[Path] = None,
                 families: Sequence[str] = ("GRU", "SASREC"), *, with_c2: bool = False,
                 c2_only: bool = False) -> dict:
    """TRAIN-02R1 F6 / TRAIN-02R2: register the tuned recipe of `families` (one or both) into TUNED_RECIPES_02B.json;
    each family's P2_<F>_STAGE1.json and P2_<F>_STAGE2.json are verified and the selection recomputed
    (v4train.tuned_recipes.verify_stages); a registered family is never replaced."""
    sys.path.insert(0, str(CODE))
    from v4train.hp_registry import REGISTRY_PATH as HP_PATH
    from v4train.tuned_recipes import REGISTRY_PATH, write_registry
    return write_registry(Path(p2_dir), families, Path(out) if out else REGISTRY_PATH,
                          Path(hp_registry) if hp_registry else HP_PATH, with_c2=with_c2, c2_only=c2_only)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("register-02b", "register-02b-c2"):
        command = argv[0]
        rp = argparse.ArgumentParser(prog=f"p2_driver.py {command}",
                                     description="write TUNED_RECIPES_02B.json from the two stage-2 files (O_EXCL)")
        rp.add_argument("--p2-dir", required=True, type=Path)
        rp.add_argument("--out", type=Path, default=None, help="default contracts/hp_registry/TUNED_RECIPES_02B.json")
        rp.add_argument("--hp-registry", type=Path, default=None)
        rp.add_argument("--with-c2", action="store_true", help="register C1 and C2 in one registry version")
        rp.add_argument("--family", nargs="+", type=str.upper, choices=("GRU", "SASREC"), default=["GRU", "SASREC"],
                        help="the family / families to register now (one at a time is allowed; never re-registered)")
        a = rp.parse_args(argv[1:])
        sys.path.insert(0, str(CODE))
        from v4train.tuned_recipes import TunedRecipeError
        try:
            print(json.dumps(register_02b(a.p2_dir, a.out, a.hp_registry, a.family,
                                          with_c2=a.with_c2, c2_only=command == "register-02b-c2"), indent=1))
            return 0
        except (P2Error, TunedRecipeError, FileExistsError) as e:
            print(f"REGISTER-02B REFUSED: {e}", file=sys.stderr, flush=True)
            return 2
    ap = argparse.ArgumentParser(description="P2 search driver (TRAIN-02 item 6)")
    ap.add_argument("--family", required=True, type=str.upper, choices=("GRU", "SASREC"))
    ap.add_argument("--stage", "--rung", dest="stage", required=True, type=int, choices=(1, 2))
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--history-policy", default="USER_HISTORY_L50")
    ap.add_argument("--hp-registry", type=Path, default=None)
    ap.add_argument("--theta0-variants", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the launcher command lines; run and write nothing")
    args = ap.parse_args(argv)
    try:
        return run_stage(args)
    except P2Error as e:
        print(f"P2 REFUSED: {e}", file=sys.stderr, flush=True)
        return 2


__all__ = ["run_values", "family_best", "exclusions", "stage1_plan", "choose_with_tie", "ckpt_avg_carry",
           "stage2_select", "launch_cmd", "load_stage", "run_stage", "P2Error", "TIE_ABS", "CARRY_ABS", "EPS"]

if __name__ == "__main__":
    raise SystemExit(main())
