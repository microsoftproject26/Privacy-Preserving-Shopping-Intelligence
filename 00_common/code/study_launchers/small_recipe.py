"""v4int.small_recipe — the SMALL (SASREC_D64_B2) mini-ladder and the SMALL recipe (INT-03 item 3; PREG1_AMENDMENT_03
§1/§3, PREG1_AMENDMENT_06 §1).

  python -B code/v4int/small_recipe.py write-ladder [--out contracts/hp_registry/HP_REGISTRY_SMALL_LADDER.json]
  python -B code/v4int/small_recipe.py stage --stage 1 --gpu G --out-dir <ladder dir> [--dry-run] [--allow-shared-gpu]
  python -B code/v4int/small_recipe.py stage --stage 2 --gpu G --out-dir <ladder dir> [--dry-run]
  python -B code/v4int/small_recipe.py register --out-dir <ladder dir> [--out contracts/hp_registry/SMALL_RECIPE_SASREC_D64_B2.json]

Design (registered values, no random draw):
  * the ladder = 4 TUNING trials of SASREC_D64_B2 at the FOUR SASRec LR midpoints of 02R2 §1, read from the verified
    HP_REGISTRY_P2R2.json (never recomputed here, never rewritten), schedule S0 and dropout [0.1] inherited from
    02B_SASREC (amendment 06 §1; cross-checked against TUNED_RECIPES_02B.json when it is present);
    ids P2S_SASREC_D64_B2_S0_s<k>; seed 2026 (stage 1) for all four, the best re-run at seed 2027 (stage 2);
    selection = the 02R1 §3 rule through v4train.p2_driver's own pure functions (stage1_plan / stage2_select, one size
    level); the SMALL recipe (id SMALL_SASREC_D64_B2) = the stage-2 tuned trial.
  * both files are NEW files (O_EXCL, self-hashed); HP_REGISTRY_P2R2.json and TUNED_RECIPES_02B.json are never touched.
Why an extension hook: the locked loaders (v4train.hp_registry.load_hp_registry / v4train.tuned_recipes) accept only
their own formats and designs, and v4train is not edited here. `install()` wraps them IN THE CALLING PROCESS ONLY: a
file whose "format" is one of the two SMALL formats is loaded and verified by this module (strict JSON, self-hash,
per-trial / per-recipe sha256, design equal to a fresh build from the verified P2R2 file); every other path is passed
to the original loader unchanged, so every existing launch is byte-for-byte unaffected. RunConfig.validate and the
trainer then apply their normal rules (peak LR / variant / dropout / schedule must equal the named trial or recipe).
A non-test SMALL-recipe run must use the production path RECIPE_PATH (the G1B pin covers 02B only; the SMALL recipe's
self-hash is recorded in every launch record instead — disclosed in the INT-03 handoff).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Optional

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

FAMILY = "SASREC"
SMALL_VARIANT = "SASREC_D64_B2"
HP_FORMAT = "PPSI_V4_HP_REGISTRY_SMALL_LADDER_1"
RECIPE_FORMAT = "PPSI_V4_SMALL_RECIPE_1"
STAGE_FORMAT = "PPSI_V4_SMALL_LADDER_STAGE_1"
HP_PATH = V4 / "contracts" / "hp_registry" / "HP_REGISTRY_SMALL_LADDER.json"
RECIPE_PATH = V4 / "contracts" / "hp_registry" / "SMALL_RECIPE_SASREC_D64_B2.json"
RECIPE_ID = "SMALL_SASREC_D64_B2"
INHERITED = {"recipe_id": "02B_SASREC", "schedule": "S0", "dropout": [0.1]}      # amendment 06 §1
SEEDS = {1: 2026, 2: 2027}
AUTHORITY = "contracts/PREG1_AMENDMENT_03.md §1/§3; contracts/PREG1_AMENDMENT_06.md §1; INT-03 item 3"


class SmallRecipeError(ValueError):
    pass


def canonical_sha256(obj: Mapping, exclude=("self_sha256",)) -> str:
    body = {k: v for k, v in obj.items() if k not in exclude}
    enc = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(enc.encode("utf-8")).hexdigest()


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(V4.resolve()).as_posix()
    except ValueError:
        return str(p)


def _strict(p: Path) -> tuple:
    p = Path(p)
    if not p.is_file():
        raise SmallRecipeError(f"{p} does not exist")
    data = p.read_bytes()
    try:
        obj = json.loads(data.decode("utf-8"),
                         parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except ValueError as e:
        raise SmallRecipeError(f"{p} is not strict JSON ({e})") from None
    if not isinstance(obj, dict):
        raise SmallRecipeError(f"{p} is not a JSON object")
    return obj, data


def file_format(path) -> Optional[str]:
    """The "format" of a JSON file, or None (missing / unreadable / not an object): used only to route a path."""
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return obj.get("format") if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ the ladder
def _p2r2(orig_load=None) -> dict:
    import v4train.hp_registry as H
    return (orig_load or _ORIG.get("load_hp_registry") or H.load_hp_registry)(H.REGISTRY_PATH)


def ladder_obj(p2r2: Optional[Mapping] = None) -> dict:
    """The registered ladder, built from the VERIFIED P2R2 registry's SASRec LR grid."""
    p2 = p2r2 or _p2r2()
    strata = p2["raw"]["lr_grid"][FAMILY]["strata"]
    if len(strata) != 4:
        raise SmallRecipeError(f"P2R2 SASRec grid has {len(strata)} strata, not 4 (02R2 §1)")
    trials = []
    for st in strata:
        k = int(st["stratum"])
        t = {"hp_id": f"P2S_{SMALL_VARIANT}_{INHERITED['schedule']}_s{k}", "family": FAMILY, "index": k + 1,
             "cell": {"model_variant": SMALL_VARIANT, "schedule": INHERITED["schedule"], "lr_stratum": k,
                      "dropout_p": INHERITED["dropout"][0]},
             "model_variant": SMALL_VARIANT, "schedule": INHERITED["schedule"], "dropout_p": INHERITED["dropout"][0],
             "dropout": list(INHERITED["dropout"]), "dropout_sites": ["all"], "lr_stratum": k,
             "log10_peak_lr": st["log10_mid"], "peak_lr": st["peak_lr"]}
        t["sha256"] = canonical_sha256(t, exclude=("sha256",))
        trials.append(t)
    obj = {"format": HP_FORMAT, "task": "INT-03 item 3 (SMALL mini-ladder)", "authority": AUTHORITY,
           "families": {FAMILY: trials},
           "source_hp_registry": {"path": _rel(Path(p2["path"])), "self_sha256": p2["self_sha256"],
                                  "lr_grid_sha256": p2["lr_grid_sha256"]},
           "inherited": dict(INHERITED), "seeds": {"stage1": SEEDS[1], "stage2": SEEDS[2]},
           "variant_levels": [SMALL_VARIANT],
           "fixed": {"effective_batch": 256, "efe": 6.0, "eval_marks": [2.0, 4.0, 6.0],
                     "train_manifest_role": "CALIBRATION", "eval_manifest_role": "CALIBRATION (MAIN never read)",
                     "run_role": "TUNING", "run_label": "HP_SEARCH", "determinism_mode": "STRICT"},
           "selection": "02R1 §3 via v4train.p2_driver.stage1_plan / stage2_select (one size level); recipe = tuned"}
    obj["self_sha256"] = canonical_sha256(obj)
    return obj


def load_ladder(path=HP_PATH) -> dict:
    """Verified ladder in the shape of v4train.hp_registry.load_hp_registry's return value."""
    obj, data = _strict(Path(path))
    if obj.get("format") != HP_FORMAT:
        raise SmallRecipeError(f"{path} is not a {HP_FORMAT} object")
    if obj.get("self_sha256") != canonical_sha256(obj):
        raise SmallRecipeError(f"{path}: self-hash mismatch")
    for t in obj.get("families", {}).get(FAMILY, []):
        if t.get("sha256") != canonical_sha256(t, exclude=("sha256",)):
            raise SmallRecipeError(f"{path}: trial {t.get('hp_id')} sha256 mismatch")
    fresh = ladder_obj()
    bad = [k for k in fresh if k != "self_sha256" and obj.get(k) != fresh[k]]
    if bad:
        raise SmallRecipeError(f"{path}: sections {bad} differ from the registered ladder (built from P2R2)")
    return {"path": str(path), "file_sha256": hashlib.sha256(data).hexdigest(), "self_sha256": obj["self_sha256"],
            "families": obj["families"], "lr_grid_sha256": obj["source_hp_registry"]["lr_grid_sha256"], "raw": obj}


def _check_inherited() -> Optional[dict]:
    """Cross-check S0 / [0.1] against 02B_SASREC when TUNED_RECIPES_02B.json is present (self-hash only)."""
    import v4train.tuned_recipes as TRM
    if not Path(TRM.REGISTRY_PATH).is_file():
        return None
    reg = (_ORIG.get("load_tuned_recipes") or TRM.load_tuned_recipes)(TRM.REGISTRY_PATH, reverify_stages=False)
    r = reg["recipes"].get(INHERITED["recipe_id"])
    if r is None or r["schedule"] != INHERITED["schedule"] or list(r["dropout"]) != INHERITED["dropout"]:
        raise SmallRecipeError(f"02B_SASREC does not carry schedule {INHERITED['schedule']} / dropout "
                               f"{INHERITED['dropout']} (amendment 06 §1)")
    return {"self_sha256": reg["self_sha256"], "recipe_sha256": r["sha256"]}


def write_ladder(out=HP_PATH) -> dict:
    from v4models.variants import write_exclusive
    cross = _check_inherited()
    obj = ladder_obj()
    sha = write_exclusive(Path(out), json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return {"path": str(out), "file_sha256": sha, "self_sha256": obj["self_sha256"], "inherited_check": cross}


# ------------------------------------------------------------------------------------------------ the SMALL recipe
def stage_dir(out_dir: Path, stage: int, hp_id: str) -> Path:
    return Path(out_dir) / FAMILY / f"stage{stage}" / f"{hp_id}_s{SEEDS[stage]}"


def recipe_obj(ladder: Mapping, stage1: Mapping, stage2: Mapping) -> dict:
    """The SMALL recipe from the verified stage files (the tuned trial of stage 2)."""
    for s in (stage1, stage2):
        if s.get("format") != STAGE_FORMAT or s.get("self_sha256") != canonical_sha256(s):
            raise SmallRecipeError("stage file not verified (format / self-hash)")
        if s["hp_registry"]["self_sha256"] != ladder["self_sha256"]:
            raise SmallRecipeError("a stage ran under another ladder file")
    sel = stage2["selection"]
    if sel.get("status") != "OK" or "tuned" not in sel:
        raise SmallRecipeError(f"stage 2 selection status {sel.get('status')}: no SMALL recipe (Fable trigger)")
    import v4train.hp_registry as H
    t = H.trial(ladder, sel["tuned"]["hp_id"])
    r = {"recipe_id": RECIPE_ID, "family": FAMILY, "model_variant": t["model_variant"], "peak_lr": t["peak_lr"],
         "dropout": list(t["dropout"]), "schedule": t["schedule"], "source_hp_id": t["hp_id"],
         "source_hp_sha256": t["sha256"], "mean_6.0": sel["tuned"]["mean_6.0"],
         "stage1_self_sha256": stage1["self_sha256"], "stage2_self_sha256": stage2["self_sha256"]}
    r["sha256"] = canonical_sha256(r, exclude=("sha256",))
    obj = {"format": RECIPE_FORMAT, "task": "INT-03 item 3 (SMALL recipe)", "authority": AUTHORITY,
           "recipes": {RECIPE_ID: r},
           "ladder_registry": {"path": _rel(Path(ladder["path"])), "self_sha256": ladder["self_sha256"],
                               "file_sha256": ladder["file_sha256"]},
           "note": "C_SMALL (launch_central --phase MAIN --recipe-id SMALL_SASREC_D64_B2) and every FL / device arm "
                   "use this one recipe (amendment 03 §2)"}
    obj["self_sha256"] = canonical_sha256(obj)
    return obj


def load_small_recipes(path=RECIPE_PATH, hp_path=None, *, reverify_stages: bool = True) -> dict:
    """Verified SMALL recipe file in the shape of v4train.tuned_recipes.load_tuned_recipes's return value."""
    obj, data = _strict(Path(path))
    if obj.get("format") != RECIPE_FORMAT or obj.get("self_sha256") != canonical_sha256(obj):
        raise SmallRecipeError(f"{path} is not a verified {RECIPE_FORMAT} object")
    lad = obj.get("ladder_registry") or {}
    lp = V4 / lad.get("path", "") if not Path(lad.get("path", "")).is_absolute() else Path(lad["path"])
    ladder = load_ladder(lp)
    if ladder["self_sha256"] != lad.get("self_sha256"):
        raise SmallRecipeError(f"{path}: registered under another ladder file")
    import v4train.hp_registry as H
    for rid, r in (obj.get("recipes") or {}).items():
        if rid != RECIPE_ID or r.get("recipe_id") != rid or r.get("sha256") != canonical_sha256(r, exclude=("sha256",)):
            raise SmallRecipeError(f"{path}: recipe {rid} id / sha256 mismatch")
        t = H.trial(ladder, r["source_hp_id"])
        if t["sha256"] != r["source_hp_sha256"] or any(t[k] != r[k] for k in ("peak_lr", "model_variant",
                                                                                 "dropout", "schedule")):
            raise SmallRecipeError(f"{path}: recipe {rid} differs from its ladder trial {r['source_hp_id']}")
    return {"path": str(path), "file_sha256": hashlib.sha256(data).hexdigest(), "self_sha256": obj["self_sha256"],
            "recipes": obj["recipes"], "version": 1, "no_c2": []}


def check_small_production(path, reg: Mapping, gate_path=None) -> dict:
    if Path(path).resolve() != RECIPE_PATH.resolve():
        raise SmallRecipeError(f"a non-test SMALL run must use the production SMALL recipe {RECIPE_PATH}, not {path}")
    return {"gate": None, "pinned": reg["self_sha256"], "note": "SMALL recipe: production path + self-hash recorded "
                                                                  "(no G1B pin covers it; INT-03 disclosure)"}


# ------------------------------------------------------------------------------------------------ the hook
_ORIG: dict = {}


def install() -> dict:
    """Route SMALL-format files through this module inside the locked loaders (this process only; idempotent)."""
    import v4train.hp_registry as H
    import v4train.tuned_recipes as TRM
    if _ORIG:
        return {"installed": True, "formats": [HP_FORMAT, RECIPE_FORMAT]}
    _ORIG.update(load_hp_registry=H.load_hp_registry, tr_load_hp=TRM.load_hp_registry,
                 load_tuned_recipes=TRM.load_tuned_recipes, check_production_registry=TRM.check_production_registry)

    def load_hp_registry(path=H.REGISTRY_PATH):
        if file_format(path) == HP_FORMAT:
            try:
                return load_ladder(path)
            except SmallRecipeError as e:
                raise H.HPRegistryError(str(e)) from None
        return _ORIG["load_hp_registry"](path)

    def load_tuned_recipes(path=TRM.REGISTRY_PATH, hp_path=TRM.HP_REGISTRY_PATH, *, reverify_stages: bool = True):
        if file_format(path) == RECIPE_FORMAT:
            try:
                return load_small_recipes(path, hp_path, reverify_stages=reverify_stages)
            except (SmallRecipeError, H.HPRegistryError) as e:
                raise TRM.TunedRecipeError(str(e)) from None
        return _ORIG["load_tuned_recipes"](path, hp_path, reverify_stages=reverify_stages)

    def check_production_registry(path, reg, gate_path=None):
        if file_format(path) == RECIPE_FORMAT:
            try:
                return check_small_production(path, reg, gate_path)
            except SmallRecipeError as e:
                raise TRM.TunedRecipeError(str(e)) from None
        return _ORIG["check_production_registry"](path, reg, gate_path)

    H.load_hp_registry = load_hp_registry
    TRM.load_hp_registry = load_hp_registry
    TRM.load_tuned_recipes = load_tuned_recipes
    TRM.check_production_registry = check_production_registry
    return {"installed": True, "formats": [HP_FORMAT, RECIPE_FORMAT]}


def uninstall() -> None:
    """Tests only: restore the original loaders."""
    if not _ORIG:
        return
    import v4train.hp_registry as H
    import v4train.tuned_recipes as TRM
    H.load_hp_registry = _ORIG["load_hp_registry"]
    TRM.load_hp_registry = _ORIG["tr_load_hp"]
    TRM.load_tuned_recipes = _ORIG["load_tuned_recipes"]
    TRM.check_production_registry = _ORIG["check_production_registry"]
    _ORIG.clear()


# ------------------------------------------------------------------------------------------------ the ladder driver
def run_stage(args, runner=None, say=print) -> int:
    """Stage 1: the 4 trials at seed 2026; stage 2: the stage-1 best at seed 2027 — each through launch_central
    (--role TUNING --hp-id <id> --hp-registry <ladder>) as its own process; then the stage file (new, self-hashed)."""
    import subprocess
    from v4train import p2_driver as P2
    install()
    runner = runner or subprocess.run
    ladder_path = Path(args.hp_registry) if args.hp_registry else HP_PATH
    ladder = load_ladder(ladder_path)
    trials = ladder["families"][FAMILY]
    stage, out_dir = int(args.stage), Path(args.out_dir)
    stage_file = out_dir / f"SMALL_LADDER_STAGE{stage}.json"
    if stage_file.exists():
        raise SmallRecipeError(f"{stage_file} exists: a finished stage is never re-run or rewritten")
    s1 = None
    if stage == 1:
        todo = [t["hp_id"] for t in trials]
    else:
        s1 = load_stage(out_dir / "SMALL_LADDER_STAGE1.json")
        if s1["selection"]["status"] != "OK":
            raise SmallRecipeError("stage 1 ended in a FABLE_TRIGGER: no stage 2")
        todo = [s1["selection"]["stage2_plan"][SMALL_VARIANT]]
    seed = SEEDS[stage]
    runs, results = [], {}
    for h in todo:
        rd = stage_dir(out_dir, stage, h)
        summ_p = rd / "RUN_SUMMARY.json"
        done = summ_p.exists() and json.loads(summ_p.read_text(encoding="utf-8")).get("status") == "DONE_TUNING"
        resume = (rd / "RUN_START.json").exists() and not done
        cmd = P2.launch_cmd(FAMILY, h, seed, args.gpu, rd, history_policy=args.history_policy,
                            hp_registry=ladder_path, theta0_variants=args.theta0_variants, resume=resume)
        if getattr(args, "allow_shared_gpu", False):
            cmd.append("--allow-shared-gpu")
        rec = {"hp_id": h, "seed": seed, "run_dir": str(rd), "cmd": cmd, "already_done": done}
        runs.append(rec)
        if args.dry_run:
            say(" ".join(f'"{c}"' if " " in c else c for c in cmd))
            continue
        if not done:
            t0 = time.time()
            p = runner(cmd, cwd=str(CODE), env=P2.child_env())
            rec.update(exit_code=int(p.returncode), wall_s=round(time.time() - t0, 1))
        else:
            rec["exit_code"] = 0
        summ = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else None
        results[h] = P2.run_values(summ, rec["exit_code"])
    if args.dry_run:
        return 0
    if stage == 1:
        sel = P2.stage1_plan(trials, results, [SMALL_VARIANT])
    else:
        sel = P2.stage2_select(trials, s1["selection"]["stage2_plan"], dict(s1["values"]), results, [SMALL_VARIANT])
    obj = {"format": STAGE_FORMAT, "stage": stage, "seed": seed, "family": FAMILY,
           "hp_registry": {"path": str(ladder_path), "file_sha256": ladder["file_sha256"],
                           "self_sha256": ladder["self_sha256"]},
           "stage1_self_sha256": None if s1 is None else s1["self_sha256"], "runs": runs, "values": results,
           "selection": sel, "run_label": "HP_SEARCH (never selection evidence)",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_new(stage_file, obj)
    say(f"[small] {stage_file.name} written; status {sel['status']}")
    return 0 if sel["status"] == "OK" else 3


def write_new(path: Path, obj: dict) -> str:
    from v4models.variants import write_exclusive
    obj = dict(obj)
    obj["self_sha256"] = canonical_sha256(obj)
    return write_exclusive(Path(path), json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")


def load_stage(path: Path) -> dict:
    obj, _ = _strict(Path(path))
    if obj.get("format") != STAGE_FORMAT or obj.get("self_sha256") != canonical_sha256(obj):
        raise SmallRecipeError(f"{path}: not a verified {STAGE_FORMAT} file")
    return obj


def register(out_dir: Path, out=RECIPE_PATH, hp_registry=None) -> dict:
    from v4models.variants import write_exclusive
    ladder = load_ladder(Path(hp_registry) if hp_registry else HP_PATH)
    obj = recipe_obj(ladder, load_stage(Path(out_dir) / "SMALL_LADDER_STAGE1.json"),
                     load_stage(Path(out_dir) / "SMALL_LADDER_STAGE2.json"))
    sha = write_exclusive(Path(out), json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return {"path": str(out), "file_sha256": sha, "self_sha256": obj["self_sha256"],
            "recipe": copy.deepcopy(obj["recipes"][RECIPE_ID])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SMALL mini-ladder / SMALL recipe (INT-03 item 3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-ladder")
    w.add_argument("--out", type=Path, default=HP_PATH)
    s = sub.add_parser("stage")
    s.add_argument("--stage", type=int, choices=(1, 2), required=True)
    s.add_argument("--gpu", type=int, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--hp-registry", type=Path, default=None)
    s.add_argument("--theta0-variants", type=Path, default=None)
    s.add_argument("--history-policy", default="USER_HISTORY_L50")
    s.add_argument("--allow-shared-gpu", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("register")
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--out", type=Path, default=RECIPE_PATH)
    r.add_argument("--hp-registry", type=Path, default=None)
    a = ap.parse_args(argv)
    import v4train  # noqa: F401  (ppsi_env first; the stage children get the pre-import environment)
    install()
    if a.cmd == "write-ladder":
        print(json.dumps(write_ladder(a.out), indent=1))
        return 0
    if a.cmd == "stage":
        return run_stage(a)
    print(json.dumps(register(a.out_dir, a.out, a.hp_registry), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
