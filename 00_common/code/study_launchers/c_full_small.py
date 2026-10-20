"""v4int.c_full_small (P2-05, owner: Phase-2 controls) - control C2 of contracts/PREG2.md §4.3
(P2_C2_CFULL_SMALL; optional, owner decision O3; Fable recommends YES, reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md
§9.3): the matched-size central reference = the SMALL recipe exactly as PRE (SMALL_SASREC_D64_B2: SASREC_D64_B2, its
registered theta0 of seed 2026, 6.0 EFE, automatic C3 continuation as PRE had) on FULL_TRAIN_5_100_TIEBREAK (PRE users
+ MAIN users) -> MAIN_VALIDATION_5_100_131072. A parameterised copy of v4int.c_full (DATA-08), which is NOT edited;
launch_central / v4train are not edited either.

  python -B code/v4int/c_full_small.py [--ext-registry <path>] -- --family SASREC --seed 2026 --phase MAIN \
         --recipe-id SMALL_SASREC_D64_B2 --gpu <id> --run-dir <dir> --history-policy USER_HISTORY_L50 [launcher args]
  python -B code/v4int/c_full_small.py --dry-run-out <file.json> [--no-providers] -- <the same launcher args>

Differences from c_full (everything else - the extension registry REGISTRY_V2_FULL_EXT.json, wrappers 1-3, the dry run
- is the c_full code with the C2 parameters):
  * REQUIRED recipe SMALL_SASREC_D64_B2 (not 02B_SASREC), seed 2026; --tuned-recipes defaults to the production SMALL
    recipe (v4int.small_recipe.RECIPE_PATH), any other path is refused; --smoke is refused (S8: it would evaluate
    MAIN_VALIDATION outside sealed/); --peak-lr is refused (the
    recipe's peak);
  * run id prefix P2_C2_CFULL_SMALL_ (default P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026);
  * labels (wrapper 4, shared with v4int.central_ft, installed by wrapper 1 after the CUDA export): run_label
    P2_C2_CFULL_SMALL instead of PRODUCTION, rows method P2_C2_CFULL_SMALL, row_status / comparison_status P2_CONTROL,
    c_matched_eligible = downstream_seed_allowed = False - C2 is a Phase-2 reference only; K20b's denominator stays
    C_FULL (no Phase-1 contest row can use C2);
  * --allow-shared-gpu only together with --coschedule-06g (06G §5 / 06K §1, as v4int.pretrain_small).
RunConfig: no new field and no changed rule (run_label is a derived property: the config hash is unchanged by the label
wrapper); population COHORT, role MAIN, sealed outputs and the automatic C3 continuation of a contest-shaped run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

TASK = "P2-05 v4int.c_full_small"
AUTHORITY = "contracts/PREG2.md §4.3; company/plans/PHASE2_IMPLEMENTATION_PLAN.md P2-05"
DEFAULT_EXT = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2_FULL_EXT.json"
RECIPE_ID = "SMALL_SASREC_D64_B2"
MODEL_VARIANT = "SASREC_D64_B2"
REQUIRED = {"family": "SASREC", "phase": "MAIN", "recipe_id": RECIPE_ID, "population": "COHORT",
            "history_policy": "USER_HISTORY_L50", "seed": 2026}
RUN_ID_PREFIX = "P2_C2_CFULL_SMALL_"
RUN_LABEL = "P2_C2_CFULL_SMALL"
RECORD_PREFIX = "P2_C2_SHIM_"
TRAIN_ID, EVAL_ID = "FULL_TRAIN_5_100_TIEBREAK", "MAIN_VALIDATION_5_100_131072"   # = v4data08.d8_common (checked)


class CFullSmallRefused(RuntimeError):
    pass


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _strict(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"),
                      parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))


def split_argv(argv: list) -> tuple:
    if "--" not in argv:
        raise CFullSmallRefused("usage: c_full_small.py [shim options] -- <launch_central arguments>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description="P2-05 C2 (C_FULL at the SMALL recipe) launcher shim")
    ap.add_argument("--ext-registry", type=Path, default=DEFAULT_EXT)
    ap.add_argument("--dry-run-out", type=Path, default=None)
    ap.add_argument("--no-providers", action="store_true", help="dry run: verify registries / manifests only")
    return ap.parse_args(argv[:i]), list(argv[i + 1:])


def recipe_path() -> Path:
    from v4int.small_recipe import RECIPE_PATH                # stdlib-only at import
    return Path(RECIPE_PATH)


def check_launch_args(args) -> str:
    """The C2 launch contract on the parsed launch_central arguments; returns the run id to use."""
    if getattr(args, "smoke", False):                     # REV-P2-05 m1/m2: checked before any data load
        raise CFullSmallRefused("--smoke is refused for C2: its evaluation manifest is MAIN_VALIDATION, and a smoke "
                                "would write a MAIN_VALIDATION number into SMOKE_EVAL.json outside sealed/ "
                                "(PREG2 §6.3 S8); the code path is c_full's (exercised by C_FULL / PRE)")
    for k, want in REQUIRED.items():
        if getattr(args, k, None) != want:
            raise CFullSmallRefused(f"C2 needs --{k.replace('_', '-')} {want} (got {getattr(args, k, None)!r})")
    for k in ("train_manifest", "eval_manifest", "role", "hp_id", "calibration_record", "continuation_budget",
              "peak_lr"):
        if getattr(args, k, None):
            raise CFullSmallRefused(f"C2 refuses --{k.replace('_', '-')}: the manifests come from the extension's "
                                    f"C_FULL role and the peak LR from recipe {RECIPE_ID}")
    for k in ("transform_state", "manifest_dir", "manifest_hashes", "header_list", "lock_path", "theta0_variants",
              "hp_registry"):
        if getattr(args, k, None) is not None:
            raise CFullSmallRefused(f"C2 refuses --{k.replace('_', '-')}: the locked tree, lock, transform state, "
                                    f"theta0 registry and recipe {RECIPE_ID} are used unchanged")
    if getattr(args, "allow_shared_gpu", False) and not getattr(args, "coschedule_06g", False):
        raise CFullSmallRefused("C2 refuses --allow-shared-gpu without --coschedule-06g (06G §5 / 06K §1)")
    if getattr(args, "coschedule_06g", False) and not getattr(args, "allow_shared_gpu", False):
        raise CFullSmallRefused("--coschedule-06g takes effect only together with --allow-shared-gpu (DECISION_LOG 208)")
    tr = getattr(args, "tuned_recipes", None)
    if tr is not None and Path(tr).resolve() != recipe_path().resolve() and not args.smoke:
        raise CFullSmallRefused(f"C2 uses the production SMALL recipe {recipe_path()} (got {tr})")
    from v4int.smoke import SMOKE_RUN_PREFIX
    default = f"{SMOKE_RUN_PREFIX if args.smoke else ''}{RUN_ID_PREFIX}{MODEL_VARIANT}_s{int(args.seed)}"
    rid = args.run_id or default
    core = rid[len(SMOKE_RUN_PREFIX):] if rid.startswith(SMOKE_RUN_PREFIX) else rid
    if not core.startswith(RUN_ID_PREFIX):
        raise CFullSmallRefused(f"a C2 run id must start with {RUN_ID_PREFIX!r} (got {rid!r})")
    return rid


def normalise_argv(largv: list, args, rid: str) -> list:
    """Adds --run-id (the checked default) and --tuned-recipes (the production SMALL recipe) when absent."""
    out = list(largv)
    if "--run-id" not in out:
        out += ["--run-id", rid]
        args.run_id = rid
    if getattr(args, "tuned_recipes", None) is None:
        out += ["--tuned-recipes", str(recipe_path())]
        args.tuned_recipes = recipe_path()
    return out


# ------------------------------------------------------------------------------------------------ the wrappers
def install(LC, ext_path: Path, record: dict, *, labels=None) -> None:
    """Wrappers 1 and 2 on the launch_central module object (wrapper 3 and the labels are installed by wrapper 1, after
    the CUDA export, because v4int.verify and the label targets import v4train)."""
    if labels is None:
        from v4int.central_ft import install_p2_label_wrappers as labels
    ext_path = Path(ext_path).resolve()
    orig_bind, orig_view = LC.bind_data_protocol_lock_v2, LC.build_trainer_registry_view

    def bind(paths, *, smoke, gate_path=LC.G1_GATE):
        ctx = orig_bind(paths, smoke=smoke, gate_path=gate_path)
        from v4data08 import d8_common as D8
        from v4data08 import d8_registry as R
        if (D8.TRAIN_ID, D8.EVAL_ID) != (TRAIN_ID, EVAL_ID):
            raise LC.LaunchRefused("c_full_small ids differ from v4data08.d8_common")
        base_path = Path(ctx["manifest_registry_path"])
        if not ext_path.is_file():
            raise LC.LaunchRefused(f"C2 extension registry {ext_path} does not exist (run the DATA-08 jobs)")
        ext, base = _strict(ext_path), _strict(base_path)
        chk = R.verify_ext_registry(ext, base, ctx["manifest_registry_file_sha256"])
        if not chk["all_ok"] or (ext.get("extends") or {}).get("path") != ctx["manifest_registry_rel_path"]:
            raise LC.LaunchRefused(f"C2 extension registry refused: {[k for k, v in chk.items() if not v]}")
        install_verify_wrapper(ext_path, base_path)
        record["labels"] = labels()
        try:
            ext_rel = str(ext_path.relative_to(Path(LC.V4).resolve())).replace("\\", "/")
        except ValueError:
            ext_rel = str(ext_path)
        record.update(ext_registry={"path": ext_rel, "file_sha256": _sha(ext_path), "self_sha256": ext["self_sha256"]},
                      base_registry={"path": ctx["manifest_registry_rel_path"],
                                     "file_sha256": ctx["manifest_registry_file_sha256"],
                                     "self_sha256": ctx["manifest_registry_self_sha256"]},
                      ext_checks=chk)
        return dict(ctx, manifest_registry_path=ext_path, manifest_registry_rel_path=ext_rel,
                    manifest_registry_file_sha256=_sha(ext_path), manifest_registry_self_sha256=ext["self_sha256"],
                    c_full_base_registry=record["base_registry"])

    def view(manifest_registry_path, run_dir, *, registry_rel_path, registry_file_sha256):
        from v4data08 import d8_registry as R
        if Path(manifest_registry_path).resolve() != ext_path:
            raise LC.LaunchRefused("the C2 trainer view is built from the extension registry only")
        ext = _strict(ext_path)
        obj = R.trainer_view(ext, ext_rel_path=registry_rel_path, ext_file_sha256=registry_file_sha256)
        chk = R.check_trainer_view(obj, ext)
        if not chk["all_ok"]:
            raise LC.LaunchRefused(f"C2 trainer view refused: {chk}")
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        p = run_dir / R.VIEW_FILE
        p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        record["trainer_view"] = {"path": str(p), "self_sha256": obj["self_sha256"], "roles": sorted(obj["roles"]),
                                  "checks": chk}
        if not record.get("dry_run"):
            record["_record_path"] = str(_write_record(run_dir, record))
        return {"path": p, "self_sha256": obj["self_sha256"], "roles": sorted(obj["roles"]),
                "source_registry_self_sha256": ext["self_sha256"], "source_registry_file_sha256": registry_file_sha256}

    bind.__wrapped__, view.__wrapped__ = orig_bind, orig_view
    LC.bind_data_protocol_lock_v2 = bind
    LC.build_trainer_registry_view = view


def install_verify_wrapper(ext_path: Path, base_path: Path) -> None:
    """Wrapper 3 = v4int.c_full's (the same extension file, the same rule; c_full's factory is reused unchanged)."""
    import v4int.verify as VF
    from v4int.c_full import make_verify_wrapper
    if getattr(VF.verify_registry, "__c_full__", False) or getattr(VF.verify_registry, "__c_full_small__", False):
        return
    w = make_verify_wrapper(VF.verify_registry, ext_path, base_path, error=VF.ManifestVerificationError)
    w.__c_full_small__ = True
    VF.verify_registry = w


def _write_record(run_dir: Path, record: dict, path=None) -> Path:
    """P2_C2_SHIM_<k>.json in the run dir (written when the view is built, rewritten with the exit code)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if path is None:
        k = 0
        while (run_dir / f"{RECORD_PREFIX}{k}.json").exists():
            k += 1
        path = run_dir / f"{RECORD_PREFIX}{k}.json"
    p = Path(path)
    p.write_text(json.dumps({k: v for k, v in record.items() if not k.startswith("_")}, indent=1, sort_keys=True,
                            default=str) + "\n", encoding="utf-8", newline="\n")
    return p


# ------------------------------------------------------------------------------------------------ dry run
def check_plan(want_manifests, tuned: dict) -> dict:
    """The C2 bindings a dry run proves: the trainer's MAIN role = (TIEBREAK, MAIN_VALIDATION) and the SMALL recipe's
    D64 variant."""
    if tuple(want_manifests) != (TRAIN_ID, EVAL_ID):
        raise CFullSmallRefused(f"the trainer view resolves MAIN to {tuple(want_manifests)}, not {(TRAIN_ID, EVAL_ID)}")
    if tuned.get("recipe_id") != RECIPE_ID or tuned.get("model_variant") != MODEL_VARIANT:
        raise CFullSmallRefused(f"C2 runs {RECIPE_ID} / {MODEL_VARIANT}, not {tuned.get('recipe_id')} / "
                                f"{tuned.get('model_variant')}")
    return {"train_manifest": TRAIN_ID, "eval_manifest": EVAL_ID, "recipe_id": RECIPE_ID,
            "model_variant": MODEL_VARIANT, "peak_lr": tuned.get("peak_lr")}


def _verify_manifests(LC, lockctx, paths) -> dict:
    from v4int.verify import verify_manifest_by_registry
    out = {}
    for mid, kind in ((TRAIN_ID, "TRAIN"), (EVAL_ID, "EVAL")):
        vm = verify_manifest_by_registry(mid, lockctx["manifest_registry_path"], paths, root=LC.V4, expect_kind=kind)
        out[mid] = {"n_checks": len(vm.checks), "n_decisions": int(vm.canonical_decision_id.size),
                    "registry": vm.registry}
        vm.columns = None
    return out


def dry_run(LC, args, shim, record: dict, *, small_install=True) -> dict:
    """launch_central.run up to the RunConfig (no trainer, no GPU work, nothing written under --run-dir): the SMALL
    recipe, the view in a temporary directory, the providers (unless --no-providers), the comparison key, the RunConfig,
    validate() (run_label P2_C2_CFULL_SMALL), and the trainer's own registry checks."""
    import tempfile
    record["environment"] = LC.export_environment(int(args.gpu))       # as the launcher, before v4train is imported
    import v4train  # noqa: F401
    if small_install:
        from v4int.small_recipe import install as _small_install
        _small_install()                                             # as launch_central.run (INT-03)
    from v4train.provider import expected_manifests
    from v4train.registry import load_manifest_registry
    run_dir = Path(args.run_dir)
    before = sorted(str(p) for p in run_dir.rglob("*")) if run_dir.exists() else None
    t0 = time.time()
    tuned = LC._recipe_values(args, args.family)
    paths = LC.data_paths(args)
    lockctx = LC.bind_data_protocol_lock_v2(paths, smoke=bool(args.smoke))
    out = {"note": "C2 dry run: no training, nothing written under run_dir", "utc": LC.utc(), "shim": record,
           "tuned_recipe": tuned}
    with tempfile.TemporaryDirectory(prefix="c_full_small_dryrun_") as td:
        view = LC.build_trainer_registry_view(lockctx["manifest_registry_path"], Path(td),
                                              registry_rel_path=lockctx["manifest_registry_rel_path"],
                                              registry_file_sha256=lockctx["manifest_registry_file_sha256"])
        vreg = load_manifest_registry(view["path"])
        want = expected_manifests("COHORT", "MAIN", registry=vreg)
        out["trainer_view"] = {"self_sha256": view["self_sha256"], "roles": view["roles"],
                               "expected_manifests": list(want)}
        out["plan"] = check_plan(want, tuned)
        if shim.no_providers:
            out["verified"] = _verify_manifests(LC, lockctx, paths)
            out["config_sha256"] = None
            out["config_note"] = "--no-providers: the comparison key (and so the config hash) needs the providers"
        else:
            from v4int.cache import peak_rss_bytes, rss_bytes
            from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
            from v4train.recipe import RunConfig
            from v4train.registry import check_provider
            tp = StoreTrainProvider(TRAIN_ID, history_policy=args.history_policy, paths=paths, log=print,
                                    manifest_registry=lockctx["manifest_registry_path"], registry_root=LC.V4)
            ep = StoreEvalProvider(EVAL_ID, history_policy=args.history_policy, paths=paths,
                                   prefetch=int(args.eval_prefetch), log=print,
                                   manifest_registry=lockctx["manifest_registry_path"], registry_root=LC.V4)
            out["bindings"] = {"train": check_provider(vreg, "MAIN", "train", tp),
                               "eval": check_provider(vreg, "MAIN", "eval", ep)}
            key = comparison_key_base(paths, tp, ep)
            extra = {"determinism_mode": LC.DETERMINISM_MODE,
                     "manifest_registry": str(run_dir / "REGISTRY_V2_TRAINER_VIEW.json"),
                     "data_version": lockctx["data_version"], "schema_registry": str(lockctx["schema_registry_path"]),
                     "recipe_id": tuned["recipe_id"], "recipe_registry": tuned["recipe_registry"],
                     "model_variant": tuned["model_variant"], "dropout": tuple(tuned["dropout"]),
                     "schedule": tuned["schedule"],
                     "theta0_init_sha256": LC._theta0_for(args, lockctx["schema_id"], tuned, int(args.seed))}
            cfg = RunConfig(run_id=args.run_id, family=args.family, seed=int(args.seed), exposure_point="end",
                            cuda_visible_devices=str(int(args.gpu)), phase="MAIN", population="COHORT",
                            peak_lr=tuned["peak_lr"], log_every=int(args.log_every),
                            checkpoint_every=int(args.checkpoint_every or 0), calibration_record=None,
                            continuation_budget=None, expected_init_sha256=args.expected_init_sha256, key=key,
                            test_overrides=bool(args.smoke), **extra)
            cfg.validate()
            if not args.smoke and cfg.run_label != RUN_LABEL:
                raise CFullSmallRefused(f"the label wrapper is not in effect: run_label {cfg.run_label!r}")
            out.update(run_id=cfg.run_id, config_sha256=cfg.sha256(), config=cfg.as_json(),
                       run_label=cfg.run_label, manifest_role=cfg.manifest_role, sealed_outputs=cfg.sealed_outputs,
                       effective_auto_continuation=cfg.effective_auto_continuation,
                       train_provider={"n_decisions": tp.n_decisions, "build": tp.build_record},
                       eval_provider={"describe_n": int(ep.manifest.n)},
                       host_rss_bytes=rss_bytes(), host_peak_rss_bytes=peak_rss_bytes())
    after = sorted(str(p) for p in run_dir.rglob("*")) if run_dir.exists() else None
    if after != before:
        raise CFullSmallRefused(f"the dry run changed {run_dir}")
    out["wall_s"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        shim, largv = split_argv(argv)
    except CFullSmallRefused as e:
        print(f"C2 REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    from v4int import launch_central as LC                  # stdlib-only at import (no torch / v4train yet)
    args = LC.parse_args(largv)
    try:
        rid = check_launch_args(args)
    except CFullSmallRefused as e:
        print(f"C2 REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    largv = normalise_argv(largv, args, rid)
    record = {"task": TASK, "authority": AUTHORITY, "utc": LC.utc(), "argv": argv, "launch_argv": largv,
              "run_id": rid, "run_label": RUN_LABEL, "ext_registry_arg": str(shim.ext_registry),
              "wrappers": ["launch_central.bind_data_protocol_lock_v2", "launch_central.build_trainer_registry_view",
                           "v4int.verify.verify_registry (extension path only)",
                           "v4train.recipe.RunConfig.run_label + v4train.trainer.CentralTrainer._row_labels "
                           "(P2_C2_CFULL_SMALL / P2_CONTROL)"],
              "c_full_small_sha256": _sha(Path(__file__)), "dry_run": shim.dry_run_out is not None}
    install(LC, shim.ext_registry, record)
    if shim.dry_run_out is not None:
        try:
            res = dry_run(LC, args, shim, record)
        except (CFullSmallRefused, LC.LaunchRefused) as e:
            print(f"C2 DRY RUN REFUSED: {e}", file=sys.stderr, flush=True)
            return 2
        shim.dry_run_out.parent.mkdir(parents=True, exist_ok=True)
        shim.dry_run_out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"dry_run": str(shim.dry_run_out), "config_sha256": res.get("config_sha256"),
                          "run_id": res.get("run_id"), "run_label": res.get("run_label"),
                          "effective_auto_continuation": res.get("effective_auto_continuation")}))
        return 0
    rc = LC.main(largv)
    record.update(exit_code=rc, utc_end=LC.utc())
    _write_record(Path(args.run_dir), record, record.get("_record_path"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
