"""v4int.c_full (owner: Data engineer, DATA-08) - the C_FULL launcher shim: runs the UNEDITED v4int.launch_central with
recipe 02B_SASREC on FULL_TRAIN_5_100_TIEBREAK -> MAIN_VALIDATION_5_100_131072 (PREG1_AMENDMENT_06 §5, 06B §6.3).

  python -B code/v4int/c_full.py [--ext-registry <path>] -- --family SASREC --seed 2026 --phase MAIN \
         --recipe-id 02B_SASREC --gpu <id> --run-dir <dir> --history-policy USER_HISTORY_L50 [launch_central args]
  python -B code/v4int/c_full.py --dry-run-out <file.json> [--no-providers] -- <the same launcher args>

Why a shim (and not the trainer's FULL_TRAIN mode): population FULL_TRAIN refuses a recipe_id, requires a calibration
record and the registered default variant / dropout / schedule (v4train.recipe.RunConfig.validate), while 02B_SASREC is
the tuned SASREC_D256_B3 recipe; launch_central also refuses population FULL_TRAIN outright. v4train / launch_central
are not edited. The shim changes ONLY which manifests the MAIN role resolves to, through three in-process wrappers
installed before launch_central.main runs:
  1. launch_central.bind_data_protocol_lock_v2: the original binds LOCK_V2 (pin, REGISTRY_V2 file sha256, S* schema);
     then REGISTRY_V2_FULL_EXT.json is verified against that exact REGISTRY_V2 (v4data08.d8_registry.
     verify_ext_registry: every REGISTRY_V2 entry canonical-JSON identical, `extends` = REGISTRY_V2's path / file /
     self hash) and the returned context names the EXTENSION as the providers' manifest registry;
  2. launch_central.build_trainer_registry_view: writes the SAME file name (REGISTRY_V2_TRAINER_VIEW.json) with roles
     MAIN / CONTINUATION = the extension's C_FULL pair (v4data08.d8_registry.trainer_view), so the unedited trainer
     (role MAIN) binds FULL_TRAIN_5_100_TIEBREAK and MAIN_VALIDATION_5_100_131072 by id, row count and hashes;
  3. v4int.verify.verify_registry (installed lazily, after the launcher's CUDA export): for the extension path only,
     the ORIGINAL function verifies REGISTRY_V2 itself under LOCK_V2 (F3 pin + B1 superseded-lock route, returning
     the real v1 lock context), then the extension must pass verify_ext_registry; any other path is delegated
     unchanged. A LOCK_V2 extension that pins REGISTRY_V2_FULL_EXT would make wrapper 3 unnecessary (Ops item).
RunConfig: no new field and no changed rule. Its sha256 differs from C_BIG's only through values the launcher already
derives: run_id (default C_FULL_02B_SASREC_s<seed>), manifest_registry (the view path in the C_FULL run dir),
cuda_visible_devices, and key.{train_manifest_id, population_hash, manifest_registry_self_sha256} (the TB manifest,
the TB band hash, the extension's self-hash). population COHORT, role MAIN, recipe 02B_SASREC (6.0 EFE, marks,
PRACTICAL_BEST, sealed outputs and the automatic C3 continuation rule of every contest-shaped run) are unchanged.
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

DEFAULT_EXT = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2_FULL_EXT.json"
REQUIRED = {"family": "SASREC", "phase": "MAIN", "recipe_id": "02B_SASREC", "population": "COHORT",
            "history_policy": "USER_HISTORY_L50"}
RUN_ID_PREFIX = "C_FULL_"
TRAIN_ID, EVAL_ID = "FULL_TRAIN_5_100_TIEBREAK", "MAIN_VALIDATION_5_100_131072"   # = v4data08.d8_common (checked)


class CFullRefused(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _strict(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"),
                      parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))


def split_argv(argv: list) -> tuple:
    if "--" not in argv:
        raise CFullRefused("usage: c_full.py [shim options] -- <launch_central arguments>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description="C_FULL launcher shim (DATA-08)")
    ap.add_argument("--ext-registry", type=Path, default=DEFAULT_EXT)
    ap.add_argument("--dry-run-out", type=Path, default=None)
    ap.add_argument("--no-providers", action="store_true", help="dry run: verify registries / manifests only")
    return ap.parse_args(argv[:i]), list(argv[i + 1:])


def check_launch_args(args, smoke_ok: bool = True) -> str:
    """The C_FULL launch contract on the parsed launch_central arguments; returns the run id to use."""
    for k, want in REQUIRED.items():
        if getattr(args, k, None) != want:
            raise CFullRefused(f"C_FULL needs --{k.replace('_', '-')} {want} (got {getattr(args, k, None)!r})")
    for k in ("train_manifest", "eval_manifest", "role", "hp_id", "calibration_record", "continuation_budget"):
        if getattr(args, k, None):
            raise CFullRefused(f"C_FULL refuses --{k.replace('_', '-')}: the manifests come from the extension's "
                               "C_FULL role and the peak LR from recipe 02B_SASREC")
    for k in ("transform_state", "manifest_dir", "manifest_hashes", "header_list", "lock_path", "theta0_variants",
              "hp_registry"):                      # REV-ANA-01 C-F2: every data / lock / HP override path
        if getattr(args, k, None) is not None:
            raise CFullRefused(f"C_FULL refuses --{k.replace('_', '-')}: the locked tree, lock, transform state and "
                               "recipe 02B_SASREC are used unchanged")
    if getattr(args, "allow_shared_gpu", False):
        raise CFullRefused("C_FULL refuses --allow-shared-gpu (06D §7: the run is alone on its card)")
    if args.smoke and not smoke_ok:
        raise CFullRefused("smoke refused here")
    from v4int.smoke import SMOKE_RUN_PREFIX
    default = f"{SMOKE_RUN_PREFIX if args.smoke else ''}{RUN_ID_PREFIX}02B_SASREC_s{int(args.seed)}"
    rid = args.run_id or default
    core = rid[len(SMOKE_RUN_PREFIX):] if rid.startswith(SMOKE_RUN_PREFIX) else rid
    if not core.startswith(RUN_ID_PREFIX):
        raise CFullRefused(f"a C_FULL run id must start with {RUN_ID_PREFIX!r} (got {rid!r})")
    return rid


# ------------------------------------------------------------------------------------------------ the wrappers
def install(LC, ext_path: Path, record: dict) -> None:
    """Wrappers 1 and 2 on the launch_central module object (wrapper 3 is installed by wrapper 1, after the CUDA
    export, because v4int.verify imports v4train)."""
    ext_path = Path(ext_path).resolve()
    orig_bind, orig_view = LC.bind_data_protocol_lock_v2, LC.build_trainer_registry_view

    def bind(paths, *, smoke, gate_path=LC.G1_GATE):
        ctx = orig_bind(paths, smoke=smoke, gate_path=gate_path)
        from v4data08 import d8_common as D8
        from v4data08 import d8_registry as R
        if (D8.TRAIN_ID, D8.EVAL_ID) != (TRAIN_ID, EVAL_ID):
            raise LC.LaunchRefused("c_full ids differ from v4data08.d8_common")
        base_path = Path(ctx["manifest_registry_path"])
        if not ext_path.is_file():
            raise LC.LaunchRefused(f"C_FULL extension registry {ext_path} does not exist (run the DATA-08 jobs)")
        ext, base = _strict(ext_path), _strict(base_path)
        chk = R.verify_ext_registry(ext, base, ctx["manifest_registry_file_sha256"])
        if not chk["all_ok"] or (ext.get("extends") or {}).get("path") != ctx["manifest_registry_rel_path"]:
            raise LC.LaunchRefused(f"C_FULL extension registry refused: {[k for k, v in chk.items() if not v]}")
        install_verify_wrapper(ext_path, base_path)
        try:
            ext_rel = str(ext_path.relative_to(LC.V4.resolve())).replace("\\", "/")
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
            raise LC.LaunchRefused("the C_FULL trainer view is built from the extension registry only")
        ext = _strict(ext_path)
        obj = R.trainer_view(ext, ext_rel_path=registry_rel_path, ext_file_sha256=registry_file_sha256)
        chk = R.check_trainer_view(obj, ext)
        if not chk["all_ok"]:
            raise LC.LaunchRefused(f"C_FULL trainer view refused: {chk}")
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


def make_verify_wrapper(orig, ext_path: Path, base_path: Path, error=ValueError):
    """Wrapper 3 as a pure factory (testable with a stub `orig`)."""
    ext_path, base_path = Path(ext_path).resolve(), Path(base_path)

    def verify_registry(registry_path, lock_self, lock_file, *, lock=None, root=None):
        if Path(registry_path).resolve() != ext_path:
            return orig(registry_path, lock_self, lock_file, lock=lock, root=root)
        base, base_file, base_self, v1_ctx = orig(base_path, lock_self, lock_file, lock=lock, root=root)
        from v4data08 import d8_registry as R
        ext = _strict(ext_path)
        chk = R.verify_ext_registry(ext, base, base_file)
        if not chk["all_ok"]:
            raise error(f"C_FULL extension registry {ext_path} fails the extension rule: "
                        f"{[k for k, v in chk.items() if not v]}")
        if not isinstance(ext.get("manifests"), dict) or not isinstance(ext.get("roles"), dict):
            raise error(f"{ext_path}: 'roles' and 'manifests' are required")
        return ext, _sha(ext_path), ext["self_sha256"], v1_ctx

    verify_registry.__wrapped__ = orig
    return verify_registry


def install_verify_wrapper(ext_path: Path, base_path: Path) -> None:
    import v4int.verify as VF
    if getattr(VF.verify_registry, "__c_full__", False):
        return
    w = make_verify_wrapper(VF.verify_registry, ext_path, base_path, error=VF.ManifestVerificationError)
    w.__c_full__ = True
    VF.verify_registry = w


def _write_record(run_dir: Path, record: dict, path=None) -> Path:
    """C_FULL_SHIM_<k>.json in the run dir (written when the view is built, rewritten with the exit code)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if path is None:
        k = 0
        while (run_dir / f"C_FULL_SHIM_{k}.json").exists():
            k += 1
        path = run_dir / f"C_FULL_SHIM_{k}.json"
    p = Path(path)
    p.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
    return p


# ------------------------------------------------------------------------------------------------ dry run
def dry_run(LC, args, shim, record: dict) -> dict:
    """launch_central.run up to the RunConfig (no trainer, no GPU work, nothing written under --run-dir): the view in
    a temporary directory, the providers (unless --no-providers), the comparison key, the RunConfig, validate(), and
    the trainer's own registry checks (load_manifest_registry, expected_manifests, check_provider)."""
    import tempfile
    record["environment"] = LC.export_environment(int(args.gpu))       # as the launcher, before v4train is imported
    import v4train  # noqa: F401
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    from v4int.verify import verify_manifest_by_registry
    from v4train.provider import expected_manifests
    from v4train.recipe import RunConfig
    from v4train.registry import check_provider, load_manifest_registry
    run_dir = Path(args.run_dir)
    before = sorted(str(p) for p in run_dir.rglob("*")) if run_dir.exists() else None
    t0 = time.time()
    tuned = LC._recipe_values(args, args.family)
    paths = LC.data_paths(args)
    lockctx = LC.bind_data_protocol_lock_v2(paths, smoke=bool(args.smoke))
    out = {"note": "C_FULL dry run: no training, nothing written under run_dir", "utc": LC.utc(), "shim": record}
    with tempfile.TemporaryDirectory(prefix="c_full_dryrun_") as td:
        view = LC.build_trainer_registry_view(lockctx["manifest_registry_path"], Path(td),
                                              registry_rel_path=lockctx["manifest_registry_rel_path"],
                                              registry_file_sha256=lockctx["manifest_registry_file_sha256"])
        vreg = load_manifest_registry(view["path"])
        want = expected_manifests("COHORT", "MAIN", registry=vreg)
        out["trainer_view"] = {"self_sha256": view["self_sha256"], "roles": view["roles"],
                               "expected_manifests": list(want)}
        if tuple(want) != (TRAIN_ID, EVAL_ID):
            raise CFullRefused(f"the trainer view resolves MAIN to {want}")
        if shim.no_providers:
            for mid, kind in ((TRAIN_ID, "TRAIN"), (EVAL_ID, "EVAL")):
                vm = verify_manifest_by_registry(mid, lockctx["manifest_registry_path"], paths, root=LC.V4,
                                                 expect_kind=kind)
                out.setdefault("verified", {})[mid] = {"n_checks": len(vm.checks), "n_decisions": int(
                    vm.canonical_decision_id.size), "registry": vm.registry}
                vm.columns = None
            out["config_sha256"] = None
            out["config_note"] = "--no-providers: the comparison key (and so the config hash) needs the providers"
        else:
            from v4int.cache import peak_rss_bytes, rss_bytes
            tp = StoreTrainProvider(TRAIN_ID, history_policy=args.history_policy, paths=paths, log=print,
                                    manifest_registry=lockctx["manifest_registry_path"], registry_root=LC.V4)
            ep = StoreEvalProvider(EVAL_ID, history_policy=args.history_policy, paths=paths,
                                   prefetch=int(args.eval_prefetch), log=print,
                                   manifest_registry=lockctx["manifest_registry_path"], registry_root=LC.V4)
            out["bindings"] = {"train": check_provider(vreg, "MAIN", "train", tp),
                               "eval": check_provider(vreg, "MAIN", "eval", ep)}
            key = comparison_key_base(paths, tp, ep)
            extra = {"determinism_mode": LC.DETERMINISM_MODE, "manifest_registry": str(run_dir / "REGISTRY_V2_TRAINER_VIEW.json"),
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
            out.update(run_id=cfg.run_id, config_sha256=cfg.sha256(), config=cfg.as_json(),
                       run_label=cfg.run_label, manifest_role=cfg.manifest_role, sealed_outputs=cfg.sealed_outputs,
                       effective_auto_continuation=cfg.effective_auto_continuation,
                       train_provider={"n_decisions": tp.n_decisions, "build": tp.build_record},
                       eval_provider={"describe_n": int(ep.manifest.n)},
                       host_rss_bytes=rss_bytes(), host_peak_rss_bytes=peak_rss_bytes())
    after = sorted(str(p) for p in run_dir.rglob("*")) if run_dir.exists() else None
    if after != before:
        raise CFullRefused(f"the dry run changed {run_dir}")
    out["wall_s"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        shim, largv = split_argv(argv)
    except CFullRefused as e:
        print(f"C_FULL REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    from v4int import launch_central as LC                  # stdlib-only at import (no torch / v4train yet)
    args = LC.parse_args(largv)
    try:
        rid = check_launch_args(args)
    except CFullRefused as e:
        print(f"C_FULL REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    if "--run-id" not in largv:
        largv += ["--run-id", rid]
        args.run_id = rid
    record = {"task": "DATA-08 v4int.c_full", "utc": LC.utc(), "argv": argv, "launch_argv": largv, "run_id": rid,
              "ext_registry_arg": str(shim.ext_registry),
              "wrappers": ["launch_central.bind_data_protocol_lock_v2", "launch_central.build_trainer_registry_view",
                           "v4int.verify.verify_registry (extension path only)"],
              "c_full_sha256": _sha(Path(__file__))}
    record["dry_run"] = shim.dry_run_out is not None
    install(LC, shim.ext_registry, record)
    if shim.dry_run_out is not None:
        try:                                    # like runs_p3/dryrun/p3_config_dryrun.py: no preflight / GPU check
            res = dry_run(LC, args, shim, record)
        except (CFullRefused, LC.LaunchRefused) as e:
            print(f"C_FULL DRY RUN REFUSED: {e}", file=sys.stderr, flush=True)
            return 2
        shim.dry_run_out.parent.mkdir(parents=True, exist_ok=True)
        shim.dry_run_out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"dry_run": str(shim.dry_run_out), "config_sha256": res.get("config_sha256"),
                          "run_id": res.get("run_id")}))
        return 0
    rc = LC.main(largv)
    record.update(exit_code=rc, utc_end=LC.utc())
    _write_record(Path(args.run_dir), record, record.get("_record_path"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
