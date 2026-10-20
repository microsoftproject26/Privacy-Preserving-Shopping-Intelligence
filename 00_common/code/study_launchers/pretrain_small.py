"""v4int.pretrain_small (owner: Data engineer, DATA-10) - the FA_WARM pretraining launcher shim: runs the UNEDITED
v4int.launch_central as one central SMALL run (recipe SMALL_SASREC_D64_B2 of contracts/hp_registry/
SMALL_RECIPE_SASREC_D64_B2.json, 6.0 EFE, seed 2026) on PRETRAIN_TRAIN_5_100_NONMAIN -> UNEW_VAL_VALIDATION_32768
(owner decision 2026-09-29, DECISION_LOG 195; pattern: v4int.c_full, DATA-08).

  python -B code/v4int/pretrain_small.py [--ext-registry <path>] -- --family SASREC --seed 2026 --phase MAIN \
         --recipe-id SMALL_SASREC_D64_B2 --gpu <id> --run-dir <dir> --history-policy USER_HISTORY_L50 [launcher args]
  python -B code/v4int/pretrain_small.py --dry-run-out <file.json> [--no-providers] -- <the same launcher args>
  python -B code/v4int/pretrain_small.py --fl-init-only -- <the same launcher args>     (write / verify FL_INIT record)

`--tuned-recipes` defaults to the production SMALL recipe file (v4int.small_recipe.RECIPE_PATH); any other path is
refused outside --smoke. The shim changes ONLY which manifests the MAIN role resolves to and how the rows are labelled,
through in-process wrappers installed before / while launch_central.main runs (launch_central, v4train, small_recipe
and c_full are not edited):
  1. launch_central.bind_data_protocol_lock_v2: the original binds LOCK_V2; then REGISTRY_V2_PRETRAIN_EXT.json is
     verified against that exact REGISTRY_V2 (v4data10.d10_registry.verify_ext_registry) and handed to the providers;
  2. launch_central.build_trainer_registry_view: the same file name, roles MAIN / CONTINUATION = the PRETRAIN pair;
  3. v4int.verify.verify_registry (extension path only): REGISTRY_V2 through the original function, then the rule;
  4. labels (installed by wrapper 1, i.e. after v4train is imported, before the RunConfig exists):
     v4train.recipe.RunConfig.run_label returns PRETRAIN_FOR_FL instead of PRODUCTION for a run id starting with
     PRETRAIN_FOR_FL_ (SMOKE_NOT_EVIDENCE is untouched), and v4train.trainer.CentralTrainer._row_labels overwrites every
     row's method / row_status / comparison_status with PRETRAIN_FOR_FL and sets c_matched_eligible =
     downstream_seed_allowed = False: no selection stage (v4train.selection refuses the unknown label), no C / C_MATCHED
     row (method != C, row_status != CENTRAL_CONTEST) and no retention row can ever use them. run_label is a derived
     property, so RunConfig.as_json / sha256 are unchanged by wrapper 4.
FL warm start: the endpoint_6.0 checkpoint (sealed/checkpoints/endpoint_6.0.pt, CONTROLLED_BUDGET weights) - no
checkpoint selection, no value read. After the launcher returns 0 the shim writes PRETRAIN_FOR_FL_INIT.json in the run
dir (path, sha256, bytes of that file; nothing from sealed/CHECKPOINTS.json, which carries values).
Automatic continuation: population COHORT + phase MAIN = a sealed contest-shaped run, so RunConfig forces the C3 rule
(STILL_IMPROVING at 6.0 -> continue to at most 8.0 EFE; auto_continuation=False is refused for a non-test run). The
continuation adds GPU time only; the FL init stays endpoint_6.0 (written at 6.0 and retained). The dry run records
effective_auto_continuation.
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

DEFAULT_EXT = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2_PRETRAIN_EXT.json"
RECIPE_ID = "SMALL_SASREC_D64_B2"
REQUIRED = {"family": "SASREC", "phase": "MAIN", "recipe_id": RECIPE_ID, "population": "COHORT",
            "history_policy": "USER_HISTORY_L50", "seed": 2026}
RUN_ID_PREFIX = "PRETRAIN_FOR_FL_"
RUN_LABEL = "PRETRAIN_FOR_FL"
ROW_STATUS = "PRETRAIN_FOR_FL"
COMPARISON_STATUS = "PRETRAIN_FOR_FL_DESCRIPTIVE"
FL_INIT_KIND = "endpoint_6.0"
FL_INIT_FILE = "PRETRAIN_FOR_FL_INIT.json"
TRAIN_ID, EVAL_ID = "PRETRAIN_TRAIN_5_100_NONMAIN", "UNEW_VAL_VALIDATION_32768"   # = v4data10.d10_common (checked)


class PretrainRefused(RuntimeError):
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
        raise PretrainRefused("usage: pretrain_small.py [shim options] -- <launch_central arguments>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description="FA_WARM pretraining launcher shim (DATA-10)")
    ap.add_argument("--ext-registry", type=Path, default=DEFAULT_EXT)
    ap.add_argument("--dry-run-out", type=Path, default=None)
    ap.add_argument("--no-providers", action="store_true", help="dry run: verify registries / manifests only")
    ap.add_argument("--fl-init-only", action="store_true", help="write PRETRAIN_FOR_FL_INIT.json for a finished run")
    return ap.parse_args(argv[:i]), list(argv[i + 1:])


def recipe_path() -> Path:
    from v4int.small_recipe import RECIPE_PATH                # stdlib-only at import (v4train imported lazily)
    return Path(RECIPE_PATH)


def check_launch_args(args) -> str:
    """The PRETRAIN launch contract on the parsed launch_central arguments; returns the run id to use."""
    for k, want in REQUIRED.items():
        if getattr(args, k, None) != want:
            raise PretrainRefused(f"PRETRAIN needs --{k.replace('_', '-')} {want} (got {getattr(args, k, None)!r})")
    for k in ("train_manifest", "eval_manifest", "role", "hp_id", "calibration_record", "continuation_budget",
              "peak_lr"):
        if getattr(args, k, None):
            raise PretrainRefused(f"PRETRAIN refuses --{k.replace('_', '-')}: the manifests come from the extension's "
                                  "PRETRAIN role and the peak LR from recipe SMALL_SASREC_D64_B2")
    for k in ("transform_state", "manifest_dir", "manifest_hashes", "header_list", "lock_path", "theta0_variants",
              "hp_registry"):
        if getattr(args, k, None) is not None:
            raise PretrainRefused(f"PRETRAIN refuses --{k.replace('_', '-')}: the locked tree, lock, transform state, "
                                  "theta0 registry and the SMALL recipe are used unchanged")
    if getattr(args, "allow_shared_gpu", False) and not getattr(args, "coschedule_06g", False):  # 06G §5 / 06K §1
        raise PretrainRefused("PRETRAIN refuses --allow-shared-gpu without --coschedule-06g (a central MAIN-phase run "
                              "is alone on its card unless co-scheduled per 06G §5 / 06K §1)")
    if getattr(args, "coschedule_06g", False) and not getattr(args, "allow_shared_gpu", False):   # FIX-DELTA-01
        raise PretrainRefused("--coschedule-06g takes effect only together with --allow-shared-gpu (DECISION_LOG 208)")
    tr = getattr(args, "tuned_recipes", None)
    if tr is not None and Path(tr).resolve() != recipe_path().resolve() and not args.smoke:
        raise PretrainRefused(f"PRETRAIN uses the production SMALL recipe {recipe_path()} (got {tr})")
    from v4int.smoke import SMOKE_RUN_PREFIX
    default = f"{SMOKE_RUN_PREFIX if args.smoke else ''}{RUN_ID_PREFIX}{RECIPE_ID}_s{int(args.seed)}"
    rid = args.run_id or default
    core = rid[len(SMOKE_RUN_PREFIX):] if rid.startswith(SMOKE_RUN_PREFIX) else rid
    if not core.startswith(RUN_ID_PREFIX):
        raise PretrainRefused(f"a PRETRAIN run id must start with {RUN_ID_PREFIX!r} (got {rid!r})")
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


# ------------------------------------------------------------------------------------------------ wrapper 4: labels
def is_pretrain_run_id(run_id) -> bool:
    return str(run_id or "").startswith(RUN_ID_PREFIX)


def pretrain_row_fields() -> dict:
    return {"method": RUN_LABEL, "row_status": ROW_STATUS, "comparison_status": COMPARISON_STATUS,
            "c_matched_eligible": False, "downstream_seed_allowed": False, "fl_init_rule": FL_INIT_KIND,
            "marks_status": "DESCRIPTIVE_NEVER_SELECTS"}


def install_label_wrappers(RC=None, TR=None) -> dict:
    """Wrapper 4 on the given (default: the real v4train.recipe / v4train.trainer) modules; idempotent."""
    if RC is None:
        import v4train.recipe as RC  # noqa: N811
    if TR is None:
        import v4train.trainer as TR  # noqa: N811
    cls, tcls = RC.RunConfig, TR.CentralTrainer
    if not getattr(cls, "__pretrain_label__", False):
        orig_prop = cls.__dict__["run_label"]

        def run_label(self):
            lab = orig_prop.fget(self)
            return RUN_LABEL if lab == "PRODUCTION" and is_pretrain_run_id(self.run_id) else lab

        run_label.__wrapped__ = orig_prop
        cls.run_label = property(run_label, doc="DATA-10 v4int.pretrain_small wrapper 4 (PRETRAIN_FOR_FL)")
        cls.__pretrain_label__ = True
    if not getattr(tcls, "__pretrain_label__", False):
        orig_rows = tcls._row_labels

        def _row_labels(self):
            lab = orig_rows(self)
            if getattr(self, "run_label", None) == RUN_LABEL:
                lab.update(pretrain_row_fields())
            return lab

        _row_labels.__wrapped__ = orig_rows
        tcls._row_labels = _row_labels
        tcls.__pretrain_label__ = True
    return {"run_label": RUN_LABEL, "row_fields": pretrain_row_fields()}


# ------------------------------------------------------------------------------------------------ wrappers 1-3
def install(LC, ext_path: Path, record: dict, *, labels=install_label_wrappers) -> None:
    """Wrappers 1 and 2 on the launch_central module object (wrappers 3 and 4 are installed by wrapper 1, after the
    CUDA export, because v4int.verify and the label targets import v4train)."""
    ext_path = Path(ext_path).resolve()
    orig_bind, orig_view = LC.bind_data_protocol_lock_v2, LC.build_trainer_registry_view

    def bind(paths, *, smoke, gate_path=LC.G1_GATE):
        ctx = orig_bind(paths, smoke=smoke, gate_path=gate_path)
        from v4data10 import d10_common as D10
        from v4data10 import d10_registry as R
        if (D10.TRAIN_ID, D10.EVAL_ID) != (TRAIN_ID, EVAL_ID):
            raise LC.LaunchRefused("pretrain_small ids differ from v4data10.d10_common")
        base_path = Path(ctx["manifest_registry_path"])
        if not ext_path.is_file():
            raise LC.LaunchRefused(f"PRETRAIN extension registry {ext_path} does not exist (run the DATA-10 jobs)")
        ext, base = _strict(ext_path), _strict(base_path)
        chk = R.verify_ext_registry(ext, base, ctx["manifest_registry_file_sha256"])
        if not chk["all_ok"] or (ext.get("extends") or {}).get("path") != ctx["manifest_registry_rel_path"]:
            raise LC.LaunchRefused(f"PRETRAIN extension registry refused: {[k for k, v in chk.items() if not v]}")
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
                    pretrain_base_registry=record["base_registry"])

    def view(manifest_registry_path, run_dir, *, registry_rel_path, registry_file_sha256):
        from v4data10 import d10_registry as R
        if Path(manifest_registry_path).resolve() != ext_path:
            raise LC.LaunchRefused("the PRETRAIN trainer view is built from the extension registry only")
        ext = _strict(ext_path)
        obj = R.trainer_view(ext, ext_rel_path=registry_rel_path, ext_file_sha256=registry_file_sha256)
        chk = R.check_trainer_view(obj, ext)
        if not chk["all_ok"]:
            raise LC.LaunchRefused(f"PRETRAIN trainer view refused: {chk}")
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
        from v4data10 import d10_registry as R
        ext = _strict(ext_path)
        chk = R.verify_ext_registry(ext, base, base_file)
        if not chk["all_ok"]:
            raise error(f"PRETRAIN extension registry {ext_path} fails the extension rule: "
                        f"{[k for k, v in chk.items() if not v]}")
        return ext, _sha(ext_path), ext["self_sha256"], v1_ctx

    verify_registry.__wrapped__ = orig
    return verify_registry


def install_verify_wrapper(ext_path: Path, base_path: Path) -> None:
    import v4int.verify as VF
    if getattr(VF.verify_registry, "__pretrain__", False):
        return
    w = make_verify_wrapper(VF.verify_registry, ext_path, base_path, error=VF.ManifestVerificationError)
    w.__pretrain__ = True
    VF.verify_registry = w


def _write_record(run_dir: Path, record: dict, path=None) -> Path:
    """PRETRAIN_SHIM_<k>.json in the run dir (written when the view is built, rewritten with the exit code)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if path is None:
        k = 0
        while (run_dir / f"PRETRAIN_SHIM_{k}.json").exists():
            k += 1
        path = run_dir / f"PRETRAIN_SHIM_{k}.json"
    p = Path(path)
    p.write_text(json.dumps({k: v for k, v in record.items() if not k.startswith("_")}, indent=1, sort_keys=True,
                            default=str) + "\n", encoding="utf-8", newline="\n")
    return p


# ------------------------------------------------------------------------------------------------ FL init record
def fl_init_record(run_dir: Path, run_id: str) -> dict:
    """PRETRAIN_FOR_FL_INIT.json: the endpoint_6.0 weights file of the run (path / sha256 / bytes only). Refuses when
    the file is absent. Never reads sealed/CHECKPOINTS.json or any value-bearing file.
    FIX-DEV-FTPRE M3: written ONCE (exclusive create). Its file sha256 is theta_pre's init_record_sha256 (FT_PRE /
    FA_WARM / the PRE re-evaluation compare it), so a later call never rewrites it: it returns the existing record
    unchanged when every field but `utc` equals this call's (same run, same endpoint_6.0 path / sha256 / bytes) and
    refuses otherwise."""
    run_dir = Path(run_dir)
    cands = [run_dir / "sealed" / "checkpoints" / f"{FL_INIT_KIND}.pt", run_dir / "checkpoints" / f"{FL_INIT_KIND}.pt"]
    p = next((c for c in cands if c.is_file()), None)
    if p is None:
        raise PretrainRefused(f"no {FL_INIT_KIND} checkpoint under {run_dir} (the run has not reached 6.0 EFE)")
    rec = {"task": "DATA-10 v4int.pretrain_small", "run_id": run_id, "run_label": RUN_LABEL,
           "fl_init": {"kind": FL_INIT_KIND, "rule": "CONTROLLED_BUDGET endpoint at 6.0 EFE; no checkpoint selection",
                       "path": str(p.relative_to(run_dir)).replace("\\", "/"), "sha256": _sha(p),
                       "bytes": p.stat().st_size},
           "train_manifest": TRAIN_ID, "eval_manifest_descriptive": EVAL_ID,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = run_dir / FL_INIT_FILE
    try:
        with open(out, "x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        return rec
    except FileExistsError:
        pass
    try:
        old = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PretrainRefused(f"{out} exists and is unreadable ({e}); the FL init record is never rewritten") from None
    same = isinstance(old, dict) and {k: v for k, v in old.items() if k != "utc"} == \
        {k: v for k, v in rec.items() if k != "utc"}
    if not same:
        raise PretrainRefused(f"{out} exists and does not describe this run's theta_pre ({p.name} sha256 "
                              f"{rec['fl_init']['sha256'][:12]}…): the FL init record is never rewritten")
    return old


# ------------------------------------------------------------------------------------------------ dry run
def dry_run(LC, args, shim, record: dict) -> dict:
    """launch_central.run up to the RunConfig (no trainer, no GPU work, nothing written under --run-dir): the SMALL
    recipe, the view in a temporary directory, the providers (unless --no-providers), the comparison key, the
    RunConfig + validate() (with wrapper 4: run_label PRETRAIN_FOR_FL), and the trainer's own registry checks."""
    import tempfile
    record["environment"] = LC.export_environment(int(args.gpu))       # as the launcher, before v4train is imported
    import v4train  # noqa: F401
    from v4int.small_recipe import install as small_install
    small_install()                                                  # as launch_central.run (INT-03)
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
    out = {"note": "PRETRAIN dry run: no training, nothing written under run_dir", "utc": LC.utc(), "shim": record,
           "tuned_recipe": tuned}
    with tempfile.TemporaryDirectory(prefix="pretrain_dryrun_") as td:
        view = LC.build_trainer_registry_view(lockctx["manifest_registry_path"], Path(td),
                                              registry_rel_path=lockctx["manifest_registry_rel_path"],
                                              registry_file_sha256=lockctx["manifest_registry_file_sha256"])
        vreg = load_manifest_registry(view["path"])
        want = expected_manifests("COHORT", "MAIN", registry=vreg)
        out["trainer_view"] = {"self_sha256": view["self_sha256"], "roles": view["roles"],
                               "expected_manifests": list(want)}
        if tuple(want) != (TRAIN_ID, EVAL_ID):
            raise PretrainRefused(f"the trainer view resolves MAIN to {want}")
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
                raise PretrainRefused(f"wrapper 4 not in effect: run_label {cfg.run_label!r}")
            out.update(run_id=cfg.run_id, config_sha256=cfg.sha256(), config=cfg.as_json(),
                       run_label=cfg.run_label, manifest_role=cfg.manifest_role, sealed_outputs=cfg.sealed_outputs,
                       effective_auto_continuation=cfg.effective_auto_continuation,
                       continuation_note=("STILL_IMPROVING at 6.0 continues automatically to at most 8.0 EFE (C3); "
                                          "the FL init stays endpoint_6.0") if cfg.effective_auto_continuation else None,
                       train_provider={"n_decisions": tp.n_decisions, "build": tp.build_record},
                       eval_provider={"describe_n": int(ep.manifest.n)},
                       host_rss_bytes=rss_bytes(), host_peak_rss_bytes=peak_rss_bytes())
    after = sorted(str(p) for p in run_dir.rglob("*")) if run_dir.exists() else None
    if after != before:
        raise PretrainRefused(f"the dry run changed {run_dir}")
    out["wall_s"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        shim, largv = split_argv(argv)
    except PretrainRefused as e:
        print(f"PRETRAIN REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    from v4int import launch_central as LC                  # stdlib-only at import (no torch / v4train yet)
    args = LC.parse_args(largv)
    try:
        rid = check_launch_args(args)
    except PretrainRefused as e:
        print(f"PRETRAIN REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    largv = normalise_argv(largv, args, rid)
    if shim.fl_init_only:
        try:
            rec = fl_init_record(Path(args.run_dir), rid)
        except PretrainRefused as e:
            print(f"PRETRAIN REFUSED: {e}", file=sys.stderr, flush=True)
            return 2
        print(json.dumps(rec["fl_init"]))
        return 0
    record = {"task": "DATA-10 v4int.pretrain_small", "utc": LC.utc(), "argv": argv, "launch_argv": largv,
              "run_id": rid, "ext_registry_arg": str(shim.ext_registry), "run_label": RUN_LABEL,
              "fl_init_rule": FL_INIT_KIND,
              "wrappers": ["launch_central.bind_data_protocol_lock_v2", "launch_central.build_trainer_registry_view",
                           "v4int.verify.verify_registry (extension path only)",
                           "v4train.recipe.RunConfig.run_label + v4train.trainer.CentralTrainer._row_labels "
                           "(PRETRAIN_FOR_FL)"],
              "pretrain_small_sha256": _sha(Path(__file__)), "dry_run": shim.dry_run_out is not None}
    install(LC, shim.ext_registry, record)
    if shim.dry_run_out is not None:
        try:
            res = dry_run(LC, args, shim, record)
        except (PretrainRefused, LC.LaunchRefused) as e:
            print(f"PRETRAIN DRY RUN REFUSED: {e}", file=sys.stderr, flush=True)
            return 2
        shim.dry_run_out.parent.mkdir(parents=True, exist_ok=True)
        shim.dry_run_out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"dry_run": str(shim.dry_run_out), "config_sha256": res.get("config_sha256"),
                          "run_id": res.get("run_id"), "run_label": res.get("run_label"),
                          "effective_auto_continuation": res.get("effective_auto_continuation")}))
        return 0
    rc = LC.main(largv)
    record.update(exit_code=rc, utc_end=LC.utc())
    if rc == 0 and not args.smoke:
        try:
            record["fl_init"] = fl_init_record(Path(args.run_dir), rid)["fl_init"]
        except PretrainRefused as e:
            record["fl_init_error"] = str(e)
    _write_record(Path(args.run_dir), record, record.get("_record_path"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
