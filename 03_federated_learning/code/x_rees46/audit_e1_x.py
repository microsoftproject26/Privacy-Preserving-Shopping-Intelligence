"""C3 (POST_HOC_EXPLORATORY): the UNCHANGED E1 MIA audit (v4int.audit_e1_mia from code_x1 = code_p2r4 bytes; PREG2 §7.1
populations / score / AUC / CP / validity rule; TRAIN-window rows only, no VALIDATION / TEST row) on the moving X models.
The four mandatory confirmatory targets (PRE null control, FA_WARM_FROZEN, P2_WSTAR, P2_C1_FT_CENTRAL_FULL) keep their
unchanged checks; four optional labels are added, each with its own source checks run under TEMPORARY pins:
  X_FA_COLD       cold FA MAIN run of laneX_COLD.sh (runs_x_rees46/COLD/MAIN): LAUNCH arm FA, no init_from, FedAdam tau 1e-6
  X_FA_COLD_DP8   cold DP MAIN run (runs_x_rees46/DP/COLD): LAUNCH arm DP, dp_z epsilon 8, tau 1e-6; endpoint only
  X2_PB100        PB100 winner MAIN run (runs_x_rees46/PB100/MAIN): reeval_p2.p2_source_fl with the X2 init pin
                  (theta_pre(100k), env X2_INIT_SHA / X2_INIT_DIGEST), recipe file sha (env X2_RECIPE_SHA), run id format
                  (env X2_RUN_ID_FMT), tau 1e-6 re-derived by launch_fl.check_p2_spec
  X2_PB100_DP8    its W* DP chain run (runs_x_rees46/PB100/DP): audit_e1_mia.check_dp_source as P2_WSTAR_DP8 under the
                  same pins; endpoint only
  python -B x_rees46/audit_e1_x.py <audit_e1_mia arguments; --targets may add the four labels> --out runs_x_rees46/E1/<new>"""
import contextlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("X_CODE") or os.getcwd())
from v4int import audit_e1_mia as E  # noqa: E402
from v4int import reeval as R  # noqa: E402
from v4int import reeval_p2 as P2  # noqa: E402
import v4int.launch_fl as LF  # noqa: E402
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import x2_admissions as X2A  # noqa: E402

XR = Path(__file__).resolve().parents[1] / "runs_x_rees46"
X_LABELS = ("X_FA_COLD", "X_FA_COLD_DP8", "X2_PB100", "X2_PB100_DP8")
E.TARGETS.update({
    "X_FA_COLD": {"kind": "fl", "class": "NON_DP", "epsilon": None, "files": (E.BEST, E.END), "mandatory": False},
    "X_FA_COLD_DP8": {"kind": "fl", "class": "DP", "epsilon": 8.0, "files": (E.END,), "mandatory": False},
    "X2_PB100": {"kind": "fl", "class": "NON_DP", "epsilon": None, "files": (E.BEST, E.END), "mandatory": False},
    "X2_PB100_DP8": {"kind": "fl", "class": "DP", "epsilon": 8.0, "files": (E.END,), "mandatory": False}})
E.TARGET_ORDER = tuple(E.TARGET_ORDER) + X_LABELS


@contextlib.contextmanager
def pins(label, ckpt=None):
    saved = (P2.PRE_CHECKPOINT_SHA256, P2.PRE_STATE_DIGEST, LF.p2_server_opt, P2.check_launch_selection,
             LF.P2_CLIENT_LRS, LF.P2_ITEM_LR_MULTS)
    adm = X2A.snapshot(LF)
    try:
        if label.startswith("X2"):
            P2.PRE_CHECKPOINT_SHA256, P2.PRE_STATE_DIGEST = os.environ["X2_INIT_SHA"], os.environ["X2_INIT_DIGEST"]
            # the launch_fl_x2 admissions of the run (x2_admissions): the MAIN run's own E / P from its LAUNCH spec;
            # the DP chain was launched with the launch_fl_x2 defaults (E 6, P 1; laneX_DP.sh pb100)
            efe, passes = X2A.run_efe_passes(ckpt) if (label == "X2_PB100" and ckpt is not None) else (6.0, 1)
            X2A.apply(LF, efe, passes)

            def recipe_check(launch, spec):
                src = launch.get("p2_recipe") or {}
                if launch.get("p2_selection") is not None or src.get("sha256") != os.environ["X2_RECIPE_SHA"]:
                    E._refuse(f"{label}: the run's recipe file is not the bound X2 recipe")
                return {"p2_recipe": dict(src), "label": "POST_HOC_EXPLORATORY"}
            P2.check_launch_selection = recipe_check
            P2.ARMS["X2_PB100"] = {"kind": "FL", "seeds": (2026, 2027), "launch_arm": "P2_W",
                                   "run_id": os.environ["X2_RUN_ID_FMT"]}
        yield
    finally:
        (P2.PRE_CHECKPOINT_SHA256, P2.PRE_STATE_DIGEST, LF.p2_server_opt, P2.check_launch_selection,
         LF.P2_CLIENT_LRS, LF.P2_ITEM_LR_MULTS) = saved
        X2A.restore(LF, adm)


def _tau(spec):
    return float(((spec.get("cfg") or {}).get("server_opt") or {}).get("tau", float("nan")))


def check_x2_dp_source(label, src, ckpt, got):
    """X2_PB100_DP8: the exploratory W* DP-chain run, bound by its OWN run dir (runs_x_rees46/PB100/DP/<run>), its own
    LAUNCH run id (= the checkpoint directory name; the launch_fl_x2 tag, never the confirmatory P2_WSTAR_DP8_SMALL_s2026)
    and the checkpoint sha256 (recorded). The other checks of audit_e1_mia.check_dp_source are kept: LAUNCH arm
    P2_WSTAR_DP8, arm-spec digest, launch_fl.check_p2_dp_spec (under the X2 pins), dp_z epsilon = the registered 8, init
    = the X2 pin (theta_pre(100k))."""
    la = src.get("launch") or {}
    spec = la.get("arm_spec") or {}
    if src.get("arm") != "P2_WSTAR_DP8" or not LF.is_p2_dp_arm("P2_WSTAR_DP8"):
        E._refuse(f"{label}: the run's LAUNCH arm is {src.get('arm')!r}, not P2_WSTAR_DP8")
    if LF.spec_digest(spec) != la.get("arm_spec_sha256") or spec.get("arm") != "P2_WSTAR_DP8":
        E._refuse(f"{label}: the LAUNCH arm spec does not hash to its arm_spec_sha256 / names another arm")
    try:
        LF.check_p2_dp_spec(spec)
    except LF.LaunchRefused as e:
        E._refuse(f"{label}: launch_fl.check_p2_dp_spec refuses the run's spec: {e}")
    rid = str(la.get("run_id") or "")
    if (Path(ckpt).parent.name != rid or not rid.startswith("P2_WSTAR_DP8_") or "_XTAU1e-06_" not in rid
            or not rid.endswith(f"_SMALL_s{E.TARGET_SEED}") or rid == f"P2_WSTAR_DP8_SMALL_s{E.TARGET_SEED}"):
        E._refuse(f"{label}: run id {rid!r} is not an exploratory X2 DP8 run (own tag, seed {E.TARGET_SEED})")
    try:
        eps = float((la.get("dp_z") or {}).get("epsilon"))
    except (TypeError, ValueError):
        eps = float("nan")
    if eps != 8.0 or LF.P2_DP_EPSILON.get("P2_WSTAR_DP8") != 8.0:
        E._refuse(f"{label}: the LAUNCH dp_z epsilon is not the registered accountant epsilon 8")
    return {"launch_arm": "P2_WSTAR_DP8", "run_id": rid, "run_dir": str(Path(ckpt).parent.parent.parent),
            "checkpoint_sha256": got, "arm_spec_sha256": la.get("arm_spec_sha256"), "check_p2_dp_spec": True,
            "accountant_epsilon": eps, "dp_z": dict(la.get("dp_z") or {}),
            "theta_pre_100k": E.check_theta_pre_init(label, spec), "label": "POST_HOC_EXPLORATORY"}


_orig_vs = E.verify_source


def verify_source_x(label, ckpt):
    if label not in X_LABELS:
        return _orig_vs(label, ckpt)
    ckpt = Path(ckpt)
    sub = {"X_FA_COLD": "COLD/MAIN", "X_FA_COLD_DP8": "DP/COLD", "X2_PB100": "PB100/MAIN", "X2_PB100_DP8": "PB100/DP"}[label]
    if (XR / sub).resolve() not in ckpt.resolve().parents:
        E._refuse(f"{label}: {ckpt} is not under runs_x_rees46/{sub}")
    got = R._sha(ckpt)
    ctx = {"plan": {"label": label}, "checks": {}, "spec": None}
    checks = {"label": "POST_HOC_EXPLORATORY"}
    args = E._args(label, ckpt, "fl")
    try:
        with pins(label, ckpt):
            if label == "X2_PB100":
                src = P2.p2_source_fl(ckpt, args, ctx)
                checks["p2_w"] = dict(ctx["checks"].get("fl") or {})
            else:
                src = R.source_fl(ckpt, args)
                spec = (src.get("launch") or {}).get("arm_spec") or {}
                if _tau(spec) != 1e-6:
                    E._refuse(f"{label}: the run's FedAdam tau is not 1e-6")
                if label == "X_FA_COLD":
                    if src.get("arm") != "FA" or spec.get("init_from") is not None:
                        E._refuse(f"{label}: not a cold FA run (arm {src.get('arm')!r})")
                elif label == "X_FA_COLD_DP8":
                    eps = float(((src.get("launch") or {}).get("dp_z") or {}).get("epsilon", "nan"))
                    if src.get("arm") != "DP" or eps != 8.0 or spec.get("init_from") is not None:
                        E._refuse(f"{label}: not a cold DP epsilon-8 run")
                    checks["dp"] = {"accountant_epsilon": eps, "dp_z": dict(src["launch"]["dp_z"])}
                else:
                    checks["dp"] = check_x2_dp_source(label, src, ckpt, got)
            checks["launch_files"] = src["launch_files"]
    except R.ReevalRefused as e:
        E._refuse(f"{label}: source check refused: {e}")
    return {"label": label, "kind": "fl", "sha256": got, "file": ckpt.name, "rule": E.FILE_RULE[ckpt.name],
            "path": str(ckpt), "src": src, "ctx": ctx, "args": args, "checks": checks}


E.verify_source = verify_source_x
_orig_rtv = E.read_target_verified


def read_target_verified_x(label, path, source, schema_self_sha256):
    with pins(label, path):
        return _orig_rtv(label, path, source, schema_self_sha256)


E.read_target_verified = read_target_verified_x

if __name__ == "__main__":
    if "--out" in sys.argv and "runs_x_rees46/E1/" not in sys.argv[sys.argv.index("--out") + 1]:
        sys.exit("audit_e1_x: --out must be under runs_x_rees46/E1/")
    raise SystemExit(E.main())
