"""X-REES46 part-B scorer (POST_HOC_EXPLORATORY, DL 539): the UNCHANGED v4int.reeval_p2 (from code_x1, a byte copy of
code_p2r4) for the X_FA runs (x_rees46/launch_fl_x.py: P2_W, FedAdam tau 1e-6, a recipe file, run dir under
runs_x_rees46/MAIN). Run ONLY through x_rees46/reeval_x.py (it checks the EXP lock and its register entry first and
exports X_LOCK / X_LOCK_SHA). In-process changes, nothing else:
  (1) arm label X_FA admitted as a P2_W FL arm, seeds 2026 / 2027, run id = X_RUN_ID_FMT (from the EXP lock)
  (2) launch_fl.p2_server_opt -> FedAdam tau X_TAU (so launch_fl.check_p2_spec re-derives the X spec exactly)
  (3) check_launch_selection -> the run's LAUNCH p2_recipe record = the lock-bound recipe file (sha256, label); a run
      bound to the registered P2_WARM_SELECTION_FINAL (W*) is refused here
  (4) p2_test_gate -> the DATA-09 gate (unchanged) + the EXP lock register binding + the arm bound in the EXP lock
  (5) every checkpoint must be under runs_x_rees46/MAIN (or runs_x_rees46/PB100/MAIN for X2)
  (6) X2 only (env X_INIT_SHA / X_INIT_DIGEST from the EXP lock): the init pin is theta_pre(100k), not theta_pre"""
import json
import os
import sys
from pathlib import Path

for k in ("X_CODE", "X_TAU", "X_RUN_ID_FMT", "X_RECIPE_SHA", "X_LOCK", "X_LOCK_SHA"):
    if not os.environ.get(k):
        sys.exit(f"reeval_p2_x: env {k} missing (run through x_rees46/reeval_x.py)")
X_TAU = float(os.environ["X_TAU"])
LABEL = "X_FA"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ["X_CODE"])
sys.path.insert(1, str(HERE))
from v4int import reeval_p2 as RP  # noqa: E402
import v4int.launch_fl as LF  # noqa: E402
import reeval_x as RX  # noqa: E402

_so = LF.p2_server_opt


def p2_server_opt_x(rec):
    so = _so(rec)
    return dict(so, tau=X_TAU) if so is not None and so.get("name") == "fedadam" else so


LF.p2_server_opt = p2_server_opt_x
if os.environ.get("X_INIT_SHA"):                    # X2 (PB100): the run's init is theta_pre(100k), bound by the EXP lock
    RP.PRE_CHECKPOINT_SHA256 = os.environ["X_INIT_SHA"]
    RP.PRE_STATE_DIGEST = os.environ["X_INIT_DIGEST"]
    import x2_admissions as X2A                      # the launch_fl_x2 admissions of THIS run's spec (proofC fix)
    if "--ckpt" not in sys.argv:
        sys.exit("reeval_p2_x: X2 mode needs --ckpt")
    _e, _p = X2A.run_efe_passes(sys.argv[sys.argv.index("--ckpt") + 1])
    for _k, _v in (("X_EFE", _e), ("X_PASSES", _p)):
        if os.environ.get(_k) and float(os.environ[_k]) != float(_v):
            sys.exit(f"reeval_p2_x: the run's {_k} {_v} is not the EXP lock's {os.environ[_k]}")
    X2A.apply(LF, _e, _p)
RP.ARMS[LABEL] = {"kind": "FL", "seeds": (2026, 2027), "launch_arm": "P2_W", "run_id": os.environ["X_RUN_ID_FMT"]}


def check_launch_recipe(launch, spec):
    if launch.get("p2_selection") is not None:
        RP._refuse("an X_FA run is a recipe-file run; this run is bound to a registered selection (W*), refused")
    src = launch.get("p2_recipe")
    if not isinstance(src, dict):
        RP._refuse("the X_FA run's LAUNCH carries no p2_recipe source record")
    if src.get("sha256") != os.environ["X_RECIPE_SHA"] or src.get("label") != spec.get("p2_label"):
        RP._refuse("the X_FA run's recipe file (sha256 / label) is not the EXP-lock-bound recipe")
    if float(spec["cfg"]["server_opt"]["tau"]) != X_TAU:
        RP._refuse("the X_FA run's FedAdam tau is not the EXP-lock tau")
    return {"p2_recipe": dict(src), "tau": X_TAU, "label": "POST_HOC_EXPLORATORY"}


RP.check_launch_selection = check_launch_recipe


def p2_test_gate_x(args, plan):
    from v4data09 import d9_gate as G
    if not RP.R.test_gate_ok(args.role):
        RP._refuse(f"TEST role {args.role} refused: the DATA-09 TEST gate is closed")
    g = G.check_test_gate(RP.AML_PATH, RP.REGISTER_PATH)
    lp, ls = Path(os.environ["X_LOCK"]), os.environ["X_LOCK_SHA"].lower()
    if RX.sha_file(lp) != ls:
        RP._refuse("the EXP lock changed since reeval_x checked it")
    lock = RX.strict(lp)
    b = RX.register_binding(ls, lock["document"])
    hits = [e for e in lock["sealed_arms"] if e["arm_label"] == LABEL and int(e["seed"]) == int(args.seed)
            and e["rule"] == args.selection_rule and e["checkpoint_sha256"] == str(args.expect_sha256)]
    if len(hits) != 1:
        RP._refuse(f"the EXP lock binds {len(hits)} entries for ({LABEL}, {args.seed}, {args.selection_rule})")
    return {"data09_gate": g.binding(), "exp_lock": {"path": str(lp), "sha256": ls}, "register_binding": b,
            "arm_binding": hits[0]["key"], "label": "POST_HOC_EXPLORATORY"}


RP.p2_test_gate = p2_test_gate_x

if __name__ == "__main__":
    if "--ckpt" in sys.argv:
        ck = Path(sys.argv[sys.argv.index("--ckpt") + 1]).resolve()
        if (RX.XR / "MAIN").resolve() not in ck.parents and (RX.XR / "PB100" / "MAIN").resolve() not in ck.parents:
            sys.exit("reeval_p2_x: the checkpoint is not under runs_x_rees46/MAIN")
    raise SystemExit(RP.main())
