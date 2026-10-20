"""X2 admissions for SCORING / AUDIT (POST_HOC_EXPLORATORY): exactly the in-process launch_fl changes of
x_rees46/launch_fl_x2.py that enter a P2_W spec, re-applied so launch_fl.check_p2_spec / check_p2_dp_spec re-derive a
PB100 X2 run's own spec (the check itself is unchanged and still compares the whole spec):
  P2_CLIENT_LRS + {x2: (0.3,)}, P2_ITEM_LR_MULTS + (1.0,), P2_FIXED local_passes = P, KNOBS endpoint_efe = E,
  exposure_budget -> rounds ceil(10 E N / (576 P)) with the launch_fl_x2 formula string, FedAdam tau 1e-6.
E and P come from the run's OWN LAUNCH arm spec (cfg.endpoint_efe, cfg.solver.passes) and, when the EXP lock gives
them (scorer_env X_EFE / X_PASSES), must equal the lock's. snapshot() / restore() make the change temporary."""
import json
from pathlib import Path

X_TAU = 1e-6


def run_efe_passes(ckpt) -> tuple:
    run_dir = Path(ckpt).resolve().parent.parent.parent
    spec = json.loads((run_dir / "LAUNCH_0.json").read_text(encoding="utf-8"))["arm_spec"]
    return float(spec["cfg"]["endpoint_efe"]), int(spec["cfg"]["solver"]["passes"])


def snapshot(F):
    return (F.P2_CLIENT_LRS, F.P2_ITEM_LR_MULTS, F.P2_FIXED["local_passes"], F.KNOBS["endpoint_efe"],
            F.exposure_budget, F.p2_server_opt)


def restore(F, s):
    F.P2_CLIENT_LRS, F.P2_ITEM_LR_MULTS = s[0], s[1]
    F.P2_FIXED["local_passes"], F.KNOBS["endpoint_efe"] = s[2], s[3]
    F.exposure_budget, F.p2_server_opt = s[4], s[5]


def apply(F, efe: float, passes: int) -> dict:
    if efe not in (6.0, 12.0, 24.0) or passes not in (1, 2):
        raise SystemExit(f"x2_admissions: E {efe} / P {passes} outside the X2 grid")
    base_eb, base_so = F.exposure_budget, F.p2_server_opt
    F.P2_CLIENT_LRS = dict(F.P2_CLIENT_LRS, x2=(0.3,))
    F.P2_ITEM_LR_MULTS = tuple(F.P2_ITEM_LR_MULTS) + (1.0,)
    F.P2_FIXED["local_passes"] = passes
    F.KNOBS["endpoint_efe"] = efe

    def exposure_budget_x(n_clients, p):              # = launch_fl_x2.exposure_budget_x
        eb = base_eb(n_clients, p)
        eb["rounds"] = -(-(int(round(10 * efe)) * int(n_clients)) // (F.KNOBS["group"] * 9 * int(p)))
        eb["endpoint_efe"] = efe
        eb["formula"] = f"X2: rounds = ceil(10 x {efe:g} x N / (576 P)) (POST_HOC_EXPLORATORY budget)"
        return eb

    def p2_server_opt_x(rec):                          # = launch_fl_x.p2_server_opt_x
        so = base_so(rec)
        return dict(so, tau=X_TAU) if so is not None and so.get("name") == "fedadam" else so

    F.exposure_budget, F.p2_server_opt = exposure_budget_x, p2_server_opt_x
    return {"x2_admissions": True, "efe": efe, "passes": passes, "tau": X_TAU}
