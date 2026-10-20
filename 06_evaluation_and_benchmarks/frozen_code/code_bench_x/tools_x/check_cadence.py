"""POST_HOC_X check (no training): for an FL arm at budgets 6 / 24 / 96 build the spec, the participation plan and the
planned LR table exactly as fl_engine.run_fl does, and count the evaluation marks the runtime would fire."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from fl.participation import ParticipationPlan  # noqa: E402
from fl.runtime import RunConfig, build_lr_table  # noqa: E402
from v4bench import data as D, recipe  # noqa: E402
from v4bench.common import default_data_root  # noqa: E402
from v4bench.fl_engine import client_keys  # noqa: E402

ds = sys.argv[1]
d = D.load(ds, default_data_root(ds))
users, keys, n_valid = client_keys(d)
N = int(sum(n_valid.values()))
for B in (6, 24, 96):
    spec = recipe.build_fl_spec("FA", seed=2026, n_clients=len(keys), budget_efe=B)
    cfg = RunConfig.from_dict(dict(spec["cfg"], run_id="x", ckpt_every=64))
    plan = ParticipationPlan.build(keys, seed=2026, manifest_hash=d.fingerprint(), group_size=64, sampling="sweep")
    tab = build_lr_table(plan, n_valid, N, cfg)
    units = int(2 * cfg.endpoint_efe)
    fired, prev = [], 0
    for i, row in enumerate(tab["rows"]):
        u = (units * int(row[3])) // int(tab["E_total"])
        if u > prev and u % cfg.eval_every_half_efe == 0:
            fired.append(u / 2.0)
        prev = u
    print(f"{ds} B{B}: rounds={spec['effective']['rounds']} (B6 x {spec['effective']['rounds'] / recipe.fa_rounds(len(keys)):.3f}) "
          f"eval_every_half_efe={cfg.eval_every_half_efe} n_evals={len(fired)} marks={fired} "
          f"E_total/N={tab['E_total'] / N:.3f} exploratory={spec['exploratory']}")
