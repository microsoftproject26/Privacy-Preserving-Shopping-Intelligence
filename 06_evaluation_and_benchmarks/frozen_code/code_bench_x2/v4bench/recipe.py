"""v4bench.recipe — the REES46 arm specifications, mirrored (not imported) from v4int.launch_fl / launch_device.

v4int.launch_fl pulls in the whole GPU launcher stack; the benchmark runner needs only its *semantics*. Each value below is
copied verbatim from launch_fl.py / contracts and a test (tests_bench/test_recipe.py) compares build_fl_spec() with
v4int.launch_fl.arm_spec() whenever that module is importable (the full r6 tree).

Mirrored semantics (all MEASURED from the files, 2026-10-01):
  tuning point  FL_TUNING_SELECTION.json winner: FedAdam eta_s 0.01 (beta 0.9 / 0.99, tau 1e-3, v_0 = tau^2), lr_shape
                FL_CONST, client SGD-m (momentum 0.9) lr 0.1, 1 local pass, local batch 16, clip 1.0, group 64, sweep
                sampling, drop-out 0.1, n_shards 2, max_attempts 2, exposure point "end", endpoint 6.0 EFE.
  rounds        launch_fl.fa_rounds(n_clients, passes=1) = ceil(60 N / (576 P)) (group 64 x 0.9 x P rows per client).
  warm arms     FA_WARM = S7 (m* x 0.25 local factor, server LR x 0.1 -> FedAdam 0.001), FA_WARM_FROZEN = S8 (factor 1,
                server LR x 0.1, frozen item_embed.weight / output_embed / output_bias); both start from theta_pre = the
                PRE endpoint_6.0 checkpoint (PRETRAIN_FOR_FL fl_init_kind endpoint_6.0).
  device        FT_C procedure with the registered cell FT_C_f0.3_P4 (SELECTION_FT_C.json): LR = 0.3 x the SMALL peak,
                4 passes, fresh AdamW, B16, clip 1.0, wd 1e-5.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

from .common import (CENTRAL_RECIPES, CONTRACTS_DEFAULT, EXPLORATORY, FL_VARIANT, PINS, SMALL_PEAK_LR, BenchRefused,
                     sha256_file)

# ---- launch_fl.py constants (verbatim) ------------------------------------------------------------------------
KNOBS = {"group": 64, "sampling": "sweep", "dropout_p": 0.1, "rounds": 6827, "batch": 16, "passes": 2, "mu": 0.01,
         "pf_lam": 1e-4, "pf_clip": 1.0, "clip": 1.0, "weight_decay": 1e-5, "m": 1024, "rounds_1024": 384,
         "s_cal_rounds": 20, "s_cal_seed": 2026, "n_shards": 2, "max_attempts": 2, "endpoint_efe": 6.0}
FEDADAM_HYPER = {"beta1": 0.9, "beta2": 0.99, "tau": 1e-3, "v_init": "tau_squared"}
EXPOSURE_POINT = "end"
Q8_CODEC = {"name": "Q8_SR_PER_TENSOR", "bits": 8, "qmax": 127, "scale": "max|delta| / 127 per tensor (FP32)",
            "rounding": "stochastic, torch.Generator seeded derive_seed(seed, 'q8', round, client_key)",
            "upload_bytes": "sum(numel) + 4 x n_tensors", "download": "dense FP32", "authority": "amendment 06B §5"}
FROZEN_ITEM_KEYS = ("item_embed.weight", "output_embed", "output_bias")
WARM_ARMS = ("FA_WARM", "FA_WARM_FROZEN")
WARM_STAGE = {"FA_WARM": "S7", "FA_WARM_FROZEN": "S8"}
WARM_LINE = {"FP_WARM": ("FP", "FA_WARM"), "FA_Q8_WARM": ("FA_Q8", "FA_WARM"),
             "FP_WARM_FROZEN": ("FP", "FA_WARM_FROZEN"), "FA_Q8_WARM_FROZEN": ("FA_Q8", "FA_WARM_FROZEN"),
             "S_CAL_WARM": ("S_CAL", "FA_WARM"), "FA_1024_WARM": ("FA_1024", "FA_WARM"), "DP8_WARM": ("DP8", "FA_WARM"),
             "S_CAL_WARM_FROZEN": ("S_CAL", "FA_WARM_FROZEN"), "FA_1024_WARM_FROZEN": ("FA_1024", "FA_WARM_FROZEN"),
             "DP8_WARM_FROZEN": ("DP8", "FA_WARM_FROZEN"),
             "PF_WARM": ("PF", "FA_WARM"), "PF_WARM_FROZEN": ("PF", "FA_WARM_FROZEN")}
ALIASES = {"DP_WARM": "DP8_WARM", "DP8_FROZEN": "DP8_WARM_FROZEN", "FA_1024_FROZEN": "FA_1024_WARM_FROZEN",
           "S_CAL_FROZEN": "S_CAL_WARM_FROZEN"}                   # B9 (4): the names the amendment uses
POISSON_ARMS = ("S_CAL", "FA_1024", "DP8")                       # amendment 07 R4 / O2 (the DP chain)
DP = {"q_nominal": 2.0 ** -7, "T": 384, "delta": 5e-6, "z": 0.5642, "s_cal_rounds": 20, "s_cal_seed": 2026}
FL_ARMS = ("FA",) + WARM_ARMS + tuple(WARM_LINE) + ("FA_Q8", "FP", "PF") + ("S_CAL", "FA_1024", "DP8")
CENTRAL_ARMS = ("C_FULL", "PRE", "C_SMALL")   # POST_HOC_X: C_SMALL = the FL phone model (D64_B2) trained centrally on ALL users
X_BUDGETS = (6, 24, 96)                       # POST_HOC_X: the FL exposure budgets (EFE) this tree accepts
FT_ARMS = ("FT_FA", "FT_FA_WARM", "FT_FA_WARM_FROZEN")
ALL_ARMS = CENTRAL_ARMS + FL_ARMS + FT_ARMS
NOT_IMPLEMENTED = ()
FT_CELL = {"cell": "FT_C_f0.3_P4", "lr_frac": 0.3, "passes": 4}          # SELECTION_FT_C.json
FT_BASE_OF = {"FT_FA": "FA", "FT_FA_WARM": "FA_WARM", "FT_FA_WARM_FROZEN": "FA_WARM_FROZEN"}

# the registered warm winners (FL_WARM_SELECTION_S7 / _S8.json "winner"; MEASURED on the server 2026-10-01)
WARM_WINNER = {"FA_WARM": {"label": "f=0.25|sf=0.1", "warm_lr_factor": 0.25, "warm_server_lr_factor": 0.1},
               "FA_WARM_FROZEN": {"label": "f=1|sf=0.1", "warm_lr_factor": 1.0, "warm_server_lr_factor": 0.1}}
TUNING_WINNER = {"label": "fedadam|eta=0.01|m=1|shape=fl_const|client=sgd_m|clr=0.1|passes=1",
                 "optimizer": "fedadam", "server_lr": 0.01, "server_tau": 1e-3, "lr_mult": 1.0, "lr_shape": "fl_const",
                 "client_opt": "sgd_m", "client_lr": 0.1, "local_passes": 1, "local_batch": 16}


def base_arm(arm: str) -> str:
    return WARM_LINE[arm][0] if arm in WARM_LINE else arm


def warm_variant(arm: str) -> Optional[str]:
    if arm in WARM_ARMS:
        return arm
    return WARM_LINE[arm][1] if arm in WARM_LINE else None


def is_poisson_arm(arm: str) -> bool:
    return base_arm(arm) in POISSON_ARMS


def dp_cohort(n_clients: int) -> int:
    """R4: the fixed DP denominator m = round(q N), q = 2^-7 (ML-1M 47 / Beauty 175 / Toys 152 / Sports 278)."""
    return int(round(DP["q_nominal"] * int(n_clients)))


def is_frozen_arm(arm: str) -> bool:
    return warm_variant(arm) == "FA_WARM_FROZEN"


def fa_rounds(n_clients: int, passes: int = 1, efe: int = 6) -> int:
    """launch_fl.fa_rounds: ceil(6.0 x N_dec / (64 x 0.9 x P x N_dec / N)) = ceil(60 N / (576 P)).
    POST_HOC_X: the same formula at an E-EFE budget, ceil(10 E N / (576 P)) (identical integers at E = 6)."""
    return -(-(10 * int(efe) * int(n_clients)) // (KNOBS["group"] * 9 * int(passes)))


def check_pins(contracts_dir=CONTRACTS_DEFAULT, *, required: bool) -> dict:
    """sha256 of the registered contract files (present on the server; absent in a bare CPU test tree)."""
    out = {}
    for rel, want in PINS.items():
        p = Path(contracts_dir) / rel
        if not p.is_file():
            if required:
                raise BenchRefused(f"registered contract {p} is absent")
            out[rel] = None
            continue
        got = sha256_file(p)
        if got != want:
            raise BenchRefused(f"registered contract {rel} sha256 {got[:12]}… != the pinned {want[:12]}…")
        out[rel] = got
    return out


def check_mirrors(contracts_dir=CONTRACTS_DEFAULT) -> dict:
    """M6: at run time prove that the mirrored constants equal the pinned files: the four sha256 pins, the S7 / S8
    winners, the FL_TUNING_SELECTION winner and the SELECTION_FT_C cell. Raises when a file is absent or differs."""
    rec = {"pins": check_pins(contracts_dir, required=True), "warm_winners": check_warm_winners(contracts_dir)}
    if any(v is None for v in rec["warm_winners"].values()):
        raise BenchRefused("FL_WARM_SELECTION_S7 / S8 files are required for a real run")
    tun = json.loads((Path(contracts_dir) / "fl" / "FL_TUNING_SELECTION.json").read_text(encoding="utf-8"))["winner"]
    for k, v in TUNING_WINNER.items():
        if k != "label" and tun.get(k) != v:
            raise BenchRefused(f"FL_TUNING_SELECTION winner {k} = {tun.get(k)!r} != the mirrored {v!r}")
    ft = json.loads((Path(contracts_dir) / "device" / "SELECTION_FT_C.json").read_text(encoding="utf-8"))
    if (ft["cell"], ft["cells"]["lr_frac"], ft["cells"]["passes"]) != (FT_CELL["cell"], FT_CELL["lr_frac"], FT_CELL["passes"]):
        raise BenchRefused("SELECTION_FT_C cell differs from the mirrored FT_C_f0.3_P4")
    rec["tuning_winner_label"] = tun["label"]
    rec["ft_cell"] = ft["cell"]
    return rec


def check_warm_winners(contracts_dir=CONTRACTS_DEFAULT) -> dict:
    """Server-side proof that WARM_WINNER equals the registered selection files' winners."""
    out = {}
    for arm, stage in WARM_STAGE.items():
        p = Path(contracts_dir) / "fl" / f"FL_WARM_SELECTION_{stage}.json"
        if not p.is_file():
            out[arm] = None
            continue
        w = json.loads(p.read_text(encoding="utf-8"))["winner"]
        mine = WARM_WINNER[arm]
        if (w["warm_lr_factor"], w["warm_server_lr_factor"]) != (mine["warm_lr_factor"], mine["warm_server_lr_factor"]):
            raise BenchRefused(f"{p.name} winner {w} != the mirrored {mine}")
        out[arm] = w["label"]
    return out


def build_fl_spec(arm: str, *, seed: int, n_clients: int, init_from: Optional[Mapping] = None,
                  clip_norm: Optional[float] = None, budget_efe: int = 6, server_lr: Optional[float] = None,
                  client_lr: Optional[float] = None, local_passes: Optional[int] = None,
                  server_tau: Optional[float] = None) -> dict:
    """The registered configuration of one FL arm (plan + fl.runtime.RunConfig kwargs), launch_fl.arm_spec semantics
    at the FL_TUNING_SELECTION winner (client SGD-m 0.1, FedAdam 0.01, FL_CONST, 1 local pass, B16)."""
    if arm in NOT_IMPLEMENTED:
        raise BenchRefused(f"{arm} is not implemented in BM-02 phase 2 (see the handoff open questions)")
    if arm not in FL_ARMS:
        raise BenchRefused(f"unknown FL arm {arm}")
    t = TUNING_WINNER
    wv = warm_variant(arm)
    if (wv is not None) != (init_from is not None):
        raise BenchRefused(f"{arm}: " + ("a warm arm needs init_from (theta_pre)" if wv else "a cold arm takes no init_from"))
    K = KNOBS
    base = float(SMALL_PEAK_LR)
    poisson = is_poisson_arm(arm)
    budget_efe = int(budget_efe)
    if budget_efe not in X_BUDGETS:
        raise BenchRefused(f"POST_HOC_X: the FL budget must be one of {X_BUDGETS} EFE")
    if poisson and (budget_efe != 6 or server_lr is not None or client_lr is not None or local_passes is not None
                    or server_tau is not None):
        raise BenchRefused("POST_HOC_X: the budget / LR / passes overrides are for the sweep arms (FA, FA_WARM*), "
                           "not the Poisson DP chain")
    passes = K["passes"] if poisson else t["local_passes"]       # DL 219: the Poisson arms keep P = 2, B = 16
    if local_passes is not None:
        if int(local_passes) < 1:
            raise BenchRefused("--local-passes must be >= 1")
        passes = int(local_passes)
    rounds = DP["T"] if poisson else fa_rounds(n_clients, passes, budget_efe)
    ca = base_arm(arm)
    solver = {"lr": t["client_lr"], "batch_size": K["batch"], "passes": passes, "clip": K["clip"],
              "weight_decay": 0.0, "fused": None, "optimizer": "sgd_m"}
    cfg = {"method": ca if ca in ("FP", "PF") else "FA", "seed": int(seed), "lr_peak_local": t["client_lr"], "solver": solver,
           "n_shards": K["n_shards"], "max_attempts": K["max_attempts"], "endpoint_efe": float(budget_efe),
           "exposure_point": EXPOSURE_POINT, "mu": K["mu"] if ca == "FP" else 0.0,
           "pf": {"lr": base, "lam": K["pf_lam"], "clip": K["pf_clip"]} if ca == "PF" else None, "dp": None,
           "dropout_p": 0.0 if poisson else K["dropout_p"], "endpoint_rounds": rounds,
           "server_opt": {"name": "fedadam", "lr": t["server_lr"], **FEDADAM_HYPER, "tau": t["server_tau"]},
           "lr_shape": t["lr_shape"]}
    plan = {"group_size": K["group"], "sampling": "sweep"}
    max_rounds = rounds
    if poisson:
        m = dp_cohort(n_clients)
        plan = {"group_size": m, "sampling": "poisson"}
        if ca == "S_CAL":
            if int(seed) not in (2026, 2027):                  # B12: S_CAL runs once PER PRE SEED
                raise BenchRefused("S_CAL runs at seeds 2026 and 2027 (one per PRE seed, addendum B12)")
            cfg["dp"] = {"clip_norm": None, "noise_multiplier": 0.0, "denominator": m}
            max_rounds = DP["s_cal_rounds"]
        else:
            if clip_norm is None:
                raise BenchRefused(f"{arm} needs the S record (--s-record)")
            cfg["dp"] = {"clip_norm": float(clip_norm), "noise_multiplier": DP["z"] if ca == "DP8" else 0.0,
                         "denominator": m}
    spec = {"arm": arm, "plan": plan, "cfg": cfg, "max_rounds": max_rounds,
            "knobs": dict(K), "lr_mult": 1.0, "lr_peak_base": base, "role": "MAIN",
            "upload_codec": dict(Q8_CODEC) if ca == "FA_Q8" else None,
            "server_opt": "fedadam", "server_lr": t["server_lr"], "lr_shape": t["lr_shape"], "client_opt": "sgd_m",
            "client_lr": t["client_lr"], "local_passes": passes}
    if poisson:
        spec["dp_accounting"] = {**{k: DP[k] for k in ("q_nominal", "T", "delta", "z")}, "m": plan["group_size"],
                                 "N": int(n_clients), "q_actual": DP["q_nominal"],
                                 "inclusion": "independent per client with probability EXACTLY 2^-7 (fl_engine.ExactQPlan); "
                                              "m = round(q N) is the fixed denominator only (B10 Q1)",
                                 "authority": "PREG1_AMENDMENT_07 R4 (registered z for identical q, T, delta)"}
        spec["poisson_local_rule"] = {"selected_local_passes": 1, "local_passes": 2, "local_batch": 16, "authority": "DL 219"}
    else:
        spec["exposure_budget"] = {"endpoint_efe": float(budget_efe), "n_clients": int(n_clients), "passes": passes,
                                   "group": K["group"], "dropout_p": K["dropout_p"], "rounds": rounds,
                                   "formula": "rounds = ceil(10 E N_clients / (576 P)) (launch_fl.fa_rounds at E EFE; "
                                              "= ceil(60 N / (576 P)) at E = 6)"}
        if budget_efe != 6:                       # 12 evaluations per run, every E / 12 EFE (= the B6 cadence at E = 6)
            cfg["eval_every_half_efe"] = budget_efe // 6
    if wv is not None:
        w = WARM_WINNER[wv]
        spec["init_from"] = dict(init_from)
        wf, wsf = w["warm_lr_factor"], w["warm_server_lr_factor"]
        if wf != 1.0:
            cfg["lr_peak_local"] = wf * cfg["lr_peak_local"]
            cfg["solver"]["lr"] = cfg["lr_peak_local"]
            spec["warm_lr_factor"] = wf
        if wsf != 1.0:                                   # launch_fl.warm_server_opt: FedAdam eta* / 10, tau unchanged
            div = {0.1: 10, 0.25: 4}[float(wsf)]
            cfg["server_opt"] = {"name": "fedadam", "lr": float(t["server_lr"]) / div, **FEDADAM_HYPER,
                                 "tau": t["server_tau"]}
            spec["warm_server_lr_factor"] = wsf
        if is_frozen_arm(arm):
            spec["frozen_item_tables"] = list(FROZEN_ITEM_KEYS)
    # ---- POST_HOC_X overrides: applied LAST, they set the EFFECTIVE value (a warm arm's factors are then bypassed)
    ov = {}
    if server_lr is not None:
        cfg["server_opt"] = dict(cfg["server_opt"], lr=float(server_lr))
        ov["server_lr"] = float(server_lr)
    if server_tau is not None:                    # FedAdam's tau (v_0 = tau^2 follows it; ServerOptConfig validates > 0)
        cfg["server_opt"] = dict(cfg["server_opt"], tau=float(server_tau))
        ov["server_tau"] = float(server_tau)
    if client_lr is not None:
        cfg["lr_peak_local"] = float(client_lr)
        cfg["solver"]["lr"] = float(client_lr)
        ov["client_lr"] = float(client_lr)
    if local_passes is not None:
        ov["local_passes"] = int(local_passes)
    if budget_efe != 6:
        ov["budget_efe"] = budget_efe
    spec["exploratory"] = EXPLORATORY
    spec["budget_efe"] = budget_efe
    spec["overrides"] = ov
    spec["effective"] = {"server_lr": float(cfg["server_opt"]["lr"]), "server_tau": float(cfg["server_opt"]["tau"]),
                         "client_lr": float(cfg["solver"]["lr"]),
                         "local_passes": int(cfg["solver"]["passes"]), "rounds": int(rounds),
                         "endpoint_efe": float(cfg["endpoint_efe"]), "eval_every_half_efe": int(cfg.get("eval_every_half_efe", 1))}
    return spec


def central_recipe(arm: str) -> dict:
    """C_FULL = 02B_SASREC (SASREC_D256_B3); PRE = SMALL_SASREC_D64_B2 (the REES46 PRETRAIN_FOR_FL recipe)."""
    key = {"C_FULL": "02B_SASREC", "PRE": "SMALL_SASREC_D64_B2", "C_SMALL": "SMALL_SASREC_D64_B2"}[arm]
    return dict(CENTRAL_RECIPES[key], recipe_id=key)


def central_schedule_note() -> str:
    return ("S0 EFE schedule (v4train.schedule), exposure point end, AdamW(0.9, 0.999, 1e-8, wd 1e-5 on ndim>=2), "
            "effective batch 256, clip 1.0, 6.0 EFE, evaluation every 0.5 EFE, PRACTICAL_BEST = strictly-best (> best + "
            "0.0002) inner-validation NDCG@10, CONTROLLED_BUDGET = the 6.0 mark")


def ft_recipe_kwargs() -> dict:
    """FTRecipe kwargs of the registered cell (arm FT_C procedure; peak_lr = the SMALL peak)."""
    return {"arm": "FT_C", "peak_lr": float(SMALL_PEAK_LR), "lr_frac": FT_CELL["lr_frac"],
            "passes": (FT_CELL["passes"],), "batch_size": 16}
