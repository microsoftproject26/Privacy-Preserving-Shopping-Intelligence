"""The arms of the benchmark runner and their recipe (hyperparameters), loaded from a JSON file.

Arms:
  central   C_FULL (central training on every user's TRAIN decisions) and PRE (central pretraining on the band users
            only; its endpoint is the warm start of the warm federated arms);
  federated FA (FedAvg-family run of the small on-device model), FA_WARM (FA started from PRE), FA_WARM_FROZEN (FA_WARM
            with the item tables frozen), FP (FedProx), PF (personalised FedAvg), FA_Q8 (8-bit uploads), and the DP
            chain S_CAL (clip-norm calibration on the unclipped trajectory), FA_1024 (Poisson sampling with clipping,
            no noise) and DP8 (DP-FedAvg); every federated arm except FA has warm and warm-frozen variants;
  device    FT_FA / FT_FA_WARM / FT_FA_WARM_FROZEN: per-user on-device fine-tuning of a finished federated base model.

The recipe holds every hyperparameter the arms use. `DEFAULT_RECIPE` has illustrative values only (not tuned for any
dataset); pass your own with `--recipe recipe.json` (a partial file is merged over the defaults). Structure:
  central   per central arm: model_variant, peak_lr, dropout
  fl        model_variant, lr_peak_base (also the personal-vector and fine-tuning reference LR), server_opt
            ("fedadam" | "fedavgm" | null = FedAvg), server_lr, server_tau / server_momentum, lr_shape ("fl_const" |
            "central"), client_lr (SGD with momentum), local_passes, local_batch
  warm      per warm arm: warm_lr_factor (client LR factor), warm_server_lr_factor (server LR factor)
  ft        lr_frac (x lr_peak_base), passes, batch_size
  dp        q_nominal (Poisson inclusion probability), T (rounds), delta, z (noise multiplier; take it from the
            accountant in ppsi.fedsim for your privacy target), s_cal_rounds
  knobs     group (clients per round), dropout_p, poisson_passes, mu (FedProx), pf_lam / pf_clip, clip, n_shards,
            max_attempts, endpoint_efe
Rounds of the sweep arms: ceil(endpoint_efe x N_clients / (group x (1 - dropout_p) x passes)), i.e. every client's
data is seen endpoint_efe times in expectation.
"""
from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

from .common import BenchRefused, canon_sha256

DEFAULT_RECIPE = {
    "central": {"C_FULL": {"model_variant": "SASREC_D256_B2", "peak_lr": 1e-3, "dropout": [0.2]},
                "PRE": {"model_variant": "SASREC_D64_B2", "peak_lr": 1e-3, "dropout": [0.2]}},
    "fl": {"model_variant": "SASREC_D64_B2", "lr_peak_base": 1e-3, "server_opt": None, "server_lr": 1.0,
           "server_tau": 1e-3, "server_momentum": 0.9, "lr_shape": "fl_const", "client_lr": 0.05,
           "local_passes": 1, "local_batch": 16},
    "warm": {"FA_WARM": {"warm_lr_factor": 1.0, "warm_server_lr_factor": 1.0},
             "FA_WARM_FROZEN": {"warm_lr_factor": 1.0, "warm_server_lr_factor": 1.0}},
    "ft": {"lr_frac": 0.1, "passes": 1, "batch_size": 16},
    "dp": {"q_nominal": 2.0 ** -7, "T": 384, "delta": 5e-6, "z": 1.0, "s_cal_rounds": 20},
    "knobs": {"group": 64, "dropout_p": 0.1, "poisson_passes": 2, "mu": 0.01, "pf_lam": 1e-4, "pf_clip": 1.0,
              "clip": 1.0, "n_shards": 2, "max_attempts": 2, "endpoint_efe": 6.0},
}
FEDADAM_HYPER = {"beta1": 0.9, "beta2": 0.99, "v_init": "tau_squared"}
EXPOSURE_POINT = "end"
Q8_CODEC = {"name": "Q8_SR_PER_TENSOR", "bits": 8, "qmax": 127, "scale": "max|delta| / 127 per tensor (FP32)",
            "rounding": "stochastic, torch.Generator seeded derive_seed(seed, 'q8', round, client_key)",
            "upload_bytes": "sum(numel) + 4 x n_tensors", "download": "dense FP32"}
FROZEN_ITEM_KEYS = ("item_embed.weight", "output_embed", "output_bias")
WARM_ARMS = ("FA_WARM", "FA_WARM_FROZEN")
BASE_FL_ARMS = ("FP", "FA_Q8", "S_CAL", "FA_1024", "DP8", "PF")
WARM_LINE = {f"{b}_{w[3:]}": (b, w) for b in BASE_FL_ARMS for w in WARM_ARMS}     # e.g. FP_WARM -> (FP, FA_WARM)
POISSON_ARMS = ("S_CAL", "FA_1024", "DP8")
FL_ARMS = ("FA", *WARM_ARMS, *WARM_LINE, "FA_Q8", "FP", "PF", "S_CAL", "FA_1024", "DP8")
CENTRAL_ARMS = ("C_FULL", "PRE")
FT_ARMS = ("FT_FA", "FT_FA_WARM", "FT_FA_WARM_FROZEN")
ALL_ARMS = CENTRAL_ARMS + FL_ARMS + FT_ARMS
FT_BASE_OF = {"FT_FA": "FA", "FT_FA_WARM": "FA_WARM", "FT_FA_WARM_FROZEN": "FA_WARM_FROZEN"}
SERVER_OPTS = (None, "fedadam", "fedavgm")


def _merge(base: dict, over: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in out:
            raise BenchRefused(f"unknown recipe key {k!r}")
        if isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def check_recipe(r: Mapping) -> dict:
    fl = r["fl"]
    if fl["server_opt"] not in SERVER_OPTS:
        raise BenchRefused(f"fl.server_opt must be one of {SERVER_OPTS}")
    for path, v in (("fl.lr_peak_base", fl["lr_peak_base"]), ("fl.client_lr", fl["client_lr"]),
                    ("fl.server_lr", fl["server_lr"]), ("dp.z", r["dp"]["z"])):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
            raise BenchRefused(f"{path} must be a finite number >= 0")
    for arm, c in r["central"].items():
        if arm not in CENTRAL_ARMS or not (c["peak_lr"] > 0):
            raise BenchRefused(f"central.{arm}: unknown arm or a non-positive peak_lr")
    for arm, w in r["warm"].items():
        if arm not in WARM_ARMS or not (w["warm_lr_factor"] > 0 and w["warm_server_lr_factor"] > 0):
            raise BenchRefused(f"warm.{arm}: unknown warm arm or a non-positive factor")
    if not 0.0 <= float(r["knobs"]["dropout_p"]) < 1.0:
        raise BenchRefused("knobs.dropout_p must be in [0, 1)")
    return dict(r)


def load_recipe(path=None) -> dict:
    """DEFAULT_RECIPE, with the JSON file at `path` (if given) merged over it; validated, with its sha256."""
    r = copy.deepcopy(DEFAULT_RECIPE)
    if path is not None:
        r = _merge(r, json.loads(Path(path).read_text(encoding="utf-8")))
    r = check_recipe(r)
    r["recipe_sha256"] = canon_sha256({k: v for k, v in r.items() if k != "recipe_sha256"})
    return r


def base_arm(arm: str) -> str:
    return WARM_LINE[arm][0] if arm in WARM_LINE else arm


def warm_variant(arm: str) -> str | None:
    if arm in WARM_ARMS:
        return arm
    return WARM_LINE[arm][1] if arm in WARM_LINE else None


def is_poisson_arm(arm: str) -> bool:
    return base_arm(arm) in POISSON_ARMS


def is_frozen_arm(arm: str) -> bool:
    return warm_variant(arm) == "FA_WARM_FROZEN"


def dp_cohort(n_clients: int, recipe: Mapping) -> int:
    """The fixed DP denominator m = round(q N)."""
    return round(recipe["dp"]["q_nominal"] * int(n_clients))


def fa_rounds(n_clients: int, passes: int, recipe: Mapping) -> int:
    """ceil(endpoint_efe x N / (group x (1 - dropout_p) x passes)), in exact decimal arithmetic."""
    k = recipe["knobs"]
    per_round = Fraction(int(k["group"])) * (1 - Fraction(str(k["dropout_p"]))) * int(passes)
    return math.ceil(Fraction(str(k["endpoint_efe"])) * int(n_clients) / per_round)


def _server_opt(fl: Mapping, factor: float = 1.0) -> dict | None:
    if fl["server_opt"] is None:
        return None
    lr = float(fl["server_lr"])
    if factor != 1.0:
        div = 1.0 / float(factor)
        lr = lr / round(div) if abs(div - round(div)) < 1e-12 else lr * float(factor)
    if fl["server_opt"] == "fedadam":
        return {"name": "fedadam", "lr": lr, **FEDADAM_HYPER, "tau": float(fl["server_tau"])}
    return {"name": "fedavgm", "lr": lr, "beta1": float(fl["server_momentum"])}


def build_fl_spec(arm: str, *, seed: int, n_clients: int, recipe: Mapping, init_from: Mapping | None = None,
                  clip_norm: float | None = None) -> dict:
    """The configuration of one federated arm: the participation plan and the ppsi.fedsim RunConfig kwargs."""
    if arm not in FL_ARMS:
        raise BenchRefused(f"unknown federated arm {arm}")
    wv = warm_variant(arm)
    if (wv is not None) != (init_from is not None):
        raise BenchRefused(f"{arm}: " + ("a warm arm needs init_from (theta_pre)" if wv else
                                         "a cold arm takes no init_from"))
    fl, K, dp = recipe["fl"], recipe["knobs"], recipe["dp"]
    base = float(fl["lr_peak_base"])
    poisson = is_poisson_arm(arm)
    passes = int(K["poisson_passes"]) if poisson else int(fl["local_passes"])
    rounds = int(dp["T"]) if poisson else fa_rounds(n_clients, passes, recipe)
    ca = base_arm(arm)
    solver = {"lr": float(fl["client_lr"]), "batch_size": int(fl["local_batch"]), "passes": passes,
              "clip": K["clip"], "weight_decay": 0.0, "fused": None, "optimizer": "sgd_m"}
    cfg = {"method": ca if ca in ("FP", "PF") else "FA", "seed": int(seed), "lr_peak_local": float(fl["client_lr"]),
           "solver": solver, "n_shards": K["n_shards"], "max_attempts": K["max_attempts"],
           "endpoint_efe": K["endpoint_efe"], "exposure_point": EXPOSURE_POINT, "mu": K["mu"] if ca == "FP" else 0.0,
           "pf": {"lr": base, "lam": K["pf_lam"], "clip": K["pf_clip"]} if ca == "PF" else None, "dp": None,
           "dropout_p": 0.0 if poisson else K["dropout_p"], "endpoint_rounds": rounds,
           "server_opt": _server_opt(fl), "lr_shape": fl["lr_shape"]}
    plan = {"group_size": K["group"], "sampling": "sweep"}
    max_rounds = rounds
    if poisson:
        m = dp_cohort(n_clients, recipe)
        plan = {"group_size": m, "sampling": "poisson"}
        if ca == "S_CAL":
            cfg["dp"] = {"clip_norm": None, "noise_multiplier": 0.0, "denominator": m}
            max_rounds = int(dp["s_cal_rounds"])
        else:
            if clip_norm is None:
                raise BenchRefused(f"{arm} needs the S record of a matching S_CAL run (--s-record)")
            cfg["dp"] = {"clip_norm": float(clip_norm), "noise_multiplier": float(dp["z"]) if ca == "DP8" else 0.0,
                         "denominator": m}
    spec = {"arm": arm, "plan": plan, "cfg": cfg, "max_rounds": max_rounds, "knobs": dict(K),
            "lr_peak_base": base, "upload_codec": dict(Q8_CODEC) if ca == "FA_Q8" else None,
            "server_opt": fl["server_opt"], "server_lr": fl["server_lr"], "lr_shape": fl["lr_shape"],
            "client_opt": "sgd_m", "client_lr": fl["client_lr"], "local_passes": passes,
            "recipe_sha256": recipe.get("recipe_sha256")}
    if poisson:
        spec["dp_accounting"] = {**{k: dp[k] for k in ("q_nominal", "T", "delta", "z")}, "m": plan["group_size"],
                                 "N": int(n_clients),
                                 "inclusion": "independent per client with probability exactly q; m = round(q N) is "
                                              "the fixed denominator only"}
    else:
        spec["exposure_budget"] = {"endpoint_efe": K["endpoint_efe"], "n_clients": int(n_clients), "passes": passes,
                                   "group": K["group"], "dropout_p": K["dropout_p"], "rounds": rounds,
                                   "formula": "ceil(endpoint_efe x N_clients / (group x (1 - dropout_p) x passes))"}
    if wv is not None:
        w = recipe["warm"][wv]
        spec["init_from"] = dict(init_from)
        wf, wsf = float(w["warm_lr_factor"]), float(w["warm_server_lr_factor"])
        if wf != 1.0:
            cfg["lr_peak_local"] = wf * cfg["lr_peak_local"]
            cfg["solver"]["lr"] = cfg["lr_peak_local"]
            spec["warm_lr_factor"] = wf
        if wsf != 1.0:
            cfg["server_opt"] = _server_opt(fl, wsf)
            spec["warm_server_lr_factor"] = wsf
        if is_frozen_arm(arm):
            spec["frozen_item_tables"] = list(FROZEN_ITEM_KEYS)
    return spec


def central_recipe(arm: str, recipe: Mapping) -> dict:
    return dict(recipe["central"][arm], recipe_id=arm)


def ft_recipe_kwargs(recipe: Mapping) -> dict:
    """ppsi.fedsim.finetune.FTRecipe kwargs of the device arms (procedure FT_C: fine-tune every trainable parameter)."""
    ft = recipe["ft"]
    return {"arm": "FT_C", "peak_lr": float(recipe["fl"]["lr_peak_base"]), "lr_frac": float(ft["lr_frac"]),
            "passes": (int(ft["passes"]),), "batch_size": int(ft["batch_size"])}
