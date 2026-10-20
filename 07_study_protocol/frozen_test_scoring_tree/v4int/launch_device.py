"""v4int.launch_device — the real-data launcher of the on-device / personalisation arms LO, FT_C, FT_FA (INT-04) and
FT_PRE (DEV-FTPRE; PREG1_AMENDMENT_06K §2, 06L-C §C7).

  python -B code/v4int/launch_device.py --arm LO|FT_C|FT_FA|FT_PRE|FT_FA_WARM|FT_FA_WARM_FROZEN --seed 2026 --gpu 0 --run-dir <dir> \
         --history-policy USER_HISTORY_L50 --role MAIN|CALIBRATION|UNSEEN_VAL|UNSEEN_HOLDOUT|SPARSE_PANEL|HEAVY_PANEL|TRUE_COLD \
                                                  |MAIN_TEST|UNSEEN_HOLDOUT_TEST|SPARSE_PANEL_TEST|HEAVY_PANEL_TEST|TRUE_COLD_TEST \
         [--select] [--selection <SELECTION.json>] [--base <ckpt> --base-kind central|fl [--base-sha256 <hex>]] \
         [--init-from <PRETRAIN_FOR_FL_INIT.json> (FT_PRE only)] [--registry-ext <REGISTRY_V2_TEST_EXT.json>] \
         [--n-shards 64] [--shards a:b] [--resume] [--allow-shared-gpu] [--smoke --smoke-clients 8] [--dry-run]

FT_PRE (DEV-FTPRE): theta_pre + the registered FT_C local fine-tune procedure with the registered FT_C selection.
  base   = theta_pre ONLY: the endpoint_6.0 checkpoint named in the PRE run's PRETRAIN_FOR_FL_INIT.json (--init-from),
           loaded through launch_fl.load_pretrain_init (the checkpoint bytes hash to the record's fl_init.sha256 before
           torch.load; kind endpoint_6.0; run label PRETRAIN_FOR_FL; run id PRETRAIN_FOR_FL_*; train manifest
           PRETRAIN_TRAIN_5_100_NONMAIN; any checkpoint trained on MAIN users - MAIN / CALIB / FULL_TRAIN manifests,
           C_BIG / C_SMALL / C_FULL / C_MATCHED runs - refuses) and launch_fl.apply_init_state (the loaded state must
           hash to the checkpoint's own state_digest). --base / --base-kind are refused for FT_PRE; --base-sha256
           optionally pins the checkpoint sha256 on top of the record. The LAUNCH `start` record carries the theta_pre
           identity (`init_from`, = launch_fl's init_from) and the init record's file sha256, which the ANA index /
           report compare with the PRE re-evaluation's own PRETRAIN_FOR_FL_INIT.json binding.
  recipe = fl.finetune with the FT_C procedure (FTRecipe arm FT_C: the same RNG stream, LR = frac x SMALL peak, fresh
           AdamW, B16, clip 1.0, wd 1e-5) and the ONE cell of contracts/device/SELECTION_FT_C.json (the default
           --selection; a real run refuses any other file); the cell id is renamed FT_PRE_f<frac>_P<passes> and the
           procedure's cell is recorded. --select is refused (FT_PRE has no selection of its own) and so is the UNSEEN_VAL
           selection panel.
  rows   = the device RESULT_ROWS format with method = arm = FT_PRE, ft_procedure FT_C and the base identity labels
           (base_source PRE, base_run_id, base_checkpoint_sha256, base_train_manifest_id, base_init_record_sha256).
  An FT_C run whose --base is a PRETRAIN_FOR_FL checkpoint is refused (a device run from theta_pre is FT_PRE).

FT_FA_WARM / FT_FA_WARM_FROZEN (FL-15; PREG1_AMENDMENT_06M_B B4, the full production stack theta_pre -> FA_WARM (or S8)
MAIN -> the registered FT_C local fine-tune -> on-device inference):
  python -B code/v4int/launch_device.py --arm FT_FA_WARM|FT_FA_WARM_FROZEN --seed <s> ... \
         --base <FA_WARM run dir>/ckpt/<run_id>/best.pt|endpoint_6.0.pt --base-kind fa_warm [--base-sha256 <hex>]
  base   = a REGISTERED FA_WARM (resp. FA_WARM_FROZEN) MAIN checkpoint only (load_warm_base: DONE_MAIN, role MAIN, arm,
           seed = --seed, arm-spec sha256, config digest, init_from = the PRE identity; anything trained centrally on
           MAIN users, theta_pre itself, a CALIBRATION screen run or another arm refuses). --init-from / --select refuse.
  recipe = exactly FT_PRE's: the FT_C procedure (FTRecipe arm FT_C) with the ONE cell of SELECTION_FT_C.json (pinned
           sha256 e47965d2…; a real run refuses any other file); cells FT_FA_WARM_f<frac>_P<passes>. Whole-model fine-tune
           (disclosed for FT_FA_WARM_FROZEN: the item tables, frozen during FL, are fine-tuned locally and never leave).
  rows   = method FT_FA_WARM / FT_FA_WARM_FROZEN, ft_procedure FT_C, base_* labels (base_source = the base arm, run id,
           checkpoint + sha256, arm-spec sha256, the theta_pre run / checkpoint, the warm factors). LAUNCH start.kind =
           FA_WARM | FA_WARM_FROZEN. The LO / FT_C / FT_FA / FT_PRE specs and digests are unchanged.

Authority: PREG1_AMENDMENT_04 §2-§3 (LO; FT_C / FT_FA whole-SMALL fine-tune, passes {1,2,4} x LR {0.1,0.3} x SMALL
peak), _06 (SASREC_D64_B2; §4 device jobs may share one A16), ARCHITECTURE_LOCK e11_9a_recipe_grid, PREG1_LOCK_VALUES
T2 lo_step (LR = peak * s(6 (t+1) / T), exposure point "end") and T3 (frozen evaluation numerics), CLIENT_CONTRACT_V1.

Arms (KNOBS; never a CLI value):
  LO     each client trains its OWN SMALL model from theta0 of the seed on its OWN support rows (fl.local_only.LORunner,
         lr_shape s_efe / exposure "end", persistent AdamW for the client's trajectory, B16, clip 1.0, wd 1e-5); one
         trajectory PER pass budget P in {1,2,4} (the s_efe schedule depends on T = P x steps-per-pass, so budgets are
         NOT nested: a 4-pass trajectory evaluated after pass 1 is not the 1-pass recipe — test_nc_lo_budgets_nested);
         bytes 0 (nothing is downloaded after the theta0 bootstrap, nothing is uploaded).
  FT_C / FT_FA  fl.finetune.FTRunner: copy the base (C_SMALL central checkpoint / an FA global FL checkpoint), fine-tune
         every trainable parameter on the client's own support at a CONSTANT LR = frac x SMALL peak, fresh AdamW; one
         trajectory per frac, evaluated after passes 1, 2, 4 (nested by construction, fl.finetune T-FT-nested);
         bytes_up 0, bytes_down_once = the base model's shared bytes.
Selection (item 2, PREG1_AMENDMENT_06A §1): the registered selection panel "M04S FIXED_20" is the UNSEEN_VAL role of
REGISTRY_V2 (support UNEW_VAL_SUPPORT_32768 -> eval UNEW_VAL_VALIDATION_32768; 12,792 evaluable users, disjoint from
MAIN, UNSEEN_HOLDOUT and every report panel). This supersedes INT-04's CALIBRATION substitute: CALIB_VALIDATION is the
rank-0-32,768 prefix of MAIN, so selecting on it would let 25% of the report users take part in the selection.
`--select` is allowed on --role UNSEEN_VAL ONLY (never on CALIBRATION, MAIN or a panel); it evaluates the full grid and
writes SELECTION.json (O_EXCL; selection_role = "UNSEEN_VAL", the eval manifest content_sha256 it selected on, best
E2E macro MRR@20, ties -> the earlier cell of the canonical order = fewer passes, then smaller LR). An FT run on any
other role REQUIRES that selection (same arm) and runs only the selected cell; LO runs its 3 cells everywhere and marks
the selected one when a --selection is given (PREG1_AMENDMENT_06E §1: `--arm LO --role UNSEEN_VAL --select` selects the
LO cell by the same rule; with that SELECTION the two other cells' rows are DESCRIPTIVE, LO_CELL_NOT_SELECTED). Applying a SELECTION.json whose selection_role != "UNSEEN_VAL" on
MAIN or a panel is refused (load_selection).
FIX-DELTA-01 (REV-DELTA-01): F7 — a non-smoke LO run on any role other than UNSEEN_VAL is refused in preflight without
--selection (check_lo_selection); F6 — before any provider / GPU work, an applied SELECTION.json must carry the
content_sha256 of the VERIFIED UNSEEN_VAL eval manifest of the registry (and of its registry entry), and a real run
needs that manifest's 12,792 users (verify_selection_manifest; LAUNCH selection.manifest_check).
Clients = the evaluation manifest's users with >= 1 rankable row (key = client_key_hex, v4base.identity over the
manifest's own verified tables); support = the same key's rows of the role's TRAIN-layout manifest (StoreTrainProvider,
canonical positions). Causality: every support row's target_ts must be < cutoff = the earliest decision_ts among the
client's evaluation-manifest rows (and < TRAIN_END for the warm roles MAIN / CALIBRATION); a violation REFUSES the run
(fl.finetune.check_support_causal, applied to LO as well). A client without support is scored with the start state
(theta0 for LO, the base for FT) — recorded as no_support.
Evaluation: per client, its rankable rows are ranked with ITS OWN weights by the frozen T3 kernel
(v4eval.ranking.rank_from_query with the frozen score_block / class_chunk / cache_bytes, head + bias of that client's
model, eval mode, batches of <= eval_batch rows of that client in ascending decision_id); the per-row ranks are
assembled into ONE full-manifest PredictionArtifact per cell (path adapter_stream, eval_stream E2E_ROWS; every
rankable row covered exactly once) and scored by the frozen v4eval.evaluate -> rows_from_result, so the rows have the
central RESULT_ROWS format. Disclosed (needs an Ops note next to the T3 eval_batch memory rule): the batches are
grouped by client, not manifest-consecutive. With one shared model the per-client ranks equal predict_adapter's
bitwise on the tiny synthetic fixture, but on the real SASREC_D64_B2 (CPU, INT-01 env) 6 / 1,251 rows differ by
exactly one rank position (float32 near-ties under another batch shape; all at ranks > 20, so @20 credit identical
there). The per-client path itself is deterministic (bitwise across runs and across resumes).
Resumable by client shard: clients (sorted keys) are cut into --n-shards contiguous shards; each finished shard writes
shards/shard_XXXX.npz (+ .json, the completion marker, carrying the spec digest and the npz sha256); a rerun with
--resume skips finished shards (a shard of another spec refuses); --shards a:b lets several processes share the work
(e.g. two jobs on one A16 with --allow-shared-gpu); the process that finds every shard present assembles the rows.
GPU efficiency: fl.local_only / fl.finetune run ONE client per worker model; batching many clients into one forward
would need per-client weight stacking (torch.func) that the locked library does not provide and that would change the
numerics — not done; throughput comes from parallel shard processes instead.
Out of scope: COLD_H / WARM_TRUNCATION (panel manifests need an E11.4 truncated-history feature builder; refused).
TEST roles (INT-07; PREG1_AMENDMENT_06D §2): MAIN_TEST / UNSEEN_HOLDOUT_TEST / SPARSE_PANEL_TEST / HEAVY_PANEL_TEST
(support = the UNCHANGED registered support manifest of the role) and TRUE_COLD_TEST (eval-only). Refused in preflight
unless the DATA-09 TEST gate is open (v4int.test_roles.test_gate_ok, before any GPU / data access); --select is refused
on every TEST role; FT needs the UNSEEN_VAL --selection (load_selection); LO applies its UNSEEN_VAL --selection when given.
The harness (CentralTrainer, MAIN VALIDATION pair, theta0 / base) is built exactly as for every other role - it never
sees the TEST view, so no dedicated harness path is needed. After it, the DATA-09 loader wrappers are installed
(test_roles.install(role=None), gate re-checked on every TEST resolution), the role is resolved under the gate, the
RETRO_TEST StoreEvalProvider runs on the gated store / window path (v4int.retro_store), the support rows come from the
support manifest (MAIN_TEST: the harness MAIN_TRAIN provider itself), cutoff = the earliest RETRO_TEST decision_ts of
the client (and TRAIN_END for MAIN_TEST, a warm role), and the key is v4int.retro_store.retro_role_key (split
RETRO_TEST; the support manifest verified through the extension for the key).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from v4int.launch_central import (G1_GATE, LaunchRefused, _json_safe, _next_launch_path, gpu_checks, utc)  # noqa: E402

ARMS = ("LO", "FT_C", "FT_FA", "FT_PRE", "FT_FA_WARM", "FT_FA_WARM_FROZEN")
LOCK_CODE = {"LO": "LO", "FT_C": "C", "FT_FA": "FA",           # the ARCHITECTURE_LOCK method the arm builds on
             "FT_PRE": "C",                                    # (FT_PRE: a central base, theta_pre, + the FT_C procedure)
             "FT_FA_WARM": "FA", "FT_FA_WARM_FROZEN": "FA"}    # (FL-15: an FA_WARM / S8 FL base + the FT_C procedure)
PARADIGM = {"LO": "ON_DEVICE_LOCAL_ONLY", "FT_C": "CLIENT_FINETUNE", "FT_FA": "CLIENT_FINETUNE",
            "FT_PRE": "CLIENT_FINETUNE", "FT_FA_WARM": "CLIENT_FINETUNE", "FT_FA_WARM_FROZEN": "CLIENT_FINETUNE"}
LIFECYCLE = {"LO": "PERSISTENT_ADAMW_PER_CLIENT", "FT_C": "FRESH_ADAMW_PER_CLIENT", "FT_FA": "FRESH_ADAMW_PER_CLIENT",
             "FT_PRE": "FRESH_ADAMW_PER_CLIENT", "FT_FA_WARM": "FRESH_ADAMW_PER_CLIENT",
             "FT_FA_WARM_FROZEN": "FRESH_ADAMW_PER_CLIENT"}
# DEV-FTPRE (06K §2 / 06L-C §C7): FT_PRE = theta_pre + the registered FT_C procedure with the registered FT_C selection
# FL-15 (06M_B B4): FT_FA_WARM / FT_FA_WARM_FROZEN = a registered FA_WARM / FA_WARM_FROZEN MAIN checkpoint + the same
FT_PROCEDURE = {"FT_C": "FT_C", "FT_FA": "FT_FA", "FT_PRE": "FT_C",   # the fl.finetune FTRecipe arm of each FT arm
                "FT_FA_WARM": "FT_C", "FT_FA_WARM_FROZEN": "FT_C"}
SELECTION_ARM = {"FT_PRE": "FT_C",                                    # an arm that applies another arm's SELECTION
                 "FT_FA_WARM": "FT_C", "FT_FA_WARM_FROZEN": "FT_C"}
FT_WARM_BASE = {"FT_FA_WARM": "FA_WARM", "FT_FA_WARM_FROZEN": "FA_WARM_FROZEN"}   # FL-15: the base arm of each
WARM_BASE_KIND = "fa_warm"                                            # FL-15: their --base-kind
WARM_BASE_FILES = ("best.pt", "endpoint_6.0.pt")                      # the FL checkpoints a DONE_MAIN run keeps
FT_PRE_SELECTION = V4 / "contracts" / "device" / "SELECTION_FT_C.json"   # registered (DL 231); the default --selection
FT_PRE_SELECTION_SHA256 = "e47965d2f4486fc2c30140a8ee14e79686a842decb091b3dff1de5646716c10c"   # its file sha256 (DL 231)
PRE_BASE_KIND = "PRE"                                                 # the LAUNCH start kind of an FT_PRE run
PRE_RUN_ID_PREFIX = "PRETRAIN_FOR_FL_"                                # = v4int.pretrain_small.RUN_ID_PREFIX
PRE_RUN_LABEL = "PRETRAIN_FOR_FL"
KNOBS = {"lo_passes": (1, 2, 4), "ft_passes": (1, 2, 4), "ft_lr_fracs": (0.1, 0.3), "batch": 16, "clip": 1.0,
         "weight_decay": 1e-5, "exposure_point": "end", "lo_lr_shape": "s_efe"}
SUPPORT_ROLES = ("MAIN", "CALIBRATION", "UNSEEN_VAL", "UNSEEN_HOLDOUT", "SPARSE_PANEL", "HEAVY_PANEL")
EVAL_ONLY_ROLES = ("TRUE_COLD",)
TEST_SUPPORT_ROLES = ("MAIN_TEST", "UNSEEN_HOLDOUT_TEST", "SPARSE_PANEL_TEST", "HEAVY_PANEL_TEST")   # INT-07 / DATA-09
TEST_EVAL_ONLY_ROLES = ("TRUE_COLD_TEST",)
TEST_ROLES = TEST_SUPPORT_ROLES + TEST_EVAL_ONLY_ROLES
OUT_OF_SCOPE_ROLES = ("COLD_H", "WARM_TRUNCATION")
WARM_ROLES = ("MAIN", "CALIBRATION", "MAIN_TEST")
SELECTION_ROLE = "UNSEEN_VAL"
SELECTION_FORMAT = "PPSI_V4_DEVICE_SELECTION_1"
SELECTION_SUBSTITUTE = ("PREG1_AMENDMENT_06A §1: M04S FIXED_20 = the UNSEEN_VAL role of REGISTRY_V2 (support "
                        "UNEW_VAL_SUPPORT_32768 -> eval UNEW_VAL_VALIDATION_32768), disjoint from MAIN, "
                        "UNSEEN_HOLDOUT and every report panel; supersedes INT-04's CALIBRATION substitute "
                        "(CALIB_VALIDATION is the rank-0-32,768 prefix of MAIN and would let report users take part "
                        "in the selection); defined in v4int.launch_device, registered by Ops in amendment 06A")
PRIMARY = ("E2E", "macro", "mrr@20")
LO_NOT_SELECTED = "LO_CELL_NOT_SELECTED"          # PREG1_AMENDMENT_06E §1 (K8 / K11 use the selected LO cell only)
FAMILY = "SASREC"
TS_KEY = "target_ts"
ARCH_LOCK = V4 / "contracts" / "ARCHITECTURE_LOCK.json"


# ------------------------------------------------------------------------------------------------ spec (pure)
def cell_id(arm: str, passes: int, frac: Optional[float] = None) -> str:
    return f"LO_P{int(passes)}" if arm == "LO" else f"{arm}_f{float(frac):g}_P{int(passes)}"


def grid_cells(arm: str) -> list:
    """Canonical cell order (the selection tie-break order): fewer passes first, then the smaller LR fraction."""
    if arm == "LO":
        return [{"cell": cell_id(arm, p), "passes": p, "lr_frac": None} for p in KNOBS["lo_passes"]]
    return [{"cell": cell_id(arm, p, f), "passes": p, "lr_frac": f}
            for p in KNOBS["ft_passes"] for f in KNOBS["ft_lr_fracs"]]


def check_role(role: str, *, select: bool, arm: str, selection: Optional[Mapping]) -> None:
    if role in OUT_OF_SCOPE_ROLES:
        raise LaunchRefused(f"--role {role}: a truncated-history panel needs an E11.4 feature builder (out of scope)")
    if "TEST" in str(role).upper() and role not in TEST_ROLES:
        raise LaunchRefused(f"--role {role} is not a registered TEST role {TEST_ROLES}")
    if role not in SUPPORT_ROLES + EVAL_ONLY_ROLES + TEST_ROLES:
        raise LaunchRefused(f"--role {role} is not a registered REGISTRY_V2 role of launch_device")
    if select and not selection_role_ok(role):
        raise LaunchRefused(f"--select is allowed on --role {SELECTION_ROLE} only (PREG1_AMENDMENT_06A §1: the "
                            f"CALIBRATION selection substitute is superseded; never select on {role}, incl. "
                            f"CALIBRATION)")
    if select and selection is not None:
        raise LaunchRefused("--select and --selection are exclusive")
    if arm in SELECTION_ARM and select:
        raise LaunchRefused(f"{arm} applies the registered {SELECTION_ARM[arm]} selection (06L-C §C7): --select refused")
    if arm in SELECTION_ARM and selection_role_ok(role):
        raise LaunchRefused(f"{arm} is never run on the {SELECTION_ROLE} selection panel (it applies the registered "
                            f"{SELECTION_ARM[arm]} selection; 06L-C §C7 / §C16)")
    if arm != "LO" and not select and selection is None:
        raise LaunchRefused(f"{arm} on {role} needs --selection (the {SELECTION_ROLE} SELECTION.json of {arm})")


def selection_role_ok(role: str) -> bool:
    return role == SELECTION_ROLE


def check_lo_selection(role: str, *, arm: str, select: bool, selection, smoke: bool) -> None:
    """FIX-DELTA-01 F7 (PREG1_AMENDMENT_06E §1; REV-DELTA-01): a non-smoke LO run on any role other than UNSEEN_VAL
    needs the UNSEEN_VAL --selection (its unselected cells are DESCRIPTIVE and the ANA index refuses an unselected LO
    run, so launching one only wastes compute). Launcher-level (preflight); the pure device_spec is unchanged."""
    if arm == "LO" and not smoke and role != SELECTION_ROLE and not select and selection is None:
        raise LaunchRefused(f"LO on {role} needs --selection (the {SELECTION_ROLE} SELECTION.json of LO, 06E §1); "
                            f"select the LO cell first with --arm LO --role {SELECTION_ROLE} --select")


def device_spec(arm: str, *, seed: int, peak_lr: float, role: str, start_digest: str, select: bool = False,
                selection: Optional[Mapping] = None, fused: Optional[bool] = None) -> dict:
    """The registered configuration of one device run (plain JSON-able values; its sha256 binds every shard)."""
    if arm not in ARMS:
        raise LaunchRefused(f"unknown arm {arm}")
    check_role(role, select=select, arm=arm, selection=selection)
    sel_arm = SELECTION_ARM.get(arm, arm)
    if selection is not None and selection.get("arm") != sel_arm:
        raise LaunchRefused(f"--selection is for {selection.get('arm')}, not {sel_arm}"
                            + (f" (the registered selection {arm} applies)" if sel_arm != arm else ""))
    cells = grid_cells(arm)
    sel_cell = None if selection is None else selection["cell"]
    proc_cell = None
    if sel_cell is not None and sel_arm != arm:                        # FT_PRE: the FT_C cell -> the same (P, frac)
        proc = {c["cell"]: c for c in grid_cells(sel_arm)}
        if sel_cell not in proc:
            raise LaunchRefused(f"selection cell {sel_cell} is not a cell of the {sel_arm} grid")
        proc_cell = sel_cell
        sel_cell = cell_id(arm, proc[sel_cell]["passes"], proc[sel_cell]["lr_frac"])
    if sel_cell is not None and sel_cell not in {c["cell"] for c in cells}:
        raise LaunchRefused(f"selection cell {sel_cell} is not a cell of the {arm} grid")
    if arm != "LO" and not select:
        cells = [c for c in cells if c["cell"] == sel_cell]            # FT: apply the ONE selected cell
    spec = {"arm": arm, "seed": int(seed), "peak_lr": float(peak_lr), "role": role, "select": bool(select),
            "selected_cell": sel_cell, "cells": cells, "start_digest": start_digest, "fused": fused,
            "knobs": {k: list(v) if isinstance(v, tuple) else v for k, v in KNOBS.items()},
            "selection_substitute": SELECTION_SUBSTITUTE if (select or selection is not None) else None}
    if arm in SELECTION_ARM:                                           # FT_PRE only: every other spec is unchanged
        spec.update(ft_procedure=FT_PROCEDURE[arm], procedure_selection_arm=sel_arm,
                    procedure_selection_cell=proc_cell,
                    procedure_selection_self_sha256=None if selection is None else selection.get("self_sha256"))
    if arm in FT_WARM_BASE:                                            # FL-15: FT_FA_WARM(_FROZEN) only
        spec["base_arm"] = FT_WARM_BASE[arm]
    return spec


def spec_digest(spec: Mapping) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()


def _canon(obj: Mapping) -> str:
    body = {k: v for k, v in obj.items() if k != "self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ runners
def make_runners(spec: Mapping, worker, start_state: Mapping):
    """[(trajectory kind, runner, {budget: cell id})]: one LO runner per pass budget; one FT runner per LR fraction."""
    from fl.client import LocalSolver
    from fl.numerics import state_digest
    arm, K = spec["arm"], KNOBS
    if state_digest(start_state) != spec["start_digest"]:
        raise LaunchRefused("the start state does not match the spec's start digest (theta0 / base provenance)")
    out = []
    if arm == "LO":
        from fl.local_only import LORecipe, LORunner
        for c in spec["cells"]:
            p = int(c["passes"])
            sol = LocalSolver(lr=float(spec["peak_lr"]), batch_size=K["batch"], passes=p, clip=K["clip"],
                              weight_decay=K["weight_decay"], fused=spec["fused"])
            rec = LORecipe(theta0_digest=spec["start_digest"], seed=int(spec["seed"]), passes=p, eval_at=(p,),
                           solver=sol, lr_shape=K["lo_lr_shape"], exposure_point=K["exposure_point"])
            out.append(("LO", LORunner(worker, start_state, rec), {p: c["cell"]}))
        return out
    from fl.finetune import FTRecipe, FTRunner
    fracs = sorted({c["lr_frac"] for c in spec["cells"]})
    for f in fracs:
        cs = [c for c in spec["cells"] if c["lr_frac"] == f]
        rec = FTRecipe(arm=FT_PROCEDURE[arm], base_digest=spec["start_digest"], seed=int(spec["seed"]),
                       peak_lr=float(spec["peak_lr"]),
                       lr_frac=f, passes=tuple(sorted(int(c["passes"]) for c in cs)), batch_size=K["batch"],
                       weight_decay=K["weight_decay"], clip=K["clip"], fused=spec["fused"])
        out.append(("FT", FTRunner(worker, start_state, rec), {int(c["passes"]): c["cell"] for c in cs}))
    return out


# ------------------------------------------------------------------------------------------------ clients (pure)
@dataclass
class ClientPlan:
    key: str
    rows: np.ndarray                 # the client's rankable manifest rows (ascending)
    stream_pos: np.ndarray           # their positions in the evaluation stream (provider positions)
    support: Optional[np.ndarray]    # support provider positions, or None
    cutoff: Optional[int]            # the support cutoff (microseconds), None without support


def build_client_plans(manifest, row_key: np.ndarray, *, row_decision_ts: Optional[np.ndarray],
                       support_pos_by_key: Optional[Mapping], warm: bool, train_end_us: Optional[int] = None) -> list:
    """One plan per evaluation user with >= 1 rankable row (sorted keys). cutoff = the earliest decision_ts among ALL
    the user's manifest rows (min TRAIN_END for warm roles)."""
    from v4data.common import TRAIN_END_US
    tend = TRAIN_END_US if train_end_us is None else int(train_end_us)
    row_key = np.asarray(row_key, dtype=object)
    if row_key.shape != (manifest.n,):
        raise ValueError("row_key must have one key per manifest row")
    rk = manifest.rankable
    stream = np.flatnonzero(manifest.eval_mask)
    order = np.argsort(row_key, kind="stable")
    sk = row_key[order]
    cut = np.flatnonzero(sk[1:] != sk[:-1]) + 1 if sk.size else np.zeros(0, dtype=np.int64)
    plans = []
    for g in np.split(order, cut) if sk.size else []:
        key = str(row_key[g[0]])
        rows = np.sort(g[rk[g]])
        if rows.size == 0:
            continue
        sup = None if support_pos_by_key is None else support_pos_by_key.get(key)
        sup = None if sup is None or len(sup) == 0 else np.sort(np.asarray(sup, dtype=np.int64))
        cutoff = None
        if sup is not None:
            if row_decision_ts is not None:
                cutoff = int(np.asarray(row_decision_ts, dtype=np.int64)[g].min())
                if warm:
                    cutoff = min(cutoff, tend)
            elif warm:
                cutoff = tend
            else:
                raise ValueError(f"client {key}: a panel with support needs the rows' decision_ts for the cutoff")
        plans.append(ClientPlan(key, rows, np.searchsorted(stream, rows).astype(np.int64), sup, cutoff))
    return plans


def check_plans_cover(manifest, plans: Sequence[ClientPlan]) -> None:
    rows = np.concatenate([p.rows for p in plans]) if plans else np.zeros(0, dtype=np.int64)
    want = np.flatnonzero(manifest.rankable)
    if rows.size != np.unique(rows).size or not np.array_equal(np.sort(rows), want):
        raise LaunchRefused("the client plans do not cover every rankable row exactly once")


def support_causal_ok(ts: np.ndarray, cutoff: Optional[int]) -> bool:
    """The causality property: no support event at or after the cutoff."""
    return ts.size == 0 or (cutoff is not None and int(np.max(ts)) < int(cutoff))


# ------------------------------------------------------------------------------------------------ T3 per-client ranks
def rank_client(adapter, batches: Sequence[Mapping], rows_chunks: Sequence[np.ndarray], manifest, frozen) -> tuple:
    """The frozen T3 kernel on the client's own rows with ITS model: (ranks, n_tied, eq_before) int64 arrays."""
    import torch
    from v4eval.ranking import rank_from_query
    W, b = adapter.head_weight(), adapter.head_bias()
    if int(W.shape[0]) != manifest.catalogue_K:
        raise LaunchRefused(f"head has {W.shape[0]} classes, the manifest catalogue_K is {manifest.catalogue_K}")
    out = [[], [], []]
    for bt, rows in zip(batches, rows_chunks):
        q = adapter.query(bt)
        tgt = torch.as_tensor(manifest.target_class[rows], dtype=torch.long, device=q.device)
        r = rank_from_query(q, W, b, tgt, score_block=frozen.score_block, class_chunk=frozen.class_chunk,
                            K_expected=manifest.catalogue_K, cache_bytes=frozen.cache_bytes)
        for i, t in enumerate((r.ranks, r.n_tied, r.eq_before)):
            out[i].append(t.cpu().numpy().astype(np.int64))
    return tuple(np.concatenate(x) if x else np.zeros(0, dtype=np.int64) for x in out)


def _chunks(plan: ClientPlan, eval_batch: int) -> list:
    return [(plan.stream_pos[s:s + eval_batch], plan.rows[s:s + eval_batch])
            for s in range(0, int(plan.rows.size), int(eval_batch))]


def _check_eval_batch(bt: Mapping, rows: np.ndarray, manifest) -> None:
    d = np.asarray(bt["decision_id"].cpu().numpy() if hasattr(bt["decision_id"], "cpu") else bt["decision_id"])
    if not np.array_equal(d.astype(np.int64), manifest.decision_id[rows]):
        raise LaunchRefused("an evaluation batch does not carry the client's manifest rows")
    if "target_class" in bt:
        t = np.asarray(bt["target_class"].cpu().numpy())
        if not np.array_equal(t.astype(np.int64), manifest.target_class[rows]):
            raise LaunchRefused("an evaluation batch's target_class disagrees with the manifest")


# ------------------------------------------------------------------------------------------------ engine
def write_json(path: Path, obj) -> None:
    from v4train.checkpoint import write_json_atomic
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(path), _json_safe(obj))


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def shard_device_record(device) -> dict:
    """REV-FL-01 M3: the evaluation device class + registered pin a shard was ranked on (every shard carries it)."""
    from v4eval.device_pin import pin_for_device
    from v4eval.frozen_config import eval_device_class
    import torch
    dev = torch.device(device)
    pin = pin_for_device(dev)
    return {"eval_device_class": eval_device_class(dev), "device_pin_sha256": pin["self_sha256"] if pin else None,
            "code_sha256": device_code_sha256()}


def device_code_sha256() -> str:
    """06E §8: the code identity of a device shard (fl/*.py + this launcher; v4eval.hashing rule)."""
    from v4eval.hashing import code_sha256
    return code_sha256(sorted((CODE / "fl").glob("*.py")) + [Path(__file__).resolve()])


def check_shard_device(rec: Mapping, want: Mapping, where: str) -> None:
    got = {k: rec.get(k, "<absent>") for k in want}
    if got != dict(want):
        raise LaunchRefused(f"{where}: shard made on {got}, this process is {dict(want)}: shards of different "
                            "device classes / pins / code are never merged (or resumed)")


def shard_bounds(n_clients: int, n_shards: int) -> list:
    n_shards = max(1, min(int(n_shards), max(1, n_clients)))
    edges = np.linspace(0, n_clients, n_shards + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_shards)]


def load_shard(sdir: Path, s: int, digest: str) -> Optional[dict]:
    js, npz = sdir / f"shard_{s:04d}.json", sdir / f"shard_{s:04d}.npz"
    if not js.is_file():
        return None
    rec = json.loads(js.read_text(encoding="utf-8"))
    if rec.get("spec_sha256") != digest:
        raise LaunchRefused(f"{js} belongs to another spec ({rec.get('spec_sha256')}): refusing to mix runs")
    if not npz.is_file() or _sha_file(npz) != rec.get("npz_sha256"):
        raise LaunchRefused(f"{npz} is missing or fails its recorded sha256")
    return rec


def execute(*, spec: Mapping, run_dir: Path, worker, start_state: Mapping, plans: Sequence[ClientPlan],
            get_support: Callable, get_eval_batch: Callable, manifest, frozen, row_fn: Optional[Callable] = None,
            evaluator_sha: Optional[str] = None, n_shards: int = 64, shard_range: Optional[tuple] = None,
            resume: bool = False, smoke_clients: Optional[int] = None, say: Callable = print,
            eval_manifest_content_sha256: Optional[str] = None) -> dict:
    """Run / resume the shards of one device run, then (when every shard exists) assemble one full-manifest artifact per
    cell, evaluate it with the frozen evaluator and write rows. `get_support(plan)` -> examples dict (with TS_KEY) and
    `get_eval_batch(stream_pos)` -> a batch on the worker's device."""
    import torch
    from fl.client import ClientData, valid_rows
    from fl.finetune import check_support_causal
    from v4eval.frozen_config import apply_runtime_flags, check_runtime_flags
    run_dir = Path(run_dir)
    digest = spec_digest(spec)
    sdir = run_dir / "shards"
    plans = list(plans)
    apply_runtime_flags(frozen)                          # the T3 flags hold for the whole run (training is STRICT too)
    check_runtime_flags(frozen, worker.device)
    check_plans_cover(manifest, plans) if smoke_clients is None else None
    if smoke_clients is not None:
        plans = plans[:int(smoke_clients)]
    bounds = shard_bounds(len(plans), n_shards)
    if not resume and sdir.is_dir() and any(sdir.glob("shard_*.json")):
        raise LaunchRefused(f"{sdir} already has finished shards: use --resume")
    lo, hi = (0, len(bounds)) if shard_range is None else (int(shard_range[0]), min(int(shard_range[1]), len(bounds)))
    runners = make_runners(spec, worker, start_state)
    cells = [c["cell"] for c in spec["cells"]]
    eb = int(frozen.eval_batch)
    dev_rec = shard_device_record(worker.device)
    t0 = time.perf_counter()
    done_now = 0
    for s in range(lo, hi):
        prev = load_shard(sdir, s, digest)
        if prev is not None:
            check_shard_device(prev, dev_rec, f"shard {s}")
            continue
        a, b = bounds[s]
        rows_all = np.concatenate([p.rows for p in plans[a:b]]) if b > a else np.zeros(0, dtype=np.int64)
        arr = {f"{c}__{f}": np.full(rows_all.size, -1, dtype=np.int64) for c in cells
               for f in ("ranks", "n_tied", "eq_before")}
        recs, off = [], 0
        for p in plans[a:b]:
            chunks = _chunks(p, eb)
            batches = []
            for sp, rows in chunks:
                bt = get_eval_batch(sp)
                _check_eval_batch(bt, rows, manifest)
                batches.append(bt)
            rchunks = [r for _, r in chunks]
            hook = lambda adapter, budget: rank_client(adapter, batches, rchunks, manifest, frozen)  # noqa: E731
            rec = {"key": p.key, "n_rows": int(p.rows.size), "cutoff": p.cutoff, "trajectories": []}
            res_by_cell = {}
            ex = None
            if p.support is not None:
                ex = get_support(p)
                idx = valid_rows(ex)
                if int(idx.numel()):
                    check_support_causal(ex, idx, cutoff=p.cutoff, ts_key=TS_KEY)     # refuses (never filters)
                else:
                    ex = None
            rec["no_support"] = ex is None
            if ex is None:                  # no support: the start state itself (theta0 / base), scored once
                worker.load_state_(start_state)
                with torch.no_grad():
                    worker.module.eval()
                    r0 = hook(worker, 0)
                worker.module.train()
                res_by_cell = {cid: r0 for cid in cells}
            for kind, runner, bmap in ([] if ex is None else runners):
                cd = ClientData(p.key, ex)
                if kind == "LO":
                    r = runner.run(cd, hook)
                    rec["trajectories"].append({"kind": "LO", "passes": list(bmap), "n_valid": r.n_valid,
                                                "steps": r.steps, "final_digest": r.final_digest,
                                                "no_label": r.no_label})
                else:
                    r = runner.run(cd, hook, support_cutoff=p.cutoff if p.cutoff is not None else 0, ts_key=TS_KEY)
                    rec["trajectories"].append({"kind": "FT", "lr_frac": r.lr_frac, "n_support_events":
                                                r.n_support_events, "updates": r.updates,
                                                "final_digest": r.final_digest, "no_support": r.no_support,
                                                "bytes_up": r.bytes_up, "bytes_down_once": r.bytes_down_once})
                for bud, cid in bmap.items():
                    res_by_cell[cid] = r.evals[bud]
            n = int(p.rows.size)
            for cid in cells:
                for i, f in enumerate(("ranks", "n_tied", "eq_before")):
                    arr[f"{cid}__{f}"][off:off + n] = res_by_cell[cid][i]
            off += n
            recs.append(rec)
        sdir.mkdir(parents=True, exist_ok=True)
        npz = sdir / f"shard_{s:04d}.npz"
        tmp = sdir / f"shard_{s:04d}.tmp.{os.getpid()}.npz"
        np.savez(tmp, rows=rows_all, **arr)
        os.replace(tmp, npz)
        write_json(sdir / f"shard_{s:04d}.json", {"shard": s, "clients": [a, b], "spec_sha256": digest,
                                                 "npz_sha256": _sha_file(npz), "records": recs, "utc": utc(),
                                                 **dev_rec})
        done_now += 1
        say(f"[device] shard {s} ({b - a} clients) written")
    wall = time.perf_counter() - t0
    status = {"spec_sha256": digest, "n_clients": len(plans), "n_shards": len(bounds), "shards_this_leg": done_now,
              "wall_s_this_leg": round(wall, 2), "device": str(worker.device)}
    have = [load_shard(sdir, s, digest) for s in range(len(bounds))]
    if smoke_clients is not None:
        write_json(run_dir / "SMOKE_SUMMARY.json", dict(status, label="SMOKE_NOT_EVIDENCE",
                                                         status="SMOKE_DONE_NOT_EVIDENCE"))
        return dict(status, status="SMOKE_DONE_NOT_EVIDENCE")
    if any(h is None for h in have):
        missing = [s for s, h in enumerate(have) if h is None]
        write_json(run_dir / "STATUS.json", dict(status, status="PARTIAL", missing_shards=missing, utc=utc()))
        return dict(status, status="PARTIAL", missing_shards=missing)
    return finalize(spec=spec, run_dir=run_dir, manifest=manifest, frozen=frozen, n_shards=len(bounds),
                    row_fn=row_fn, evaluator_sha=evaluator_sha, status=status, say=say,
                    eval_manifest_content_sha256=eval_manifest_content_sha256)


def assemble(manifest, frozen, sdir: Path, n_shards: int, cell: str, model_sha256: str, device_class: str,
             device_pin_sha256: Optional[str] = None):
    from v4eval.predict import PATH_ADAPTER, _new_artifact
    eval_cfg = frozen.eval_config(PATH_ADAPTER, device_class)
    if device_pin_sha256:                                # INT-06: exactly what v4eval.predict adds on the pinned class
        eval_cfg["device_pin_sha256"] = device_pin_sha256
    art = _new_artifact(manifest, model_sha256, eval_cfg)
    seen = np.zeros(manifest.n, dtype=bool)
    for s in range(n_shards):
        z = np.load(sdir / f"shard_{s:04d}.npz")
        rows = z["rows"]
        if bool(np.any(seen[rows])):
            raise LaunchRefused(f"shard {s}: a manifest row is covered twice")
        seen[rows] = True
        art.ranks[rows] = z[f"{cell}__ranks"]
        art.n_tied[rows] = z[f"{cell}__n_tied"]
        art.eq_before[rows] = z[f"{cell}__eq_before"]
    if not np.array_equal(seen, manifest.rankable):
        raise LaunchRefused("the shards do not cover every rankable row exactly once")
    art.header["eval_config"]["eval_stream"] = "E2E_ROWS"
    art.header["eval_config"]["per_client_models"] = True
    return art


def select_cell(values: Mapping[str, float], order: Sequence[str]) -> str:
    """Best primary value; ties -> the earlier cell of the canonical order."""
    best = None
    for c in order:
        v = float(values[c])
        if best is None or v > values[best]:
            best = c
    return best


def finalize(*, spec, run_dir: Path, manifest, frozen, n_shards: int, row_fn, evaluator_sha, status, say,
            eval_manifest_content_sha256: Optional[str] = None) -> dict:
    from v4eval.evaluate import evaluate
    from v4eval.frozen_config import eval_device_class
    from v4eval.values import Undefined
    import torch
    run_dir, sdir = Path(run_dir), Path(run_dir) / "shards"
    digest = spec_digest(spec)
    recs = [json.loads((sdir / f"shard_{s:04d}.json").read_text(encoding="utf-8")) for s in range(n_shards)]
    set_sha = hashlib.sha256(json.dumps([r["npz_sha256"] for r in recs]).encode()).hexdigest()
    dev_cls = eval_device_class(torch.device(status["device"]))
    from v4eval.device_pin import pin_for_device            # INT-06: rows on the registered A40 class are labelled
    pin = pin_for_device(torch.device(status["device"]))    # MEASURED_PINNED_DEVICE (fails closed on a wrong GPU)
    pin_sha = pin["self_sha256"] if pin else None
    for s, r in enumerate(recs):                            # M3: every shard on this process's class / pin / code
        check_shard_device(r, {"eval_device_class": dev_cls, "device_pin_sha256": pin_sha,
                               "code_sha256": device_code_sha256()}, f"finalize shard {s}")
    values, all_rows, results = {}, [], {}
    for c in spec["cells"]:
        cid = c["cell"]
        msha = hashlib.sha256(f"PER_CLIENT_MODELS|{digest}|{cid}|{set_sha}".encode()).hexdigest()
        art = assemble(manifest, frozen, sdir, n_shards, cid, msha, dev_cls, device_pin_sha256=pin_sha)
        res = evaluate(manifest, art, config=frozen, evaluator_code_sha256=evaluator_sha)
        v = res[PRIMARY[0]][PRIMARY[1]][PRIMARY[2]]
        if isinstance(v, Undefined):
            raise LaunchRefused(f"{cid}: primary metric UNDEFINED ({v})")
        values[cid] = float(v)
        results[cid] = res
        write_json(run_dir / "cells" / f"{cid}.result.json", res)
        np.save(run_dir / "cells" / f"{cid}.ranks_int32.npy", art.ranks.astype(np.int32))
    selected = spec.get("selected_cell")
    if spec.get("select"):
        selected = select_cell(values, [c["cell"] for c in spec["cells"]])
    for c in spec["cells"]:
        cid = c["cell"]
        if row_fn is not None:
            all_rows += row_fn(results[cid], c, cid == selected)
    n_nosup = sum(1 for r in recs for x in r["records"] if x.get("no_support"))
    summary = dict(status, arm=spec["arm"], role=spec["role"], spec=dict(spec), spec_sha256=digest,
                   shard_set_sha256=set_sha, n_no_support=n_nosup, selected_cell=selected,
                   bytes={"up": 0, "down_after_bootstrap": 0 if spec["arm"] == "LO" else "base shared bytes once"},
                   **({"device_pin_sha256": pin_sha} if pin_sha else {}))
    if row_fn is not None:
        write_json(run_dir / "RESULT_ROWS.json", {"arm": spec["arm"], "role": spec["role"], "spec_sha256": digest,
                                                  "rows": all_rows})
    if spec.get("select"):
        if not eval_manifest_content_sha256:
            raise LaunchRefused("--select needs the eval manifest content_sha256 (the UNSEEN_VAL panel it selected on)")
        obj = {"format": SELECTION_FORMAT, "arm": spec["arm"], "seed": spec["seed"], "selection_role": spec["role"],
               "eval_manifest_content_sha256": eval_manifest_content_sha256,
               "cell": selected, "cells": [c for c in spec["cells"] if c["cell"] == selected][0],
               "values_primary": values, "primary": "/".join(PRIMARY), "rule": "max; ties -> canonical order",
               "spec_sha256": digest, "shard_set_sha256": set_sha, "substitute": SELECTION_SUBSTITUTE}
        obj["self_sha256"] = _canon(obj)
        from v4models.variants import write_exclusive
        write_exclusive(run_dir / "SELECTION.json", json.dumps(obj, indent=1, sort_keys=True) + "\n")
        summary["selection"] = {"file": "SELECTION.json", "self_sha256": obj["self_sha256"]}
    write_json(run_dir / "RUN_SUMMARY.json", dict(summary, status="DONE"))
    write_json(run_dir / "STATUS.json", {"spec_sha256": digest, "status": "DONE", "utc": utc()})
    say(f"[device] {spec['arm']} {spec['role']}: DONE ({len(spec['cells'])} cells)")
    return dict(summary, status="DONE", values=values)


def load_selection(path: Path) -> dict:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"--selection {path} unreadable ({e})") from e
    if obj.get("format") != SELECTION_FORMAT or obj.get("self_sha256") != _canon(obj):
        raise LaunchRefused(f"--selection {path} is not a verified {SELECTION_FORMAT} object")
    if obj.get("selection_role") != SELECTION_ROLE:
        raise LaunchRefused(f"--selection was made on {obj.get('selection_role')}, not {SELECTION_ROLE}")
    return obj


UNSEEN_VAL_USERS = 12792                          # REGISTRY_V2 UNEW_VAL_VALIDATION_32768 n_users (06A §1)


def check_selection_manifest(selection: Mapping, *, binding: Mapping, n_users: Optional[int], smoke: bool,
                             registry_entry: Optional[Mapping] = None) -> dict:
    """FIX-DELTA-01 F6 (REV-FL-01 MINOR): a SELECTION.json applies only if the eval manifest content_sha256 it selected
    on equals the VERIFIED binding of the registry's UNSEEN_VAL eval manifest (and the registry entry's recorded
    content_sha256); a real (non-smoke) run also needs that manifest to hold the registered 12,792 users."""
    got, want = selection.get("eval_manifest_content_sha256"), binding.get("content_sha256")
    if not got or got != want:
        raise LaunchRefused(f"--selection eval_manifest_content_sha256 {str(got)[:12]}… is not the verified "
                            f"{SELECTION_ROLE} eval manifest {binding.get('manifest_id')} ({str(want)[:12]}…)")
    if registry_entry is not None and registry_entry.get("content_sha256") != want:
        raise LaunchRefused(f"the registry entry of {binding.get('manifest_id')} records content_sha256 "
                            f"{str(registry_entry.get('content_sha256'))[:12]}…, the verified manifest {want[:12]}…")
    if not smoke and (n_users is None or int(n_users) != UNSEEN_VAL_USERS):
        raise LaunchRefused(f"the {SELECTION_ROLE} eval manifest {binding.get('manifest_id')} has {n_users} users, "
                            f"not the registered {UNSEEN_VAL_USERS}")
    return {"manifest_id": binding.get("manifest_id"), "content_sha256": want, "n_users": n_users,
            "users_checked": not smoke, "authority": "FIX-DELTA-01 F6"}


def verify_selection_manifest(selection: Mapping, registry_path: Path, reg_kw: Mapping, *, smoke: bool) -> dict:
    """check_selection_manifest against the registry's UNSEEN_VAL eval manifest, verified through v4int.verify
    (tables re-read and hashed; users = the verified users-table row count, else the distinct eval_user_key)."""
    from v4int.providers import _verify, manifest_binding_of
    reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    mid = reg["roles"][SELECTION_ROLE]["eval"]
    vm = _verify(mid, reg_kw.get("paths"), "EVAL", reg_kw.get("manifest_registry"), reg_kw.get("registry_root"),
                 reg_kw.get("log", print))
    users = (vm.tables or {}).get("users", {}).get("row_count")
    if users is None and vm.columns is not None and "eval_user_key" in vm.columns:
        users = int(np.unique(np.asarray(vm.columns["eval_user_key"])).size)
    return check_selection_manifest(selection, binding=manifest_binding_of(vm), n_users=users, smoke=smoke,
                                    registry_entry=(reg.get("manifests") or {}).get(mid))


# ------------------------------------------------------------------------------------------------ rows
def device_rows(result: Mapping, key_base: Mapping, labels: Mapping, *, spec: Mapping, cell: Mapping, selected: bool,
                down_bytes: int, overrides: Optional[Mapping] = None) -> list:
    """rows_from_result (central RESULT_ROWS format) relabelled for a device arm."""
    from v4eval.rows import rows_from_result
    arm = spec["arm"]
    rows = rows_from_result(result, key_base, method="LO" if arm == "LO" else arm, seed=int(spec["seed"]),
                            selection_rule=None, efe=None)
    rule = ("SELECTED_ON_UNSEEN_VAL" if selected and spec.get("selected_cell") else
            "SELECTION_WINNER" if selected else "GRID_CELL")
    out = []
    for r in rows:
        r = dict(r, **labels)
        r.update(method=arm, method_code=arm, training_paradigm=PARADIGM[arm], optimizer_lifecycle=LIFECYCLE[arm],
                 local_personalization="WHOLE_MODEL" if arm != "LO" else "LOCAL_ONLY_MODEL",
                 row_status="DEVICE_SELECTION" if spec.get("select") else "DOWNSTREAM_ARM",
                 comparison_status="DEVICE_SELECTION" if spec.get("select") else "DOWNSTREAM_RETENTION",
                 c_matched_eligible=False, downstream_seed_allowed=False, up_bytes=0, down_bytes=int(down_bytes),
                 device_cell=cell["cell"], device_passes=int(cell["passes"]), device_lr_frac=cell["lr_frac"],
                 device_selection_rule=rule, device_selected=bool(selected), arm_spec_sha256=spec_digest(spec),
                 selection_substitute=spec.get("selection_substitute"), **(overrides or {}))
        if arm == "LO" and spec.get("selected_cell") and not spec.get("select") and not selected:
            r.update(row_status=LO_NOT_SELECTED, comparison_status="DESCRIPTIVE",   # 06E §1: only the UNSEEN_VAL-
                     device_selection_rule=LO_NOT_SELECTED)                         # selected LO cell is evidence
        if arm in SELECTION_ARM:                          # DEV-FTPRE: the procedure and the cell it was selected as
            r.update(ft_procedure=spec.get("ft_procedure"), procedure_selection_arm=spec.get("procedure_selection_arm"),
                     procedure_selection_cell=spec.get("procedure_selection_cell"))
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ PF hook (item 3)
def pf_personal_fn(store, manifest, hex_by_euk: np.ndarray, device) -> Callable:
    """predict_adapter's personal_fn for PF: row -> eval_user_key -> client_key_hex -> p_u (0 for a user the store has
    never seen). The shared head is unchanged, so the frozen single-model adapter path applies as is."""
    import torch
    did = manifest.decision_id

    def fn(bt: Mapping):
        d = np.asarray(bt["decision_id"].cpu().numpy(), dtype=np.int64)
        rows = np.searchsorted(did, d)
        keys = hex_by_euk[manifest.eval_user_key[rows]]
        return torch.stack([store.p(str(k)) for k in keys]).to(device)
    return fn


def pf_evaluator(tr, hex_by_euk: np.ndarray) -> Callable:
    """evaluate_pf(mark, store) for launch_fl.execute: the harness trainer's frozen T3 path with the personal component
    (predict_adapter personal_fn) — the rows keep the harness format (method C relabelled by launch_fl.fl_rows)."""
    from fl.numerics import state_digest
    from v4eval.evaluate import evaluate
    from v4eval.frozen_config import apply_runtime_flags, check_runtime_flags
    from v4eval.predict import predict_adapter
    from v4eval.rows import rows_from_result

    def evaluate_pf(mark: float, store):
        m, fz = tr.evalp.manifest, tr.frozen
        msha = hashlib.sha256(f"PF|{state_digest(tr.adapter.broadcast_state())}|{store.digest()}".encode()).hexdigest()
        apply_runtime_flags(fz)
        try:
            check_runtime_flags(fz, tr.device)
            art = predict_adapter(tr.adapter, tr.evalp.batches(fz.eval_batch), m, config=fz, model_sha256=msha,
                                  personal_fn=pf_personal_fn(store, m, hex_by_euk, tr.device))
            res = evaluate(m, art, config=fz, evaluator_code_sha256=tr.evaluator_sha)
        finally:
            tr._apply_determinism()
        rows = rows_from_result(res, tr.key_base, method="PF", seed=tr.cfg.seed, selection_rule=None,
                                efe=tr.state.exposures / tr.N)
        rows = [dict(r, **tr._row_labels(), pf_personal_component="EVALUATED") for r in rows]
        return float(res["E2E"]["macro"]["mrr@20"]), rows, art.ranks, res
    return evaluate_pf


# ------------------------------------------------------------------------------------------------ real wiring
def parse_args(argv=None) -> argparse.Namespace:
    from v4hist.policy import HISTORY_POLICIES
    ap = argparse.ArgumentParser(description="PPSI V4 on-device / personalisation launcher (INT-04)")
    ap.add_argument("--arm", required=True, type=str.upper, choices=ARMS)
    ap.add_argument("--family", default=FAMILY, type=str.upper)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--history-policy", required=True, choices=HISTORY_POLICIES)
    ap.add_argument("--role", default="MAIN", type=str.upper)
    ap.add_argument("--select", action="store_true", help=f"--role {SELECTION_ROLE} only: grid + SELECTION.json")
    ap.add_argument("--selection", type=Path, default=None)
    ap.add_argument("--base", type=Path, default=None, help="FT: C_SMALL central checkpoint or FA FL checkpoint")
    ap.add_argument("--base-kind", choices=("central", "fl", WARM_BASE_KIND), default=None,
                    help="central (FT_C), fl (FT_FA), fa_warm (FL-15: FT_FA_WARM / FT_FA_WARM_FROZEN)")
    ap.add_argument("--base-sha256", default=None)
    ap.add_argument("--init-from", type=Path, default=None,
                    help="FT_PRE: the PRE run's PRETRAIN_FOR_FL_INIT.json (theta_pre = its endpoint_6.0 checkpoint)")
    ap.add_argument("--small-recipe", type=Path, default=None)
    ap.add_argument("--theta0-variants", type=Path, default=None)
    ap.add_argument("--n-shards", type=int, default=64)
    ap.add_argument("--shards", default=None, help="a:b — process shards [a, b) only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-clients", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--lock-path", type=Path, default=None)
    ap.add_argument("--manifest-hashes", type=Path, default=None)
    ap.add_argument("--header-list", type=Path, default=None)
    ap.add_argument("--transform-state", type=Path, default=None)
    ap.add_argument("--registry-ext", type=Path, default=None,
                    help="TEST roles: the DATA-09 REGISTRY_V2_TEST_EXT.json (default v4int.test_roles.DEFAULT_EXT)")
    return ap.parse_args(argv)


def parse_shards(s: Optional[str]) -> Optional[tuple]:
    if s is None:
        return None
    try:
        a, b = (int(x) for x in s.split(":"))
    except ValueError as e:
        raise LaunchRefused(f"--shards {s!r} must be a:b") from e
    if not 0 <= a < b:
        raise LaunchRefused("--shards a:b needs 0 <= a < b")
    return a, b


def check_ft_pre_selection_file(path: Path, *, smoke: bool, arm: str = "FT_PRE") -> dict:
    """DEV-FTPRE: the selection FT_PRE applies is the registered FT_C selection (contracts/device/SELECTION_FT_C.json):
    a verified PPSI_V4_DEVICE_SELECTION_1 object made on UNSEEN_VAL for arm FT_C; a real (non-smoke) run refuses any
    file whose bytes differ from the registered file, and (FIX-DEV-FTPRE M5) any file whose sha256 is not the pinned
    DL 231 value FT_PRE_SELECTION_SHA256 (a replaced contracts copy is refused, not trusted by path).
    FL-15 (06M_B B4): the same check for FT_FA_WARM / FT_FA_WARM_FROZEN (`arm`; FT_PRE's messages unchanged)."""
    p = Path(path)
    if not p.is_file():
        raise LaunchRefused(f"{arm} --selection {p} does not exist (the registered {FT_PRE_SELECTION.name})")
    obj = load_selection(p)
    if obj.get("arm") != SELECTION_ARM[arm]:
        raise LaunchRefused(f"{arm} applies the registered {SELECTION_ARM[arm]} selection, --selection is for "
                            f"{obj.get('arm')!r} (06L-C §C7)")
    got = _sha_file(p)
    reg = _sha_file(FT_PRE_SELECTION) if Path(FT_PRE_SELECTION).is_file() else None
    if not smoke and got != reg:
        raise LaunchRefused(f"{arm} applies the registered {FT_PRE_SELECTION} only (sha256 {str(reg)[:12]}…); "
                            f"--selection {p} is {got[:12]}… (06L-C §C7: 'the registered FT_C procedure with "
                            "SELECTION_FT_C')")
    if not smoke and got != FT_PRE_SELECTION_SHA256:
        raise LaunchRefused(f"{arm} --selection {p} is {got[:12]}…, not the DL 231 registered SELECTION_FT_C.json "
                            f"(pinned sha256 {FT_PRE_SELECTION_SHA256[:12]}…; 06L-C §C7)")
    return {"path": str(p), "file_sha256": got, "registered_path": str(FT_PRE_SELECTION), "registered_sha256": reg,
            "is_registered_file": got == reg, "self_sha256": obj["self_sha256"], "cell": obj["cell"]}


def check_ft_pre_args(args) -> dict:
    """DEV-FTPRE (torch-free): FT_PRE takes its base ONLY from --init-from (the PRE run's PRETRAIN_FOR_FL_INIT.json),
    never --base / --base-kind; --select is refused; --selection defaults to the registered SELECTION_FT_C.json (a real
    run refuses another file). --init-from is refused for every other arm."""
    init = getattr(args, "init_from", None)
    if args.arm != "FT_PRE":
        if init is not None:
            raise LaunchRefused("--init-from is for --arm FT_PRE only (theta_pre + the FT_C procedure; 06L-C §C7)")
        return {}
    if args.base is not None or getattr(args, "base_kind", None) is not None:
        raise LaunchRefused("FT_PRE's base is theta_pre from --init-from <PRETRAIN_FOR_FL_INIT.json>: --base / "
                            "--base-kind are refused (06K §2: never a checkpoint trained on MAIN users)")
    if init is None:
        raise LaunchRefused("FT_PRE needs --init-from <PRETRAIN_FOR_FL_INIT.json> (the PRE run's theta_pre record, "
                            "06K §1-§2)")
    if not Path(init).is_file():
        raise LaunchRefused(f"--init-from {init} does not exist")
    if args.select:
        raise LaunchRefused("FT_PRE applies the registered FT_C selection (06L-C §C7): --select refused")
    if args.selection is None:
        args.selection = FT_PRE_SELECTION
    sel = check_ft_pre_selection_file(args.selection, smoke=bool(args.smoke))
    return {"ft_pre": {"init_record": str(init), "init_record_sha256": _sha_file(Path(init)), "selection": sel}}


def check_ft_warm_args(args) -> dict:
    """FL-15 (06M_B B4; torch-free): FT_FA_WARM / FT_FA_WARM_FROZEN take their base ONLY from --base <an FA_WARM /
    FA_WARM_FROZEN MAIN checkpoint> --base-kind fa_warm (verified by load_warm_base at run time); --init-from and
    --select are refused; --selection defaults to the registered SELECTION_FT_C.json (a real run refuses any other file:
    check_ft_pre_selection_file, the DL 231 sha256 pin). --base-kind fa_warm is refused for every other arm."""
    if args.arm not in FT_WARM_BASE:
        if getattr(args, "base_kind", None) == WARM_BASE_KIND:
            raise LaunchRefused(f"--base-kind {WARM_BASE_KIND} is the FT_FA_WARM / FT_FA_WARM_FROZEN base (06M_B B4)")
        return {}
    if getattr(args, "init_from", None) is not None:
        raise LaunchRefused(f"{args.arm} starts from its FA_WARM MAIN checkpoint (--base), not theta_pre: --init-from "
                            "refused (theta_pre + FT_C is FT_PRE)")
    if args.base is None or getattr(args, "base_kind", None) != WARM_BASE_KIND:
        raise LaunchRefused(f"{args.arm} needs --base <the {FT_WARM_BASE[args.arm]} MAIN checkpoint "
                            f"(ckpt/<run_id>/best.pt | endpoint_6.0.pt)> --base-kind {WARM_BASE_KIND} (06M_B B4)")
    if not Path(args.base).is_file():
        raise LaunchRefused(f"--base {args.base} does not exist")
    if args.select:
        raise LaunchRefused(f"{args.arm} applies the registered FT_C selection (06M_B B4 / 06L-C §C7): --select refused")
    if args.selection is None:
        args.selection = FT_PRE_SELECTION
    sel = check_ft_pre_selection_file(args.selection, smoke=bool(args.smoke), arm=args.arm)
    return {"ft_fa_warm": {"base": str(args.base), "base_arm": FT_WARM_BASE[args.arm], "selection": sel}}


def load_pre_base(adapter, init_json: Path, *, model_variant: str, want_sha256: Optional[str] = None) -> dict:
    """DEV-FTPRE: theta_pre into `adapter` from the PRE run's PRETRAIN_FOR_FL_INIT.json, through launch_fl's FL-12
    loader (checkpoint bytes = the record's fl_init sha256 / bytes, hashed BEFORE torch.load; endpoint_6.0 of a
    PRETRAIN_FOR_FL run on PRETRAIN_TRAIN_5_100_NONMAIN; every MAIN-trained checkpoint refused) and apply_init_state
    (the loaded state hashes to the checkpoint's own state_digest). Returns the load_base-shaped record + the identity."""
    from fl.numerics import state_digest
    from v4int import launch_fl as LF
    state, ident, lrec = LF.load_pretrain_init(Path(init_json), model_variant=model_variant)
    LF.check_init_identity(ident)                       # the MAIN-users refusal, re-asserted by this launcher
    if want_sha256 and ident["checkpoint_sha256"] != str(want_sha256).lower():
        raise LaunchRefused(f"--base-sha256 {want_sha256} is not theta_pre's checkpoint sha256 "
                            f"{ident['checkpoint_sha256']} (the init record's)")
    d = LF.apply_init_state(adapter, state, ident["state_digest"])
    st = adapter.broadcast_state(clone=True)
    if state_digest(st) != d:
        raise LaunchRefused("theta_pre: the start state does not reproduce the loaded digest")
    return {"state": st, "digest": d, "kind": PRE_BASE_KIND, "path": lrec["checkpoint"],
            "file_sha256": ident["checkpoint_sha256"], "run_id": ident["run_id"], "init_from": dict(ident),
            "init_record": lrec["init_record"], "init_record_sha256": lrec["init_record_sha256"]}


def load_warm_base(adapter, path: Path, *, base_arm: str, seed: int, want_sha256: Optional[str] = None) -> dict:
    """FL-15 (06M_B B4): the FT_FA_WARM / FT_FA_WARM_FROZEN base = a REGISTERED FA_WARM / FA_WARM_FROZEN MAIN checkpoint,
    verified before its weights are used: the file is <run_dir>/ckpt/<run_id>/best.pt | endpoint_6.0.pt of a DONE_MAIN
    run (RUN_SUMMARY status, checkpoints list); ARM_SPEC.json / RUN_SUMMARY.json agree on run id / arm / arm-spec
    sha256 (= launch_fl.spec_digest of the recorded spec); arm = base_arm, role MAIN (a CALIBRATION screen run never
    qualifies), cfg seed = --seed (the paired FT_FA_WARM - FA_WARM delta); launch_fl.check_warm_spec / spec_tuning
    pass (init_from = the PRE run's endpoint_6.0 identity); nothing trained centrally on MAIN users (launch_fl
    _main_trained on the run id and the init identity: C_BIG / C_SMALL / C_FULL / C_MATCHED, MAIN / CALIB /
    FULL_TRAIN manifests) and not theta_pre itself; the payload is fl weights of that run with config_digest =
    launch_fl.fl_run_config(spec).digest(); optional --base-sha256 pin; the loaded state reproduces the checkpoint's
    theta digest. Returns the load_base-shaped record + the base identity."""
    from fl.checkpoint import load_checkpoint
    from fl.numerics import state_digest
    from v4int import launch_fl as LF
    p = Path(path)
    if not p.is_file():
        raise LaunchRefused(f"--base {p} does not exist")
    if p.name not in WARM_BASE_FILES or p.parent.parent.name != "ckpt":
        raise LaunchRefused(f"--base {p}: not <run_dir>/ckpt/<run_id>/{' | '.join(WARM_BASE_FILES)} of an FL run")
    run_id, run_dir = p.parent.name, p.parent.parent.parent
    try:
        side = json.loads((run_dir / "ARM_SPEC.json").read_text(encoding="utf-8"))
        summ = json.loads((run_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"--base {p}: {run_dir} ARM_SPEC.json / RUN_SUMMARY.json unreadable ({e})") from e
    spec = summ.get("arm_spec") or {}
    if summ.get("status") != "DONE_MAIN":
        raise LaunchRefused(f"--base {p}: the run's status is {summ.get('status')!r}, not DONE_MAIN")
    if side.get("run_id") != run_id or summ.get("run_id") != run_id:
        raise LaunchRefused(f"--base {p}: run id {side.get('run_id')!r} / {summ.get('run_id')!r} != the checkpoint "
                            f"directory's {run_id!r}")
    if spec.get("arm") != base_arm or side.get("arm") != base_arm or summ.get("arm") != base_arm:
        raise LaunchRefused(f"--base {p}: arm {spec.get('arm')!r} is not {base_arm} (06M_B B4)")
    if spec.get("role", "MAIN") != "MAIN":
        raise LaunchRefused(f"--base {p}: role {spec.get('role')!r}; the base is a MAIN run (never a CALIBRATION screen)")
    digest = LF.spec_digest(spec)
    if digest != side.get("arm_spec_sha256") or digest != summ.get("arm_spec_sha256") or \
            side.get("arm_spec") != spec:
        raise LaunchRefused(f"--base {p}: the arm spec does not hash to the recorded arm_spec_sha256")
    LF.check_warm_spec(spec)                              # init_from = the PRE run's endpoint_6.0 (MAIN-users refusal)
    LF.spec_tuning(spec)
    ident = dict(spec["init_from"])
    mk = LF._main_trained([ident.get("train_manifest")], [run_id, ident.get("run_id"), ident.get("run_label")])
    if mk is not None:
        raise LaunchRefused(f"--base {p}: trained centrally on MAIN users' data ({mk}); never a base (06K §2)")
    if run_id.startswith(PRE_RUN_ID_PREFIX):
        raise LaunchRefused(f"--base {p}: theta_pre itself ({run_id}); theta_pre + FT_C is FT_PRE")
    if int(spec["cfg"]["seed"]) != int(seed):
        raise LaunchRefused(f"--base {p}: seed {spec['cfg']['seed']} != --seed {seed} (FT_FA_WARM - FA_WARM is paired "
                            "per seed)")
    if p.name not in (summ.get("checkpoints") or []):
        raise LaunchRefused(f"--base {p}: {p.name} is not among the run's recorded checkpoints {summ.get('checkpoints')}")
    sha = _sha_file(p)
    if want_sha256 and sha != str(want_sha256).lower():
        raise LaunchRefused(f"--base {p}: sha256 {sha} != --base-sha256 {want_sha256}")
    obj = load_checkpoint(p)
    if obj.get("kind") != "weights" or obj.get("run_id") != run_id:
        raise LaunchRefused(f"--base {p}: not the fl weights checkpoint of {run_id}")
    if obj.get("config_digest") != LF.fl_run_config(spec, run_id, 1).digest():
        raise LaunchRefused(f"--base {p}: config_digest {str(obj.get('config_digest'))[:12]}… is not the recorded "
                            "arm spec's fl RunConfig digest")
    adapter.load_state_(obj["theta"])
    state = adapter.broadcast_state(clone=True)
    d = state_digest(state)
    if d != state_digest(obj["theta"]):
        raise LaunchRefused(f"--base {p}: the loaded state digest {d[:12]}… != the checkpoint's theta digest")
    return {"state": state, "digest": d, "kind": base_arm, "path": str(p), "file_sha256": sha, "run_id": run_id,
            "checkpoint": p.name, "base_kind": WARM_BASE_KIND, "base_arm_spec_sha256": digest,
            "base_round": obj.get("round"), "base_efe": obj.get("efe"), "init_from": ident,
            "warm_lr_factor": float(spec.get("warm_lr_factor", 1.0)),
            "warm_server_lr_factor": float(spec.get("warm_server_lr_factor", 1.0)),
            "frozen_item_tables": spec.get("frozen_item_tables")}


def refuse_theta_pre_base(arm: str, start_rec: Mapping) -> None:
    """A device run from theta_pre is FT_PRE (06L-C §C7): an FT_C / FT_FA base that is a PRETRAIN_FOR_FL checkpoint
    (run id PRETRAIN_FOR_FL_* ) refuses, and FT_PRE's start must be the theta_pre identity.
    FL-15: an FT_FA_WARM(_FROZEN) start must be a verified base of its base arm (load_warm_base kind)."""
    rid = str((start_rec or {}).get("run_id") or "")
    if arm != "FT_PRE" and rid.startswith(PRE_RUN_ID_PREFIX):
        raise LaunchRefused(f"{arm} --base is theta_pre ({rid}): a device run from theta_pre is FT_PRE (use --arm FT_PRE "
                            "--init-from <PRETRAIN_FOR_FL_INIT.json>; 06L-C §C7)")
    if arm in FT_WARM_BASE and ((start_rec or {}).get("kind") != FT_WARM_BASE[arm]
                                or (start_rec or {}).get("base_kind") != WARM_BASE_KIND):
        raise LaunchRefused(f"{arm}'s start is not a verified {FT_WARM_BASE[arm]} MAIN checkpoint: "
                            f"{ {k: v for k, v in (start_rec or {}).items() if k != 'init_from'} }")
    if arm == "FT_PRE":
        ident = (start_rec or {}).get("init_from") or {}
        if (start_rec or {}).get("kind") != PRE_BASE_KIND or ident.get("run_label") != PRE_RUN_LABEL or \
                not rid.startswith(PRE_RUN_ID_PREFIX) or ident.get("run_id") != rid:
            raise LaunchRefused(f"FT_PRE's start is not the theta_pre identity: {dict(start_rec or {})}")


WARM_RUN_ID_PREFIX = "FL_FA_WARM"                                     # launch_fl run ids of FA_WARM / FA_WARM_FROZEN


def refuse_warm_base(arm: str, base_path, start_rec: Mapping) -> None:
    """REV-FL-15 m3: an FT_C / FT_FA base that is an FA_WARM / FA_WARM_FROZEN checkpoint refuses (a device run from a
    warm FL model is FT_FA_WARM / FT_FA_WARM_FROZEN with --base-kind fa_warm; as FT_FA it would contaminate the FT_FA
    contrast). Detected by the checkpoint's run id (FL_FA_WARM*) and, for the FL layout <run>/ckpt/<run_id>/<file>, by
    the run's ARM_SPEC.json / RUN_SUMMARY.json arm (any run id, e.g. a renamed run)."""
    from v4int import launch_fl as LF
    rid = str((start_rec or {}).get("run_id") or "")
    arms = set()
    p = Path(base_path)
    if p.parent.parent.name == "ckpt":
        run_dir = p.parent.parent.parent
        for name, key in (("ARM_SPEC.json", ("arm",)), ("RUN_SUMMARY.json", ("arm_spec", "arm"))):
            try:
                obj = json.loads((run_dir / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for k in key:
                obj = obj.get(k) if isinstance(obj, Mapping) else None
            if obj:
                arms.add(str(obj))
    warm = rid.upper().startswith(WARM_RUN_ID_PREFIX) or bool(arms & set(LF.WARM_ARMS))
    if warm:
        raise LaunchRefused(f"{arm} --base is a warm FL checkpoint (run id {rid!r}, arm {sorted(arms) or '?'}): a device "
                            "run from FA_WARM / FA_WARM_FROZEN is --arm FT_FA_WARM / FT_FA_WARM_FROZEN --base-kind "
                            f"{WARM_BASE_KIND} (06M_B B4; REV-FL-15 m3)")


def resolve_start(args, harness, worker, *, theta0: str, model_variant: str) -> tuple:
    """(start state, base record or None, LAUNCH start record) of a device run: LO = theta0 of the seed (the untrained
    harness), FT_PRE = theta_pre from --init-from (load_pre_base), FT_C / FT_FA = --base (load_base); then
    refuse_theta_pre_base (a device run from theta_pre is FT_PRE, both ways). run() takes its start from here only."""
    base = None
    if args.arm == "LO":
        start = harness.adapter.broadcast_state(clone=True)       # theta0 of the seed (the harness is untrained)
        start_rec = {"kind": "theta0", "init_sha256": harness.built.init_sha256, "theta0_registered": theta0}
    elif args.arm == "FT_PRE":                                    # DEV-FTPRE: theta_pre from the PRE init record
        base = load_pre_base(worker, args.init_from, model_variant=model_variant, want_sha256=args.base_sha256)
        start = base["state"]
        start_rec = {k: v for k, v in base.items() if k != "state"}
    elif args.arm in FT_WARM_BASE:                                # FL-15 (06M_B B4): the registered FA_WARM checkpoint
        base = load_warm_base(worker, args.base, base_arm=FT_WARM_BASE[args.arm], seed=int(args.seed),
                              want_sha256=args.base_sha256)
        start = base["state"]
        start_rec = {k: v for k, v in base.items() if k != "state"}
    else:
        base = load_base(worker, args.base, args.base_kind, args.base_sha256)
        start = base["state"]
        start_rec = {k: v for k, v in base.items() if k != "state"}
        refuse_warm_base(args.arm, args.base, start_rec)          # REV-FL-15 m3: a warm base is FT_FA_WARM(_FROZEN)
    refuse_theta_pre_base(args.arm, start_rec)                    # a device run from theta_pre is FT_PRE (both ways)
    return start, base, start_rec


def ft_pre_row_labels(start_rec: Mapping, selection_rec: Optional[Mapping]) -> dict:
    """The base identity labels of FT_PRE rows (none of them a comparison-key field)."""
    ident = start_rec["init_from"]
    return {"base_source": PRE_BASE_KIND, "base_run_id": ident["run_id"],
            "base_checkpoint_sha256": ident["checkpoint_sha256"], "base_train_manifest_id": ident["train_manifest"],
            "base_init_record_sha256": start_rec["init_record_sha256"],
            "ft_selection_self_sha256": None if selection_rec is None else selection_rec.get("self_sha256")}


def ft_warm_row_labels(start_rec: Mapping, selection_rec: Optional[Mapping]) -> dict:
    """FL-15 (06M_B B4): the base identity labels of FT_FA_WARM / FT_FA_WARM_FROZEN rows (none a comparison-key
    field): the FA_WARM run / checkpoint and, through its init_from, the theta_pre it started from."""
    ident = start_rec["init_from"]
    return {"base_source": start_rec["kind"], "base_run_id": start_rec["run_id"],
            "base_checkpoint": start_rec["checkpoint"], "base_checkpoint_sha256": start_rec["file_sha256"],
            "base_arm_spec_sha256": start_rec["base_arm_spec_sha256"], "base_training_paradigm": "FEDERATED",
            "base_init_run_id": ident["run_id"], "base_init_checkpoint_sha256": ident["checkpoint_sha256"],
            "base_init_train_manifest_id": ident["train_manifest"],
            "base_warm_lr_factor": start_rec["warm_lr_factor"],
            "base_warm_server_lr_factor": start_rec["warm_server_lr_factor"],
            "ft_selection_self_sha256": None if selection_rec is None else selection_rec.get("self_sha256")}


def preflight(args, *, runner=None, environ=None, gate_path: Path = G1_GATE) -> dict:
    """Torch-free refusals + the shared GPU checks."""
    from v4hist.policy import HistoryPolicyError, require_policy
    from v4sup.gpu import default_runner
    env = os.environ if environ is None else environ
    rec = {"utc": utc(), "argv": list(sys.argv), "host": platform.node(), "smoke": bool(args.smoke), "arm": args.arm,
           "role": args.role, "allow_shared_gpu": bool(args.allow_shared_gpu)}
    try:
        rec["history_policy"] = require_policy(args.history_policy)
    except HistoryPolicyError as e:
        raise LaunchRefused(str(e)) from e
    if args.gpu is None or int(args.gpu) < 0:
        raise LaunchRefused("an explicit physical GPU id (--gpu >= 0) is required")
    if args.family != FAMILY:
        raise LaunchRefused(f"--family {args.family}: the device arms run the ARCHITECTURE_LOCK winner ({FAMILY}) only")
    rec.update(check_ft_pre_args(args))                  # DEV-FTPRE (before the role check: the default selection)
    rec.update(check_ft_warm_args(args))                 # FL-15 (06M_B B4): FT_FA_WARM / FT_FA_WARM_FROZEN
    check_role(args.role, select=bool(args.select), arm=args.arm,
               selection={"arm": args.arm} if args.selection is not None else None)
    check_lo_selection(args.role, arm=args.arm, select=bool(args.select), selection=args.selection,
                       smoke=bool(args.smoke))
    if args.role in TEST_ROLES:                         # INT-07: the TEST gate before any GPU / data access
        from v4int import test_roles as TR
        if not TR.test_gate_ok(args.role):
            raise LaunchRefused(f"--role {args.role} refused: the DATA-09 TEST gate is closed (ALL_METHODS_LOCK + its "
                                "MANIFEST_BUILD TEST_EXPOSURE_REGISTER entry)")
        rec["test_gate"] = {"required": True, "role": args.role}
    elif getattr(args, "registry_ext", None) is not None:
        raise LaunchRefused("--registry-ext is for the TEST roles only")
    if args.arm == "LO" and args.base is not None:
        raise LaunchRefused("LO starts from theta0: --base is for FT_C / FT_FA")
    if args.arm not in ("LO", "FT_PRE"):
        want = "central" if args.arm == "FT_C" else "fl" if args.arm == "FT_FA" else WARM_BASE_KIND   # FL-15
        if args.base is None or args.base_kind != want:
            raise LaunchRefused(f"{args.arm} needs --base <checkpoint> --base-kind {want}")
    if int(args.n_shards) < 1:
        raise LaunchRefused("--n-shards must be >= 1")
    parse_shards(args.shards)
    if args.smoke:
        if not 1 <= int(args.smoke_clients) <= 64:
            raise LaunchRefused("--smoke-clients must be in [1, 64]")
        rec["g1_gate"] = {"required": False, "reason": "smoke (SMOKE_NOT_EVIDENCE)"}
    else:
        if not Path(gate_path).is_file():
            raise LaunchRefused(f"non-smoke run refused: {gate_path} does not exist (G1 not approved)")
        rec["g1_gate"] = {"required": True, "path": str(gate_path),
                          "sha256": hashlib.sha256(Path(gate_path).read_bytes()).hexdigest()}
    if args.dry_run:
        rec["gpu_check"] = {"skipped": "dry-run"}
        return rec
    rec["gpu_check"] = gpu_checks(args, runner or default_runner, env)
    return rec


def assert_arm_allowed(arm: str, family: str, lock_path: Path = ARCH_LOCK) -> dict:
    from v4train.selection import LosingFamilyError, assert_downstream_allowed
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    decision = {"status": lock.get("decision_status"), "winner_family": lock.get("winner_family"),
                "forbidden_downstream_families": lock.get("forbidden_downstream_families", [])}
    code = LOCK_CODE[arm]
    try:
        assert_downstream_allowed(decision, family, code)
    except LosingFamilyError as e:
        raise LaunchRefused(str(e)) from e
    if code not in (lock.get("allowed_subsequent_method_codes") or []):
        raise LaunchRefused(f"method {code} is not an allowed subsequent method of the ARCHITECTURE_LOCK")
    return {"path": str(lock_path), "sha256": hashlib.sha256(Path(lock_path).read_bytes()).hexdigest(),
            "winner_family": lock.get("winner_family"), "lock_method_code": code,
            "grid": lock.get("e11_9a_recipe_grid")}


def load_base(adapter, path: Path, kind: str, want_sha256: Optional[str] = None) -> dict:
    """Load the FT base into `adapter`: central = v4train checkpoint {"meta", "model"}; fl = fl.checkpoint weights."""
    import torch
    from fl.numerics import state_digest
    p = Path(path)
    sha = _sha_file(p)
    if want_sha256 and sha != want_sha256:
        raise LaunchRefused(f"--base {p}: sha256 {sha} != --base-sha256 {want_sha256}")
    if kind == "central":
        obj = torch.load(p, map_location="cpu", weights_only=True)
        adapter.module.load_state_dict(obj["model"], strict=True)
        rec_digest = (obj.get("meta") or {}).get("state_digest")
    else:
        from fl.checkpoint import load_checkpoint
        obj = load_checkpoint(p)
        adapter.load_state_(obj["theta"])
        rec_digest = state_digest(obj["theta"])
    state = adapter.broadcast_state(clone=True)
    d = state_digest(state)
    if rec_digest is not None and rec_digest != d:
        raise LaunchRefused(f"--base {p}: loaded state digest {d} != the checkpoint's recorded {rec_digest}")
    return {"state": state, "digest": d, "path": str(p), "file_sha256": sha, "kind": kind,
            "run_id": obj.get("run_id") or (obj.get("meta") or {}).get("run_id")}


def _read_cols(manifest_dir, vm, table: str, cols) -> dict:
    from v4base.identity import _read_parquet_columns
    info = vm.tables[table]
    return _read_parquet_columns(Path(manifest_dir) / info["file"], info["file_sha256"], list(cols))


def train_target_ts_by_position(tp, manifest_dir) -> np.ndarray:
    c = _read_cols(manifest_dir, tp.verified, "decisions", ["target_ts_us"])
    return np.asarray(c["target_ts_us"], dtype=np.int64)[tp.verified.canonical_rows]


def eval_row_arrays(ep, manifest_dir) -> tuple:
    """(client_key_hex per manifest row, decision_ts_us per manifest row) from the manifest's own verified tables."""
    from v4base.identity import eval_client_key_hex_by_eval_user_key
    m = ep.manifest
    hexes = eval_client_key_hex_by_eval_user_key(ep, manifest_dir=manifest_dir)
    c = _read_cols(manifest_dir, ep.verified, "decisions", ["decision_id", "decision_ts_us"])
    did = np.asarray(c["decision_id"], dtype=np.int64)
    o = np.argsort(did, kind="stable")
    if not np.array_equal(did[o], m.decision_id):
        raise LaunchRefused("the evaluation decisions table does not hold exactly the manifest's decision_ids")
    return hexes[m.eval_user_key], np.asarray(c["decision_ts_us"], dtype=np.int64)[o], hexes


def eval_batch_at(ep, pos: np.ndarray) -> dict:
    """StoreEvalProvider._make for arbitrary stream positions (same store read, offsets, renorm and checks)."""
    from v4int.providers import ProviderIntegrityError, add_session_offsets
    pos = np.asarray(pos, dtype=np.int64)
    b = add_session_offsets(ep.store.batch(ep.ds, pos), ep.store, ep.ds, pos)
    if ep.renorm is not None:
        b = ep.renorm.apply(b)
    if not (np.array_equal(b["decision_id"].numpy(), ep._ids[pos]) and
            np.array_equal(b["target_class"].numpy(), ep._target[pos])):
        raise ProviderIntegrityError(f"{ep.manifest_id}: an evaluation batch disagrees with the manifest")
    return b


def _to(bt: Mapping, dev) -> dict:
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in bt.items()}


def run(args, pre: dict, env_rec: dict, say: Callable = print) -> int:
    import v4train
    import ppsi_env
    snap = ppsi_env.verify_snapshot()
    iso = ppsi_env.assert_isolated()
    if v4train.LAUNCH_CUDA_VISIBLE_DEVICES != str(int(args.gpu)):
        raise LaunchRefused("the CUDA_VISIBLE_DEVICES export did not precede the v4train import")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise LaunchRefused("expected exactly one visible CUDA device")
    torch.set_num_threads(int(args.n_threads))
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    from fl.numerics import state_digest
    from v4data.common import dataset_root
    from v4eval.frozen_config import apply_runtime_flags, check_runtime_flags
    from v4int.launch_central import (DETERMINISM_MODE, _theta0_for, bind_data_protocol_lock_v2,
                                      build_trainer_registry_view, data_paths, load_production_schema)
    from v4int.launch_fl import _small_recipe, client_index
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    from v4int.smoke import SMOKE_RUN_PREFIX
    from v4models import build
    from v4train.provider import expected_manifests
    from v4train.recipe import RunConfig
    from v4train.trainer import CentralTrainer
    allowed = assert_arm_allowed(args.arm, args.family)
    recipe = _small_recipe(args)
    selection = load_selection(args.selection) if args.selection is not None else None
    if args.arm == "FT_PRE":                                      # DEV-FTPRE: re-checked at run time (registered file)
        check_ft_pre_selection_file(args.selection, smoke=bool(args.smoke))
    elif args.arm in FT_WARM_BASE:                                # FL-15: the same registered SELECTION_FT_C pin
        check_ft_pre_selection_file(args.selection, smoke=bool(args.smoke), arm=args.arm)
    run_dir = Path(args.run_dir)
    run_id = args.run_id or f"{SMOKE_RUN_PREFIX if args.smoke else ''}DEV_{args.arm}_{args.role}_s{args.seed}"
    paths = data_paths(args)
    lockctx = bind_data_protocol_lock_v2(paths, smoke=bool(args.smoke))
    view = build_trainer_registry_view(lockctx["manifest_registry_path"], run_dir,
                                       registry_rel_path=lockctx["manifest_registry_rel_path"],
                                       registry_file_sha256=lockctx["manifest_registry_file_sha256"])
    reg_kw = dict(history_policy=args.history_policy, paths=paths, log=say,
                  manifest_registry=lockctx["manifest_registry_path"], registry_root=V4)
    sel_manifest = None if selection is None else verify_selection_manifest(      # F6: before any provider / GPU work
        selection, lockctx["manifest_registry_path"], reg_kw, smoke=bool(args.smoke))
    train_id, eval_id = expected_manifests("COHORT", "MAIN", registry=str(view["path"]))
    tp = StoreTrainProvider(train_id, **reg_kw)                   # the harness pair (MAIN) — the model's population
    ep = StoreEvalProvider(eval_id, prefetch=0, **reg_kw)
    key = comparison_key_base(paths, tp, ep)
    theta0 = _theta0_for(argparse.Namespace(theta0_variants=args.theta0_variants, smoke=args.smoke),
                         lockctx["schema_id"], {"model_variant": recipe["model_variant"]}, int(args.seed))
    cfg = RunConfig(run_id=run_id, family=FAMILY, seed=int(args.seed), exposure_point="end",
                    cuda_visible_devices=str(int(args.gpu)), phase="MAIN", population="COHORT",
                    peak_lr=recipe["peak_lr"], log_every=50, checkpoint_every=0, key=key,
                    test_overrides=bool(args.smoke), determinism_mode=DETERMINISM_MODE,
                    manifest_registry=str(view["path"]), data_version=lockctx["data_version"],
                    schema_registry=str(lockctx["schema_registry_path"]), recipe_id=recipe["recipe_id"],
                    recipe_registry=recipe["recipe_registry"], model_variant=recipe["model_variant"],
                    dropout=tuple(recipe["dropout"]), schedule=recipe["schedule"], theta0_init_sha256=theta0)
    schema = load_production_schema(lockctx["schema_id"], lockctx["schema_self_sha256"], lockctx["schema_registry_path"])
    tr = CentralTrainer(cfg, schema, tp, ep, run_dir / "eval_harness", device="cuda:0", v1_root=dataset_root())
    worker = build(FAMILY, schema, int(args.seed), "cuda:0", v1_root=dataset_root(), variant=recipe["model_variant"],
                   dropout=tuple(recipe["dropout"])).adapter
    start, base, start_rec = resolve_start(args, tr, worker, theta0=theta0, model_variant=recipe["model_variant"])
    spec = device_spec(args.arm, seed=int(args.seed), peak_lr=recipe["peak_lr"], role=args.role,
                       start_digest=state_digest(start), select=bool(args.select), selection=selection, fused=True)
    roles = json.loads(Path(lockctx["manifest_registry_path"]).read_text(encoding="utf-8"))["roles"]
    gate, test_rec = None, None
    mdir = eval_mdir = paths.manifest_dir
    if args.role in TEST_ROLES:                                   # INT-07: the gated TEST route
        gate, rtp, rep, key_train, eval_mdir, test_rec = retro_role_providers(args, lockctx, reg_kw, paths, tp)
        roles = dict(roles, **{args.role: test_rec["pair"]})
    elif args.role == "MAIN":
        rtp, rep = tp, ep
    else:
        r = roles[args.role]
        rtp = StoreTrainProvider(r["train"], **reg_kw) if "train" in r else None
        rep = StoreEvalProvider(r["eval"], prefetch=0, **reg_kw)
    row_key, row_dts, _ = eval_row_arrays(rep, eval_mdir)
    sup_pos = ts_by_pos = None
    if rtp is not None:
        _keys, sup_pos = client_index(rtp, mdir)
        ts_by_pos = train_target_ts_by_position(rtp, mdir)
    plans = build_client_plans(rep.manifest, row_key, row_decision_ts=row_dts, support_pos_by_key=sup_pos,
                               warm=args.role in WARM_ROLES)
    kb, over = dict(tr.key_base), {}
    if gate is not None:
        from v4int.retro_store import retro_role_key
        kb, over = retro_role_key(tr.key_base, comparison_key_base(paths, key_train, rep), gate, role=args.role,
                                  eval_manifest_content_sha256=rep.manifest_binding["content_sha256"])
    elif args.role != "MAIN":
        rk = comparison_key_base(paths, rtp or tp, rep)
        for f in ("population_hash", "candidate_order_hash", "train_manifest_id", "eval_manifest_id"):
            kb[f] = rk[f]
        over = {"manifest_role": args.role, "train_manifest_id": rk["train_manifest_id"],
                "eval_manifest_id": rk["eval_manifest_id"]}
    labels = tr._row_labels()
    down = 0 if args.arm == "LO" else int(worker.manifest.shared_bytes)          # the one base download (FT)
    if args.arm == "FT_PRE":                                      # DEV-FTPRE: the base identity on every row
        over = dict(over, **ft_pre_row_labels(start_rec, selection))
    elif args.arm in FT_WARM_BASE:                                # FL-15 (06M_B B4): the FA_WARM base identity
        over = dict(over, **ft_warm_row_labels(start_rec, selection))

    def row_fn(res, cell, selected):
        return device_rows(res, kb, labels, spec=spec, cell=cell, selected=selected, down_bytes=down,
                           overrides=dict(over, device_run_id=run_id))

    def get_support(p):
        b = rtp.batch(p.support)
        b[TS_KEY] = torch.as_tensor(ts_by_pos[p.support], dtype=torch.long)
        return b

    fz = tr.frozen
    apply_runtime_flags(fz)
    check_runtime_flags(fz, dev)
    launch = {"task": "INT-04 launch_device", "label": "SMOKE_NOT_EVIDENCE" if args.smoke else "PRODUCTION",
              "run_id": run_id, "arm": args.arm, "role": args.role, "spec": spec, "spec_sha256": spec_digest(spec),
              "start": start_rec, "architecture_lock": allowed, "small_recipe": recipe,
              "selection": None if selection is None else {"path": str(args.selection),
                                                           "self_sha256": selection["self_sha256"],
                                                           "manifest_check": sel_manifest,
                                                           **({"file_sha256": _sha_file(Path(args.selection))}
                                                              if args.arm in SELECTION_ARM else {})},
              "history_policy": args.history_policy, "comparison_key": kb, "role_manifests": roles.get(args.role),
              "test_role": test_rec,
              "clients": {"n": len(plans), "n_no_support": sum(p.support is None for p in plans)},
              "preflight": pre, "environment": env_rec, "data_paths": paths.as_json(),
              "snapshot": {"verified_files": len(snap["files"]), "isolated_modules": len(iso)},
              "device": {"physical_gpu": int(args.gpu), "name": torch.cuda.get_device_name(dev)},
              **({"device_pin_sha256": tr.device_pin_sha256} if tr.device_pin_sha256 else {}),
              "torch": torch.__version__, "shards": {"n": int(args.n_shards), "range": args.shards},
              "mode": "resume" if args.resume else "fresh"}
    lp = _next_launch_path(run_dir)
    write_json(lp, launch)
    say(f"[device] {run_id}: {lp.name} written; {len(plans)} clients")
    out = execute(spec=spec, run_dir=run_dir, worker=worker, start_state=start, plans=plans, get_support=get_support,
                  get_eval_batch=lambda sp: _to(eval_batch_at(rep, sp), dev), manifest=rep.manifest, frozen=fz,
                  row_fn=row_fn, evaluator_sha=tr.evaluator_sha, n_shards=int(args.n_shards),
                  shard_range=parse_shards(args.shards), resume=bool(args.resume),
                  smoke_clients=int(args.smoke_clients) if args.smoke else None, say=say,
                  eval_manifest_content_sha256=rep.manifest_binding["content_sha256"])
    return 0 if out["status"] in ("DONE", "PARTIAL", "SMOKE_DONE_NOT_EVIDENCE") else 1


def retro_role_providers(args, lockctx: Mapping, reg_kw: Mapping, paths, tp) -> tuple:
    """INT-07: (gate, support provider or None, RETRO_TEST provider, key-train object, eval manifest dir, record) of a
    TEST role. Called AFTER the harness is built (the harness never sees the extension). The DATA-09 wrappers are
    installed with role=None (no trainer view); the role is resolved under the gate; the support rows of MAIN_TEST are
    the harness MAIN_TRAIN provider's (the same manifest; checked by content sha), of the other support roles a
    StoreTrainProvider of the unchanged support manifest verified through the extension; the eval-only
    TRUE_COLD_TEST has no support. The key's train side is always verified through the extension."""
    from v4int import retro_store as RS
    from v4int import test_roles as TR
    from v4int.providers import StoreEvalProvider, StoreTrainProvider
    ext_path = Path(args.registry_ext) if args.registry_ext is not None else Path(TR.DEFAULT_EXT)
    base_path = Path(lockctx["manifest_registry_path"])
    try:
        TR.install(ext_path, role=None, eager=True)
        res = TR.resolve_role(args.role, ext_path=ext_path, base_path=base_path)
    except Exception as e:                                          # noqa: BLE001
        raise LaunchRefused(f"--role {args.role} refused: {type(e).__name__}: {e}") from e
    gate = res["gate"]
    pair = {k: res[k] for k in ("train", "eval") if res.get(k)}
    ext_kw = dict(reg_kw, manifest_registry=ext_path)
    rep = StoreEvalProvider(pair["eval"], prefetch=0, test_gate=gate, **ext_kw)
    train_id = pair.get("train") or tp.manifest_id
    key_train = RS.train_key_ref(train_id, history_policy=reg_kw["history_policy"], paths=paths,
                                 manifest_registry=ext_path, registry_root=reg_kw.get("registry_root"))
    if "train" not in pair:
        rtp = None
    elif pair["train"] == tp.manifest_id:
        if tp.manifest_binding["content_sha256"] != key_train.manifest_binding["content_sha256"]:
            raise LaunchRefused("the harness train manifest differs from the extension's support manifest")
        rtp = tp
    else:
        rtp = StoreTrainProvider(pair["train"], **ext_kw)
    eval_mdir = RS.retro_manifest_dir(ext_path, pair["eval"], reg_kw.get("registry_root"))
    rec = {"role": args.role, "pair": pair, "test_gate": gate.binding(),
           "registry_ext": {"path": str(ext_path), "self_sha256": res["registry_self_sha256"]},
           "eval_manifest_content_sha256": rep.manifest_binding["content_sha256"],
           "support_from": None if rtp is None else ("harness_train_provider" if rtp is tp else "extension")}
    return gate, rtp, rep, key_train, eval_mdir, rec


def main(argv=None) -> int:
    from v4int.launch_central import export_environment
    args = parse_args(argv)
    try:
        pre = preflight(args)
        if args.dry_run:
            print(json.dumps({"preflight": pre, "cells": grid_cells(args.arm), "knobs": KNOBS}, indent=1,
                             default=str))
            return 0
        env_rec = export_environment(int(args.gpu))
        return run(args, pre, env_rec)
    except LaunchRefused as e:
        print(f"LAUNCH REFUSED: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
