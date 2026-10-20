"""v4int.central_ft (P2-05, owner: Phase-2 controls) - control C1 of contracts/PREG2.md §4.2: a central
fine-tune of theta_pre (the PRE run's endpoint_6.0, checkpoint sha256 cf08588c..., state digest ee33345b...) with the
SMALL recipe shape, run by the UNEDITED v4int.launch_central (the shim pattern of v4int.pretrain_small / v4int.c_full).

  python -B code/v4int/central_ft.py --role CALIBRATION --peak-lr 1e-4 [--p2-freeze-item-tables] \
         --init-from <PRETRAIN_FOR_FL_INIT.json> -- --family SASREC --seed 2026 --phase MAIN \
         --recipe-id SMALL_SASREC_D64_B2 --gpu <id> --run-dir <dir> --history-policy USER_HISTORY_L50 [launcher args]
  python -B code/v4int/central_ft.py --role MAIN --peak-lr <x> [--p2-freeze-item-tables] --init-from <...> \
         --c1-selection contracts/fl/P2_C1_SELECTION.json --c1-selection-sha256 <file sha256> -- <the same args>
  python -B code/v4int/central_ft.py --dry-run-out <file.json> [--no-providers] <shim options> -- <launcher args>

Arms (labels, never a Phase-1 row): P2_C1_FT_CENTRAL_FULL (every parameter trained) and P2_C1_FT_CENTRAL_FROZEN
(--p2-freeze-item-tables: item_embed.weight, output_embed, output_bias frozen = the central analogue of S8 / W*'s trained
subset). Roles: MAIN (MAIN_TRAIN_5_100_131072 -> MAIN_VALIDATION_5_100_131072; seeds 2026 / 2027) and CALIBRATION
(CALIB_TRAIN_5_100_32768 -> CALIB_VALIDATION_5_100_32768; the LR screen, seed 2026). Budget axis = the FL arms': the
trainer's MAIN phase (6.0 EFE on the role's TRAIN decisions, 12 half-EFE marks 0.5 .. 6.0, PRACTICAL_BEST = the
strictly-best of those marks, CONTROLLED_BUDGET = 6.0); theta_pre is never evaluated as a mark, so "mark 0" can never be
a PRACTICAL_BEST candidate. No continuation (fixed 6.0 EFE). Run id (derived, never free):
  [SMOKE_NOT_EVIDENCE__]P2_C1_FT_CENTRAL_<FULL|FROZEN>_<MAIN|CALIB>_lr<peak>_s<seed>   (peak as 1e-4, 3e-5, ...)
Peak LR (--peak-lr): CALIBRATION takes a value of the registered screen {1e-5, 3e-5, 1e-4, 3e-4} or its one-step edge
extension {3e-6, 1e-3} (PREG2 §4.2 / §5.7); MAIN takes the same grid and must equal the variant's winner in the
write-once contracts/fl/P2_C1_SELECTION.json (P2-06), bound by its file sha256 (--c1-selection-sha256, required for a
non-smoke MAIN run; hashed before and after verification; format / self_sha256 / stage "C1" verified, then re-verified
by v4int.p2_select.load_p2_selection; the peak = variants.<FULL|FROZEN>.winner_peak_lr).

How (launch_central, v4train, launch_fl and fl are NOT edited; every wrapper is installed in THIS process only and acts
on the P2_C1_* / P2_C2_* run ids only, so every other run is byte-for-byte unaffected):
  1. launch_central._recipe_values: the SMALL recipe is loaded and verified by the original; the shim's peak replaces
     the recipe's (the recipe's own peak is recorded as recipe_peak_lr in the LAUNCH record);
  2. launch_central.bind_data_protocol_lock_v2: the original, then (v4train now imported, after the CUDA export) the
     class wrappers 4-6 are installed;
  3. launch_central.build_trainer_registry_view: MAIN = the original view; CALIBRATION = the same file name with roles
     MAIN / CONTINUATION / CALIBRATION = REGISTRY_V2's CALIBRATION pair (entries copied byte-for-byte, self-hashed with
     v4train.registry's own rule), so the unedited trainer runs its MAIN phase (6.0 EFE, 12 marks, best / endpoint
     weights) on the CALIB manifests; the providers still verify against REGISTRY_V2 itself;
  4. labels (shared with v4int.c_full_small): RunConfig.run_label returns the arm label instead of PRODUCTION for a
     P2 run id, and CentralTrainer._row_labels sets method = the arm label, row_status / comparison_status P2_CONTROL,
     c_matched_eligible = downstream_seed_allowed = False (v4train.selection refuses the label everywhere);
  5. RunConfig._validate_train02 (P2_C1 run ids only): the recipe rule is checked on a copy whose peak is the recipe's,
     then the actual peak must be the shim's peak and a registered grid value;
  6. CentralTrainer (P2_C1 run ids only): TO-CHECK of PHASE2_IMPLEMENTATION_PLAN P2-05 RESOLVED - CentralTrainer has no
     initial-state argument (it always builds theta0 from (variant, seed) and checks it against THETA0_VARIANTS), so
     the wrapper runs right after the constructor and BEFORE step 0 (state.updates == 0, empty AdamW state):
     theta_pre is loaded by launch_fl.load_pretrain_init (the record's checkpoint bytes hashed before torch.load; PRE
     endpoint_6.0 / PRETRAIN_FOR_FL / PRETRAIN_TRAIN_5_100_NONMAIN; every MAIN-trained checkpoint refused) and must
     carry the pinned checkpoint sha256 and state digest; launch_fl.apply_init_state loads it strictly and proves the
     broadcast state hashes to that digest. FROZEN: fl.runtime.freeze_item_tables (the S8 freeze: requires_grad False,
     fixed buffers, no recentering) and a fresh AdamW over the trained parameters only (same recipe values). The
     continuation is switched off (trainer.auto_continue False; continue_run refused; a STILL_IMPROVING run's `latest`
     is deleted after RESULT_ROWS / RUN_SUMMARY). The init (record sha256, checkpoint sha256 / bytes, state digest),
     the frozen keys, the trained parameter count and the peak go into RUN_START.json["p2_c1"], every checkpoint meta,
     every summary and every row (p2_* fields; key initialization_block P2_THETA_PRE|...); a resume re-applies the init
     and refuses when RUN_START's p2_c1 identity differs.
Nothing here reads a metric value except read_c1_run (for the P2-06 selection tool; CALIB runs only, a MAIN run
is refused before any value-bearing file is opened: PREG2 §6.3 S8).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Optional

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

TASK = "P2-05 v4int.central_ft"
AUTHORITY = "contracts/PREG2.md §4.2; company/plans/PHASE2_IMPLEMENTATION_PLAN.md P2-05"
PRE_CHECKPOINT_SHA256 = "cf08588c11f019cc760f129ba7ba6b23806297660e0170f9938436a5cb88cf78"
PRE_STATE_DIGEST = "ee33345be4f20b61a53cd80fa11e1af702031bfcd1425af490825cd0019dc54a"
RECIPE_ID = "SMALL_SASREC_D64_B2"
MODEL_VARIANT = "SASREC_D64_B2"
REQUIRED = {"family": "SASREC", "phase": "MAIN", "recipe_id": RECIPE_ID, "population": "COHORT",
            "history_policy": "USER_HISTORY_L50"}
ROLES = ("MAIN", "CALIBRATION")
ROLE_TAG = {"MAIN": "MAIN", "CALIBRATION": "CALIB"}
SEEDS = {"MAIN": (2026, 2027), "CALIBRATION": (2026,)}
SCREEN_LRS = (1e-5, 3e-5, 1e-4, 3e-4)                       # PREG2 §4.2
EDGE_LRS = (3e-6, 1e-3)                                     # §4.2 extension values (§5.7: one step, once)
LR_GRID = tuple(sorted(SCREEN_LRS + EDGE_LRS))
FROZEN_ITEM_KEYS = ("item_embed.weight", "output_embed", "output_bias")     # = fl.runtime.FROZEN_ITEM_KEYS (checked)
C1_ARMS = {"FULL": "P2_C1_FT_CENTRAL_FULL", "FROZEN": "P2_C1_FT_CENTRAL_FROZEN"}
C2_ARM = "P2_C2_CFULL_SMALL"
# run-id prefix -> arm label (the label wrapper 4 serves both P2-05 shims)
P2_RUN_PREFIXES = {f"{C1_ARMS['FULL']}_": C1_ARMS["FULL"], f"{C1_ARMS['FROZEN']}_": C1_ARMS["FROZEN"],
                   f"{C2_ARM}_": C2_ARM}
ROW_STATUS = "P2_CONTROL"
COMPARISON_STATUS = "P2_CONTROL"
CONTINUATION_RULE = "DISABLED_FIXED_6.0_EFE"
INIT_BLOCK = "P2_THETA_PRE"
C1_SELECTION_PATH = V4 / "contracts" / "fl" / "P2_C1_SELECTION.json"
C1_SELECTION_STAGE = "C1"
C1_SELECTION_FORMAT = "PPSI_V4_P2_C1_SELECTION_1"           # = v4int.p2_select.FORMATS["C1"]
RECORD_PREFIX = "P2_C1_SHIM_"
SMOKE_PREFIX = "SMOKE_NOT_EVIDENCE__"                       # = v4int.smoke.SMOKE_RUN_PREFIX (checked in tests)

_CTX: Optional[dict] = None                                 # the shim context of THIS process (set by install())


class CentralFTRefused(RuntimeError):
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


def canonical_self_sha256(obj: Mapping, key: str = "self_sha256") -> str:
    """= v4train.registry.canonical_self_sha256 (stdlib copy: usable before the CUDA export)."""
    body = {k: v for k, v in obj.items() if k != key}
    enc = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(enc.encode("utf-8")).hexdigest()


def lr_tag(lr: float) -> str:
    return f"{float(lr):.0e}".replace("-0", "-")


def in_grid(lr: float) -> bool:
    return round(float(lr), 12) in {round(x, 12) for x in LR_GRID}


def core_run_id(run_id) -> str:
    rid = str(run_id or "")
    return rid[len(SMOKE_PREFIX):] if rid.startswith(SMOKE_PREFIX) else rid


def p2_label_of_run_id(run_id) -> Optional[str]:
    core = core_run_id(run_id)
    for prefix, label in P2_RUN_PREFIXES.items():
        if core.startswith(prefix):
            return label
    return None


def c1_variant_of_run_id(run_id) -> Optional[str]:
    lab = p2_label_of_run_id(run_id)
    return next((v for v, a in C1_ARMS.items() if a == lab), None)


def c1_run_id(variant: str, role: str, peak_lr: float, seed: int, smoke: bool = False) -> str:
    return f"{SMOKE_PREFIX if smoke else ''}{C1_ARMS[variant]}_{ROLE_TAG[role]}_lr{lr_tag(peak_lr)}_s{int(seed)}"


# ------------------------------------------------------------------------------------------------ argv / contract
def split_argv(argv: list) -> tuple:
    if "--" not in argv:
        raise CentralFTRefused("usage: central_ft.py [shim options] -- <launch_central arguments>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description="P2-05 C1 central fine-tune shim")
    ap.add_argument("--role", required=True, type=str.upper, choices=ROLES)
    ap.add_argument("--peak-lr", required=True, type=float)
    ap.add_argument("--p2-freeze-item-tables", action="store_true",
                    help="P2_C1_FT_CENTRAL_FROZEN: item_embed.weight / output_embed / output_bias frozen")
    ap.add_argument("--init-from", required=True, type=Path, help="theta_pre: the PRE run's PRETRAIN_FOR_FL_INIT.json")
    ap.add_argument("--c1-selection", type=Path, default=None, help=f"MAIN: {C1_SELECTION_PATH} (registered path)")
    ap.add_argument("--c1-selection-sha256", default=None, help="MAIN: the selection file's sha256 (binding)")
    ap.add_argument("--dry-run-out", type=Path, default=None)
    ap.add_argument("--no-providers", action="store_true", help="dry run: verify registries / init / selection only")
    return ap.parse_args(argv[:i]), list(argv[i + 1:])


def variant_of(shim) -> str:
    return "FROZEN" if shim.p2_freeze_item_tables else "FULL"


def recipe_path() -> Path:
    from v4int.small_recipe import RECIPE_PATH                # stdlib-only at import
    return Path(RECIPE_PATH)


def check_launch_args(args, shim) -> str:
    """The C1 launch contract on the parsed shim / launch_central arguments; returns the (derived) run id."""
    for k, want in REQUIRED.items():
        if getattr(args, k, None) != want:
            raise CentralFTRefused(f"C1 needs --{k.replace('_', '-')} {want} (got {getattr(args, k, None)!r})")
    if shim.role == "MAIN" and getattr(args, "smoke", False):    # REV-P2-05 m1: checked before any data load
        raise CentralFTRefused("--smoke is refused with --role MAIN: a smoke evaluates MAIN_VALIDATION into "
                               "SMOKE_EVAL.json, and no Phase-2 MAIN value may exist outside sealed/ (PREG2 §6.3 S8); "
                               "smoke the CALIBRATION role instead")
    if int(args.seed) not in SEEDS[shim.role]:
        raise CentralFTRefused(f"C1 {shim.role} uses seed {' / '.join(map(str, SEEDS[shim.role]))} "
                               f"(PREG2 §4.2), not {args.seed}")
    for k in ("train_manifest", "eval_manifest", "role", "hp_id", "calibration_record", "continuation_budget",
              "peak_lr"):
        if getattr(args, k, None):
            raise CentralFTRefused(f"C1 refuses launcher --{k.replace('_', '-')}: the manifests come from the shim's "
                                   "--role, the peak LR from the shim's --peak-lr")
    for k in ("transform_state", "manifest_dir", "manifest_hashes", "header_list", "lock_path", "theta0_variants",
              "hp_registry"):
        if getattr(args, k, None) is not None:
            raise CentralFTRefused(f"C1 refuses --{k.replace('_', '-')}: the locked tree, lock, transform state, "
                                   "theta0 registry and the SMALL recipe are used unchanged")
    if getattr(args, "continue_run", False):
        raise CentralFTRefused("C1 has no continuation (fixed 6.0 EFE, PREG2 §4.2): --continue-run refused")
    if getattr(args, "allow_shared_gpu", False) and not getattr(args, "coschedule_06g", False):
        raise CentralFTRefused("C1 refuses --allow-shared-gpu without --coschedule-06g (06G §5 / 06K §1)")
    if getattr(args, "coschedule_06g", False) and not getattr(args, "allow_shared_gpu", False):
        raise CentralFTRefused("--coschedule-06g takes effect only together with --allow-shared-gpu (DECISION_LOG 208)")
    tr = getattr(args, "tuned_recipes", None)
    if tr is not None and Path(tr).resolve() != recipe_path().resolve() and not args.smoke:
        raise CentralFTRefused(f"C1 uses the production SMALL recipe {recipe_path()} (got {tr})")
    if not in_grid(shim.peak_lr):
        raise CentralFTRefused(f"--peak-lr {shim.peak_lr} is not a registered C1 value {LR_GRID} (PREG2 §4.2 / §5.7)")
    if shim.role == "CALIBRATION" and (shim.c1_selection is not None or shim.c1_selection_sha256):
        raise CentralFTRefused("the CALIBRATION screen takes no --c1-selection (it produces it)")
    rid = c1_run_id(variant_of(shim), shim.role, shim.peak_lr, int(args.seed), smoke=bool(args.smoke))
    if args.run_id and args.run_id != rid:
        raise CentralFTRefused(f"the C1 run id is derived ({rid}); --run-id {args.run_id} refused")
    return rid


def normalise_argv(largv: list, args, rid: str) -> list:
    out = list(largv)
    if "--run-id" not in out:
        out += ["--run-id", rid]
        args.run_id = rid
    if getattr(args, "tuned_recipes", None) is None:
        out += ["--tuned-recipes", str(recipe_path())]
        args.tuned_recipes = recipe_path()
    return out


def _p2_select_loader():
    """The P2-06 verifier of a written selection file (format, self_sha256, labels = recipes, the decision re-derived
    from the recorded candidates): v4int.p2_select.load_p2_selection(path, stage)."""
    try:
        from v4int.p2_select import load_p2_selection
    except ImportError as e:
        raise CentralFTRefused(f"the C1 selection is verified by v4int.p2_select (P2-06), which is not importable "
                               f"({e})") from None
    return load_p2_selection


def bind_c1_selection(path, *, variant: str, peak_lr: float, expect_sha256: Optional[str],
                      registered: Path = C1_SELECTION_PATH, reverify=None) -> dict:
    """MAIN: the write-once P2-06 file contracts/fl/P2_C1_SELECTION.json, bound by its file sha256 (hashed before and
    after the checks): registered path only, strict JSON, format PPSI_V4_P2_C1_SELECTION_1, self_sha256 (the canonical
    rule = launch_fl._canon), stage C1, then the P2-06 verifier (`reverify`, default v4int.p2_select.load_p2_selection:
    the recorded decision re-derived from the recorded candidates), and variants.<FULL|FROZEN>.winner_peak_lr ==
    --peak-lr (the P2-06 output layout)."""
    p = Path(path if path is not None else registered)
    if p.resolve() != Path(registered).resolve():
        raise CentralFTRefused(f"--c1-selection {p} is not the registered {registered} (a copy never qualifies)")
    if not p.is_file():
        raise CentralFTRefused(f"the C1 selection {p} does not exist (run the CALIBRATION screen and P2-06 first)")
    sha = _sha(p)
    if not expect_sha256:
        raise CentralFTRefused("a non-smoke MAIN C1 run binds the selection by its sha256: --c1-selection-sha256")
    if sha != str(expect_sha256).lower():
        raise CentralFTRefused(f"{p} sha256 {sha} != --c1-selection-sha256 {expect_sha256}")
    try:
        obj = _strict(p)
    except (OSError, ValueError) as e:
        raise CentralFTRefused(f"{p} is not strict JSON ({e})") from None
    if not isinstance(obj, dict) or obj.get("format") != C1_SELECTION_FORMAT:
        raise CentralFTRefused(f"{p} is not a {C1_SELECTION_FORMAT} file")
    if obj.get("self_sha256") != canonical_self_sha256(obj):
        raise CentralFTRefused(f"{p}: self_sha256 missing or wrong")
    if obj.get("stage") != C1_SELECTION_STAGE:
        raise CentralFTRefused(f"{p}: stage {obj.get('stage')!r} is not {C1_SELECTION_STAGE}")
    loader = reverify if reverify is not None else _p2_select_loader()
    try:
        loader(p, C1_SELECTION_STAGE)
    except Exception as e:  # noqa: BLE001  (P2SelectionRefused or any verification failure refuses the launch)
        raise CentralFTRefused(f"{p}: the P2-06 verifier refused it ({type(e).__name__}: {e})") from None
    win = ((obj.get("variants") or {}).get(variant) or {})
    try:
        chosen = float(win["winner_peak_lr"])
    except (KeyError, TypeError, ValueError):
        raise CentralFTRefused(f"{p}: no variants.{variant}.winner_peak_lr") from None
    if win.get("run_label") not in (None, C1_ARMS[variant]):
        raise CentralFTRefused(f"{p}: variants.{variant}.run_label {win.get('run_label')!r} != {C1_ARMS[variant]}")
    if round(chosen, 12) != round(float(peak_lr), 12):
        raise CentralFTRefused(f"--peak-lr {peak_lr} != the C1 selection's {variant} winner {chosen}")
    if _sha(p) != sha:
        raise CentralFTRefused(f"{p} changed while it was verified")
    return {"path": str(p), "sha256": sha, "self_sha256": obj["self_sha256"], "format": obj["format"],
            "stage": obj["stage"], "variant": variant, "peak_lr": chosen, "flags": obj.get("flags"),
            "reverified_by": getattr(loader, "__module__", "?") + "." + getattr(loader, "__name__", "?")}


# ------------------------------------------------------------------------------------------------ wrapper 4: labels
def p2_row_fields(label: str) -> dict:
    return {"method": label, "row_status": ROW_STATUS, "comparison_status": COMPARISON_STATUS,
            "c_matched_eligible": False, "downstream_seed_allowed": False, "p2_arm": label, "p2_authority": AUTHORITY}


def install_p2_label_wrappers(RC=None, TR=None) -> dict:
    """Wrapper 4 on the given (default: the real) v4train.recipe / v4train.trainer modules; idempotent. Acts on P2 run
    ids only; run_label is a derived property, so RunConfig.as_json / sha256 are unchanged."""
    if RC is None:
        import v4train.recipe as RC  # noqa: N811
    if TR is None:
        import v4train.trainer as TR  # noqa: N811
    cls, tcls = RC.RunConfig, TR.CentralTrainer
    if not getattr(cls, "__p2_label__", False):
        orig_prop = cls.__dict__["run_label"]

        def run_label(self):
            lab = orig_prop.fget(self)
            p2 = p2_label_of_run_id(self.run_id)
            return p2 if lab == "PRODUCTION" and p2 is not None else lab

        run_label.__wrapped__ = orig_prop
        cls.run_label = property(run_label, doc="P2-05 wrapper 4 (P2 control labels)")
        cls.__p2_label__ = True
    if not getattr(tcls, "__p2_label__", False):
        orig_rows = tcls._row_labels

        def _row_labels(self):
            lab = orig_rows(self)
            p2 = p2_label_of_run_id(getattr(getattr(self, "cfg", None), "run_id", None))
            if p2 is not None:
                lab.update(p2_row_fields(p2))
                lab.update(getattr(self, "_p2_row_extra", None) or {})
            return lab

        _row_labels.__wrapped__ = orig_rows
        tcls._row_labels = _row_labels
        tcls.__p2_label__ = True
    return {"prefixes": dict(P2_RUN_PREFIXES), "row_status": ROW_STATUS}


# ------------------------------------------------------------------------------------------------ wrapper 5: peak LR
def make_validate_wrapper(orig, error=ValueError, ctx_getter=lambda: _CTX):
    """Wrapper 5 as a pure factory: for a P2_C1 run id the recipe rule is checked on a copy carrying the recipe's peak,
    and the actual peak must be the shim's peak and a registered grid value; any other run id is delegated unchanged."""
    def _validate_train02(self, fam):
        if c1_variant_of_run_id(self.run_id) is None:
            return orig(self, fam)
        ctx = ctx_getter()
        if not ctx or ctx.get("recipe_peak_lr") is None:
            raise error(f"{self.run_id}: a P2_C1 run id needs the v4int.central_ft shim context (peak LR binding)")
        if round(float(self.peak_lr), 12) != round(float(ctx["peak_lr"]), 12) or not in_grid(self.peak_lr):
            raise error(f"P2 C1 peak LR {self.peak_lr} is not the shim's {ctx['peak_lr']} / not in {LR_GRID}")
        return orig(dataclasses.replace(self, peak_lr=float(ctx["recipe_peak_lr"])), fam)

    _validate_train02.__wrapped__ = orig
    return _validate_train02


def install_validate_wrapper(RC=None) -> None:
    if RC is None:
        import v4train.recipe as RC  # noqa: N811
    cls = RC.RunConfig
    if getattr(cls, "__p2_c1_peak__", False):
        return
    cls._validate_train02 = make_validate_wrapper(cls._validate_train02, error=RC.RecipeError)
    cls.__p2_c1_peak__ = True


# ------------------------------------------------------------------------------------------------ wrapper 6: trainer
def apply_c1_init(tr, ctx: Mapping) -> dict:
    """Load theta_pre into a freshly constructed CentralTrainer BEFORE step 0 (see the module docstring, item 6)."""
    import torch
    import v4int.launch_fl as F
    err = ctx.get("error") or CentralFTRefused
    variant = c1_variant_of_run_id(tr.cfg.run_id)
    if variant is None:
        raise err(f"{tr.cfg.run_id} is not a P2_C1 run id")
    if (variant == "FROZEN") != bool(ctx["freeze"]):
        raise err(f"{tr.cfg.run_id}: the run id says {variant} but --p2-freeze-item-tables is {bool(ctx['freeze'])}")
    if tr.state.updates or tr.state.exposures or len(tr.opt.state):
        raise err("theta_pre must be loaded before step 0 (fresh trainer, empty AdamW state)")
    if round(float(tr.cfg.peak_lr), 12) != round(float(ctx["peak_lr"]), 12):
        raise err(f"trainer peak {tr.cfg.peak_lr} != the shim's {ctx['peak_lr']}")
    try:
        state, ident, launch = F.load_pretrain_init(Path(ctx["init_record"]), model_variant=tr.built.variant)
    except F.LaunchRefused as e:
        raise err(f"theta_pre refused: {e}") from None
    pins = {"checkpoint_sha256": ctx.get("pin_checkpoint_sha256", PRE_CHECKPOINT_SHA256),
            "state_digest": ctx.get("pin_state_digest", PRE_STATE_DIGEST)}
    bad = {k: ident.get(k) for k, want in pins.items() if ident.get(k) != want}
    if bad:
        raise err(f"theta_pre refused: {bad} differ from the registered PRE init {pins} (PREG2 §4.2)")
    try:
        got = F.apply_init_state(tr.adapter, state, ident["state_digest"])
    except F.LaunchRefused as e:
        raise err(str(e)) from None
    total = int(tr.adapter.manifest.shared_numel)
    freeze = None
    if variant == "FROZEN":
        from fl.runtime import FROZEN_ITEM_KEYS as FK
        from fl.runtime import freeze_item_tables
        if tuple(FK) != FROZEN_ITEM_KEYS:
            raise err(f"fl.runtime.FROZEN_ITEM_KEYS {FK} != {FROZEN_ITEM_KEYS}")
        freeze = freeze_item_tables(tr.adapter, FROZEN_ITEM_KEYS)
        r = tr.cfg.recipe                                     # a fresh AdamW (same recipe values) on the trained set
        tr.opt = torch.optim.AdamW(tr.adapter.param_groups(r.weight_decay), lr=tr.cfg.peak_lr, betas=r.betas,
                                   eps=r.eps, fused=r.fused)
    trained = int(sum(p.numel() for p in tr.adapter.shared_parameters()))
    tr.auto_continue = False
    role = ctx["role"]
    reg_role = role                                           # the REGISTRY_V2 role of the manifests
    identity = {"arm": C1_ARMS[variant], "variant": variant, "role": role, "peak_lr": float(tr.cfg.peak_lr),
                "frozen_item_tables": list(FROZEN_ITEM_KEYS) if freeze else None,
                "init_checkpoint_sha256": ident["checkpoint_sha256"], "init_checkpoint_bytes": ident["checkpoint_bytes"],
                "init_state_digest": ident["state_digest"], "init_record_sha256": launch["init_record_sha256"],
                "init_run_id": ident["run_id"], "continuation": CONTINUATION_RULE}
    rec = {"task": TASK, "authority": AUTHORITY, "identity": identity, "init_identity": dict(ident),
           "init_record": launch["init_record"], "loaded_state_digest": got,
           "applied": "after CentralTrainer.__init__, before step 0 (CentralTrainer has no initial-state argument)",
           "theta0_built_sha256": tr.built.init_sha256, "freeze": freeze, "trained_numel": trained,
           "total_shared_numel": total, "recipe_peak_lr": ctx.get("recipe_peak_lr"),
           "selection": ctx.get("selection"), "manifest_role": reg_role,
           "mark0_rule": "theta_pre is not evaluated as a mark; PRACTICAL_BEST over marks 0.5 .. 6.0 only"}
    tr._p2_c1 = rec
    tr._p2_row_extra = {"p2_role": role, "p2_peak_lr": float(tr.cfg.peak_lr),
                        "p2_frozen_item_tables": identity["frozen_item_tables"], "p2_trained_numel": trained,
                        "p2_init": INIT_BLOCK, "p2_init_checkpoint_sha256": ident["checkpoint_sha256"],
                        "p2_init_state_digest": ident["state_digest"], "p2_continuation": CONTINUATION_RULE,
                        "manifest_role": reg_role}
    tr.key_base["initialization_block"] = (f"{INIT_BLOCK}|SASREC|checkpoint_sha256={ident['checkpoint_sha256']}|"
                                           f"state_digest={ident['state_digest']}")
    tr.key_base["manifest_role"] = reg_role
    return rec


def install_trainer_wrappers(TR=None, ctx_getter=lambda: _CTX) -> None:
    """Wrapper 6 on v4train.trainer.CentralTrainer (idempotent); every method delegates unchanged for a non-C1 run."""
    if TR is None:
        import v4train.trainer as TR  # noqa: N811
    tcls = TR.CentralTrainer
    if getattr(tcls, "__p2_c1__", False):
        return
    o_init, o_start, o_meta, o_summary = tcls.__init__, tcls._run_start_record, tcls._meta, tcls._summary
    o_check, o_cont, o_finish = tcls._check_start_identity, tcls.continue_run, tcls._finish_main

    def is_c1(self) -> bool:
        return c1_variant_of_run_id(self.cfg.run_id) is not None

    def __init__(self, config, *a, **kw):
        o_init(self, config, *a, **kw)
        if is_c1(self):
            ctx = ctx_getter()
            if not ctx:
                raise TR.TrainerError(f"{config.run_id}: a P2_C1 run needs the v4int.central_ft shim context")
            apply_c1_init(self, ctx)

    def _run_start_record(self, disk):
        out = o_start(self, disk)
        if is_c1(self):
            out["p2_c1"] = dict(self._p2_c1)
        return out

    def _meta(self, **extra):
        out = o_meta(self, **extra)
        if is_c1(self):
            out["p2_c1"] = dict(self._p2_c1["identity"])
        return out

    def _summary(self, status, **extra):
        out = o_summary(self, status, **extra)
        if is_c1(self):
            out["p2_c1"] = dict(self._p2_c1["identity"], trained_numel=self._p2_c1["trained_numel"])
            out["p2_trainable_numel"] = int(self._p2_c1["trained_numel"])     # read by v4int.p2_select.c1_run_adapter
        return out

    def _check_start_identity(self):
        start = o_check(self)
        if is_c1(self):
            was = (start.get("p2_c1") or {}).get("identity")
            if was != self._p2_c1["identity"]:
                raise TR.TrainerError(f"resume refused: the P2 C1 identity changed ({was} -> "
                                      f"{self._p2_c1['identity']})")
        return start

    def continue_run(self):
        if is_c1(self):
            raise TR.TrainerError("a P2 C1 control has no continuation (fixed 6.0 EFE, PREG2 §4.2)")
        return o_cont(self)

    def _finish_main(self):
        o_finish(self)
        if not is_c1(self) or not self.ckpt.exists("latest"):
            return
        from v4train.checkpoint import read_json, write_json_atomic
        sha = self._latest_sha()                     # STILL_IMPROVING at 6.0: no continuation -> drop the resumable
        sp = self.art / "RUN_SUMMARY.json"
        summ = read_json(sp)
        summ["checkpoints"].pop("latest", None)
        summ.update(latest_kept=False, latest_deleted_sha256=sha, continuation=None, p2_continuation=CONTINUATION_RULE)
        write_json_atomic(sp, summ)
        self._public_summary(summ)
        self.ckpt.delete("latest", f"P2-05 C1: resumable deleted at 6.0 ({CONTINUATION_RULE}; PREG2 §4.2)")
        self._event(event="P2_C1_NO_CONTINUATION", still_improving=bool(summ.get("still_improving")))

    for name, fn, orig in (("__init__", __init__, o_init), ("_run_start_record", _run_start_record, o_start),
                           ("_meta", _meta, o_meta), ("_summary", _summary, o_summary),
                           ("_check_start_identity", _check_start_identity, o_check),
                           ("continue_run", continue_run, o_cont), ("_finish_main", _finish_main, o_finish)):
        fn.__wrapped__ = orig
        setattr(tcls, name, fn)
    tcls.__p2_c1__ = True


# ------------------------------------------------------------------------------------------------ wrappers 1-3
def calibration_view_obj(reg: Mapping, *, registry_rel_path: str, registry_file_sha256: str) -> dict:
    """The CALIBRATION trainer view: roles MAIN / CONTINUATION / CALIBRATION = REGISTRY_V2's CALIBRATION pair, the two
    manifest entries byte-for-byte, self-hashed with v4train.registry's rule."""
    from v4train.registry import with_self_sha256
    pair = dict(reg["roles"]["CALIBRATION"])
    roles = {r: dict(pair) for r in ("MAIN", "CONTINUATION", "CALIBRATION")}
    return with_self_sha256({
        "data_version": reg["data_version"], "roles": roles,
        "manifests": {mid: dict(reg["manifests"][mid]) for mid in sorted(set(pair.values()))},
        "derived_from": {"path": registry_rel_path, "file_sha256": registry_file_sha256,
                         "self_sha256": reg.get("self_sha256"),
                         "note": "v4int.central_ft (P2-05 C1, role CALIBRATION): MAIN / CONTINUATION / CALIBRATION = "
                                 "the source registry's CALIBRATION pair (the trainer's MAIN phase on the CALIB "
                                 "manifests); every manifest field byte-identical to the source registry"}})


def install(LC, ctx: dict, record: dict) -> None:
    """Wrappers 1-3 on the launch_central module object; wrappers 4-6 are installed by wrapper 2 (after the CUDA
    export, because they import v4train). Sets this process's shim context."""
    global _CTX
    ctx.setdefault("error", LC.LaunchRefused)
    _CTX = ctx
    orig_recipe, orig_bind, orig_view = LC._recipe_values, LC.bind_data_protocol_lock_v2, LC.build_trainer_registry_view

    def recipe_values(args, fam):
        t = orig_recipe(args, fam)
        if t.get("recipe_id") != RECIPE_ID or t.get("model_variant") != MODEL_VARIANT:
            raise LC.LaunchRefused(f"C1 runs the SMALL recipe {RECIPE_ID} ({MODEL_VARIANT}), not {t.get('recipe_id')}")
        ctx["recipe_peak_lr"] = float(t["peak_lr"])
        record["recipe"] = {"recipe_id": t["recipe_id"], "recipe_sha256": t.get("recipe_sha256"),
                            "recipe_peak_lr": float(t["peak_lr"]), "peak_lr": float(ctx["peak_lr"])}
        return dict(t, peak_lr=float(ctx["peak_lr"]), recipe_peak_lr=float(t["peak_lr"]),
                    p2_peak_lr_source=("contracts/fl/P2_C1_SELECTION.json" if ctx["role"] == "MAIN"
                                       else "C1 CALIBRATION screen grid (PREG2 §4.2)"))

    def bind(paths, *, smoke, gate_path=LC.G1_GATE):
        lockctx = orig_bind(paths, smoke=smoke, gate_path=gate_path)
        record["labels"] = install_p2_label_wrappers()
        install_validate_wrapper()
        install_trainer_wrappers()
        record["class_wrappers"] = ["RunConfig.run_label", "CentralTrainer._row_labels", "RunConfig._validate_train02",
                                    "CentralTrainer.__init__/_run_start_record/_meta/_summary/_check_start_identity/"
                                    "continue_run/_finish_main"]
        return lockctx

    def view(manifest_registry_path, run_dir, *, registry_rel_path, registry_file_sha256):
        run_dir = Path(run_dir)
        if ctx["role"] == "MAIN":
            out = orig_view(manifest_registry_path, run_dir, registry_rel_path=registry_rel_path,
                            registry_file_sha256=registry_file_sha256)
        else:
            reg = _strict(Path(manifest_registry_path))
            obj = calibration_view_obj(reg, registry_rel_path=registry_rel_path,
                                       registry_file_sha256=registry_file_sha256)
            run_dir.mkdir(parents=True, exist_ok=True)
            p = run_dir / "REGISTRY_V2_TRAINER_VIEW.json"
            p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            out = {"path": p, "self_sha256": obj["self_sha256"], "roles": sorted(obj["roles"]),
                   "source_registry_self_sha256": reg.get("self_sha256"),
                   "source_registry_file_sha256": registry_file_sha256}
        record["trainer_view"] = {"path": str(out["path"]), "self_sha256": out["self_sha256"], "roles": out["roles"],
                                  "role": ctx["role"]}
        if not record.get("dry_run"):
            record["_record_path"] = str(_write_record(run_dir, record))
        return out

    recipe_values.__wrapped__, bind.__wrapped__, view.__wrapped__ = orig_recipe, orig_bind, orig_view
    LC._recipe_values = recipe_values
    LC.bind_data_protocol_lock_v2 = bind
    LC.build_trainer_registry_view = view


def _write_record(run_dir: Path, record: dict, path=None) -> Path:
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


def check_init_record(init_json: Path) -> dict:
    """Dry-run / pre-launch check of theta_pre on CPU: launch_fl.load_pretrain_init + the registered pins."""
    import v4int.launch_fl as F
    try:
        _state, ident, launch = F.load_pretrain_init(Path(init_json), model_variant=MODEL_VARIANT)
    except F.LaunchRefused as e:
        raise CentralFTRefused(f"theta_pre refused: {e}") from None
    if (ident["checkpoint_sha256"], ident["state_digest"]) != (PRE_CHECKPOINT_SHA256, PRE_STATE_DIGEST):
        raise CentralFTRefused(f"theta_pre refused: {ident['checkpoint_sha256'][:12]}… / {ident['state_digest'][:12]}… "
                               "is not the registered PRE init (cf08588c… / ee33345b…)")
    return {"identity": ident, "init_record_sha256": launch["init_record_sha256"]}


# ------------------------------------------------------------------------------------------------ dry run
def dry_run(LC, args, shim, record: dict) -> dict:
    """launch_central.run up to the RunConfig (no trainer, no GPU work, nothing written under --run-dir)."""
    import tempfile
    record["environment"] = LC.export_environment(int(args.gpu))
    import v4train  # noqa: F401
    from v4int.small_recipe import install as small_install
    small_install()
    from v4train.provider import expected_manifests
    from v4train.registry import load_manifest_registry
    run_dir = Path(args.run_dir)
    before = sorted(str(p) for p in run_dir.rglob("*")) if run_dir.exists() else None
    t0 = time.time()
    out = {"note": "P2 C1 dry run: no training, nothing written under run_dir", "utc": LC.utc(), "shim": record}
    out["init"] = check_init_record(shim.init_from)
    tuned = LC._recipe_values(args, args.family)
    paths = LC.data_paths(args)
    lockctx = LC.bind_data_protocol_lock_v2(paths, smoke=bool(args.smoke))
    reg = _strict(Path(lockctx["manifest_registry_path"]))
    want_pair = tuple(reg["roles"][shim.role][k] for k in ("train", "eval"))
    with tempfile.TemporaryDirectory(prefix="central_ft_dryrun_") as td:
        view = LC.build_trainer_registry_view(lockctx["manifest_registry_path"], Path(td),
                                              registry_rel_path=lockctx["manifest_registry_rel_path"],
                                              registry_file_sha256=lockctx["manifest_registry_file_sha256"])
        vreg = load_manifest_registry(view["path"])
        want = tuple(expected_manifests("COHORT", "MAIN", registry=vreg))
        out["trainer_view"] = {"self_sha256": view["self_sha256"], "roles": view["roles"], "expected_manifests": want}
        if want != want_pair:
            raise CentralFTRefused(f"the trainer view resolves MAIN to {want}, not the {shim.role} pair {want_pair}")
        out["tuned_recipe"] = tuned
        if shim.no_providers:
            out["config_sha256"] = None
            out["config_note"] = "--no-providers: the comparison key (and so the config hash) needs the providers"
        else:
            out.update(_dry_config(LC, args, lockctx, view, tuned, vreg))
    after = sorted(str(p) for p in run_dir.rglob("*")) if run_dir.exists() else None
    if after != before:
        raise CentralFTRefused(f"the dry run changed {run_dir}")
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def _dry_config(LC, args, lockctx, view, tuned, vreg) -> dict:
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    from v4train.recipe import RunConfig
    from v4train.registry import check_provider
    paths = LC.data_paths(args)
    tp = StoreTrainProvider(vreg.role("MAIN")[0], history_policy=args.history_policy, paths=paths, log=print,
                            manifest_registry=lockctx["manifest_registry_path"], registry_root=LC.V4)
    ep = StoreEvalProvider(vreg.role("MAIN")[1], history_policy=args.history_policy, paths=paths,
                           prefetch=int(args.eval_prefetch), log=print,
                           manifest_registry=lockctx["manifest_registry_path"], registry_root=LC.V4)
    key = comparison_key_base(paths, tp, ep)
    extra = {"determinism_mode": LC.DETERMINISM_MODE, "manifest_registry": str(view["path"]),
             "data_version": lockctx["data_version"], "schema_registry": str(lockctx["schema_registry_path"]),
             "recipe_id": tuned["recipe_id"], "recipe_registry": tuned["recipe_registry"],
             "model_variant": tuned["model_variant"], "dropout": tuple(tuned["dropout"]), "schedule": tuned["schedule"],
             "theta0_init_sha256": LC._theta0_for(args, lockctx["schema_id"], tuned, int(args.seed))}
    cfg = RunConfig(run_id=args.run_id, family=args.family, seed=int(args.seed), exposure_point="end",
                    cuda_visible_devices=str(int(args.gpu)), phase="MAIN", population="COHORT",
                    peak_lr=tuned["peak_lr"], log_every=int(args.log_every),
                    checkpoint_every=int(args.checkpoint_every or 0), calibration_record=None,
                    continuation_budget=None, expected_init_sha256=args.expected_init_sha256, key=key,
                    test_overrides=bool(args.smoke), **extra)
    cfg.validate()
    return {"run_id": cfg.run_id, "config_sha256": cfg.sha256(), "config": cfg.as_json(), "run_label": cfg.run_label,
            "bindings": {"train": check_provider(vreg, "MAIN", "train", tp), "eval": check_provider(vreg, "MAIN",
                                                                                                  "eval", ep)},
            "effective_auto_continuation": False, "continuation_note": f"{CONTINUATION_RULE} (P2-05 trainer wrapper)",
            "train_provider": {"n_decisions": tp.n_decisions}, "eval_provider": {"describe_n": int(ep.manifest.n)}}


# ------------------------------------------------------------------------------------------------ the P2-06 reader
EVAL_MARKS = tuple(0.5 * i for i in range(1, 13))
CALIB_EVAL_ID = "CALIB_VALIDATION_5_100_32768"                # REGISTRY_V2 role CALIBRATION (eval)


def read_c1_run(run_dir: Path) -> dict:
    """A finished C1 CALIBRATION run, verified (for the P2-06 selection tool; it READS the run's sealed CALIB values):
    RUN_START p2_c1 identity with role CALIBRATION and the CALIB eval manifest - checked BEFORE any value-bearing file is
    opened, so a MAIN run is refused unread (PREG2 §6.3 S8) -, sealed RUN_SUMMARY DONE_MAIN with the 12 marks 0.5 ..
    6.0, no `latest` kept, exactly one PRACTICAL_BEST
    E2E / macro / mrr@20 row labelled with the arm (row_status P2_CONTROL) whose value equals the run's evaluation at
    its PRACTICAL_BEST mark (>= 0.5: theta_pre is never a candidate). Non-finite -> value None, finite False."""
    import math
    rd = Path(run_dir)
    art = rd / "sealed" if (rd / "sealed").is_dir() else rd
    try:
        start = _strict(rd / "RUN_START.json")                 # config and hashes only (no outcome)
    except (OSError, ValueError) as e:
        raise CentralFTRefused(f"{rd}: RUN_START unreadable ({e})") from None
    ident = (start.get("p2_c1") or {}).get("identity") or {}
    arm = ident.get("arm")
    if arm not in C1_ARMS.values() or c1_variant_of_run_id(start.get("run_id")) != ident.get("variant"):
        raise CentralFTRefused(f"{rd}: not a P2 C1 run (RUN_START p2_c1 identity {ident})")
    calib_eval = (start.get("eval_provider") or {}).get("manifest_id")
    if ident.get("role") != "CALIBRATION" or calib_eval != CALIB_EVAL_ID:
        raise CentralFTRefused(f"{rd}: role {ident.get('role')} / eval manifest {calib_eval}: the selection reads CALIB "
                               "rows only; a Phase-2 MAIN value stays sealed (PREG2 §6.3 S8) - nothing was opened")
    try:
        summ = _strict(art / "RUN_SUMMARY.json")
        rr = json.loads((art / "RESULT_ROWS.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CentralFTRefused(f"{rd}: RUN_SUMMARY / RESULT_ROWS unreadable ({e})") from None
    if summ.get("status") != "DONE_MAIN" or "latest" in (summ.get("checkpoints") or {}):
        raise CentralFTRefused(f"{rd}: status {summ.get('status')} / latest kept: not a finished fixed-6.0 C1 run")
    if (summ.get("p2_c1") or {}).get("arm") != arm:
        raise CentralFTRefused(f"{rd}: RUN_SUMMARY p2_c1 differs from RUN_START")
    marks = [round(float(e["mark"]), 6) for e in summ.get("evals") or []]
    if marks != [round(m, 6) for m in EVAL_MARKS]:
        raise CentralFTRefused(f"{rd}: expected the 12 evaluations at marks 0.5 .. 6.0, got {marks}")
    hits = [r for r in rr.get("rows", []) if r.get("selection_rule") == "PRACTICAL_BEST"
            and r.get("aggregation") == "macro" and r.get("target_denominator") == "E2E"
            and r.get("metric") == "mrr" and int(r.get("k", -1)) == 20]
    if len(hits) != 1:
        raise CentralFTRefused(f"{rd}: {len(hits)} PRACTICAL_BEST E2E macro mrr@20 rows (expected exactly 1)")
    row = hits[0]
    if row.get("eval_manifest_id") not in (None, CALIB_EVAL_ID):
        raise CentralFTRefused(f"{rd}: the PRACTICAL_BEST row is on {row.get('eval_manifest_id')}, not CALIB")
    if row.get("method") != arm or row.get("row_status") != ROW_STATUS or row.get("c_matched_eligible") is not False:
        raise CentralFTRefused(f"{rd}: the PRACTICAL_BEST row is not a P2 C1 control row")
    pb = summ.get("practical_best") or {}
    mark = float(pb.get("mark", -1))
    if mark < 0.5 or abs(float(row.get("efe", -1)) - float(pb.get("efe", -2))) > 1e-9:
        raise CentralFTRefused(f"{rd}: PRACTICAL_BEST mark {mark} is not one of 0.5 .. 6.0 / differs from the row")
    try:
        value = float(row.get("value"))
    except (TypeError, ValueError):
        value = float("nan")
    finite = math.isfinite(value)
    if finite and value != float(pb.get("value")):
        raise CentralFTRefused(f"{rd}: the row value differs from the run's evaluation at its PRACTICAL_BEST mark")
    return {"run_dir": str(rd), "run_id": start.get("run_id"), "arm": arm, "variant": ident["variant"],
            "role": ident["role"], "peak_lr": ident["peak_lr"], "seed": start.get("seed"),
            "trained_numel": start["p2_c1"].get("trained_numel"), "frozen_item_tables": ident["frozen_item_tables"],
            "value": value if finite else None, "finite": finite, "mark": mark,
            "result_rows_sha256": _sha(art / "RESULT_ROWS.json"), "run_start_sha256": _sha(rd / "RUN_START.json")}


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        shim, largv = split_argv(argv)
    except CentralFTRefused as e:
        print(f"C1 REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    from v4int import launch_central as LC                  # stdlib-only at import (no torch / v4train yet)
    args = LC.parse_args(largv)
    try:
        rid = check_launch_args(args, shim)
        selection = None
        if shim.role == "MAIN":                                   # (a MAIN smoke was refused above, S8)
            selection = bind_c1_selection(shim.c1_selection, variant=variant_of(shim), peak_lr=shim.peak_lr,
                                          expect_sha256=shim.c1_selection_sha256)
        init_path = Path(shim.init_from).resolve()
        if not init_path.is_file():
            raise CentralFTRefused(f"--init-from {init_path} does not exist")
    except CentralFTRefused as e:
        print(f"C1 REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    largv = normalise_argv(largv, args, rid)
    variant = variant_of(shim)
    record = {"task": TASK, "authority": AUTHORITY, "utc": LC.utc(), "argv": argv, "launch_argv": largv,
              "run_id": rid, "arm": C1_ARMS[variant], "variant": variant, "role": shim.role,
              "peak_lr": float(shim.peak_lr), "init_from": str(init_path), "init_record_sha256": _sha(init_path),
              "pins": {"checkpoint_sha256": PRE_CHECKPOINT_SHA256, "state_digest": PRE_STATE_DIGEST},
              "selection": selection, "continuation": CONTINUATION_RULE,
              "wrappers": ["launch_central._recipe_values", "launch_central.bind_data_protocol_lock_v2",
                           "launch_central.build_trainer_registry_view"],
              "central_ft_sha256": _sha(Path(__file__)), "dry_run": shim.dry_run_out is not None}
    ctx = {"role": shim.role, "peak_lr": float(shim.peak_lr), "freeze": bool(shim.p2_freeze_item_tables),
           "init_record": str(init_path), "selection": selection, "recipe_peak_lr": None}
    install(LC, ctx, record)
    if shim.dry_run_out is not None:
        try:
            res = dry_run(LC, args, shim, record)
        except (CentralFTRefused, LC.LaunchRefused) as e:
            print(f"C1 DRY RUN REFUSED: {e}", file=sys.stderr, flush=True)
            return 2
        shim.dry_run_out.parent.mkdir(parents=True, exist_ok=True)
        shim.dry_run_out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"dry_run": str(shim.dry_run_out), "config_sha256": res.get("config_sha256"),
                          "run_id": res.get("run_id", rid), "run_label": res.get("run_label")}))
        return 0
    rc = LC.main(largv)
    record.update(exit_code=rc, utc_end=LC.utc())
    _write_record(Path(args.run_dir), record, record.get("_record_path"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
