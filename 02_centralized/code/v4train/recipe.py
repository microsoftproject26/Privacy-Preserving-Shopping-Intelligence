"""v4train.recipe — the frozen central recipe (Fable G0(b) C8) and the per-run configuration.

Executed optimizer values, transcribed (G0B C8 L28; FEATURE_SCHEMA_CLEAN_V2 executed_optimizer_reference;
g0/receipts_res02/EXECUTED_RECIPES.md): AdamW (fused kernel), betas (0.9, 0.999), eps 1e-8, decoupled weight decay 1e-5
on parameters with ndim >= 2 and 0 elsewhere (fl.adapter.BaseAdapter.param_groups), global gradient-norm clip 1.0
after the 1/n scaling, executed peak LR 1e-3 for both families, effective batch 256, label smoothing 0; dropout GRU
0.20 input / 0.10 readout and SASRec 0.2 (checked against the built module, never set here). Schedule and stopping:
v4train.schedule (G0 D6; G0B C3). Calibration: peak in {1.0, 0.5} x 1e-3, 2.0 EFE, seed 2026 (G0 D6 L37).

PREG1 lock (contracts/PREG1_LOCK_VALUES.json, binding):
  * T2 P1: `exposure_point` is a REQUIRED key ("end"); a configuration without it is refused (no code default);
  * T3: the evaluation numerics come from ONE frozen EvalConfig (v4eval.frozen_config.load()); a non-test run must use
    exactly that object;
  * T1 tie-break: `population = "FULL_TRAIN"` runs the SAME recipe on the full-TRAIN comparison union (N = 40,180,608,
    M_FULL_TRAIN -> the full warm VALIDATION manifest) with the cohort-calibrated peak, no continuation and no new
    calibration; its rows are ARCHITECTURE_TIEBREAK / HISTORICAL_REFERENCE.
PREG1_AMENDMENT_01 item 4 (TRAIN-01R1): `determinism_mode` is a REQUIRED key, "STRICT" (use_deterministic_algorithms
(True), warn_only off; the launcher must export CUBLAS_WORKSPACE_CONFIG ":4096:8" or ":16:8") or "WARN_ONLY"
(warn_only on). A value other than DETERMINISM_MODE_LOCK needs `test_overrides`. The precision id of a run is derived
from its mode (`precision_id`), never a constant claiming deterministic algorithms regardless of mode.
A RunConfig is frozen before the run and hashed (sha256 of its canonical JSON); every checkpoint carries the hash and
a resume with any other configuration is refused. Values that differ from the registered recipe (a smaller effective
batch, another seed, a smaller N, the "start" point, another EvalConfig) are allowed ONLY with `test_overrides=True`,
which is recorded in the hash and in every record.
TRAIN-01R2 (G1 residuals R1, R3): `manifest_registry` (the path of a manifest-role registry, v4train.registry) and
`data_version` (which must equal the registry's) are REQUIRED keys; no manifest id is hard-coded in the trainer. The
run's manifest role (`manifest_role`) is the optional explicit `role` key, else derived: population FULL_TRAIN ->
FULL_TRAIN, COHORT -> the phase (MAIN | CALIBRATION; CONTINUATION is entered only by continue_run). An explicit role
must agree with population and phase (FEATURE_SEARCH_SCREEN: COHORT / MAIN; FULL_TRAIN: FULL_TRAIN; MAIN / CALIBRATION:
COHORT and that phase). `schema_registry` optionally names the derived-schema registry (default
contracts/derived_schemas/REGISTRY.json; only read for a derived schema). The run label of every row is derived and
cannot be set by the caller: test_overrides -> SMOKE_NOT_EVIDENCE; role FEATURE_SEARCH_SCREEN -> FEATURE_SEARCH_SCREEN;
the full-TRAIN tie-break or a HISTORICAL data version -> HISTORICAL_REFERENCE; otherwise PRODUCTION. The
FEATURE_SEARCH_SCREEN role runs the locked main settings WITHOUT test_overrides (seed 2026 / 2027, the executed peak
1e-3 exactly, B 256, 6 EFE, STRICT), with no calibration record and no continuation.
TRAIN-02 (PREG1_AMENDMENT_02R1; Fable follow-up ruling (d)): optional keys, each None by default and left out of the
config hash while None (so every pre-TRAIN-02 config keeps its sha256):
  * `model_variant` (v4models.variants; None = the family's current build GRU_D128 / SASREC_D256_B2), `dropout`
    (per site: GRU (input, readout), SASRec (p,); None = the recipe's 0.20 / 0.10, 0.2), `schedule` ("S0" | "S1";
    None = S0) and `theta0_init_sha256` (the registered theta0 of the variant x seed, THETA0_VARIANTS.json; the built
    theta0 must equal it);
  * `hp_id` / `hp_registry` (None = contracts/hp_registry/HP_REGISTRY_P2R2.json, TRAIN-02R1 / PREG1_AMENDMENT_02R2;
    the superseded HP_REGISTRY_P2.json is refused): validate() accepts a peak LR, dropout,
    schedule and model variant that differ from the registered recipe ONLY when all four equal the trial `hp_id` of the
    HP registry (self-hash and trial sha256 verified, v4train.hp_registry); otherwise the existing rules apply
    unchanged (peak in {1.0, 0.5} x 1e-3; the recipe dropout; S0; the current build), with or without test_overrides.
  * role TUNING (the P2 search): population COHORT and phase CALIBRATION only, so its manifests are the registry's
    CALIBRATION role (CALIB_TRAIN / CALIB_VALIDATION); it needs `hp_id`; it runs the 6.0-EFE clock (train phase
    TUNING: evaluations at 2 / 4 / 6, snapshots at 5.0 / 5.5 / 6.0, CKPT_AVG, no continuation, no calibration record);
    without test_overrides: seed 2026 / 2027, B 256, STRICT and `theta0_init_sha256` required. Its run label is
    HP_SEARCH, which v4train.selection refuses everywhere.
  * TRAIN-02R1 F6 (P3): `recipe_id` / `recipe_registry` (None = contracts/hp_registry/TUNED_RECIPES_02B.json, written
    after P2 by `p2_driver register-02b`): a MAIN-phase COHORT run (role None / MAIN) may carry a tuned recipe; its
    model variant, peak LR, dropout and schedule must then equal that recipe exactly (registry self-hash, recipe sha256
    and its source HP trial verified, v4train.tuned_recipes); it takes no calibration record, and without
    test_overrides it needs theta0_init_sha256 and (TRAIN-02R2 F2) the production registry path whose self-hash is
    pinned in company/gates/G1B_APPROVED.json. A TUNING run, or an hp_id, never carries a recipe_id, and an HP-trial id
    is not a recipe id. Without recipe_id / hp_id the registered-recipe rules apply unchanged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional

from .evalconfig import EvalConfig, load_frozen_eval_config
from .schedule import CALIBRATION_END, DEFAULT_SCHEDULE, LR_EXPOSURE_POINT, LR_EXPOSURE_POINTS, MAIN_END

POPULATIONS = ("COHORT", "FULL_TRAIN")
FULL_TRAIN_N = 40_180_608                      # PREG1 T1 tiebreak.population (N_EFE) of the HISTORICAL data version
# TRAIN-01R2: manifest roles, row run labels and the historical data version (the old M01/M05/M02/M06 population)
ROLES = ("MAIN", "CALIBRATION", "CONTINUATION", "FULL_TRAIN", "FEATURE_SEARCH_SCREEN")   # manifest-registry roles
RUN_ROLES = ROLES + ("TUNING",)                      # TRAIN-02: TUNING is a run role on the registry's CALIBRATION role
RUN_LABELS = ("PRODUCTION", "SMOKE_NOT_EVIDENCE", "FEATURE_SEARCH_SCREEN", "HISTORICAL_REFERENCE", "HP_SEARCH")
TRAIN02_OPTIONAL_KEYS = ("model_variant", "dropout", "schedule", "theta0_init_sha256", "hp_id", "hp_registry",
                         "recipe_id", "recipe_registry", "auto_continuation")
HISTORICAL_DATA_VERSION = "CLEAN_V2_HISTORICAL_262144"
HISTORICAL_DATA_VERSIONS = (HISTORICAL_DATA_VERSION,)


class RecipeError(ValueError):
    pass


def _hp_default_path():
    from .hp_registry import REGISTRY_PATH
    return REGISTRY_PATH


@dataclass(frozen=True)
class CentralRecipe:
    optimizer: str = "AdamW"
    fused: bool = True
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-5
    weight_decay_scope: str = "ndim>=2"
    grad_clip: float = 1.0
    executed_peak_lr: float = 1e-3
    effective_batch: int = 256
    label_smoothing: float = 0.0
    gru_dropout: tuple = (0.20, 0.10)
    sasrec_dropout: float = 0.2
    calibration_multipliers: tuple = (1.0, 0.5)
    calibration_seed: int = 2026
    seeds: tuple = (2026, 2027)
    source: str = "G0B_PROTOCOL_FABLE_POSITION.md C8 L28; G0_PLAN_FABLE_POSITION.md D6 L36-L38; G0B C3 L18"


RECIPE = CentralRecipe()
DETERMINISM_MODES = ("STRICT", "WARN_ONLY")
DETERMINISM_MODE_LOCK = "STRICT"               # PROPOSED, PREG1_AMENDMENT_01 item 4; confirmed at G1
CUBLAS_WORKSPACE_CONFIGS_STRICT = (":4096:8", ":16:8")   # STRICT refuses to start without one of these (launcher env)
PRECISION_ID_BASE = "FP32_STRICT:matmul_tf32=off,cudnn_tf32=off,float32_matmul_precision=highest"


def precision_id(determinism_mode: str) -> str:
    """The precision id of a run: the strict-FP32 base plus the run's ACTUAL determinism mode (from its config)."""
    if determinism_mode not in DETERMINISM_MODES:
        raise RecipeError(f"determinism_mode must be one of {DETERMINISM_MODES}, not {determinism_mode!r}")
    return f"{PRECISION_ID_BASE},deterministic_algorithms={determinism_mode}"


PRECISION_ID = precision_id(DETERMINISM_MODE_LOCK)   # the id under the lock value; a run uses RunConfig.precision_id


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    family: str
    seed: int
    exposure_point: Optional[str] = None      # REQUIRED ("end", PREG1 T2 P1); None is refused by validate()
    cuda_visible_devices: Optional[str] = None  # REQUIRED: the launcher's explicit CUDA_VISIBLE_DEVICES (FL-02 trap)
    determinism_mode: Optional[str] = None    # REQUIRED: STRICT | WARN_ONLY (PREG1_AMENDMENT_01 item 4); None refused
    manifest_registry: Optional[str] = None   # REQUIRED (TRAIN-01R2): path of the manifest-role registry JSON
    data_version: Optional[str] = None        # REQUIRED (TRAIN-01R2): must equal the registry's data_version
    schema_registry: Optional[str] = None     # derived-schema registry (None: contracts/derived_schemas/REGISTRY.json)
    role: Optional[str] = None                # registry role (None: derived); FEATURE_SEARCH_SCREEN only explicitly
    phase: str = "MAIN"                      # MAIN | CALIBRATION (CONTINUATION is entered by continue_run)
    population: str = "COHORT"                # COHORT (the contest / the screen) | FULL_TRAIN (the T1 tie-break)
    peak_lr: float = 1e-3
    effective_batch: int = 256
    microbatch: int = 0                       # 0 = one pass; else exact accumulation in microbatches of this size
    log_every: int = 50                       # periodic monitor sync (the only host syncs of the training step)
    checkpoint_every: int = 0                 # extra `latest` saves every n updates (0: at evaluation marks only)
    eval_config: EvalConfig = field(default_factory=load_frozen_eval_config)   # the ONE frozen evaluation config
    continuation_budget: Optional[str] = None # path of a budget file; its sha256 is recorded at run start
    calibration_record: Optional[str] = None  # path of the CALIBRATION_<family>.json the main peak came from
    expected_init_sha256: Optional[str] = None
    key: Mapping = field(default_factory=dict)  # comparison-key base for result rows (data/feature/population hashes)
    test_overrides: bool = False
    recipe: CentralRecipe = RECIPE
    # TRAIN-02 (all optional; left out of the config hash while None)
    model_variant: Optional[str] = None       # v4models.variants name (None: the family's current build)
    dropout: Optional[tuple] = None           # per-site dropout (None: the recipe's)
    schedule: Optional[str] = None            # "S0" | "S1" (None: S0, the lock's knots)
    theta0_init_sha256: Optional[str] = None  # the registered theta0 of (variant, seed) (THETA0_VARIANTS.json)
    hp_id: Optional[str] = None               # a trial of the HP registry (role TUNING only)
    hp_registry: Optional[str] = None         # the HP registry path (None: contracts/hp_registry/HP_REGISTRY_P2R2.json)
    recipe_id: Optional[str] = None           # TRAIN-02R1 F6: a tuned recipe of TUNED_RECIPES_02B.json (MAIN only)
    recipe_registry: Optional[str] = None     # the tuned-recipes registry path (None: the contracts path)
    auto_continuation: Optional[bool] = None  # TRAIN-03: None = automatic C3 continuation for a non-test contest run

    def validate(self) -> None:
        r = self.recipe
        fam = str(self.family).upper()
        if fam not in ("GRU", "SASREC"):
            raise RecipeError(f"unknown family {self.family!r}")
        if self.exposure_point is None:
            raise RecipeError("exposure_point is a required run-config key (PREG1 T2 P1: never rely on a code default)")
        if self.exposure_point not in LR_EXPOSURE_POINTS:
            raise RecipeError(f"exposure_point must be one of {LR_EXPOSURE_POINTS}")
        if self.exposure_point != LR_EXPOSURE_POINT and not self.test_overrides:
            raise RecipeError(f"exposure_point is locked to {LR_EXPOSURE_POINT!r} (PREG1 T2)")
        if not self.cuda_visible_devices:
            raise RecipeError("cuda_visible_devices is a required run-config key: the launcher must export "
                              "CUDA_VISIBLE_DEVICES explicitly (snap_v1/reuse/train.py silently defaults it to '0')")
        if self.determinism_mode is None:
            raise RecipeError("determinism_mode is a required run-config key (PREG1_AMENDMENT_01 item 4: the mode is "
                              "locked and recorded, never a code default)")
        if self.determinism_mode not in DETERMINISM_MODES:
            raise RecipeError(f"determinism_mode must be one of {DETERMINISM_MODES}, not {self.determinism_mode!r}")
        if self.determinism_mode != DETERMINISM_MODE_LOCK and not self.test_overrides:
            raise RecipeError(f"determinism_mode is locked to {DETERMINISM_MODE_LOCK!r} (PREG1_AMENDMENT_01 item 4)")
        if not isinstance(self.manifest_registry, str) or not self.manifest_registry:
            raise RecipeError("manifest_registry is a required run-config key (TRAIN-01R2: the manifest-role registry "
                              "path; no manifest id is hard-coded in the trainer)")
        if not isinstance(self.data_version, str) or not self.data_version:
            raise RecipeError("data_version is a required run-config key (TRAIN-01R2: it must equal the registry's)")
        if self.schema_registry is not None and (not isinstance(self.schema_registry, str) or not self.schema_registry):
            raise RecipeError("schema_registry must be None or a path")
        if self.role is not None:
            if self.role not in RUN_ROLES or self.role == "CONTINUATION":
                raise RecipeError(f"role must be one of {tuple(r for r in RUN_ROLES if r != 'CONTINUATION')} "
                                  "(CONTINUATION is entered by continue_run)")
            want = {"MAIN": ("COHORT", "MAIN"), "CALIBRATION": ("COHORT", "CALIBRATION"),
                    "FULL_TRAIN": ("FULL_TRAIN", "MAIN"), "FEATURE_SEARCH_SCREEN": ("COHORT", "MAIN"),
                    "TUNING": ("COHORT", "CALIBRATION")}[self.role]
            if (self.population, self.phase) != want:
                raise RecipeError(f"role {self.role} needs population {want[0]} and phase {want[1]}, "
                                  f"not {self.population} / {self.phase}")
        if self.phase not in ("MAIN", "CALIBRATION"):
            raise RecipeError("a run starts in phase MAIN or CALIBRATION (CONTINUATION only via continue_run)")
        if self.population not in POPULATIONS:
            raise RecipeError(f"population must be one of {POPULATIONS}")
        if r != RECIPE:
            raise RecipeError("the central recipe is frozen (C8); a different recipe object is refused")
        self._validate_train02(fam)
        if self.microbatch < 0 or self.log_every < 1 or self.checkpoint_every < 0:
            raise RecipeError("microbatch >= 0, log_every >= 1, checkpoint_every >= 0")
        if not isinstance(self.eval_config, EvalConfig):
            raise RecipeError("eval_config must be a v4train.evalconfig.EvalConfig (the one frozen evaluation config)")
        self.eval_config.validate()
        if self.population == "FULL_TRAIN":
            if self.phase != "MAIN":
                raise RecipeError("the full-TRAIN tie-break is a main run (no new calibration, PREG1 T1)")
            if self.continuation_budget is not None:
                raise RecipeError("the full-TRAIN tie-break has no continuation (PREG1 T1)")
            if self.calibration_record is None and not self.test_overrides:
                raise RecipeError("the full-TRAIN tie-break uses the cohort-calibrated peak: calibration_record needed")
        if self.role == "FEATURE_SEARCH_SCREEN":
            if self.continuation_budget is not None:
                raise RecipeError("a FEATURE_SEARCH_SCREEN run has no continuation")
            if self.calibration_record is not None:
                raise RecipeError("a FEATURE_SEARCH_SCREEN run uses the executed peak 1e-3, not a calibration record")
            if not self.test_overrides and round(self.peak_lr, 12) != round(r.executed_peak_lr, 12):
                raise RecipeError(f"a FEATURE_SEARCH_SCREEN run uses the executed peak {r.executed_peak_lr} exactly")
        if self.eval_config != load_frozen_eval_config():             # always: no test override (PREG1 T3)
            raise RecipeError("eval_config must be the frozen v4eval config (PREG1 T3)")
        if not self.test_overrides:
            if self.effective_batch != r.effective_batch:
                raise RecipeError(f"effective batch is {r.effective_batch} (C8); {self.effective_batch} refused")
            if self.role == "TUNING":
                if self.seed not in r.seeds:
                    raise RecipeError(f"a TUNING run uses seed {r.seeds[0]} (stage 1) or {r.seeds[1]} (stage 2)")
            elif self.phase == "CALIBRATION" and self.seed != r.calibration_seed:
                raise RecipeError(f"calibration uses seed {r.calibration_seed}")
            if self.phase == "MAIN" and self.seed not in r.seeds:
                raise RecipeError(f"main runs use seeds {r.seeds}")
        if self.effective_batch < 1:
            raise RecipeError("effective batch must be >= 1")

    @property
    def sealed_outputs(self) -> bool:
        """TRAIN-03 (Fable staggered-P3 ruling, condition 1): a MAIN-phase COHORT contest run (role None / MAIN) writes
        every metric-bearing output under <run_dir>/sealed/ — by construction, test runs included. TUNING,
        CALIBRATION, the feature-search screen and the full-TRAIN tie-break keep their TRAIN-02 layout."""
        return self.phase == "MAIN" and self.population == "COHORT" and self.role in (None, "MAIN")

    @property
    def effective_auto_continuation(self) -> bool:
        """TRAIN-03 (condition 4): a sealed contest run continues automatically on STILL_IMPROVING (registered C3
        rule, 6.0 -> at most 8.0) unless it is a test run that did not ask for it."""
        if not self.sealed_outputs:
            return False
        return bool(self.auto_continuation) if self.auto_continuation is not None else not self.test_overrides

    def _validate_train03(self) -> None:
        if self.auto_continuation is not None and not isinstance(self.auto_continuation, bool):
            raise RecipeError("auto_continuation must be None or a bool")
        if self.auto_continuation and not self.sealed_outputs:
            raise RecipeError("auto_continuation is for a MAIN-phase COHORT contest run only")
        if self.sealed_outputs and not self.test_overrides and self.auto_continuation is False:
            raise RecipeError("a contest run continues automatically on STILL_IMPROVING (Fable staggered-P3 ruling, "
                              "condition 4): auto_continuation False is refused")
        if self.effective_auto_continuation and self.continuation_budget is not None:
            raise RecipeError("an automatic continuation follows the registered C3 rule (ceiling 8.0); a "
                              "continuation_budget file is refused")

    def _validate_train02(self, fam: str) -> None:
        self._validate_train03()
        """The TRAIN-02 keys: model variant, dropout, schedule, theta0 and the HP-registry rule (see the docstring)."""
        from v4models import variants as V
        from .schedule import SCHEDULES
        r = self.recipe
        try:
            vname, _ = V.resolve(fam, self.model_variant)
            drop = V.dropout_of(fam, vname, self.dropout)
        except V.VariantError as e:
            raise RecipeError(str(e)) from None
        if self.dropout is not None and (not isinstance(self.dropout, tuple) or
                                         tuple(float(x) for x in self.dropout) != self.dropout):
            raise RecipeError("dropout must be a tuple of floats (GRU (input, readout); SASRec (p,))")
        sched = DEFAULT_SCHEDULE if self.schedule is None else self.schedule
        if sched not in SCHEDULES:
            raise RecipeError(f"schedule must be one of {tuple(SCHEDULES)}, not {self.schedule!r}")
        for k in ("theta0_init_sha256",):
            v = getattr(self, k)
            if v is not None and (not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)):
                raise RecipeError(f"{k} must be a 64-hex sha256")
        if self.theta0_init_sha256 and self.expected_init_sha256 and \
                self.theta0_init_sha256 != self.expected_init_sha256:
            raise RecipeError("theta0_init_sha256 and expected_init_sha256 differ")
        if self.hp_registry is not None and (not isinstance(self.hp_registry, str) or not self.hp_registry):
            raise RecipeError("hp_registry must be None or a path")
        if self.role == "TUNING":
            if self.hp_id is None:
                raise RecipeError("a TUNING run needs hp_id (a trial of the HP registry, TRAIN-02)")
            if self.continuation_budget is not None:
                raise RecipeError("a TUNING run has no continuation")
            if self.calibration_record is not None:
                raise RecipeError("a TUNING run takes its peak from its HP trial, not a calibration record")
            if not self.test_overrides and not self.theta0_init_sha256:
                raise RecipeError("a TUNING run needs theta0_init_sha256 (the registered theta0 of its variant x seed, "
                                  "THETA0_VARIANTS.json)")
        if self.recipe_registry is not None and (not isinstance(self.recipe_registry, str) or not self.recipe_registry):
            raise RecipeError("recipe_registry must be None or a path")
        if self.recipe_id is not None:
            if self.role == "TUNING" or self.hp_id is not None:
                raise RecipeError("recipe_id is for MAIN runs (P3); a TUNING run uses hp_id, never a recipe_id")
            if self.phase != "MAIN" or self.population != "COHORT" or self.role not in (None, "MAIN"):
                raise RecipeError("recipe_id is valid for a MAIN-phase COHORT run only")
            if self.calibration_record is not None:
                raise RecipeError("a tuned-recipe run takes its peak LR from the recipe, not a calibration record")
            if not self.test_overrides and not self.theta0_init_sha256:
                raise RecipeError("a tuned-recipe run needs theta0_init_sha256 (THETA0_VARIANTS.json)")
            from . import tuned_recipes as TRM
            try:
                tpath = self.recipe_registry or TRM.REGISTRY_PATH
                treg = TRM.load_tuned_recipes(tpath, self.hp_registry or _hp_default_path())
                if not self.test_overrides:                          # TRAIN-02R2 F2: production file, pinned at G1b
                    TRM.check_production_registry(tpath, treg)
                rec = TRM.recipe(treg, self.recipe_id)
            except TRM.TunedRecipeError as e:
                raise RecipeError(f"tuned recipe refused: {e}") from None
            if rec["family"] != fam:
                raise RecipeError(f"recipe {self.recipe_id} is a {rec['family']} recipe, not {fam}")
            got = {"peak_lr": float(self.peak_lr), "model_variant": vname, "dropout": list(drop), "schedule": sched}
            want = {k: rec[k] for k in got}
            if got != want:
                bad = sorted(k for k in got if got[k] != want[k])
                raise RecipeError(f"{bad} differ from tuned recipe {self.recipe_id} ({ {k: want[k] for k in bad} })")
            return
        if self.hp_id is not None:
            if self.role != "TUNING":
                raise RecipeError("hp_id is valid for role TUNING only")
            from .hp_registry import HPRegistryError, REGISTRY_PATH, load_hp_registry, trial
            try:
                reg = load_hp_registry(self.hp_registry or REGISTRY_PATH)
                t = trial(reg, self.hp_id)
            except HPRegistryError as e:
                raise RecipeError(f"HP registry refused: {e}") from None
            if t["family"] != fam:
                raise RecipeError(f"hp_id {self.hp_id} is a {t['family']} trial, not {fam}")
            got = {"peak_lr": float(self.peak_lr), "model_variant": vname, "dropout": list(drop), "schedule": sched}
            want = {k: t[k] for k in got}
            if got != want:
                bad = sorted(k for k in got if got[k] != want[k])
                raise RecipeError(f"{bad} differ from HP trial {self.hp_id} ({ {k: want[k] for k in bad} }); a value "
                                  "outside the registered recipe is accepted only when it equals the named trial")
            return
        allowed = tuple(round(m * r.executed_peak_lr, 12) for m in r.calibration_multipliers)
        if round(self.peak_lr, 12) not in allowed:
            raise RecipeError(f"peak LR {self.peak_lr} not in the registered set {allowed}")
        if vname != V.DEFAULT_VARIANT[fam]:
            raise RecipeError(f"model variant {vname} is not the registered build {V.DEFAULT_VARIANT[fam]} (only an HP "
                              "trial named by hp_id may differ)")
        want_drop = tuple(r.gru_dropout) if fam == "GRU" else (r.sasrec_dropout,)
        if drop != tuple(float(x) for x in want_drop):
            raise RecipeError(f"dropout {drop} is not the executed {want_drop} (only an HP trial named by hp_id may "
                              "differ)")
        if sched != DEFAULT_SCHEDULE:
            raise RecipeError(f"schedule {sched} is not the registered {DEFAULT_SCHEDULE} (only an HP trial named by "
                              "hp_id may differ)")

    @property
    def family_u(self) -> str:
        return str(self.family).upper()

    @property
    def run_role(self) -> str:
        """The role of the run (TUNING, or the registry role it starts in)."""
        if self.role is not None:
            return self.role
        return "FULL_TRAIN" if self.population == "FULL_TRAIN" else self.phase

    @property
    def train_phase(self) -> str:
        """The trainer's phase: TUNING for a TUNING run (the 6.0 clock), else the config phase."""
        return "TUNING" if self.role == "TUNING" else self.phase

    @property
    def effective_variant(self) -> str:
        from v4models import variants as V
        return V.resolve(self.family_u, self.model_variant)[0]

    @property
    def effective_dropout(self) -> tuple:
        from v4models import variants as V
        return V.dropout_of(self.family_u, self.model_variant, self.dropout)

    @property
    def effective_schedule(self) -> str:
        return DEFAULT_SCHEDULE if self.schedule is None else self.schedule

    @property
    def end_efe(self) -> float:
        if self.role == "TUNING":
            return MAIN_END
        return CALIBRATION_END if self.phase == "CALIBRATION" else MAIN_END

    @property
    def precision_id(self) -> str:
        return precision_id(self.determinism_mode)

    @property
    def manifest_role(self) -> str:
        """The registry role a run STARTS in (continue_run switches to CONTINUATION)."""
        if self.role == "TUNING":
            return "CALIBRATION"                       # TRAIN-02: TUNING uses the registry's CALIBRATION manifests
        if self.role is not None:
            return self.role
        return "FULL_TRAIN" if self.population == "FULL_TRAIN" else self.phase

    @property
    def run_label(self) -> str:
        """The run label of every row of this run (derived; the caller cannot set it)."""
        if self.test_overrides:
            return "SMOKE_NOT_EVIDENCE"
        if self.role == "TUNING":
            return "HP_SEARCH"
        if self.manifest_role == "FEATURE_SEARCH_SCREEN":
            return "FEATURE_SEARCH_SCREEN"
        if self.population == "FULL_TRAIN" or self.data_version in HISTORICAL_DATA_VERSIONS:
            return "HISTORICAL_REFERENCE"
        return "PRODUCTION"

    def as_json(self) -> dict:
        d = asdict(self)
        d["family"] = self.family_u
        d["key"] = dict(self.key)
        for k in TRAIN02_OPTIONAL_KEYS:                # left out while None: pre-TRAIN-02 config hashes unchanged
            if d.get(k) is None:
                d.pop(k, None)
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.as_json(), sort_keys=True, allow_nan=False,
                                         default=list).encode("utf-8")).hexdigest()


__all__ = ["RecipeError", "CentralRecipe", "RECIPE", "PRECISION_ID", "RunConfig", "POPULATIONS", "FULL_TRAIN_N",
           "DETERMINISM_MODES", "DETERMINISM_MODE_LOCK", "CUBLAS_WORKSPACE_CONFIGS_STRICT", "PRECISION_ID_BASE",
           "precision_id", "ROLES", "RUN_ROLES", "RUN_LABELS", "HISTORICAL_DATA_VERSION", "HISTORICAL_DATA_VERSIONS",
           "TRAIN02_OPTIONAL_KEYS"]
