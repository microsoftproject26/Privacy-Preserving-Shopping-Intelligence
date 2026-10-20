"""v4train.trainer — the central trainer: one frozen recipe for either family (GRU / SASREC).

Lifecycle of a run directory (all records strict JSON, written atomically; the event log is append-only):
  run()            fresh start: disk guard, RUN_START.json (config + hashes, recorded BEFORE any outcome), train to the
                   phase end (MAIN 6.0 EFE; CALIBRATION 2.0 EFE), evaluating at every 0.5-EFE mark
  resume()         continue a live run from `latest` (bitwise: weights, AdamW state, RNG states, data cursor,
                   exposures, schedule position and stop state are restored; the pass permutation is recomputed from
                   (seed, pass) and checked against its recorded sha256)
  continue_run()   6.0 -> at most 8.0 for a STILL_IMPROVING main run whose budget file was recorded at run start
Per update (no host sync; package 10 §1): take the next effective batch of the pass permutation (tail at its own
size), count its loss-contributing rows on the host (CPU tensors), set lr = peak * s(EFE at the run config's REQUIRED
exposure_point: "end" by the PREG1 T2 lock, i.e. peak * s((a + n) / N); the LR must equal the entry of the planned LR
table hashed in RUN_START, pin P2), forward +
backward in exact microbatch accumulation (sum / n_eff), clip the global norm to 1.0, AdamW step, adapter.post_step()
(SASRec output-table recentering). The monitor keeps device accumulators and converts them to numbers only every
`log_every` updates, at checkpoints and at marks (loss, pre-clip gradient norm mean/max, non-finite flag, head
common-mode drift, data wait, LR). A non-finite loss or gradient found at a sync fails the run (status FAILED); the
last valid `latest` is kept.
At each mark: evaluate through v4eval with the run's frozen EvalConfig (PREG1 T3: v4eval.frozen_config.load(); score
block, class chunk, batch, cache bytes, runtime flag check; no value is hard-coded here) on the named manifest (M02
for MAIN / CONTINUATION, M06 for CALIBRATION; for the FULL_TRAIN tie-break M_FULL_TRAIN -> the full warm VALIDATION
manifest, rows labelled ARCHITECTURE_TIEBREAK / HISTORICAL_REFERENCE, values sealed out of STATUS / EVENTS, no
continuation, the 6.0 resumable always deleted),
record the result rows, update the strictly-best tracker, save `best` (MAIN) / `best_extended` (CONTINUATION) on a
strict improvement, `endpoint_6.0` at 6.0, then `latest`; best is written before latest so a crash between the two
replays the mark deterministically. Retention and the STILL_IMPROVING decision follow v4train.schedule / checkpoint.
Determinism mode (PREG1_AMENDMENT_01 item 4; TRAIN-01R1): the run config's REQUIRED `determinism_mode` is applied at
construction, at every run / resume / continuation start and after every evaluation (STRICT: deterministic
algorithms with warn_only off, refused unless the launcher exported CUBLAS_WORKSPACE_CONFIG ":4096:8" or ":16:8";
WARN_ONLY: warn_only on). The torch state is read back and verified after applying it, at every monitor sync and in
every record. The mode and the CUBLAS setting captured at construction are recorded in RUN_START, every checkpoint,
RUN_SUMMARY and every result row (with the mode-derived precision_id); a resume whose mode or CUBLAS setting differs
from the run start record or the checkpoint is refused.
Manifest-role registry and schemas (TRAIN-01R2; G1 residuals R1, R3): the run config's REQUIRED `manifest_registry`
is loaded and verified by its self-hash (v4train.registry) and its data_version must equal the config's. The run's
role (MAIN | CALIBRATION | FULL_TRAIN | FEATURE_SEARCH_SCREEN at start, CONTINUATION in continue_run) must be in the
registry, and each provider must match that role's registry entry (manifest id, the manifest_binding hashes, the row
count, the fingerprint when the entry has one). No manifest id or count is hard-coded; the only count check left is
the historical tie-break N = 40,180,608 on the HISTORICAL data version. The schema is verified (the base Clean V2
schema from its file, or a derived schema through contracts/derived_schemas/REGISTRY.json). The registry record (file
sha256, self-hash, data_version, role, manifest ids) and the schema record (schema_id, schema_self_sha256) go into
RUN_START, every checkpoint's meta, RUN_SUMMARY, the comparison-key base and every result row, with the derived
run_label (PRODUCTION | SMOKE_NOT_EVIDENCE | FEATURE_SEARCH_SCREEN | HISTORICAL_REFERENCE). A resume or continuation
whose registry or schema differs from the run start record or the checkpoint is refused. A FEATURE_SEARCH_SCREEN run
has no continuation and its 6.0 resumable is always deleted; its rows are never C_MATCHED or downstream seeds. A
non-test run takes its calibrated peak only from a PRODUCTION calibration record of its data version.
TRAIN-02 (PREG1_AMENDMENT_02R1): the model is built as the config's model variant with its dropout
(v4models.build(..., variant, dropout); the dropout is checked against the built module); `theta0_init_sha256`, when
given, must equal the built theta0; model_variant and theta0_init_sha256 enter the comparison-key base and every row
(the R4 addition). The LR follows the config's schedule (S0 default, S1 for an HP trial). Role TUNING (run label
HP_SEARCH): the registry's CALIBRATION manifests, train phase TUNING on the 6.0 clock with evaluations at 2 / 4 / 6
(no best / endpoint weights), weights-only snapshots at 5.0 / 5.5 / 6.0, and at the end: RESULT_ROWS (the 6.0
endpoint, CONTROLLED_BUDGET), the resumable deleted (no continuation), then v4train.ckpt_avg (the uniform average
evaluated through this trainer's own _evaluate, rows CKPT_AVG, snapshots and average deleted with sha256 logged, the
disk peak reported). A crash during CKPT_AVG resumes it (RUN_SUMMARY status PENDING_CKPT_AVG). The HP trial record
(hp_id, trial and registry sha256) and the model-variant registry record are in RUN_START; a resume refuses a changed
HP or variant registry file.
TRAIN-03 (Fable staggered-P3 ruling; company/reviews/STAGGERED_P3_FABLE_RULING.md):
  condition 1, SEAL BY CONSTRUCTION — a MAIN-phase COHORT contest run (RunConfig.sealed_outputs) writes every
  metric-bearing artifact under <run_dir>/sealed/ (self.art): the mark rows / results / ranks (sealed/evals/), the full
  metrics.jsonl (loss, gradient norms, eval values), RESULT_ROWS(_EXTENDED).json, the full RUN_SUMMARY.json and the
  checkpoint store (CHECKPOINTS.json, RETENTION_LOG.jsonl, checkpoints/ — their metas and train states carry values).
  Outside sealed/ such a run writes only: RUN_START.json (config and hashes, no outcome), STATUS.json and EVENTS.jsonl
  with mark index / EFE / timestamps / the STILL_IMPROVING boolean, a reduced metrics.jsonl (update, EFE, wall
  seconds, the loss-finite flag) and a public RUN_SUMMARY.json stub (status, marks done, STILL_IMPROVING, hashes).
  condition 4, AUTOMATIC CONTINUATION — at 6.0 the STILL_IMPROVING decision is computed here from the sealed values
  (StopTracker); a run with RunConfig.effective_auto_continuation continues at once, in the same process, under the
  registered C3 rule (continuation knots ((7.5, 0.1), (8.0, 0.01)), ceiling 8.0; the same rule for S0 and S1 recipes);
  only the boolean is logged outside sealed/. Continuation rows stay PRACTICAL_BEST_EXTENDED (sealed), never selection.
Every other run (TUNING, CALIBRATION, the screen, the tie-break) keeps self.art == run_dir: its layout is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

import v4models
from fl.numerics import assert_strict_fp32, enforce_strict_fp32, numerics_receipt, state_digest
from v4eval import evaluator_code_sha256
from v4eval.evaluate import evaluate
from v4eval.device_pin import accepted_classes, pin_for_device
from v4eval.hashing import code_sha256
from v4eval.frozen_config import apply_runtime_flags, check_runtime_flags
from v4eval.predict import predict_adapter
from v4eval.rows import SELECTION_RULES, rows_from_result
from v4eval.values import Undefined, to_jsonable
from v4models import variants as model_variants
from v4models.build import TOPOLOGY, build
from v4models.monitors import common_mode_report
from v4sup.disk import disk_check
from v4sup.status import append_event, write_status_atomic

from .evalconfig import EvalConfig, frozen as frozen_eval, runtime_device_class
from .checkpoint import CheckpointError, CheckpointStore, append_jsonl, read_json, sha256_file, write_json_atomic
from .objective import OBJECTIVE_ID, accumulate_step, host_contributing
from .order import ORDER_POLICY, pass_permutation, permutation_sha256
from .recipe import CUBLAS_WORKSPACE_CONFIGS_STRICT, FULL_TRAIN_N, HISTORICAL_DATA_VERSIONS, RunConfig
from .registry import RegistryError, check_provider, load_manifest_registry, verify_schema
from .selection import SelectionError, accept_calibration_record
from .schedule import (CEILING, EVAL_EVERY, MAIN_END, StopTracker, eval_marks, lr_for_update, mark_exposures,
                       phase_end, planned_lr_table, snapshot_marks)

_HERE = Path(__file__).resolve().parent
BUDGET_SCHEMA = "V4_CONTINUATION_BUDGET_1"


class TrainerError(RuntimeError):
    pass


class DiskGuardError(TrainerError):
    pass


class TrainingDiverged(TrainerError):
    pass


class InjectedInterruption(RuntimeError):
    """Test-only: raised by a fault plan after a given update, simulating a crash mid-run."""


def trainer_code_sha256() -> str:
    return code_sha256(sorted(_HERE.glob("*.py")))


def models_code_sha256() -> str:
    return code_sha256(sorted(Path(v4models.__file__).resolve().parent.glob("*.py")))


@dataclass
class TrainState:
    phase: str
    updates: int = 0
    exposures: int = 0
    pass_index: int = 0
    pos: int = 0
    perm_sha256: dict = field(default_factory=dict)      # pass index (str) -> permutation sha256
    evals: list = field(default_factory=list)
    done: bool = False
    ceiling: float = MAIN_END
    snapshots: Optional[list] = None           # TRAIN-02 (TUNING only): the snapshot marks already saved
    disk_peak_bytes: Optional[int] = None      # TRAIN-02 (TUNING only): the weight-file disk peak so far

    def as_dict(self) -> dict:
        d = {"phase": self.phase, "updates": self.updates, "exposures": self.exposures,
             "pass_index": self.pass_index, "pos": self.pos, "perm_sha256": dict(self.perm_sha256),
             "evals": [dict(e) for e in self.evals], "done": self.done, "ceiling": self.ceiling}
        if self.snapshots is not None:                 # TUNING runs only (other runs keep the TRAIN-01R2 record)
            d["snapshots"] = [float(x) for x in self.snapshots]
            d["disk_peak_bytes"] = int(self.disk_peak_bytes or 0)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainState":
        snaps = d.get("snapshots")
        return cls(phase=d["phase"], updates=int(d["updates"]), exposures=int(d["exposures"]),
                   pass_index=int(d["pass_index"]), pos=int(d["pos"]), perm_sha256=dict(d["perm_sha256"]),
                   evals=[dict(e) for e in d["evals"]], done=bool(d["done"]), ceiling=float(d["ceiling"]),
                   snapshots=None if snaps is None else [float(x) for x in snaps],
                   disk_peak_bytes=None if snaps is None else int(d.get("disk_peak_bytes", 0)))


class Monitor:
    """Device-side accumulators; converted to host numbers only in flush() (the periodic sync)."""

    def __init__(self):
        self._reset()

    def _reset(self):
        self.loss_sum = None
        self.n = 0
        self.gn_sum = None
        self.gn_max = None
        self.nonfinite = None
        self.updates = 0
        self.wait_s = 0.0
        self.step_s = 0.0
        self.lr = None

    def add(self, loss_sum: torch.Tensor, n: int, grad_norm: torch.Tensor, wait_s: float, step_s: float, lr: float):
        bad = ~torch.isfinite(loss_sum) | ~torch.isfinite(grad_norm)
        if self.loss_sum is None:
            self.loss_sum, self.gn_sum, self.gn_max, self.nonfinite = loss_sum.detach(), grad_norm.detach(), \
                grad_norm.detach(), bad
        else:
            self.loss_sum = self.loss_sum + loss_sum.detach()
            self.gn_sum = self.gn_sum + grad_norm.detach()
            self.gn_max = torch.maximum(self.gn_max, grad_norm.detach())
            self.nonfinite = self.nonfinite | bad
        self.n += int(n)
        self.updates += 1
        self.wait_s += wait_s
        self.step_s += step_s
        self.lr = lr

    def flush(self, adapter, state: TrainState, N: int, attempt: int) -> Optional[dict]:
        if self.updates == 0:
            return None
        nonfinite = bool(self.nonfinite.item())
        rec = {"kind": "monitor", "attempt": attempt, "phase": state.phase, "update": state.updates,
               "exposures": state.exposures, "efe": state.exposures / N, "window_updates": self.updates,
               "lr_last": self.lr, "nonfinite": nonfinite,
               "loss_mean": (None if nonfinite else self.loss_sum.item() / self.n),
               "grad_norm_mean_preclip": (None if nonfinite else self.gn_sum.item() / self.updates),
               "grad_norm_max_preclip": (None if nonfinite else self.gn_max.item()),
               "data_wait_s": self.wait_s, "step_wall_s": self.step_s}
        if not nonfinite:
            h = common_mode_report(adapter)["head"]
            rec.update({"head_common_mode_norm": h["common_mode_norm"], "head_common_mode_ratio": h["ratio"],
                        "head_b_mean": h["b_mean"]})
        self._reset()
        return rec


@dataclass
class EvalOutcome:
    value: float
    rows: list
    result: Optional[dict] = None
    ranks: Optional[np.ndarray] = None
    secondary: dict = field(default_factory=dict)


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


class CentralTrainer:
    def __init__(self, config: RunConfig, schema, train, evalp, run_dir, *, device: str = "cpu",
                 K: Optional[int] = None, poc=None, catalogue=None, v1_root=None,
                 disk_usage_fn: Callable = shutil.disk_usage, fault: Optional[Callable[[int], None]] = None,
                 checkpoint_fault: Optional[Callable[[str, str], None]] = None,
                 eval_override: Optional[Callable] = None):
        config.validate()
        self.cfg = config
        self.family = config.family_u
        self.schema = schema
        self.train, self.evalp = train, evalp
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.disk_usage_fn = disk_usage_fn
        self._fault = fault
        self._eval_override = eval_override
        # -- determinism mode (PREG1_AMENDMENT_01 item 4): the launcher's CUBLAS setting, then apply + verify ----------
        self.cublas_workspace_config = self._check_cublas(config)
        self._apply_determinism()
        # -- data identity and roles ------------------------------------------------------------------
        if train.split != "TRAIN":
            raise TrainerError(f"training provider split {train.split!r} refused (TRAIN only)")
        self.population = config.population
        self.tiebreak = config.population == "FULL_TRAIN"
        self.screen = config.manifest_role == "FEATURE_SEARCH_SCREEN"
        self.tuning = config.role == "TUNING"                 # TRAIN-02: the P2 search (registry role CALIBRATION)
        self.run_role = config.run_role
        self.run_label = config.run_label
        # -- the manifest-role registry (TRAIN-01R2): verified, same data version, the role's ids and bindings ---------
        try:
            self.registry = load_manifest_registry(config.manifest_registry)
        except RegistryError as e:
            raise TrainerError(f"manifest registry refused: {e}") from e
        if self.registry.data_version != config.data_version:
            raise TrainerError(f"data_version {config.data_version!r} differs from the manifest registry's "
                               f"{self.registry.data_version!r}")
        self.role = config.manifest_role
        self.bindings = self._check_role(self.role)
        self.N = int(train.n_decisions)
        if self.tiebreak and not config.test_overrides and config.data_version in HISTORICAL_DATA_VERSIONS \
                and self.N != FULL_TRAIN_N:
            raise TrainerError(f"the full-TRAIN tie-break has N = {FULL_TRAIN_N}, the provider has {self.N}")
        # -- the schema: the base Clean V2 schema or a registered derived schema (TRAIN-01R2) --------------------------
        try:
            self.schema_record = verify_schema(schema, config.schema_registry)
        except RegistryError as e:
            raise TrainerError(f"schema refused: {e}") from e
        did = np.asarray(train.decision_ids(), dtype=np.int64)
        if did.shape != (self.N,) or (self.N > 1 and not bool(np.all(did[1:] > did[:-1]))):
            raise TrainerError("the provider's canonical order must be strictly ascending decision_id")
        if 2 * config.effective_batch > self.N:
            raise TrainerError("effective batch must be <= N / 2 (at most one 0.5-EFE mark per update)")
        self.planned = self._plan(config.train_phase, 0)
        self.planned_offset = 0
        # -- TRAIN-02: the HP trial and the model-variant registry (hash-checked) ------------------------------------
        self.hp = self._hp_record()
        self.tuned_recipe = self._recipe_record()                   # TRAIN-02R1 F6 (P3 MAIN runs)
        self.variant_registry = self._variant_registry_record()
        # -- GPU visibility: an explicit CUDA_VISIBLE_DEVICES from the launcher, equal to the config (FL-02) -----
        self.cuda_visible = self._check_cuda_visible(config)
        # -- the frozen evaluation flags and the runtime device class (PREG1 T3) ---------------------------
        self.frozen = frozen_eval()                       # v4eval.frozen_config.load(): the object v4eval needs
        if EvalConfig.from_frozen(self.frozen) != config.eval_config:
            raise TrainerError("the run's EvalConfig differs from v4eval.frozen_config.load() (PREG1 T3)")
        self.eval_device_class = runtime_device_class(self.device)
        self.device_pin = pin_for_device(self.device)
        self.device_pin_sha256 = self.device_pin["self_sha256"] if self.device_pin else None
        if self.eval_device_class not in accepted_classes(self.frozen.device_class, self.device_pin) and not config.test_overrides:
            raise TrainerError(f"evaluation device class {self.eval_device_class} != frozen "
                               f"{self.frozen.device_class} (PREG1 T3)")
        if self.device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != \
                self.frozen.flags["CUBLAS_WORKSPACE_CONFIG"]:
            raise TrainerError("CUBLAS_WORKSPACE_CONFIG must be set by the launcher before CUDA starts (T3)")
        # -- model (fresh, deterministic per seed; SASRec theta0 recentered, C7) -------------------------
        self.built = build(self.family, schema, config.seed, device, K=K, poc=poc, catalogue=catalogue,
                           v1_root=v1_root, variant=config.effective_variant, dropout=config.effective_dropout)
        if config.expected_init_sha256 and self.built.init_sha256 != config.expected_init_sha256:
            raise TrainerError(f"theta0 sha256 {self.built.init_sha256} != registered {config.expected_init_sha256}")
        if config.theta0_init_sha256 and self.built.init_sha256 != config.theta0_init_sha256:
            raise TrainerError(f"theta0 sha256 {self.built.init_sha256} of {config.effective_variant} seed "
                               f"{config.seed} != the registered theta0_init_sha256 {config.theta0_init_sha256}")
        if self.built.variant != config.effective_variant:
            raise TrainerError(f"built variant {self.built.variant} != configured {config.effective_variant}")
        for f, mine in (("model_variant", self.built.variant), ("theta0_init_sha256", self.built.init_sha256)):
            if config.key.get(f) not in (None, mine):
                raise TrainerError(f"config.key {f} {config.key.get(f)!r} differs from the built model's {mine!r}")
        built_ssh = getattr(self.built, "schema_self_sha256", None)       # set by v4models for a projecting schema
        if built_ssh is not None and built_ssh != self.schema_record["schema_self_sha256"]:
            raise TrainerError("the model was not built from the verified schema")
        self.adapter = self.built.adapter
        self.module = self.built.module
        self._check_dropout()
        if evalp.manifest.catalogue_K != self.built.K:
            raise TrainerError("evaluation manifest catalogue_K differs from the model's K")
        cand = evalp.manifest.header.get("candidate_order_hash")
        if cand and config.key.get("candidate_order_hash") not in (None, cand):
            raise TrainerError("result-row candidate_order_hash differs from the evaluation manifest's")
        r = config.recipe
        self.opt = torch.optim.AdamW(self.adapter.param_groups(r.weight_decay), lr=config.peak_lr, betas=r.betas,
                                     eps=r.eps, fused=r.fused)
        self.config_sha256 = config.sha256()
        self.sealed = config.sealed_outputs                         # TRAIN-03 condition 1 (by construction)
        self.art = self.run_dir / "sealed" if self.sealed else self.run_dir     # every metric-bearing artifact
        self.auto_continue = config.effective_auto_continuation     # TRAIN-03 condition 4
        self.ckpt = CheckpointStore(self.art, self.config_sha256, fault=checkpoint_fault)
        self.monitor = Monitor()
        self.tracker = StopTracker(tolerance=0.0002)
        self.state = TrainState(phase=config.train_phase, ceiling=phase_end(config.train_phase),
                                snapshots=[] if self.tuning else None, disk_peak_bytes=0 if self.tuning else None)
        self._perm_cache = {}
        self.attempt = 0
        self.evaluator_sha = evaluator_code_sha256()
        self.weights_bytes = 4 * self.adapter.manifest.shared_numel
        self.key_base = self._key_base()

    # -------------------------------------------------------------------------------------------- setup helpers
    def _plan(self, phase: str, start_exposures: int, end_efe: Optional[float] = None) -> dict:
        """The planned per-update LR table (PREG1 T2 P2): hashed; min LR > 0; the main phase ends at 6N with 0.01 peak."""
        t = planned_lr_table(self.cfg.peak_lr, self.N, self.cfg.effective_batch, phase, self.cfg.exposure_point,
                             start_exposures=start_exposures, end_efe=end_efe,
                             schedule=self.cfg.effective_schedule if phase != "CONTINUATION" else "S0")
        if not self.cfg.test_overrides:
            if not t["min_lr"] or t["min_lr"] <= 0.0:
                raise TrainerError(f"planned LR table has a non-positive LR ({t['min_lr']}; pin P2)")
            if phase in ("MAIN", "TUNING") and (t["final_exposures"] != 6 * self.N or
                                    abs(t["last_lr"] - 0.01 * self.cfg.peak_lr) > 1e-12 * self.cfg.peak_lr):
                raise TrainerError("the final main update must end at 6N exposures with LR 0.01 * peak (pin P2)")
        return t

    @staticmethod
    def _check_cuda_visible(config: RunConfig) -> dict:
        """snap_v1/reuse/train.py setdefaults CUDA_VISIBLE_DEVICES to "0" on import, so the variable is always set after
        ppsi_env: refuse a launch that did not export it (captured by v4train before that import) and any mismatch with
        the explicit run-config value."""
        import v4train
        env = os.environ.get("CUDA_VISIBLE_DEVICES")
        launch = v4train.LAUNCH_CUDA_VISIBLE_DEVICES
        if env is None:
            raise TrainerError("CUDA_VISIBLE_DEVICES is unset: the launcher must export it explicitly")
        if launch is None:
            raise TrainerError("CUDA_VISIBLE_DEVICES was not exported by the launcher (it was unset before ppsi_env; "
                               "snap_v1/reuse/train.py then defaulted it to '0'); refuse to start")
        if env != config.cuda_visible_devices:
            raise TrainerError(f"CUDA_VISIBLE_DEVICES={env!r} differs from the run config's "
                               f"{config.cuda_visible_devices!r}")
        return {"config": config.cuda_visible_devices, "env": env, "at_v4train_import": launch}

    @staticmethod
    def _check_cublas(config: RunConfig) -> str:
        """The CUBLAS_WORKSPACE_CONFIG the launcher exported (cuBLAS reads it when CUDA initializes). STRICT refuses to
        start without ":4096:8" or ":16:8" (checked on every device, so a CPU launch behaves as the server's). A CUDA
        run is further pinned to the T3 value below."""
        env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if config.determinism_mode == "STRICT" and env not in CUBLAS_WORKSPACE_CONFIGS_STRICT:
            raise TrainerError(f"determinism_mode STRICT needs CUBLAS_WORKSPACE_CONFIG in "
                               f"{CUBLAS_WORKSPACE_CONFIGS_STRICT} exported by the launcher; it is {env!r}")
        return env if env else "UNSET"

    def _apply_determinism(self) -> None:
        """Apply the run's determinism mode (STRICT: warn_only off; WARN_ONLY: warn_only on) with the strict-FP32
        flags, then read the torch state back (raises when it is not the configured mode)."""
        enforce_strict_fp32(deterministic=True, warn_only=(self.cfg.determinism_mode == "WARN_ONLY"))
        self._determinism_record()

    def _determinism_record(self) -> dict:
        """The mode as recorded (config value, CUBLAS setting captured at construction, torch state read back now)."""
        eff = {"deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
               "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled())}
        want = {"deterministic_algorithms": True, "warn_only": self.cfg.determinism_mode == "WARN_ONLY"}
        if eff != want:
            raise TrainerError(f"effective determinism state {eff} is not the configured determinism_mode "
                               f"{self.cfg.determinism_mode} ({want})")
        return {"mode": self.cfg.determinism_mode, "cublas_workspace_config": self.cublas_workspace_config,
                "torch_effective": eff}

    def _check_resume_determinism(self, recorded: Optional[dict], where: str) -> None:
        """Refuse a resume whose determinism mode or CUBLAS setting differs from the recorded one."""
        if not isinstance(recorded, dict):
            raise TrainerError(f"resume refused: the {where} records no determinism mode (PREG1_AMENDMENT_01 item 4)")
        now = (self.cfg.determinism_mode, self.cublas_workspace_config)
        was = (recorded.get("mode"), recorded.get("cublas_workspace_config"))
        if now != was:
            raise TrainerError(f"resume refused: determinism mode / CUBLAS_WORKSPACE_CONFIG {now} differ from the "
                               f"{where}'s {was}")

    def _check_role(self, role: str) -> dict:
        """Both providers against the registry entry of `role` (manifest id, binding hashes, row count, fingerprint)."""
        try:
            return {"train": check_provider(self.registry, role, "train", self.train),
                    "eval": check_provider(self.registry, role, "eval", self.evalp)}
        except RegistryError as e:
            raise TrainerError(f"{self.cfg.population}/{role}: {e}") from e

    def _registry_record(self) -> dict:
        """The manifest-registry record of RUN_START / checkpoints / summaries (the run's START role)."""
        return dict(self.registry.record(), role=self.role, train_manifest=self.bindings["train"]["manifest_id"],
                    eval_manifest=self.bindings["eval"]["manifest_id"])

    def _schema_ident(self) -> dict:
        return {"schema_id": self.schema_record["schema_id"],
                "schema_self_sha256": self.schema_record["schema_self_sha256"]}

    def _check_resume_registry(self, recorded: Optional[dict], where: str) -> None:
        """Refuse a resume whose manifest registry (file sha256, self-hash, data version) differs from the record."""
        rec = (recorded or {}).get("manifest_registry") if isinstance(recorded, dict) else None
        if not isinstance(rec, dict):
            raise TrainerError(f"resume refused: the {where} records no manifest registry (TRAIN-01R2)")
        now = {k: self.registry.record()[k] for k in ("file_sha256", "self_sha256", "data_version")}
        was = {k: rec.get(k) for k in now}
        if now != was:
            raise TrainerError(f"resume refused: the manifest registry changed since the {where} ({was} -> {now})")

    def _check_resume_schema(self, recorded: Optional[dict], where: str) -> None:
        """Refuse a resume whose schema (schema_id, schema_self_sha256) differs from the recorded one."""
        rec = (recorded or {}).get("schema") if isinstance(recorded, dict) else None
        if not isinstance(rec, dict):
            raise TrainerError(f"resume refused: the {where} records no schema (TRAIN-01R2)")
        now = self._schema_ident()
        was = {k: rec.get(k) for k in now}
        if now != was:
            raise TrainerError(f"resume refused: the schema changed since the {where} ({was} -> {now})")

    def _check_dropout(self) -> None:
        """The built module's dropout must be the config's (the recipe's executed values, or the HP trial's)."""
        want = tuple(float(x) for x in self.cfg.effective_dropout)
        m = self.module
        if self.family == "GRU":
            got = (float(m.input_dropout.p), float(m.readout_dropout.p))
            if got != want:
                raise TrainerError(f"GRU dropout {got} != registered {want}")
        elif (float(m.cfg.dropout),) != want:
            raise TrainerError(f"SASRec dropout {m.cfg.dropout} != registered {want}")

    def _hp_record(self) -> Optional[dict]:
        """TRAIN-02: the HP trial of a TUNING run (the registry is verified again here, then recorded)."""
        if self.cfg.hp_id is None:
            return None
        from .hp_registry import REGISTRY_PATH, HPRegistryError, load_hp_registry, trial, trial_record
        try:
            reg = load_hp_registry(self.cfg.hp_registry or REGISTRY_PATH)
            t = trial(reg, self.cfg.hp_id)
        except HPRegistryError as e:
            raise TrainerError(f"HP registry refused: {e}") from e
        return dict(trial_record(reg, t), trial=t)

    def _recipe_record(self) -> Optional[dict]:
        """TRAIN-02R1 F6: the tuned recipe of a MAIN run (TUNED_RECIPES_02B.json, verified again, then recorded)."""
        if self.cfg.recipe_id is None:
            return None
        from .hp_registry import REGISTRY_PATH as HP_PATH
        from . import tuned_recipes as TRM
        try:
            path = self.cfg.recipe_registry or TRM.REGISTRY_PATH
            reg = TRM.load_tuned_recipes(path, self.cfg.hp_registry or HP_PATH)
            pin = None if self.cfg.test_overrides else TRM.check_production_registry(path, reg)   # TRAIN-02R2 F2
            r = TRM.recipe(reg, self.cfg.recipe_id)
        except TRM.TunedRecipeError as e:
            raise TrainerError(f"tuned recipe refused: {e}") from e
        return dict(TRM.recipe_record(reg, r), recipe=r, g1b_pin=pin)

    def _variant_registry_record(self) -> Optional[dict]:
        """TRAIN-02: the model-variant registry (contracts/model_variants), verified against v4models.variants. It is
        required for a non-current variant and for a non-test TUNING run; otherwise recorded when present."""
        p = model_variants.REGISTRY_PATH
        need = (not model_variants.is_current_build(self.cfg.effective_variant)) or \
            (self.tuning and not self.cfg.test_overrides)
        if not p.is_file() and not need:
            return None
        try:
            r = model_variants.load_variant_registry(p)
        except model_variants.VariantError as e:
            raise TrainerError(f"model variant registry refused: {e}") from e
        return {"path": r["path"], "file_sha256": r["file_sha256"], "self_sha256": r["self_sha256"],
                "variant": self.cfg.effective_variant}

    def _key_base(self) -> dict:
        k = dict(self.cfg.key)
        need = ("data_protocol_hash", "feature_contract_hash", "split", "label_policy_hash", "population_hash",
                "candidate_order_hash")
        missing = [f for f in need if not k.get(f)]
        if missing:
            raise TrainerError(f"config.key lacks comparison-key fields {missing}")
        if k["split"] != "VALIDATION":
            raise TrainerError("central training evaluates on VALIDATION only")
        topo = {"topology": TOPOLOGY[self.family], "widths": self.built.widths.as_json(),
                "manifest_digest": self.built.manifest_digest, "schema_self_sha256": getattr(self.schema, "self_sha256",
                                                                                              None)}
        if not model_variants.is_current_build(self.built.variant):      # the current builds keep their topology hash
            topo["model_variant"] = model_variants.topology(self.built.variant)
        k.update({"family": self.family, "topology_hash": _sha_obj(json.loads(json.dumps(topo, default=list))),
                  "initialization_block": f"I_META_RANDOM|{self.family}|seed={self.cfg.seed}|"
                                          f"theta0_sha256={self.built.init_sha256}",
                  "objective_id": OBJECTIVE_ID, "precision_id": self.cfg.precision_id,
                  "determinism_mode": self.cfg.determinism_mode,
                  "cublas_workspace_config": self.cublas_workspace_config,
                  "training_paradigm": "CENTRAL", "optimizer_lifecycle": "PERSISTENT_ADAMW"})
        k.update(self._data_identity_fields())
        return k

    def _data_identity_fields(self) -> dict:
        """TRAIN-01R2 row-key fields: the registry, its data version and role, the manifests, the schema, the label."""
        return {"data_version": self.registry.data_version, "manifest_role": self.role,
                "manifest_registry_sha256": self.registry.file_sha256,
                "manifest_registry_self_sha256": self.registry.self_sha256,
                "train_manifest_id": self.bindings["train"]["manifest_id"],
                "eval_manifest_id": self.bindings["eval"]["manifest_id"],
                "schema_id": self.schema_record["schema_id"],
                "schema_self_sha256": self.schema_record["schema_self_sha256"], "run_label": self.run_label,
                "model_variant": self.built.variant, "theta0_init_sha256": self.built.init_sha256,
                "run_role": self.run_role,
                **({"recipe_id": self.tuned_recipe["recipe_id"], "recipe_sha256": self.tuned_recipe["recipe_sha256"]}
                   if getattr(self, "tuned_recipe", None) else {})}

    def _row_labels(self) -> dict:
        """Extra row fields: the PREG1 T3 comparison-key additions, the determinism mode (PREG1_AMENDMENT_01 item 4;
        v4eval's make_row keeps only its own key fields, so they are added here), the T1 tie-break status and the
        TRAIN-01R2 data identity (registry, data version, role, schema) with the derived run_label."""
        lab = {"exposure_point": self.cfg.exposure_point, "population_scope": self.population,
               "determinism_mode": self.cfg.determinism_mode,
               "cublas_workspace_config": self.cublas_workspace_config}
        lab.update(self._data_identity_fields())
        if self.device_pin_sha256:
            lab["device_pin_sha256"] = self.device_pin_sha256
        if self.tiebreak:
            lab.update(row_status="ARCHITECTURE_TIEBREAK", population_scope="ALL",
                       comparison_status="HISTORICAL_REFERENCE", c_matched_eligible=False,
                       downstream_seed_allowed=False)
        elif self.tuning:
            lab.update(row_status="HP_SEARCH", comparison_status="HP_SEARCH", population_scope="CALIBRATION",
                       c_matched_eligible=False, downstream_seed_allowed=False)
        elif self.screen:
            lab.update(row_status="FEATURE_SEARCH_SCREEN", comparison_status="FEATURE_SEARCH_SCREEN",
                       population_scope="FEATURE_SEARCH_SCREEN", c_matched_eligible=False,
                       downstream_seed_allowed=False)
        else:
            lab.update(row_status="CENTRAL_CONTEST", comparison_status="SAME_CONDITIONS_BLOCK_CANDIDATE",
                       c_matched_eligible=self.run_label == "PRODUCTION",
                       downstream_seed_allowed=self.run_label == "PRODUCTION")
        if self.hp is not None:
            lab.update(hp_id=self.hp["hp_id"], hp_sha256=self.hp["hp_sha256"],
                       hp_registry_self_sha256=self.hp["hp_registry_self_sha256"],
                       schedule=self.cfg.effective_schedule, dropout=list(self.cfg.effective_dropout))
        return lab

    def _perm(self, p: int) -> np.ndarray:
        if p not in self._perm_cache:
            perm = pass_permutation(self.cfg.seed, p, self.N)
            sha = permutation_sha256(perm)
            rec = self.state.perm_sha256.get(str(p))
            if rec is not None and rec != sha:
                raise TrainerError(f"pass {p} permutation sha256 {sha} != recorded {rec} (order policy drift)")
            self.state.perm_sha256[str(p)] = sha
            self._perm_cache = {p: perm}
        return self._perm_cache[p]

    def _planned_peak_gib(self, phase: str) -> float:
        w = self.weights_bytes
        latest = 3 * w                                   # weights + AdamW m, v
        peak = {"MAIN": 2 * latest + 2 * w, "CALIBRATION": 2 * latest, "CONTINUATION": 2 * latest + w,
                "TUNING": 2 * latest + 4 * w}[phase]           # TUNING: + 3 snapshots + the CKPT_AVG average
        return peak / 2 ** 30

    def _disk_peak(self) -> int:
        """The weight-file disk peak since the run start (TUNING runs; CheckpointStore + the restored train state)."""
        return int(max(self.state.disk_peak_bytes or 0, self.ckpt.peak_bytes))

    def _disk(self, phase: str, allow_pause: bool) -> dict:
        d = disk_check(self.run_dir if self.run_dir.exists() else self.run_dir.parent, self._planned_peak_gib(phase),
                       disk_usage_fn=self.disk_usage_fn)
        if d.state == "REFUSE" or (d.state == "PAUSE" and not allow_pause):
            raise DiskGuardError(f"disk guard {d.state}: {d.reason}")
        return {"state": d.state, "free_gib": d.free_gib, "reason": d.reason}

    def _event(self, **ev) -> None:
        append_event(self.run_dir / "EVENTS.jsonl", {"run_id": self.cfg.run_id, **ev})

    def _status(self, status: str, **extra) -> None:
        payload = to_jsonable({"run_id": self.cfg.run_id, "status": status, "phase": self.state.phase,
                               "updates": self.state.updates, "efe": self.state.exposures / self.N,
                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})
        for i in range(40):                              # v4sup atomic status write; Windows replace retried
            try:
                write_status_atomic(self.run_dir / "STATUS.json", payload)
                return
            except PermissionError:
                if os.name != "nt" or i == 39:
                    raise
                time.sleep(0.05)

    def _meta(self, **extra) -> dict:
        return {"run_id": self.cfg.run_id, "family": self.family, "seed": self.cfg.seed,
                "config_sha256": self.config_sha256, "init_sha256": self.built.init_sha256,
                "updates": self.state.updates, "exposures": self.state.exposures,
                "efe": self.state.exposures / self.N, "determinism": self._determinism_record(),
                "manifest_registry": self._registry_record(), "schema": self._schema_ident(),
                "run_label": self.run_label, "run_role": self.run_role, "model_variant": self.built.variant,
                **({"device_pin_sha256": self.device_pin_sha256} if self.device_pin_sha256 else {}),
                **extra}

    # -------------------------------------------------------------------------------------------- lifecycle
    def run(self) -> dict:
        assert_strict_fp32()
        self._apply_determinism()
        if (self.run_dir / "RUN_START.json").exists():
            raise TrainerError("run directory already holds a run (use resume(); evidence is never overwritten)")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        disk = self._disk(self.cfg.train_phase, allow_pause=False)
        start = self._run_start_record(disk)
        write_json_atomic(self.run_dir / "RUN_START.json", start)
        self._event(event="START", phase=self.cfg.phase)
        torch.manual_seed(int(self.cfg.seed))              # the dropout stream (global RNG), restored on resume
        self.module.train()
        self._status("RUNNING")
        return self._loop()

    def _run_start_record(self, disk: dict) -> dict:
        budget = None
        if self.cfg.continuation_budget:
            bp = Path(self.cfg.continuation_budget)
            budget = {"path": str(bp), "sha256": sha256_file(bp), "content": self._read_budget(bp)}
        calib = None
        if self.cfg.calibration_record:
            cp = Path(self.cfg.calibration_record)
            rec = read_json(cp)
            if rec.get("family") != self.family:
                raise TrainerError("calibration record is for another family")
            if not self.cfg.test_overrides:                  # a non-test run: a PRODUCTION calibration only (R3)
                try:
                    accept_calibration_record(rec, family=self.family, data_version=self.cfg.data_version)
                except SelectionError as e:
                    raise TrainerError(f"calibration record refused: {e}") from e
            if round(float(rec["chosen_peak_lr"]), 12) != round(self.cfg.peak_lr, 12):
                raise TrainerError(f"main peak LR {self.cfg.peak_lr} != calibrated {rec['chosen_peak_lr']}")
            calib = {"path": str(cp), "sha256": sha256_file(cp), "chosen_peak_lr": rec["chosen_peak_lr"],
                     "run_label": rec.get("run_label"), "data_version": rec.get("data_version")}
        return {"run_id": self.cfg.run_id, "config": self.cfg.as_json(), "config_sha256": self.config_sha256,
                "family": self.family, "seed": self.cfg.seed, "phase": self.cfg.phase, "N": self.N,
                "init_sha256": self.built.init_sha256, "raw_init_sha256": self.built.raw_init_sha256,
                "theta0_recentered": self.built.recentered, "manifest_digest": self.built.manifest_digest,
                "train_provider": {"manifest_id": self.train.manifest_id, "fingerprint": self.train.fingerprint},
                "eval_provider": {"manifest_id": self.evalp.manifest.manifest_id,
                                  "fingerprint": self.evalp.fingerprint, "hashes": self.evalp.manifest.hashes},
                "order_policy": ORDER_POLICY, "schedule": "v4train.schedule (G0 D6; G0B C3)",
                "exposure_point": self.cfg.exposure_point, "cuda_visible_devices": self.cuda_visible,
                "determinism": self._determinism_record(), "precision_id": self.cfg.precision_id,
                "eval_config": self.cfg.eval_config.as_json(),
                "eval_config_sha256": self.cfg.eval_config.sha256(), "eval_device_class": self.eval_device_class,
                **({"device_pin_sha256": self.device_pin_sha256} if self.device_pin_sha256 else {}),
                "population": self.population, "planned_lr_table": {k: self.planned[k] for k in (
                    "sha256", "n_updates", "min_lr", "last_lr", "final_exposures", "point", "phase", "schedule")},
                "run_role": self.run_role, "train_phase": self.cfg.train_phase, "model_variant": self.built.variant,
                "dropout": list(self.cfg.effective_dropout), "lr_schedule_id": self.cfg.effective_schedule,
                "theta0_init_sha256": self.cfg.theta0_init_sha256, "shared_numel": self.adapter.manifest.shared_numel,
                "hp": self.hp, "model_variant_registry": self.variant_registry, "tuned_recipe": self.tuned_recipe,
                "evaluator_code_sha256": self.evaluator_sha, "trainer_code_sha256": trainer_code_sha256(),
                "models_code_sha256": models_code_sha256(), "schema_self_sha256": getattr(self.schema, "self_sha256",
                                                                                        None),
                "numerics": numerics_receipt(), "torch": torch.__version__, "device": str(self.device),
                "continuation_budget": budget, "calibration_record": calib, "disk": disk,
                "key_base": self.key_base, "test_overrides": self.cfg.test_overrides,
                "data_version": self.cfg.data_version, "manifest_role": self.role, "run_label": self.run_label,
                "manifest_registry": dict(self._registry_record(), bindings=self.bindings),
                "schema": dict(self.schema_record),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def _check_start_identity(self) -> dict:
        start = read_json(self.run_dir / "RUN_START.json")
        self._check_resume_determinism(start.get("determinism"), "run start record")
        self._check_resume_registry(start, "run start record")
        self._check_resume_schema(start, "run start record")
        idx = self.ckpt.index.get("latest")
        if idx is not None:
            self._check_resume_determinism(idx.get("meta", {}).get("determinism"), "latest checkpoint index")
            self._check_resume_registry(idx.get("meta", {}), "latest checkpoint index")
            self._check_resume_schema(idx.get("meta", {}), "latest checkpoint index")
        checks = {"config_sha256": (start["config_sha256"], self.config_sha256),
                  "device_pin_sha256": (start.get("device_pin_sha256"), self.device_pin_sha256),
                  "init_sha256": (start["init_sha256"], self.built.init_sha256),
                  "train_fingerprint": (start["train_provider"]["fingerprint"], self.train.fingerprint),
                  "eval_fingerprint": (start["eval_provider"]["fingerprint"], self.evalp.fingerprint),
                  "trainer_code_sha256": (start["trainer_code_sha256"], trainer_code_sha256()),
                  "models_code_sha256": (start["models_code_sha256"], models_code_sha256()),
                  "evaluator_code_sha256": (start["evaluator_code_sha256"], self.evaluator_sha)}
        was_hp, was_var = start.get("hp") or {}, start.get("model_variant_registry") or {}
        checks["hp_registry_file_sha256"] = (was_hp.get("hp_registry_file_sha256"),
                                             (self.hp or {}).get("hp_registry_file_sha256"))
        checks["hp_sha256"] = (was_hp.get("hp_sha256"), (self.hp or {}).get("hp_sha256"))
        checks["model_variant_registry_file_sha256"] = (was_var.get("file_sha256"),
                                                        (self.variant_registry or {}).get("file_sha256"))
        was_rec = start.get("tuned_recipe") or {}
        # TRAIN-03: the RECIPE entry is bound, not the registry file (a part-2 registration adds SASRec and leaves the
        # GRU entry byte-identical, so a GRU contest run resumes across it; the registry is re-verified and pinned)
        checks["tuned_recipe_sha256"] = (was_rec.get("recipe_sha256"), (self.tuned_recipe or {}).get("recipe_sha256"))
        bad = {k: v for k, v in checks.items() if v[0] != v[1]}
        if bad:
            raise TrainerError(f"resume refused, identity changed: {sorted(bad)}")
        return start

    def _restore(self, obj: dict) -> None:
        self.module.load_state_dict(obj["model"], strict=True)
        self.opt.load_state_dict(obj["optimizer"])
        self._restore_rng(obj["rng"])
        self.state = TrainState.from_dict(obj["train_state"])
        self.tracker = StopTracker.from_state(obj["stop_state"])
        self._perm_cache = {}
        if self.state.pos:
            self._perm(self.state.pass_index)            # re-derives and checks the in-progress pass

    def _restore_rng(self, rng: dict) -> None:
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])

    def resume(self) -> dict:
        assert_strict_fp32()
        self._apply_determinism()
        self._check_start_identity()
        status = read_json(self.run_dir / "STATUS.json")
        if str(status.get("status", "")).startswith("DONE"):
            raise TrainerError("the run is finished; nothing to resume")
        sp = self.art / "RUN_SUMMARY.json"
        if sp.exists() and not self.ckpt.exists("latest"):
            summary = read_json(sp)                      # crashed after the retention deletion: only the status is due
            if str(summary.get("status", "")).startswith("DONE"):
                self._status(summary["status"], recovered=True)
                return summary
            if self.tuning and summary.get("status") == "PENDING_CKPT_AVG":     # TRAIN-02: finish the CKPT_AVG step
                self.attempt = int(summary.get("attempt", 0)) + 1
                self._event(event="RESUME_CKPT_AVG", attempt=self.attempt)
                self.state = TrainState.from_dict(summary["train_state"])
                return self._complete_tuning(summary)
        self._disk(self.cfg.train_phase, allow_pause=True)
        obj = self.ckpt.load("latest")
        self._check_resume_determinism(obj["meta"].get("determinism"), "latest checkpoint")
        self._check_resume_registry(obj["meta"], "latest checkpoint")
        self._check_resume_schema(obj["meta"], "latest checkpoint")
        self._restore(obj)
        if self.state.phase == "CONTINUATION":
            self._check_role("CONTINUATION")
        if self.state.phase == "CONTINUATION":           # an interrupted continuation: its own planned table
            main_updates = self.planned["n_updates"]
            self.planned = self._plan("CONTINUATION", mark_exposures(MAIN_END, self.N), end_efe=self.state.ceiling)
            self.planned_offset = main_updates
        self.attempt = int(obj["meta"].get("attempt", 0)) + 1
        self._event(event="RESUME", from_update=self.state.updates, attempt=self.attempt)
        self.module.train()
        self._status("RUNNING", attempt=self.attempt)
        return self._loop()

    @staticmethod
    def _read_budget(path: Path) -> dict:
        b = read_json(path)
        if b.get("schema") != BUDGET_SCHEMA:
            raise TrainerError(f"budget file schema must be {BUDGET_SCHEMA}")
        mx = float(b.get("max_efe", 0))
        if not (MAIN_END < mx <= CEILING) or abs(mx * 2 - round(mx * 2)) > 1e-9:
            raise TrainerError(f"budget max_efe {mx} must be a 0.5 multiple in ({MAIN_END}, {CEILING}]")
        return b

    def continue_run(self) -> dict:
        assert_strict_fp32()
        self._apply_determinism()
        if self.tuning:
            raise TrainerError("a TUNING run has no continuation (PREG1_AMENDMENT_02R1 §2)")
        start = self._check_start_identity()
        summary = read_json(self.art / "RUN_SUMMARY.json")
        if summary.get("status") != "DONE_MAIN" or summary.get("phase") != "MAIN":
            raise TrainerError("continuation needs a finished main run")
        if self.tiebreak:
            raise TrainerError("the full-TRAIN tie-break has no continuation (PREG1 T1)")
        if self.screen:
            raise TrainerError("a FEATURE_SEARCH_SCREEN run has no continuation")
        self._check_role("CONTINUATION")                  # the registry's CONTINUATION role serves these providers
        if not summary.get("still_improving"):
            raise TrainerError("continuation refused: the run is not STILL_IMPROVING at 6.0")
        b = start.get("continuation_budget")
        if not b:
            raise TrainerError("continuation refused: no budget file was recorded at run start (pre-recorded rule)")
        bp = Path(b["path"])
        if not bp.is_file() or sha256_file(bp) != b["sha256"]:
            raise TrainerError("continuation refused: the budget file changed after it was recorded")
        budget = self._read_budget(bp)
        if budget.get("run_id") != self.cfg.run_id:
            raise TrainerError("continuation budget is for another run")
        self._disk("CONTINUATION", allow_pause=False)
        obj = self.ckpt.load("latest")
        self._check_resume_determinism(obj["meta"].get("determinism"), "latest checkpoint")
        self._check_resume_registry(obj["meta"], "latest checkpoint")
        self._check_resume_schema(obj["meta"], "latest checkpoint")
        self._restore(obj)
        if abs(self.state.exposures - MAIN_END * self.N) > 0:
            raise TrainerError("the resumable is not the 6.0 state")
        self.state.phase = "CONTINUATION"
        self.state.done = False
        self.state.ceiling = float(budget["max_efe"])
        self.planned = self._plan("CONTINUATION", self.state.exposures, end_efe=self.state.ceiling)
        self.planned_offset = self.state.updates
        self.attempt = int(obj["meta"].get("attempt", 0)) + 1
        self._event(event="CONTINUATION_START", ceiling=self.state.ceiling, budget_sha256=b["sha256"])
        self.module.train()
        self._status("RUNNING", attempt=self.attempt)
        return self._loop()

    # -------------------------------------------------------------------------------------------- training loop
    def _pending_marks(self) -> list:
        done = {round(e["mark"], 6) for e in self.state.evals}
        marks = [m for m in eval_marks(self.state.phase) if m not in done and m <= self.state.ceiling + 1e-9]
        if self.state.snapshots is not None:              # TRAIN-02 (TUNING): the weights-only snapshot marks too
            sdone = {round(x, 6) for x in self.state.snapshots}
            marks = sorted(set(marks) | {m for m in snapshot_marks(self.state.phase)
                                         if m not in sdone and m <= self.state.ceiling + 1e-9})
        return marks

    def _phase_complete(self) -> bool:
        ev = [e for e in self.state.evals if e["phase"] == self.state.phase]
        if not ev:
            return False
        last = ev[-1]["mark"]
        if self.state.phase == "MAIN":
            return abs(last - MAIN_END) < 1e-9
        if self.state.phase == "CALIBRATION":
            return abs(last - phase_end("CALIBRATION")) < 1e-9
        if self.state.phase == "TUNING":
            return abs(last - MAIN_END) < 1e-9 and MAIN_END in (self.state.snapshots or [])
        return self.tracker.continuation_should_stop() or last >= self.state.ceiling - 1e-9

    def _finish_phase(self) -> None:
        {"MAIN": self._finish_main, "CALIBRATION": self._finish_calibration,
         "CONTINUATION": self._finish_continuation, "TUNING": self._finish_tuning}[self.state.phase]()

    def _loop(self) -> dict:
        marks = self._pending_marks()
        shared = self.adapter.shared_parameters()
        r = self.cfg.recipe
        B, N = self.cfg.effective_batch, self.N
        try:
            assert_strict_fp32()
            if not self.state.done and self._phase_complete():
                self._finish_phase()                     # a crash after the last mark's `latest` save: finish again
            while not self.state.done:
                if not marks:
                    marks = self._pending_marks()         # TRAIN-03: an automatic continuation opened new marks
                if not marks:
                    raise TrainerError("no pending mark but the phase is not finished")
                perm = self._perm(self.state.pass_index)
                start, stop = self.state.pos, min(N, self.state.pos + B)
                t0 = time.perf_counter()
                batch = self.train.batch(perm[start:stop])
                wait = time.perf_counter() - t0
                n = host_contributing(batch)
                if n != stop - start:
                    raise TrainerError("a training batch holds rows that are not loss-eligible")
                t1 = time.perf_counter()
                lr = lr_for_update(self.cfg.peak_lr, self.state.exposures, n, N, self.state.phase,
                                   point=self.cfg.exposure_point,
                                   schedule=self.cfg.effective_schedule if self.state.phase != "CONTINUATION" else "S0")
                i = self.state.updates - self.planned_offset
                if i >= len(self.planned['lr']) or self.planned['lr'][i] != lr:
                    raise TrainerError(f'update {self.state.updates}: LR {lr} differs from the planned table')
                for g in self.opt.param_groups:
                    g["lr"] = lr
                self.opt.zero_grad(set_to_none=True)
                loss_sum = accumulate_step(self.adapter.scores, batch, n, self.cfg.microbatch, self.device)
                gn = torch.nn.utils.clip_grad_norm_(shared, r.grad_clip)
                self.opt.step()
                self.adapter.post_step()
                self.state.exposures += n
                self.state.updates += 1
                self.state.pos = stop
                if stop == N:
                    self.state.pass_index += 1
                    self.state.pos = 0
                self.monitor.add(loss_sum, n, gn, wait, time.perf_counter() - t1, lr)
                if self.state.exposures >= mark_exposures(marks[0], N):
                    self._on_mark(marks.pop(0))
                else:
                    if self.state.updates % self.cfg.log_every == 0:
                        self._flush()
                    if self.cfg.checkpoint_every and self.state.updates % self.cfg.checkpoint_every == 0:
                        self._save_latest()
                if self._fault is not None:
                    self._fault(self.state.updates)
        except InjectedInterruption:
            self._event(event="INTERRUPTED", update=self.state.updates)
            raise
        except Exception as e:
            self._event(event="FAILED", update=self.state.updates, error=f"{type(e).__name__}: {e}")
            self._status("FAILED", error=f"{type(e).__name__}: {e}")
            raise
        return read_json(self.art / "RUN_SUMMARY.json")

    def _public_metrics(self, rec: dict) -> None:
        """TRAIN-03: the unsealed metrics.jsonl of a sealed run carries no loss / gradient / eval value."""
        keep = {k: rec[k] for k in ("kind", "attempt", "phase", "update", "exposures", "efe", "window_updates",
                                    "step_wall_s", "data_wait_s", "mark") if k in rec}
        if "nonfinite" in rec:
            keep["loss_finite"] = not rec["nonfinite"]
        keep["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        append_jsonl(self.run_dir / "metrics.jsonl", keep)

    def _flush(self) -> None:
        rec = self.monitor.flush(self.adapter, self.state, self.N, self.attempt)
        if rec is None:
            return
        append_jsonl(self.art / "metrics.jsonl", rec)
        if self.sealed:
            self._public_metrics(rec)
        if rec["nonfinite"]:
            raise TrainingDiverged(f"non-finite loss or gradient in the window ending at update {rec['update']}")
        assert_strict_fp32()
        self._determinism_record()                       # the configured mode is still in force (no drift)

    def _rng(self) -> dict:
        return {"torch_cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    def _save_latest(self) -> None:
        self._flush()
        d = disk_check(self.run_dir, 2 * 3 * self.weights_bytes / 2 ** 30, disk_usage_fn=self.disk_usage_fn)
        if d.state == "REFUSE":
            raise DiskGuardError(f"disk guard before the resumable write: {d.reason}")
        if self.state.snapshots is not None:
            self.state.disk_peak_bytes = self._disk_peak()
        self.ckpt.save("latest", {"meta": self._meta(attempt=self.attempt), "model": self.module.state_dict(),
                                  "optimizer": self.opt.state_dict(), "rng": self._rng(),
                                  "train_state": self.state.as_dict(), "stop_state": self.tracker.state()})

    def _save_weights(self, kind: str, **meta) -> str:
        return self.ckpt.save(kind, {"meta": self._meta(state_digest=state_digest(self.adapter.broadcast_state()),
                                                         **meta),
                                     "model": self.module.state_dict()})

    # -------------------------------------------------------------------------------------------- evaluation
    def _evaluate(self, mark: float) -> EvalOutcome:
        if self._eval_override is not None:
            return self._eval_override(self, mark)
        m = self.evalp.manifest
        fz = self.frozen                                 # the one frozen evaluation config (no value hard-coded here)
        model_sha = state_digest(self.adapter.broadcast_state())
        apply_runtime_flags(fz)                          # T3 evaluation flags (deterministic, warn_only=False)
        try:
            check_runtime_flags(fz, self.device)
            art = predict_adapter(self.adapter, self.evalp.batches(fz.eval_batch), m, config=fz,
                                  model_sha256=model_sha)
            res = evaluate(m, art, config=fz, evaluator_code_sha256=self.evaluator_sha)
        finally:
            self._apply_determinism()                     # back to the training flags in the run's determinism mode
        v = res["E2E"]["macro"]["mrr@20"]
        if isinstance(v, Undefined):
            raise TrainerError(f"primary metric UNDEFINED on {m.manifest_id}: {v}")
        rows = rows_from_result(res, self.key_base, method="C", seed=self.cfg.seed, selection_rule=None,
                                efe=self.state.exposures / self.N)
        lab = self._row_labels()
        rows = [dict(r, **lab) for r in rows]
        sec = {"ndcg@10": res["E2E"]["macro"]["ndcg@10"], "hr@10": res["E2E"]["macro"]["hr@10"]}
        return EvalOutcome(value=float(v), rows=rows, result=res, ranks=art.ranks, secondary=sec)

    def _on_snapshot(self, mark: float) -> None:
        """TRAIN-02 (TUNING): a weights-only snapshot mark with no evaluation (5.0, 5.5)."""
        self._flush()
        self._save_snapshot(mark)
        self._save_latest()
        self._event(event="SNAPSHOT", mark=mark)
        self._status("RUNNING", last_snapshot=mark)

    def _save_snapshot(self, mark: float) -> None:
        from .ckpt_avg import snapshot_kind
        self._save_weights(snapshot_kind(mark), mark=mark)
        self.state.snapshots.append(float(mark))

    def _on_mark(self, mark: float) -> None:
        if self.tuning and round(mark, 6) not in eval_marks("TUNING"):
            return self._on_snapshot(mark)
        self._flush()
        efe = self.state.exposures / self.N
        out = self._evaluate(mark)
        if not math.isfinite(out.value):
            raise TrainerError("non-finite primary metric")
        phase = self.state.phase
        improved = self.tracker.update(mark, out.value, phase)
        tag = f"{mark:.1f}"
        rows_path = self.art / "evals" / f"eval_{tag}.rows.json"
        write_json_atomic(rows_path, {"mark": mark, "efe": efe, "rows": out.rows})
        if out.result is not None:
            write_json_atomic(self.art / "evals" / f"eval_{tag}.result.json", out.result)
        if out.ranks is not None:
            np.save(self.art / "evals" / f"eval_{tag}.ranks_int32.npy", np.asarray(out.ranks, dtype=np.int32))
        rec = {"mark": mark, "efe": efe, "exposures": self.state.exposures, "updates": self.state.updates,
               "value": out.value, "improved": improved, "phase": phase, "rows_file": f"evals/eval_{tag}.rows.json",
               "secondary": {k: (None if isinstance(v, Undefined) else float(v)) for k, v in out.secondary.items()}}
        self.state.evals.append(rec)
        append_jsonl(self.art / "metrics.jsonl", {"kind": "eval", "attempt": self.attempt, **rec})
        if self.sealed:
            self._public_metrics({"kind": "mark", "attempt": self.attempt, "phase": phase, "mark": mark, "efe": efe,
                                  "update": self.state.updates, "exposures": self.state.exposures})
        if phase == "MAIN":
            if improved:
                self._save_weights("best", mark=mark, value=out.value)
            if abs(mark - MAIN_END) < 1e-9:
                self._save_weights("endpoint_6.0", mark=mark, value=out.value)
        elif phase == "CONTINUATION" and improved:
            self._save_weights("best_extended", mark=mark, value=out.value)
        elif phase == "TUNING" and round(mark, 6) in snapshot_marks("TUNING"):
            self._save_snapshot(mark)                    # 6.0: the endpoint weights are the last snapshot
        self._save_latest()
        shown = "SEALED_UNTIL_STAGE_COMPLETE" if self.tiebreak else out.value
        if self.sealed:                                   # TRAIN-03 condition 1: mark index / EFE only
            self._event(event="EVAL", mark=mark, efe=efe)
            self._status("RUNNING", last_mark=mark)
        else:
            self._event(event="EVAL", mark=mark, value=shown, improved=(None if self.tiebreak else improved))
            self._status("RUNNING", last_mark=mark, last_value=shown)
        if self._phase_complete():
            self._finish_phase()

    # -------------------------------------------------------------------------------------------- finishing
    def _eval_rec(self, mark: float) -> dict:
        for e in self.state.evals:
            if abs(e["mark"] - mark) < 1e-9:
                return e
        raise TrainerError(f"no evaluation at {mark}")

    def _relabel(self, rec: dict, rule: str, checkpoint_id: str) -> list:
        if rule not in SELECTION_RULES:
            raise TrainerError(f"unknown selection rule {rule}")
        rows = read_json(self.art / rec["rows_file"])["rows"]
        out = []
        for r in rows:
            r = dict(r)
            r.update(selection_rule=rule, checkpoint_id=checkpoint_id, exposures=rec["exposures"], efe=rec["efe"])
            out.append(r)
        return out

    def _latest_sha(self) -> Optional[str]:
        rec = self.ckpt.index.get("latest")
        return rec["sha256"] if rec else None

    def _finish_main(self) -> None:
        """Idempotent: RESULT_ROWS, then RUN_SUMMARY, then the retention deletion, then the DONE status."""
        still = self.tracker.still_improving()
        keep_latest = still and not self.tiebreak and not self.screen   # no continuation for the tie-break / screen
        best = self._eval_rec(self.tracker.main_best_mark)
        end = self._eval_rec(MAIN_END)
        rows = (self._relabel(best, "PRACTICAL_BEST", f"{self.cfg.run_id}:best@{best['mark']:.1f}")
                + self._relabel(end, "CONTROLLED_BUDGET", f"{self.cfg.run_id}:endpoint_6.0"))
        write_json_atomic(self.art / "RESULT_ROWS.json", {"run_id": self.cfg.run_id, "rows": rows})
        self.state.done = True
        to_delete = None if keep_latest else self._latest_sha()
        summary = self._summary("DONE_MAIN", still_improving=still,
                                label=("BUDGET_LIMITED_STILL_IMPROVING" if still else "STOPPED_AT_CONTROLLED_ENDPOINT"),
                                practical_best={"mark": best["mark"], "value": best["value"], "efe": best["efe"]},
                                controlled_budget={"mark": end["mark"], "value": end["value"], "efe": end["efe"]},
                                latest_deleted_sha256=to_delete, latest_kept=keep_latest,
                                population=self.population,
                                row_status=self._row_labels()["row_status"])
        if not keep_latest:
            summary["checkpoints"].pop("latest", None)
        auto = bool(keep_latest and self.auto_continue)
        summary["continuation"] = "AUTO_C3" if auto else None
        write_json_atomic(self.art / "RUN_SUMMARY.json", summary)
        self._public_summary(summary)
        if not keep_latest:
            self.ckpt.delete("latest", "E12.1.8 / DISK rule 2: resumable deleted at 6.0 "
                             + ("(tie-break: no continuation)" if self.tiebreak else
                                "(feature-search screen: no continuation)" if self.screen and still else
                                "(not STILL_IMPROVING)"))
        self._event(event="DONE_MAIN", still_improving=still, continuation=("AUTO_C3" if auto else None))
        if auto:                                          # TRAIN-03 condition 4: no human reads a value to decide
            self._status("RUNNING", still_improving=True, continuation="AUTO_C3")
            self._begin_continuation(CEILING, source={"rule": "C3 (G0B C3; E12.1.6)", "automatic": True,
                                                      "knots": [[7.5, 0.1], [8.0, 0.01]], "ceiling": CEILING})
        else:
            self._status("DONE_MAIN", still_improving=still)

    def _begin_continuation(self, ceiling: float, *, source: dict) -> None:
        """Switch the live run (restored at the 6.0 state) into the CONTINUATION phase (C3 knots, `ceiling`)."""
        self._check_role("CONTINUATION")
        self._disk("CONTINUATION", allow_pause=False)
        if abs(self.state.exposures - MAIN_END * self.N) > 0:
            raise TrainerError("the resumable is not the 6.0 state")
        self.state.phase = "CONTINUATION"
        self.state.done = False
        self.state.ceiling = float(ceiling)
        self.planned = self._plan("CONTINUATION", self.state.exposures, end_efe=self.state.ceiling)
        self.planned_offset = self.state.updates
        self._event(event="CONTINUATION_START", ceiling=self.state.ceiling, **{k: v for k, v in source.items()
                                                                               if k not in ("knots", "ceiling")})
        self.module.train()

    def _public_summary(self, summary: dict) -> None:
        """TRAIN-03 condition 1: the unsealed RUN_SUMMARY.json of a sealed run (no metric value)."""
        if not self.sealed:
            return
        pub = {k: summary.get(k) for k in ("run_id", "status", "phase", "family", "seed", "config_sha256",
                                           "init_sha256", "N", "updates", "exposures", "efe", "run_label",
                                           "data_version", "manifest_role", "run_role", "model_variant",
                                           "continuation", "continuation_stop_mark")}
        pub["still_improving"] = bool(summary["still_improving"]) if "still_improving" in summary else None
        pub["marks_done"] = [e["mark"] for e in summary.get("evals", [])]
        pub.update(sealed=True, sealed_dir="sealed", determinism=summary.get("determinism"),
                   utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   note="TRAIN-03 seal: every metric value is in sealed/ (read only after the G2 unseal gate)")
        write_json_atomic(self.run_dir / "RUN_SUMMARY.json", pub)

    def _finish_calibration(self) -> None:
        self.state.done = True
        summary = self._summary("DONE_CALIBRATION", latest_deleted_sha256=self._latest_sha(),
                                value_at_end=self._eval_rec(phase_end("CALIBRATION"))["value"], weights_retained=[])
        summary["checkpoints"] = {}
        write_json_atomic(self.run_dir / "RUN_SUMMARY.json", summary)
        self.ckpt.delete("latest", "DISK rule 1: calibration keeps no weights")
        left = self.ckpt.weight_files()
        if left:
            raise TrainerError(f"calibration left weight files {left}")
        self._event(event="DONE_CALIBRATION")
        self._status("DONE_CALIBRATION")

    def _finish_tuning(self) -> None:
        """TRAIN-02 (TUNING, idempotent): RESULT_ROWS (the 6.0 endpoint), RUN_SUMMARY PENDING_CKPT_AVG, the resumable
        deleted (no continuation), then CKPT_AVG and the DONE_TUNING summary / status."""
        end = self._eval_rec(MAIN_END)
        rows = [dict(r, checkpoint_variant="ENDPOINT")
                for r in self._relabel(end, "CONTROLLED_BUDGET", f"{self.cfg.run_id}:endpoint_6.0")]
        write_json_atomic(self.run_dir / "RESULT_ROWS.json", {"run_id": self.cfg.run_id, "rows": rows,
                                                              "note": "HP_SEARCH rows; never selection evidence"})
        self.state.done = True
        to_delete = self._latest_sha()
        summary = self._summary("PENDING_CKPT_AVG", endpoint={"mark": end["mark"], "value": end["value"],
                                                               "efe": end["efe"]},
                                values_at_marks={f"{e['mark']:.1f}": e["value"] for e in self.state.evals},
                                latest_deleted_sha256=to_delete, latest_kept=False, population=self.population,
                                row_status="HP_SEARCH", attempt=self.attempt, train_state=self.state.as_dict())
        summary["checkpoints"].pop("latest", None)
        write_json_atomic(self.run_dir / "RUN_SUMMARY.json", summary)
        self.ckpt.delete("latest", "TRAIN-02 TUNING: resumable deleted at 6.0 (no continuation)")
        self._complete_tuning(summary)

    def _complete_tuning(self, summary: dict) -> dict:
        from .ckpt_avg import run_ckpt_avg
        rec = run_ckpt_avg(self)
        summary = dict(summary, status="DONE_TUNING",
                       ckpt_avg={k: rec.get(k) for k in ("value", "endpoint_value", "delta_vs_endpoint", "status", "outcome",
                                                         "rows_file", "deleted_sha256")},
                       disk_peak_bytes=max(int(rec.get("disk_peak_bytes") or 0), self._disk_peak()),
                       weight_files_left=self.ckpt.weight_files(),
                       checkpoints={k: {"sha256": v["sha256"], "file": v["file"]} for k, v in self.ckpt.index.items()})
        write_json_atomic(self.run_dir / "RUN_SUMMARY.json", summary)
        self._event(event="DONE_TUNING", value=summary["endpoint"]["value"], ckpt_avg=rec.get("value"))
        self._status("DONE_TUNING", endpoint_value=summary["endpoint"]["value"], ckpt_avg_value=rec.get("value"))
        return summary

    def _finish_continuation(self) -> None:
        self.state.done = True
        best = self._eval_rec(self.tracker.best_mark)
        ck = "best_extended" if best["phase"] == "CONTINUATION" else "best"
        rows = self._relabel(best, "PRACTICAL_BEST_EXTENDED", f"{self.cfg.run_id}:{ck}@{best['mark']:.1f}")
        write_json_atomic(self.art / "RESULT_ROWS_EXTENDED.json", {"run_id": self.cfg.run_id, "rows": rows,
                                                                   "note": "secondary; never used for family "
                                                                           "selection or a matched block (E12.1.7)"})
        prev = read_json(self.art / "RUN_SUMMARY.json")
        summary = self._summary("DONE_CONTINUATION", still_improving=prev["still_improving"], label=prev["label"],
                                practical_best=prev["practical_best"], controlled_budget=prev["controlled_budget"],
                                practical_best_extended={"mark": best["mark"], "value": best["value"],
                                                         "efe": best["efe"], "checkpoint": ck},
                                continuation_stop_mark=self.state.evals[-1]["mark"],
                                continuation=prev.get("continuation"),
                                latest_deleted_sha256=self._latest_sha())
        summary["checkpoints"].pop("latest", None)
        write_json_atomic(self.art / "RUN_SUMMARY.json", summary)
        self._public_summary(summary)
        self.ckpt.delete("latest", "continuation finished: resumable deleted")
        self._event(event="DONE_CONTINUATION")
        self._status("DONE_CONTINUATION")

    def _summary(self, status: str, **extra) -> dict:
        return {"run_id": self.cfg.run_id, "status": status, "phase": self.state.phase, "family": self.family,
                "seed": self.cfg.seed, "peak_lr": self.cfg.peak_lr, "config_sha256": self.config_sha256,
                "init_sha256": self.built.init_sha256, "N": self.N, "updates": self.state.updates,
                "exposures": self.state.exposures, "efe": self.state.exposures / self.N,
                "evals": [dict(e) for e in self.state.evals], "final_state_digest":
                    state_digest(self.adapter.broadcast_state()),
                "final_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                "checkpoints": {k: {"sha256": v["sha256"], "file": v["file"]} for k, v in self.ckpt.index.items()},
                "determinism": self._determinism_record(), "eval_every_efe": EVAL_EVERY,
                "run_label": self.run_label, "data_version": self.registry.data_version, "manifest_role": self.role,
                "manifest_registry": self._registry_record(), "schema": self._schema_ident(),
                "run_role": self.run_role, "model_variant": self.built.variant,
                **({"device_pin_sha256": self.device_pin_sha256} if self.device_pin_sha256 else {}),
                "shared_numel": self.adapter.manifest.shared_numel,
                "hp": None if self.hp is None else {k: v for k, v in self.hp.items() if k != "trial"},
                "hp_trial": None if self.hp is None else self.hp["trial"], **extra}


__all__ = ["CentralTrainer", "TrainerError", "DiskGuardError", "TrainingDiverged", "InjectedInterruption",
           "EvalOutcome", "TrainState", "Monitor", "trainer_code_sha256", "models_code_sha256", "BUDGET_SCHEMA",
           "RegistryError"]
