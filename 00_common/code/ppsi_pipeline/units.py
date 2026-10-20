"""The registry of training units: what each stage can skip, resume or start.

A *training unit* is one run directory of the study that a launcher creates under a run root on
the server (`runs_ds3/FL_T/FA_s2026`, `runs_p3/GRU/02B_GRU_s2026`, ...). The curated copy of
every run directory lives under `<stage>/results/runs/<phase>/...` in this repository, and the
published checkpoints are extracted into those same folders, so the skip / resume / start
decision is made on the results tree (`PPSI_RESULTS_ROOT`), while the launcher command refers to
the server tree (`PPSI_RUNS_ROOT`).

Every entry records only flags that exist in the launchers (`00_common/code/v4int/launch_*.py`,
`02_centralized/code/v4fs/run_fs.py`, the `x_rees46` lane wrappers, the extensions-tree
wrappers). The complete argv of every historical run is itself a sealed record of the study
(each run's `LAUNCH_*.json` / `ARM_SPEC.json` / `STATUS.json` on the server, and the lane
scripts under `03_federated_learning/scripts/tau_fixed_lanes/`), so commands whose arm-specific
flag values are pinned in those records carry a `<...>` placeholder instead of an invented
value. The run-directory names below are the curated names of `_build/TREE.txt`.

`tree` records which code tree the launcher is pinned to: the main-study launchers verify the
SHA-256 of their own code (`fl_code_sha256`, the snap_v1 manifest, the G1 lock pin) and refuse
to run from a tree whose bytes differ, so they are marked "original" (the flat `ppsi_v4` tree
whose deployment manifests are kept in `07_study_protocol/deployment_manifests/`); the
closure-phase wrappers ran from the extensions tree (`PPSI_EXTENSIONS_ROOT`).
"""
from __future__ import annotations

from dataclasses import dataclass

STAGE_ORDER = ("01_data", "02_centralized", "03_federated_learning", "04_on_device",
               "05_privacy_and_security", "06_evaluation_and_benchmarks")

TREE_ORIGINAL = "original"      # the flat ppsi_v4 server tree (07_study_protocol records it)
TREE_EXTENSIONS = "extensions"  # the closure-phase extensions tree (PPSI_EXTENSIONS_ROOT)

# Launcher command templates, in the shape the server tree runs them. ``--run-dir`` / ``--out``
# is filled with the server-side run directory; the orchestrator appends ``--resume`` when it
# decides to resume. Everything inside <...> is pinned in the sealed run records and must be
# filled from them, never invented here.
FS = ("python -B code/v4fs/run_fs.py --family {family} --schema <schema_id> --seed {seed} "
      "--gpu 0 --out {run_dir}")
CENTRAL = ("python -B code/v4int/launch_central.py --family {family} --seed {seed} "
           "--phase {phase} --gpu 0 --run-dir {run_dir} --history-policy USER_HISTORY_L50 "
           "<recipe flags>")
FL = ("python -B code/v4int/launch_fl.py --arm {arm} --seed {seed} --gpu 0 --run-dir {run_dir} "
      "--history-policy USER_HISTORY_L50 <arm flags>")
DEVICE = ("python -B code/v4int/launch_device.py --arm {arm} --seed {seed} --gpu 0 "
          "--run-dir {run_dir} --history-policy USER_HISTORY_L50 <arm flags>")
LANE = "bash laneX_{lane}.sh  (supervisor in scripts/tau_fixed_lanes/; per-job argv inside)"
EXT_FL = ("python {wrapper} --run-dir <extensions-tree run dir> <arm flags>  "
          "(PPSI_EXTENSIONS_ROOT)")
EXT_DEVICE = ("python launch_device.py --run-dir <extensions-tree run dir> <arm flags>  "
              "(PPSI_EXTENSIONS_ROOT)")
PRETRAIN_EXT = ("python -m experiment_extensions.pretraining.pretrain_resume resume "
                "--amendment <sealed amendment json> --amendment-sha256 <sha> "
                "--review <sealed review json> --review-sha256 <sha>")

FL_CKPT = ("ckpt/*/latest.pt",)                 # launch_fl: only latest.pt is a resume source
CENTRAL_CKPT = ("checkpoints/latest.pt", "sealed/checkpoints/latest.pt")
DEVICE_CKPT = ("shards/shard_*.json",)          # launch_device: resume is shard skip
SUMMARY = ("RUN_SUMMARY.json", "sealed/RUN_SUMMARY.json")


@dataclass(frozen=True)
class Unit:
    """One training unit: a run directory, its launcher and how resume works."""

    id: str                           # "<stage>/<uid>", unique in the registry
    run: str                          # run directory relative to <stage>/results/runs/
    server_run: str                   # the same run under PPSI_RUNS_ROOT ("" when unrecorded)
    command: str                      # launcher command template (see module docstring)
    resume: str = "none"              # the resume flag the launcher really supports
    resumable: tuple[str, ...] = ()   # globs (relative to the run dir) that make it resumable
    summary_names: tuple[str, ...] = SUMMARY
    tree: str = TREE_ORIGINAL
    note: str = ""

    @property
    def stage(self) -> str:
        return self.id.split("/", 1)[0]


def _fill(template: str, **values: str) -> str:
    """Substitute the known placeholders of a command template, leaving ``{run_dir}`` for
    `render_command` to fill with the server-side run directory."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def _u(stage: str, uid: str, run: str, server_run: str, command: str, resume: str = "none",
       resumable: tuple[str, ...] = (), tree: str = TREE_ORIGINAL, note: str = "") -> Unit:
    return Unit(id=f"{stage}/{uid}", run=run, server_run=server_run, command=command,
                resume=resume, resumable=resumable, tree=tree, note=note)


def _fl_units(stage: str, arms: tuple[tuple[str, tuple[int, ...]], ...], run_root: str,
              phase: str = "main_study", tree: str = TREE_ORIGINAL, note: str = "") -> list[Unit]:
    """One unit per (arm, seed); the run directory is `<root>/<arm>_s<seed>`."""
    return [
        _u(stage, f"{arm.lower()}_s{seed}", f"{phase}/{run_root}/{arm}_s{seed}",
           f"{run_root}/{arm}_s{seed}", _fill(FL, arm=arm, seed=str(seed)),
           resume="--resume", resumable=FL_CKPT, tree=tree, note=note)
        for arm, seeds in arms for seed in seeds
    ]


# ----------------------------------------------------------------------------------------- 01_data
def _fs_units() -> list[Unit]:
    """The feature-search campaign (D08). ``v4fs/run_fs.py`` has no resume flag."""
    runs = [("stageA", "L0S_CLEAN_V2_SESSION"),
            ("stageA", "L3S_NEAR_DUPLICATE_SESSION"),
            ("stageB", "L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7")]
    units = []
    for stage_dir, schema in runs:
        tag = schema.split("_", 1)[0].lower()
        for family in ("GRU", "SASREC"):
            for seed in (2026, 2027):
                name = f"{schema}__{family}__{seed}"
                units.append(_u(
                    "01_data", f"fs_{stage_dir}_{tag}_{family.lower()}_{seed}",
                    f"main_study/runs_fs/{stage_dir}/{name}", f"runs_fs/{stage_dir}/{name}",
                    _fill(FS, family=family, seed=str(seed)),
                    note="feature-search screen (D08); run_fs.py cannot resume"))
    return units


# --------------------------------------------------------------------------------- 02_centralized
def _central_units() -> list[Unit]:
    stage = "02_centralized"
    units: list[Unit] = []
    for family, rid in (("GRU", "02B_GRU"), ("GRU", "02B_GRU_C2"), ("SASREC", "02B_SASREC"),
                        ("SASREC", "02B_SASREC_C2")):
        for seed in (2026, 2027):
            units.append(_u(
                stage, f"p3_{rid.lower()}_s{seed}", f"main_study/runs_p3/{family}/{rid}_s{seed}",
                f"runs_p3/{family}/{rid}_s{seed}", _fill(CENTRAL, family=family, seed=str(seed),
                                                                   phase="MAIN"),
                resume="--resume", resumable=CENTRAL_CKPT,
                note="sealed P3 backbone-contest run (C10/C11)"))
    for seed in (2026, 2027):
        units.append(_u(stage, f"csmall_s{seed}", f"main_study/runs_ds/C_SMALL_s{seed}",
                        f"runs_ds/C_SMALL_s{seed}", _fill(CENTRAL, family="SASREC", seed=str(seed),
                                                                    phase="MAIN"),
                        resume="--resume", resumable=CENTRAL_CKPT,
                        note="sealed C_SMALL matched-size reference (C17)"))
        units.append(_u(stage, f"cfull_s{seed}",
                        f"main_study/runs_ds3/CFULL/C_FULL_02B_SASREC_s{seed}",
                        f"runs_ds3/CFULL/C_FULL_02B_SASREC_s{seed}",
                        _fill(CENTRAL, family="SASREC", seed=str(seed), phase="MAIN"),
                        resume="--resume", resumable=CENTRAL_CKPT,
                        note="sealed C_FULL cloud reference (C18), 1,348,682-user population"))
    for stage_dir, rid in (("stage1", "P2S_SASREC_D64_B2_S0_s0_s2026"),
                           ("stage1", "P2S_SASREC_D64_B2_S0_s1_s2026"),
                           ("stage1", "P2S_SASREC_D64_B2_S0_s2_s2026"),
                           ("stage1", "P2S_SASREC_D64_B2_S0_s3_s2026"),
                           ("stage2", "P2S_SASREC_D64_B2_S0_s1_s2027")):
        units.append(_u(
            stage, f"small_ladder_{rid}", f"main_study/runs_ds/small_ladder/SASREC/{stage_dir}/{rid}",
            f"runs_ds/small_ladder/SASREC/{stage_dir}/{rid}",
            _fill(CENTRAL, family="SASREC", seed=rid[-4:], phase="CALIBRATION"),
            resume="--resume", resumable=CENTRAL_CKPT, note="SMALL mini-ladder TUNING run (C16)"))
    units.append(_u(stage, "pre_theta_s2026",
                    "main_study/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026",
                    "runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026",
                    _fill(CENTRAL, family="SASREC", seed="2026", phase="MAIN"), resume="--resume",
                    resumable=CENTRAL_CKPT,
                    note="theta_pre, the 1,217,610-user server pretrain (C19)"))
    units.append(_u(stage, "fullpre_ext_s2027",
                    "main_study/new_runs/full_pre_seed2027/"
                    "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_FULLPRE_EXT_s2027",
                    "new_runs/full_pre_seed2027/"
                    "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_FULLPRE_EXT_s2027",
                    PRETRAIN_EXT, resume="--resume", resumable=CENTRAL_CKPT,
                    note="the s2027 full pretrain; amendment-gated manual resume that re-invokes "
                         "launch_central with --resume (experiment_extensions/pretraining)"))
    return units


# ---------------------------------------------------------------------------- 03_federated_learning
def _fl_stage_units() -> list[Unit]:
    stage = "03_federated_learning"
    arms = (("FA", (2026, 2027)), ("FA_WARM", (2026, 2027)), ("FA_WARM_FROZEN", (2026, 2027)),
            ("FP_WARM_FROZEN", (2026, 2027)), ("PF_WARM_FROZEN", (2026, 2027)),
            ("FA_Q8_WARM_FROZEN", (2026, 2027)), ("FA_1024_WARM_FROZEN", (2026, 2027)),
            ("DP8_FROZEN", (2026, 2027)), ("DP1_WARM_FROZEN", (2026,)),
            ("N33_FP", (2026, 2027)), ("N33_PF", (2026, 2027)), ("N33_FA_Q8", (2026, 2027)),
            ("N33_FA_1024", (2026, 2027)), ("N33_DP8", (2026, 2027)), ("N33_DP1", (2026,)))
    units = _fl_units(stage, arms, "runs_ds3/FL_T", note="Phase-1 MAIN federated arm")
    # The two clip-norm calibration runs carry no seed suffix / a fixed seed in the run tree.
    units.append(_u(stage, "s_cal", "main_study/runs_ds3/FL_T/S_CAL", "runs_ds3/FL_T/S_CAL",
                    _fill(FL, arm="S_CAL", seed="2026"), resume="--resume", resumable=FL_CKPT,
                    note="cold S_CAL clip-norm calibration (P1-21)"))
    units.append(_u(stage, "s_cal_warm_frozen_s2026",
                    "main_study/runs_ds3/FL_T/S_CAL_WARM_FROZEN_s2026",
                    "runs_ds3/FL_T/S_CAL_WARM_FROZEN_s2026",
                    _fill(FL, arm="S_CAL", seed="2026"), resume="--resume", resumable=FL_CKPT,
                    note="warm-frozen S_CAL clip-norm calibration (P1-21)"))
    p2 = (("P2_WSTAR_SMALL", (2026, 2027)), ("P2_WSTAR_FA1024_SMALL", (2026, 2027)),
          ("P2_WSTAR_DP8_SMALL", (2026, 2027)), ("P2_WSTAR_DP1_SMALL", (2026,)),
          ("P2_WSTAR_S_CAL_SMALL", (2026,)))
    units += _fl_units(stage, p2, "runs_p2/MAIN", note="Phase-2 RUN-3 arm (P2-06)")
    for seed in (2026, 2027):
        units.append(_u(stage, f"ft_p2_wstar_s{seed}",
                        f"main_study/runs_p2/MAIN/FT_P2_WSTAR_s{seed}",
                        f"runs_p2/MAIN/FT_P2_WSTAR_s{seed}",
                        _fill(DEVICE, arm="FT_P2_WSTAR", seed=str(seed)), resume="--resume",
                        resumable=DEVICE_CKPT, note="Phase-2 device fine-tune of W* (P2-11)"))
    units.append(_u(stage, "pre_pb100k_s2026",
                    "main_study/runs_exp_pb100/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_PB100K_s2026",
                    "runs_exp_pb100/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_PB100K_s2026",
                    _fill(CENTRAL, family="SASREC", seed="2026", phase="MAIN"), resume="--resume",
                    resumable=CENTRAL_CKPT, note="the 100,000-user pretrain (X11)"))
    units += _fl_units(stage, (("FA_WARM_PB100K", (2026,)), ("FA_WARM_FROZEN_PB100K", (2026,))),
                       "runs_exp_pb100", note="PB100 warm arms (X12)")
    # Closure-phase tau-fixed lanes (X-REES46): x3 = cold/DP lanes, x2 = the tuned PB100 winner,
    # x = the warm unfixed-items lane; all run from the extensions tree.
    x_lanes = (("x_fa_cold", "runs_x_rees46/COLD/MAIN/X_FA_COLD_eta0.001_tau1e-6_E6",
                _fill(LANE, lane="COLD"), "cold lane (X09)"),
               ("x_fa_unf", "runs_x_rees46/MAIN/X_FA_UNF_eta3e-06_tau1e-6",
                _fill(EXT_FL, wrapper="launch_fl_x.py"), "warm unfrozen-items lane (X08)"),
               ("x2_pb100", "runs_x_rees46/PB100/MAIN/X2_eta3e-05_c0.1_ilr0.1_tau1e-6_E6_P1",
                _fill(EXT_FL, wrapper="launch_fl_x2.py"), "tuned PB100 winner (X13/X14)"))
    for uid, root, command, note in x_lanes:
        for seed in (2026, 2027):
            units.append(_u(stage, f"{uid}_s{seed}", f"main_study/{root}_s{seed}",
                            f"{root}_s{seed}", command, resume="--resume", resumable=FL_CKPT,
                            tree=TREE_EXTENSIONS, note=f"{note}, seed {seed}"))
    local_passes = ("cold_FA_eta0.0003_P2_E6", "cold_FA_eta0.0003_P4_E6", "cold_FA_eta0.001_P2_E6",
                    "cold_FA_eta0.001_P4_E6", "cold_FP_eta0.001_P2_E6", "cold_PF_eta0.001_P2_E6",
                    "pb100_FA_eta0.0001_P4_E6", "pb100_FA_eta3e-05_P4_E6", "pb100_FP_eta3e-05_P2_E6",
                    "pb100_PF_eta3e-05_P2_E6")
    for name in local_passes:  # the curated run dirs carry the fixed _s2026 suffix
        units.append(_u(stage, f"local_passes_{name}", f"closure_phase/LOCAL_PASSES/{name}_s2026",
                        "", _fill(EXT_FL, wrapper="launch_fl_x2.py"), resume="--resume",
                        resumable=FL_CKPT, tree=TREE_EXTENSIONS,
                        note="closure-phase local-passes cell (extensions tree)"))
    for variant in ("FROZEN", "UNFROZEN"):  # the closure-phase pretrain-seed matrix (curated
        for seed in (2026, 2027):           # under this stage's results tree)
            name = f"EXT_FULLPRE2027_{variant}_FL_s{seed}"
            units.append(_u(stage, f"seed_matrix_{variant.lower()}_s{seed}",
                            f"main_study/new_runs/full_pre_seed_matrix/{name}", "",
                            _fill(EXT_FL, wrapper="launch_fl_x2.py"), resume="--resume",
                            resumable=FL_CKPT, tree=TREE_EXTENSIONS,
                            note="closure-phase pretrain-seed matrix (extensions tree)"))
    return units


# ------------------------------------------------------------------------------------ 04_on_device
def _device_units() -> list[Unit]:
    stage = "04_on_device"
    arms = (("LO", (2026, 2027)), ("FT_C", (2026, 2027)), ("FT_FA", (2026, 2027)),
            ("FT_FA_WARM", (2026, 2027)), ("FT_FA_WARM_FROZEN", (2026, 2027)),
            ("FT_PRE", (2026, 2027)))
    units = [
        _u(stage, f"{arm.lower()}_s{seed}", f"main_study/runs_ds3/DEV/{arm}_s{seed}",
           f"runs_ds3/DEV/{arm}_s{seed}", _fill(DEVICE, arm=arm, seed=str(seed)),
           resume="--resume", resumable=DEVICE_CKPT, note="device arm (P1-16/P1-17)")
        for arm, seeds in arms for seed in seeds
    ]
    for arm in ("LO", "FT_C"):  # the UNSEEN_HOLDOUT descriptive legs of P1-17
        units.append(_u(stage, f"{arm.lower()}_unseen_holdout_s2026",
                        f"main_study/runs_ds3/DEV/{arm}_UNSEEN_HOLDOUT_s2026",
                        f"runs_ds3/DEV/{arm}_UNSEEN_HOLDOUT_s2026",
                        _fill(DEVICE, arm=arm, seed="2026") + " --role UNSEEN_HOLDOUT",
                        resume="--resume", resumable=DEVICE_CKPT, note="P1-17 holdout leg"))
    for sel in ("SEL_FT_C", "SEL_LO", "SEL_FT_FA"):
        units.append(_u(stage, sel.lower(), f"main_study/runs_ds3/DEV/{sel}",
                        f"runs_ds3/DEV/{sel}", _fill(DEVICE, arm=sel, seed="2026"),
                        resume="--resume", resumable=DEVICE_CKPT,
                        note="P1-15 selection cells (launch_device --select)"))
    for arm in ("FT_X_COLD_MAIN", "FT_X_PB100_MAIN"):
        for seed in (2026, 2027):
            units.append(_u(stage, f"{arm.lower()}_s{seed}", f"closure_phase/{arm}_s{seed}", "",
                            EXT_DEVICE, resume="--resume", resumable=DEVICE_CKPT,
                            tree=TREE_EXTENSIONS,
                            note="closure-phase fine-tune of the tau-fixed winners"))
    return units


# --------------------------------------------------------------------------- 05_privacy_and_security
def _privacy_units() -> list[Unit]:
    stage = "05_privacy_and_security"
    units = [
        _u(stage, f"dp8_cold_s{seed}", f"main_study/runs_x_rees46/DP/COLD/DP8_s{seed}",
           f"runs_x_rees46/DP/COLD/DP8_s{seed}", _fill(LANE, lane="DP"),
           resume="--resume", resumable=FL_CKPT, tree=TREE_EXTENSIONS,
           note="X15 DP8 redo on the cold tau-fixed winner")
        for seed in (2026, 2027)
    ]
    units.append(_u(stage, "dp_s_cal", "main_study/runs_x_rees46/DP/COLD/S_CAL",
                    "runs_x_rees46/DP/COLD/S_CAL", _fill(LANE, lane="DP"),
                    resume="--resume", resumable=FL_CKPT, tree=TREE_EXTENSIONS,
                    note="clip-norm calibration of the X15 DP chain"))
    return units


# ---------------------------------------------------------------- 06_evaluation_and_benchmarks
# The benchmark campaigns are sealed studies: every dataset's run set is frozen by a LOCK.json
# and scored once (B13-B15), and the v4bench runner is plan-driven with no resume flag, so this
# stage registers no re-runnable training units. Rebuild its tables with --from-results.
STAGE_NOTES = {
    "06_evaluation_and_benchmarks": "no training units: the benchmark run sets are sealed by "
                                    "per-dataset LOCK.json records (B13-B15); use --from-results",
}


def _build_registry() -> dict[str, tuple[Unit, ...]]:
    stages = {
        "01_data": _fs_units(),
        "02_centralized": _central_units(),
        "03_federated_learning": _fl_stage_units(),
        "04_on_device": _device_units(),
        "05_privacy_and_security": _privacy_units(),
        "06_evaluation_and_benchmarks": (),
    }
    registry: dict[str, tuple[Unit, ...]] = {}
    ids: list[str] = []
    for stage in STAGE_ORDER:
        units = tuple(stages[stage])
        ids += [u.id for u in units]
        registry[stage] = units
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate unit ids in the registry")
    return registry


REGISTRY: dict[str, tuple[Unit, ...]] = _build_registry()
