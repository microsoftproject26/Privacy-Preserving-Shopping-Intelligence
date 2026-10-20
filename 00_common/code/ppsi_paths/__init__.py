"""ppsi_paths: one place for every filesystem root the research code uses.

The research code was written on one JupyterHub server and hard-coded its paths there
(`/home/jovyan/...`). This module replaces those strings. Each root can be changed with an
environment variable. When no variable is set, every root keeps its legacy server value, so the
code behaves exactly as before on the original server.

Roots (environment variable -> default):

    REPO_ROOT            PPSI_REPO_ROOT            this repository (computed from this file)
    SERVER_HOME          PPSI_SERVER_HOME          /home/jovyan
    SERVER_BASE          PPSI_SERVER_BASE          /home/jovyan/EID PROJECT NEW   (the owned physical copy)
    DELIVERY_REVIEW_ROOT PPSI_DELIVERY_REVIEW_ROOT <SERVER_BASE>/DELIVERY_REVIEW
    EXTENSIONS_ROOT      PPSI_EXTENSIONS_ROOT      <DELIVERY_REVIEW_ROOT>/EXTENSIONS_20261013
    USER_HOME            PPSI_USER_HOME            <SERVER_HOME>/ahmedabdelhamed01
    WORKSPACE_ROOT       PPSI_WORKSPACE_ROOT       <USER_HOME>/Privacy-Preserving-Shopping-Intelligence
    PILOT_ROOT           PPSI_PILOT_ROOT           <WORKSPACE_ROOT>/.local_ai     (pilot workspaces: gru_v1, sasrec_v1, ...)
    PROJECT_ROOT         PPSI_PROJECT_ROOT         <PILOT_ROOT>/ppsi_v4           (the ppsi_v4 workspace)
    LEGACY_CODE_ROOT     PPSI_LEGACY_CODE_ROOT     <PROJECT_ROOT>/code            (the old flat code tree)
    RUNS_ROOT            PPSI_RUNS_ROOT            <PROJECT_ROOT>
    SHARED_DATA_ROOT     PPSI_SHARED_DATA_ROOT     <SERVER_HOME>/shared_data
    DATA_ROOT            PPSI_DATA_ROOT            <SHARED_DATA_ROOT>/rees46_nextitem_v1  (frozen dataset, read-only)
    VENVS_ROOT           PPSI_VENVS_ROOT           <SERVER_HOME>/venvs
    RESULTS_ROOT         PPSI_RESULTS_ROOT         <REPO_ROOT>   (stage results live in RESULTS_ROOT/<stage>/results)

A default in angle brackets follows the root it names, so setting PPSI_SERVER_HOME alone moves every
server root. `legacy(path_str)` maps any legacy absolute path string to the configured roots, and
`locate(rel)` finds where a path of the old flat ppsi_v4 workspace (for example `code/tests_fl` or
`tools/p2_lock.py`) now lives in this repository. `prototype_path(rel)` does the same for a path of the
prototype repository layout (`ppsi/training/core.py`, `scripts/mvp/run_mvp.py`, `docs/evidence/...`).

The stage code and research test folders carry descriptive names (`central_trainer`, `tests_fl` ->
`tests_federated_core`), while the import names they are imported by stay the legacy ones
(`from v4train import ...` keeps working) because the research code is hash-pinned. `FOLDER_NAMES` is
the one place that maps the old names to the new folders; `locate()` uses it to resolve old names,
and `legacy_rel()` uses its inverse so receipts keep recording the names they always used.

This module uses only the standard library and does no filesystem work at import time.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

__all__ = [
    "REPO_ROOT", "STAGES", "SERVER_HOME", "SERVER_BASE", "DELIVERY_REVIEW_ROOT", "EXTENSIONS_ROOT",
    "USER_HOME", "WORKSPACE_ROOT", "PILOT_ROOT", "PROJECT_ROOT", "LEGACY_CODE_ROOT", "RUNS_ROOT",
    "SHARED_DATA_ROOT", "DATA_ROOT", "VENVS_ROOT", "RESULTS_ROOT", "LEGACY_ALIASES", "LEGACY",
    "legacy", "legacy_path", "physical", "locate", "code_dir", "legacy_rel", "results_dir", "describe",
    "PROTOTYPE_LAYOUT", "prototype_rel", "prototype_path", "prototype_name", "prototype_view",
    "prototype_root", "FOLDER_NAMES",
]

# ----------------------------------------------------------------------------------------- legacy values
# The literal server paths the code used before this module existed. They are the defaults below and
# the prefixes `legacy()` recognises. They are records of the old layout: never change them.
_H = "/home/jovyan"
LEGACY = {
    "SERVER_HOME": _H,
    "SERVER_BASE": f"{_H}/EID PROJECT NEW",
    "DELIVERY_REVIEW_ROOT": f"{_H}/EID PROJECT NEW/DELIVERY_REVIEW",
    "EXTENSIONS_ROOT": f"{_H}/EID PROJECT NEW/DELIVERY_REVIEW/EXTENSIONS_20261013",
    "USER_HOME": f"{_H}/ahmedabdelhamed01",
    "WORKSPACE_ROOT": f"{_H}/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence",
    "PILOT_ROOT": f"{_H}/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai",
    "PROJECT_ROOT": f"{_H}/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4",
    "LEGACY_CODE_ROOT": f"{_H}/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/code",
    "SHARED_DATA_ROOT": f"{_H}/shared_data",
    "DATA_ROOT": f"{_H}/shared_data/rees46_nextitem_v1",
    "VENVS_ROOT": f"{_H}/venvs",
}

STAGES = ("00_common", "01_data", "02_centralized", "03_federated_learning", "04_on_device",
          "05_privacy_and_security", "06_evaluation_and_benchmarks", "07_study_protocol")

# ----------------------------------------------------------------------------------- descriptive folder names
# The stage folders renamed the code packages and research test suites of the old flat `code/` tree
# to descriptive names; the KEY of each entry is the name the folder had in that tree and the name
# the package is still IMPORTED by (`from v4train import ...`), which cannot change because the
# research code is hash-pinned. `locate()` / `code_dir()` resolve a key to the new folder;
# `legacy_rel()` maps a new folder back to the key so receipts keep recording the old names.
FOLDER_NAMES = {
    # code packages (import name -> folder name)
    "v4int": "study_launchers",
    "v4sup": "run_supervisor",
    "experiment_extensions": "closure_experiments",
    "census_v2": "data_census",
    "featstat_v1": "feature_statistics",
    "g0_export": "export_feasibility_check",
    "v4data": "dataset_builder",
    "v4data06": "dataset_manifests_v2",
    "v4data08": "training_manifest_full",
    "v4data09": "test_manifests",
    "v4data10": "pretraining_manifest",
    "v4fs_data": "feature_search_samples",
    "v4hist": "history_policy",
    "v4trunc": "truncated_history_features",
    "snap_v1": "pilot_code_snapshot",
    "v4base": "baselines",
    "v4fs": "feature_search",
    "v4models": "model_variants",
    "v4train": "central_trainer",
    "fl": "federated_core",
    "x_rees46": "tau_fixed_reruns",
    "v4dev": "onnx_device_protocol",
    "e3_audit": "update_leakage_audit",
    "v4bench": "benchmark_runner",
    "v4bench_w2": "benchmark_converters",
    "v4eval": "evaluator",
    # research test suites
    "tests_int": "tests_study_launchers",
    "tests_sup": "tests_run_supervisor",
    "tests_data": "tests_dataset_builder",
    "tests_data06": "tests_dataset_manifests_v2",
    "tests_data08": "tests_training_manifest_full",
    "tests_data09": "tests_test_manifests",
    "tests_data10": "tests_pretraining_manifest",
    "tests_fs_data": "tests_feature_search_samples",
    "tests_hist": "tests_history_policy",
    "tests_trunc": "tests_truncated_history_features",
    "tests_base": "tests_baselines",
    "tests_fs": "tests_feature_search",
    "tests_models": "tests_model_variants",
    "tests_train": "tests_central_trainer",
    "tests_train_gpu": "tests_central_trainer_gpu",
    "tests_fl": "tests_federated_core",
    "tests_fl_independent": "tests_federated_core_independent",
    "tests_dev": "tests_onnx_device_protocol",
    "tests_ana": "tests_analysis",
    "tests_bench": "tests_benchmark_runner",
    "tests_bench_w2": "tests_benchmark_converters",
    "tests_eval": "tests_evaluator",
}

# folder name in the stage layout -> old (recorded / import) name; the inverse of FOLDER_NAMES
_OLD_NAMES = {new: old for old, new in FOLDER_NAMES.items()}


def _env(name: str, default) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else Path(default)


# ----------------------------------------------------------------------------------------- roots
REPO_ROOT = _env("PPSI_REPO_ROOT", Path(__file__).resolve().parents[3])

SERVER_HOME = _env("PPSI_SERVER_HOME", LEGACY["SERVER_HOME"])
SERVER_BASE = _env("PPSI_SERVER_BASE", SERVER_HOME / "EID PROJECT NEW")
DELIVERY_REVIEW_ROOT = _env("PPSI_DELIVERY_REVIEW_ROOT", SERVER_BASE / "DELIVERY_REVIEW")
EXTENSIONS_ROOT = _env("PPSI_EXTENSIONS_ROOT", DELIVERY_REVIEW_ROOT / "EXTENSIONS_20261013")

USER_HOME = _env("PPSI_USER_HOME", SERVER_HOME / "ahmedabdelhamed01")
WORKSPACE_ROOT = _env("PPSI_WORKSPACE_ROOT", USER_HOME / "Privacy-Preserving-Shopping-Intelligence")
PILOT_ROOT = _env("PPSI_PILOT_ROOT", WORKSPACE_ROOT / ".local_ai")
PROJECT_ROOT = _env("PPSI_PROJECT_ROOT", PILOT_ROOT / "ppsi_v4")
LEGACY_CODE_ROOT = _env("PPSI_LEGACY_CODE_ROOT", PROJECT_ROOT / "code")
RUNS_ROOT = _env("PPSI_RUNS_ROOT", PROJECT_ROOT)

SHARED_DATA_ROOT = _env("PPSI_SHARED_DATA_ROOT", SERVER_HOME / "shared_data")
DATA_ROOT = _env("PPSI_DATA_ROOT", SHARED_DATA_ROOT / "rees46_nextitem_v1")
VENVS_ROOT = _env("PPSI_VENVS_ROOT", SERVER_HOME / "venvs")

RESULTS_ROOT = _env("PPSI_RESULTS_ROOT", REPO_ROOT)

# Read-only virtual spellings of SERVER_BASE on the server (the code maps them to SERVER_BASE).
LEGACY_ALIASES = (SERVER_HOME.as_posix() + "/ahmedabdelhamed01", SERVER_HOME.as_posix() + "/ahmed")

# legacy prefix -> configured root, longest prefix first (a prefix matches whole path components only)
_PREFIXES = (
    (LEGACY["EXTENSIONS_ROOT"], EXTENSIONS_ROOT),
    (LEGACY["DELIVERY_REVIEW_ROOT"], DELIVERY_REVIEW_ROOT),
    (LEGACY["SERVER_BASE"], SERVER_BASE),
    (LEGACY["LEGACY_CODE_ROOT"], LEGACY_CODE_ROOT),
    (LEGACY["PROJECT_ROOT"], PROJECT_ROOT),
    (LEGACY["PILOT_ROOT"], PILOT_ROOT),
    (LEGACY["WORKSPACE_ROOT"], WORKSPACE_ROOT),
    (LEGACY["USER_HOME"], USER_HOME),
    (LEGACY["DATA_ROOT"], DATA_ROOT),
    (LEGACY["SHARED_DATA_ROOT"], SHARED_DATA_ROOT),
    (LEGACY["VENVS_ROOT"], VENVS_ROOT),
    (LEGACY["SERVER_HOME"], SERVER_HOME),
)


def legacy(path_str) -> str:
    """Map a legacy absolute server path string to the configured roots.

    With no environment variable set the result equals the input, so the original server sees the
    same strings as before. A string under no legacy prefix is returned unchanged. Trailing slashes
    are kept.
    """
    s = os.fspath(path_str)
    probe = s.replace("\\", "/")
    for prefix, root in _PREFIXES:
        if probe == prefix or probe.startswith(prefix + "/"):
            return root.as_posix() + probe[len(prefix):]
    return s


def legacy_path(path_str) -> Path:
    """`legacy()` as a Path."""
    return Path(legacy(path_str))


def physical(path) -> Path:
    """Map the read-only virtual spellings (LEGACY_ALIASES) to the owned physical copy SERVER_BASE."""
    value = os.fspath(path)
    probe = value.replace("\\", "/")
    for alias in LEGACY_ALIASES:
        if probe == alias or probe.startswith(alias + "/"):
            return Path(SERVER_BASE.as_posix() + probe[len(alias):])
    return Path(value)


def locate(rel) -> Path:
    """Where a path of the old flat ppsi_v4 workspace now lives.

    `rel` is relative to the old workspace root, e.g. `code/tests_fl`, `code/v4int/reeval.py` or
    `tools/p2_noop_check.py`. Code moved into the stage folders, so `code/<x>` is searched in
    `<stage>/code`, `<stage>/tests/research` and `<stage>/tools`, and `tools/<x>` in `<stage>/tools`
    and the root `tools/`. The first component is also tried under its descriptive folder name
    (FOLDER_NAMES), so the old names resolve to the renamed folders. Anything not found in the
    repository (contracts, company records, data, runs) still lives in the server workspace and
    resolves to PROJECT_ROOT / rel.
    """
    parts = PurePosixPath(str(rel).replace("\\", "/")).parts
    candidates = []
    if parts and parts[0] == "code" and len(parts) > 1:
        # the renamed folders first, then the literal name (ppsi, tools that never moved)
        first_names = [FOLDER_NAMES[parts[1]]] if parts[1] in FOLDER_NAMES else []
        first_names.append(parts[1])
        for first in first_names:
            candidates += _stage_candidates(first, parts[2:])
    elif parts and parts[0] == "tools" and len(parts) > 1:
        rest = Path(*parts[1:])
        candidates += [REPO_ROOT / stage / "tools" / rest for stage in STAGES] + [REPO_ROOT / "tools" / rest]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT.joinpath(*parts)


def _stage_candidates(first: str, rest_parts: tuple[str, ...]) -> list[Path]:
    """Every place the stage layout may hold `first/<rest...>`: code packages, research test
    suites and per-stage tools, across all stages."""
    rest = Path(first, *rest_parts)
    return [REPO_ROOT / stage / root / rest for stage in STAGES for root in ("code", "tests/research", "tools")]


def code_dir(name: str) -> Path:
    """New location of the old `ppsi_v4/code/<name>` folder or file (a package or a tests_* folder)."""
    return locate(f"code/{name}")


def legacy_rel(path) -> str:
    """The old `ppsi_v4/code`-relative name of a file in this repository, e.g. `v4data/common.py` or
    `tests_int/intkit.py` (the inverse of `code_dir`). A renamed folder is mapped back to its old
    name, so receipts keep recording the names they always used."""
    p = Path(path).resolve()
    for stage in STAGES:
        for root in (REPO_ROOT / stage / "code", REPO_ROOT / stage / "tests" / "research"):
            if p == root or root in p.parents:
                parts = p.relative_to(root).parts
                if parts and parts[0] in _OLD_NAMES:
                    parts = (_OLD_NAMES[parts[0]],) + parts[1:]
                return PurePosixPath(*parts).as_posix()
    raise ValueError(f"{path} is not under a stage code/ or tests/research/ folder")


# ----------------------------------------------------------------------------------------- prototype layout
# The public prototype repository kept ppsi/, scripts/, tests/, docs/ and results/ at its root. Its records,
# manifests and evidence files name files by those repository-relative paths (`ppsi/training/core.py`,
# `scripts/federated/fl_synthetic_smoke.py`, `docs/evidence/s2-se-05/...`), and the names stay as recorded.
# Old prefix -> new repository-relative location. config/, fixtures/, artifacts/, notebooks/, uv.lock and
# pyproject.toml did not move.
_S = dict(zip(("c0", "d1", "c2", "f3", "o4", "p5", "e6"), STAGES))
PROTOTYPE_LAYOUT = {
    "ppsi": f"{_S['c0']}/code/ppsi",
    **{f"ppsi/{pkg}": f"{stage}/code/ppsi/{pkg}" for stage, pkgs in (
        (_S["d1"], ("data", "features")),
        (_S["c2"], ("central", "models", "seqrec", "training", "baselines")),
        (_S["f3"], ("federated", "fedsim")),
        (_S["o4"], ("ondevice", "deployment")),
        (_S["p5"], ("privacy",)),
        (_S["e6"], ("evaluation", "benchmarks", "benchrun"))) for pkg in pkgs},
    **{f"scripts/{item}": f"{stage}/scripts/prototype/{item}" for stage, items in (
        (_S["d1"], ("data",)),
        (_S["c2"], ("model", "baselines", "t2", "t3", "t5", "t8", "training_smoke.py")),
        (_S["f3"], ("federated", "mvp")),
        (_S["o4"], ("deployment",)),
        (_S["e6"], ("evaluation", "experiments"))) for item in items},
    **{f"scripts/{tool}": f"tools/{tool}" for tool in (
        "check.py", "env_smoke.py", "studio.py", "checkpoint_manifest.py", "generate_training_artifact_manifest.py")},
    **{f"tests/{item}": f"{stage}/tests/prototype/{item}" for stage, items in (
        (_S["c0"], ("test_measurements.py", "test_temporal_leakage.py")),
        (_S["d1"], ("features",)),
        (_S["c2"], ("baselines", "central", "models", "seqrec", "training")),
        (_S["f3"], ("federated", "fedsim", "mvp")),
        (_S["o4"], ("deployment", "ondevice")),
        (_S["p5"], ("privacy",)),
        (_S["e6"], ("benchmarks", "benchrun", "evaluation", "experiments"))) for item in items},
    "docs": "docs/prototype",
    "results/benchmarks": f"{_S['e6']}/results/benchmarks",
    **{f"results/rees46/{name}": f"{_S['e6']}/results/rees46_confirmatory/{name}" for name in (
        "confirmatory_phase1_arms.csv", "confirmatory_phase1_contrasts.csv", "confirmatory_phase2_contrasts.csv",
        "README.md")},
    "results/rees46/exploratory_test_contrasts.csv": f"{_S['f3']}/results/exploratory_test_contrasts.csv",
    "results/rees46/exploratory_warm_tau_grid_calibration.csv":
        f"{_S['f3']}/results/exploratory_warm_tau_grid_calibration.csv",
    "results/rees46/device_latency.csv": f"{_S['o4']}/results/device_latency.csv",
    **{f"results/rees46/{name}": f"{_S['p5']}/results/{name}" for name in (
        "privacy_dp_cost.csv", "privacy_membership_inference.csv", "secure_aggregation.csv")},
}
del _S


def prototype_rel(rel) -> str:
    """The current repository-relative location of a path of the prototype layout.

    `prototype_rel("ppsi/training/core.py")` is `02_centralized/code/ppsi/training/core.py`. The longest
    matching prefix wins and prefixes match whole path components; a path under no moved prefix (config/,
    fixtures/, uv.lock, ...) is returned unchanged.
    """
    probe = PurePosixPath(str(rel).replace("\\", "/")).as_posix()
    best = ""
    for old in PROTOTYPE_LAYOUT:
        if (probe == old or probe.startswith(old + "/")) and len(old) > len(best):
            best = old
    return PROTOTYPE_LAYOUT[best] + probe[len(best):] if best else probe


def prototype_path(rel, root=None) -> Path:
    """Where a path of the prototype layout lives now (see `prototype_rel`).

    Results resolve under RESULTS_ROOT, everything else under REPO_ROOT. A `root` other than REPO_ROOT
    (a copy laid out the old way, such as a test's tmp_path) is joined with `rel` unchanged.
    """
    if root is not None and Path(root).resolve() != REPO_ROOT.resolve():
        return Path(root) / rel
    new = prototype_rel(rel)
    base = RESULTS_ROOT if PurePosixPath(new).parts[1:2] == ("results",) else REPO_ROOT
    return base / new


def prototype_name(path, root=None) -> Path:
    """The prototype-layout name of a file in this repository: the inverse of `prototype_path`.

    `prototype_name(REPO_ROOT / "02_centralized/code/ppsi/training/core.py")` is `ppsi/training/core.py`,
    the name records and manifests use. A `root` other than REPO_ROOT gives `path.relative_to(root)`.
    """
    p = Path(path)
    if root is not None and Path(root).resolve() != REPO_ROOT.resolve():
        return p.relative_to(root)
    rel = None
    for base in (RESULTS_ROOT, REPO_ROOT):
        for candidate, anchor in ((p, base), (p.resolve(), base.resolve())):
            try:
                rel = candidate.relative_to(anchor).as_posix()
                break
            except ValueError:
                continue
        if rel is not None:
            break
    if rel is None:
        raise ValueError(f"{path} is not in the repository {REPO_ROOT}")
    best = ""
    for old, new in PROTOTYPE_LAYOUT.items():
        if (rel == new or rel.startswith(new + "/")) and len(new) > len(PROTOTYPE_LAYOUT.get(best, "")):
            best = old
    return Path(best + rel[len(PROTOTYPE_LAYOUT[best]):]) if best else Path(rel)


def prototype_view(dest, rels) -> Path:
    """Copy each prototype-layout path in `rels` (a file or a folder) to `dest/<rel>`, byte for byte.

    Some frozen prototype records pin the bytes of the code that verifies them, and that code finds its
    files from its own location in the old layout (`scripts/experiments/validate_experiment_contracts.py`).
    It cannot be edited without failing its own check, so it runs unchanged against a view of the current
    files laid out the old way. Caches (`__pycache__`, `.pytest_cache`) are not copied. Returns `dest`.
    """
    import shutil

    dest = Path(dest)
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
    for rel in rels:
        source, target = prototype_path(rel), dest / rel
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            raise FileNotFoundError(f"{rel} (looked for {source})")
    return dest


@contextmanager
def prototype_root(rels):
    """A temporary `prototype_view` of `rels`, removed on exit: a root for code that joins legacy names onto it.

        with prototype_root(DEFAULT_TRAINER_CORE_FILES) as root:
            manifest = build_trainer_core_manifest(root)
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ppsi_prototype_view_") as tmp:
        yield prototype_view(tmp, rels)


def results_dir(stage: str) -> Path:
    """`RESULTS_ROOT/<stage>/results` (results are merged back from the server, never written here)."""
    return RESULTS_ROOT / stage / "results"


def describe() -> dict:
    """Every configured root as a string (for logs and smoke checks)."""
    return {name: globals()[name].as_posix() for name in (
        "REPO_ROOT", "SERVER_HOME", "SERVER_BASE", "DELIVERY_REVIEW_ROOT", "EXTENSIONS_ROOT", "USER_HOME",
        "WORKSPACE_ROOT", "PILOT_ROOT", "PROJECT_ROOT", "LEGACY_CODE_ROOT", "RUNS_ROOT", "SHARED_DATA_ROOT",
        "DATA_ROOT", "VENVS_ROOT", "RESULTS_ROOT")}
