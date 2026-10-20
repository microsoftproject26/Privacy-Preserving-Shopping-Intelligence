"""fo_env.py (owner: Opus) — import isolation and constants for SASREC_FEATURE_OPT_V1.

Import this module FIRST. It puts the byte-identical campaign snapshot (`code/snap_sasrec`, hashes
in SNAPSHOT_MANIFEST.json) at the front of sys.path and imports `paths` from it. `assert_isolated()`
proves that every campaign/reuse module in use was loaded from the snapshot and never from the live
`sasrec_v1/code` tree, which a running campaign process could depend on.

The study writes only under its own workspace: FO_ROOT on the server, or the directory given in
FO_ROOT_OVERRIDE for local tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
SNAP = CODE / "snap_sasrec"
FO_SERVER_ROOT = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/feature_opt_v1")
FO_ROOT = Path(os.environ.get("FO_ROOT_OVERRIDE", str(CODE.parent)))
LIVE_SASREC_CODE = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/code")

if str(SNAP) not in sys.path:
    sys.path.insert(0, str(SNAP))
import paths  # noqa: E402  (snapshot copy: adds snap reuse + vendor to sys.path, thread defaults)

# FO-02 (probes stage, 2026-09-26T07:35Z, server only): the reused GRU `train.py` hard-codes
# gru_v1/code and inserts it at sys.path[0] when imported (eval_full imports train), so every later
# lazy import (full_store, and train's own model/objective/evaluator/window_cache) resolved in the GRU
# tree. Pre-import every reuse module from the snapshot, dependencies first, then drop the foreign code
# directories from sys.path so later lazy imports stay in the snapshot. The snapshot is not edited.
FOREIGN_CODE_DIRS = (Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1/code"),
                     Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/code"))
# order matters: train's own imports (model, objective, evaluator, window_cache) must already be cached, and
# eval_full must be imported BEFORE train (eval_full imports train; once train has run, gru_v1/code is first
# on sys.path, so an eval_full imported afterwards would resolve in the GRU tree).
REUSE_PREIMPORT = ("rees46_eda.config", "rees46_eda.loaders", "index_utils", "window_cache", "full_store",
                   "evaluator", "model", "objective", "eval_full", "train")


def _is_foreign(entry: str) -> bool:
    e = entry.replace("\\", "/").rstrip("/")
    return any(e == d.as_posix() or e.startswith(d.as_posix() + "/") for d in FOREIGN_CODE_DIRS)


import importlib  # noqa: E402

for _m in REUSE_PREIMPORT:
    importlib.import_module(_m)
sys.path[:] = [p for p in sys.path if not _is_foreign(p)]

ISOLATED_MODULES = ("paths", "sasrec_model", "sasrec_data", "sasrec_objective", "sasrec_schedule",
                    "precision", "train_sasrec", "full_store", "eval_full", "evaluator", "train",
                    "rees46_eda.loaders", "rees46_eda.config", "synthetic_store",
                    "index_utils", "window_cache", "model", "objective", "baselines_full", "lock_and_eval",
                    "verify_frozen")

# frozen-task constants (verified against the dataset documents by tests)
K, L = paths.K, paths.L
PAD, MISSING, OOV = paths.PAD, paths.MISSING, paths.OOV
N_TRAIN, N_COHORT = paths.N_TRAIN, paths.N_COHORT
DATASET_ROOT = paths.DATASET_ROOT
NORMALIZERS_PATH = DATASET_ROOT / "mappings" / "normalizers.json"
FORBIDDEN_SPLITS = ("VALIDATION_CONFIRM", "TEST", "LABEL_GRACE")


def assert_isolated() -> dict:
    """Every loaded campaign module must come from the snapshot; returns {module: file}."""
    snap = SNAP.resolve()
    seen = {}
    for name in ISOLATED_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        f = Path(mod.__file__).resolve()
        if not f.is_relative_to(snap):                    # component-wise (Review C LOW fix)
            raise RuntimeError(f"isolation breach: module {name} loaded from {f}, not from the snapshot {snap}")
        seen[name] = str(f)
    bad = [p for p in sys.path if _is_foreign(p)]         # FO-02: a foreign code dir must never come back
    if bad:
        raise RuntimeError(f"isolation breach: foreign code directories on sys.path: {bad}")
    return seen


def verify_snapshot() -> dict:
    """Re-hash every snapshot file against SNAPSHOT_MANIFEST.json; raise on any drift."""
    man = json.loads((SNAP / "SNAPSHOT_MANIFEST.json").read_text())
    bad = [rel for rel, h in man["files"].items()
           if hashlib.sha256((SNAP / rel).read_bytes()).hexdigest() != h]
    if bad:
        raise RuntimeError(f"snapshot drift in {bad}")
    return man


def refuse_split(split: str) -> None:
    if split in FORBIDDEN_SPLITS:
        raise PermissionError(f"the feature study never reads {split} (selection uses VALIDATION_SELECT only)")


def study_path(*parts) -> Path:
    """A path inside this study's workspace; refuses anything that escapes it."""
    p = (FO_ROOT.joinpath(*parts)).resolve()
    if not p.is_relative_to(FO_ROOT.resolve()):          # component-wise, not a string prefix
        raise PermissionError(f"{p} is outside the study workspace {FO_ROOT}")
    return p


def write_json_once(path, obj) -> None:
    """Serialise FIRST (strict JSON, NaN/Inf rejected), then create the file exclusively: a rejected
    object never leaves a stray file, and an existing file is never overwritten."""
    text = json.dumps(obj, indent=1, allow_nan=False)
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
