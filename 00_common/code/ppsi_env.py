"""ppsi_env.py (owner: Opus, lead) — import isolation for PPSI V4 (acceptance test T43; lesson of FO-02).

Import FIRST. Puts the byte snapshot `02_centralized/code/pilot_code_snapshot` (hashes in SNAPSHOT_MANIFEST.json, copied from the
feature study's reviewed snapshot) at the front of sys.path, pre-imports every reused module from it (eval_full before train, because the
reused GRU train.py inserts gru_v1/code at sys.path[0] when imported), then scrubs every foreign code tree from sys.path.
`assert_isolated()` proves all reused modules come from the snapshot and no foreign tree is importable.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import ppsi_paths

CODE = Path(__file__).resolve().parent
SNAP = ppsi_paths.code_dir("snap_v1")  # 02_centralized/code/pilot_code_snapshot here; code/snap_v1 in the flat research layout
_FOREIGN = ("gru_v1", "sasrec_v1", "feature_opt_v1", "sasrec_vs_systems_v1")
# The pilot workspaces' code trees under the configured PILOT_ROOT. The byte-locked snapshot still inserts the
# legacy literal paths (snap_v1/reuse/train.py), so those spellings are scrubbed as well.
FOREIGN_CODE_DIRS = tuple(dict.fromkeys(
    Path(root) / w / "code" for root in (ppsi_paths.PILOT_ROOT, ppsi_paths.LEGACY["PILOT_ROOT"]) for w in _FOREIGN))
REUSE_PREIMPORT = ("rees46_eda.config", "rees46_eda.loaders", "index_utils", "window_cache", "full_store",
                   "evaluator", "model", "objective", "eval_full", "train")
ISOLATED_MODULES = ("paths", "sasrec_model", "sasrec_data", "sasrec_objective", "sasrec_schedule", "precision",
                    "train_sasrec", "synthetic_store", "ckpt_io") + REUSE_PREIMPORT


def _is_foreign(entry: str) -> bool:
    e = entry.replace("\\", "/").rstrip("/")
    return any(e == d.as_posix() or e.startswith(d.as_posix() + "/") for d in FOREIGN_CODE_DIRS)


if str(SNAP) not in sys.path:
    sys.path.insert(0, str(SNAP))
import paths  # noqa: E402  (snapshot copy: adds snap reuse + vendor)

for _m in REUSE_PREIMPORT:
    importlib.import_module(_m)
sys.path[:] = [p for p in sys.path if not _is_foreign(p)]


def assert_isolated() -> dict:
    snap = SNAP.resolve()
    seen = {}
    for name in ISOLATED_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        f = Path(mod.__file__).resolve()
        if not f.is_relative_to(snap):
            raise RuntimeError(f"isolation breach: module {name} loaded from {f}, not from {snap}")
        seen[name] = str(f)
    bad = [p for p in sys.path if _is_foreign(p)]
    if bad:
        raise RuntimeError(f"isolation breach: foreign code directories on sys.path: {bad}")
    return seen


def verify_snapshot() -> dict:
    man = json.loads((SNAP / "SNAPSHOT_MANIFEST.json").read_text())
    bad = [rel for rel, h in man["files"].items() if hashlib.sha256((SNAP / rel).read_bytes()).hexdigest() != h]
    if bad:
        raise RuntimeError(f"snapshot drift in {bad}")
    return man
