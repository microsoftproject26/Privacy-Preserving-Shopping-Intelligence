"""v4bench.common — paths, registered pins, refusals and small I/O helpers (no torch import here).

CPU only: the process must start with CUDA_VISIBLE_DEVICES="" (run.py sets it before any import that touches torch;
snap_v1/reuse/train.py would otherwise default it to "0").
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

DATASETS = ("s3_beauty", "s3_sports", "s3_toys", "ml_1m")                       # Wave 1 (the default)
WAVE2_DATASETS = ("amazon2023_video_games", "amazon2023_baby_products",
                  "amazon2023_beauty_and_personal_care")                              # Wave 2 (last_out x 3; B9: MBHT dropped)
ALL_DATASETS = DATASETS + WAVE2_DATASETS
WAVE2_ROOT_DEFAULT = Path(__file__).resolve().parents[2] / "bench_proc_w2"
DATA_ROOT_DEFAULT = V4 / "bench_raw" / "spotify-sequential-capacity-probes" / "data" / "processed"
RUNS_ROOT_DEFAULT = V4 / "runs_bench_x"              # POST_HOC_X: this tree writes ONLY under runs_bench_x
EXPLORATORY = "POST_HOC_X"                           # every spec of this tree carries {"exploratory": EXPLORATORY}
X_RUNS_ROOT_NAME = "runs_bench_x"


def check_x_runs_root(runs_root) -> Path:
    """POST_HOC_X: refuse any runs root outside runs_bench_x (the confirmatory runs_bench* / runs_p2* / runs_ds3 trees
    are read-only for this study; a --pre-run may still READ a PRE run there)."""
    r = Path(runs_root).resolve()
    if X_RUNS_ROOT_NAME not in r.parts:
        raise BenchRefused(f"POST_HOC_X tree: runs root {r} is not under {X_RUNS_ROOT_NAME}/ (confirmatory roots are read-only)")
    return r
CONTRACTS_DEFAULT = V4 / "contracts"

# The 15 % pretraining band (BENCHMARK_PLAN_V1 §2 / §14): a seeded permutation of the sorted external user ids.
BAND_SEED = 2026
BAND_FRAC = 0.15
BAND_TAG = 0x42414E44                # "BAND": SeedSequence entropy tag (never shared with another stream)

# Registered REES46 contracts this runner takes (file sha256; DECISION_LOG 292 / 231; FL_TUNING_SELECTION S4).
PINS = {
    "fl/FL_TUNING_SELECTION.json": "9bca186e5025f5caa9e762f3c425fb3151923b79e60f01e6ecb063aac9c715ec",
    "fl/FL_WARM_SELECTION_S7.json": "d67a0a851665495b092abbba4456f26a3423d6ef21665f5a0e2b65e2d78dc000",
    "fl/FL_WARM_SELECTION_S8.json": "8b79d3052799dfa0ba208dce5db138da88aaa7024ba6511a374c59ce8c55a98a",
    "device/SELECTION_FT_C.json": "e47965d2f4486fc2c30140a8ee14e79686a842decb091b3dff1de5646716c10c",
}

# The REES46 central recipes (contracts/hp_registry; TUNED_RECIPES_02B.json 02B_SASREC sha 9a253087…,
# SMALL_RECIPE_SASREC_D64_B2.json SMALL_SASREC_D64_B2 sha fdf47b93…). Values copied verbatim (MEASURED 2026-10-01).
CENTRAL_RECIPES = {
    "02B_SASREC": {"model_variant": "SASREC_D256_B3", "peak_lr": 0.00040005642964899735, "dropout": [0.1],
                   "schedule": "S0", "recipe_sha256": "9a2530874a3abd096eb2f9c9f4b02f012f4015ac80137bf9328b4f69c1527f6d"},
    "SMALL_SASREC_D64_B2": {"model_variant": "SASREC_D64_B2", "peak_lr": 0.0007114121116984968, "dropout": [0.1],
                            "schedule": "S0",
                            "recipe_sha256": "fdf47b931d564946b4be153395097607aa4b895272b2da1d863a2503b8f0ca39"},
}
SMALL_PEAK_LR = CENTRAL_RECIPES["SMALL_SASREC_D64_B2"]["peak_lr"]   # = FL_TUNING_SELECTION lr_peak_base
FL_VARIANT = "SASREC_D64_B2"         # the phone model (REES46 SMALL)
CENTRAL_VARIANT = "SASREC_D256_B3"   # the central ceiling (REES46 C_FULL)
MAX_LEN = 50                         # USER_HISTORY_L50 (Track A)

PRIMARY_METRIC = "ndcg@10"           # BENCHMARK_PLAN_V1 §13: primary benchmark metric (selection on inner validation)
METRICS = ("ndcg@10", "hr@10", "ndcg@20", "hr@20", "mrr@20")
GROUPS = ("ALL", "BAND", "OTHERS")
THREADS_DEFAULT = 16


class BenchRefused(RuntimeError):
    """A refusal of the benchmark runner (configuration, provenance or the TEST gate)."""


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def canon_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str,
                                     allow_nan=False).encode()).hexdigest()


def _safe(o):
    if isinstance(o, Mapping):
        return {str(k): _safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_safe(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item") and callable(o.item) and getattr(o, "ndim", 1) == 0:
        return o.item()
    return o


def write_json(path, obj) -> None:
    """Atomic strict-JSON write (tmp + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_safe(obj), indent=1, sort_keys=True, allow_nan=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_exclusive(path, obj) -> None:
    """O_EXCL strict-JSON write: an existing file is never overwritten (write-once records)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(_safe(obj), indent=1, sort_keys=True, allow_nan=False, default=str) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise BenchRefused(f"{path} exists: write-once records are never overwritten") from None
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic_exclusive(path, obj) -> None:
    """Write-once AND atomic: the full file is written + fsynced under a temp name, then hard-linked to `path` (fails when
    `path` exists); a crash never leaves a partial `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_safe(obj), indent=1, sort_keys=True, allow_nan=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise BenchRefused(f"{path} exists: write-once records are never overwritten") from None
    finally:
        tmp.unlink(missing_ok=True)


def append_jsonl(path, rec: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_safe(rec), sort_keys=True, allow_nan=False) + "\n")


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bench_code_sha256() -> str:
    """One identity for the benchmark code: sha256 over (relative path, sha256) of every v4bench/*.py file."""
    rows = [(p.relative_to(CODE).as_posix(), sha256_file(p)) for p in sorted(HERE.glob("*.py"))]
    return canon_sha256(rows)


def refuse_overwrite(run_dir) -> None:
    """M7: a finished run (RESULT.json with a DONE_* status) is never rerun or overwritten."""
    p = Path(run_dir) / "RESULT.json"
    if p.is_file():
        st = read_json(p).get("status", "")
        if str(st).startswith("DONE") or str(st).startswith("INCOMPLETE"):
            raise BenchRefused(f"{p} exists (status {st}): a finished run is never overwritten")


class WallCap:
    """B10 Q4: an optional wall-clock cap in hours; a run that hits it is INCOMPLETE and not eligible for anything."""

    def __init__(self, hours):
        self.t0 = time.perf_counter()
        self.limit = None if not hours else float(hours) * 3600.0

    def hit(self) -> bool:
        return self.limit is not None and (time.perf_counter() - self.t0) > self.limit


def run_dir_for(runs_root, dataset: str, arm: str, seed: int, smoke: bool = False) -> Path:
    base = Path(runs_root) / dataset
    return (base / "SMOKE" if smoke else base) / f"{arm}_s{int(seed)}"


def check_dataset(name: str) -> str:
    if name not in ALL_DATASETS:
        raise BenchRefused(f"dataset {name!r} is not a benchmark dataset {ALL_DATASETS}")
    return name


def default_data_root(name: str) -> Path:
    """Wave 1: the eSASRec release processed dir; Wave 2: $V/bench_proc_w2 (same <ds>/leave_one_out layout)."""
    return WAVE2_ROOT_DEFAULT if check_dataset(name) in WAVE2_DATASETS else DATA_ROOT_DEFAULT
