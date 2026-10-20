"""P2 stage-file fixtures for the tuned-recipe tests (TRAIN-02R2): real P2_<F>_STAGE1/2.json files produced by
v4train.p2_driver.run_stage with a stand-in runner that writes the RUN_SUMMARY a finished TUNING run would write (no
training). Stage-1 values rise with the trial index (0.20 + 0.01 i), so the best trial of each size level is its last;
stage 2 re-runs give the same value; SASRec CKPT_AVG values are +0.001 (carried), GRU's equal the endpoint (not
carried). The tuned recipes are then P2_GRU_D256_S1_s4 and P2_SASREC_D384_B2_S1_s2.
(tests_int keeps its own vendored copy, tests_int/p2kit_int.py; the two suites never import each other.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import v4train.p2_driver as D
from v4train import hp_registry as H

HP = H.load_hp_registry()
NP = {"GRU_D128": 20_962_210, "GRU_D256": 41_500_000, "SASREC_D256_B2": 83_092_004, "SASREC_D256_B3": 83_900_000,
      "SASREC_D384_B2": 126_000_000}
TRIALS = {t["hp_id"]: t for fam in HP["families"].values() for t in fam}
V4 = Path(D.__file__).resolve().parents[2]
P2_BASE = Path(__file__).resolve().parent / "_p2work"      # TRAIN-02R3: P2 output lives under the V4 root
SESSION_ENV = "PPSI_TESTS_TRAIN_BASETEMP"                  # set by tests_train/conftest.py (pytest_configure)


def session_key() -> str:
    """FIX-DELTA-01 F5: one P2 work dir per pytest session, keyed by its --basetemp (else by the process id), so
    concurrent suites never share (or delete) each other's P2 fixture dirs."""
    src = os.environ.get(SESSION_ENV) or f"pid{os.getpid()}"
    return hashlib.sha256(src.encode()).hexdigest()[:12]


P2_ROOT = P2_BASE / session_key()                          # this session's dir (cleanup() removes only it)


def under_v4(p) -> Path:
    """A P2 directory under the V4 root (02b stores V4-relative paths): a path outside V4 maps to P2_ROOT/<hash>."""
    r = Path(p).resolve()
    try:
        r.relative_to(V4)
        return r
    except ValueError:
        return P2_ROOT / hashlib.sha256(str(r).encode()).hexdigest()[:16]


def cleanup() -> None:
    shutil.rmtree(P2_ROOT, ignore_errors=True)            # this session's dir only
    try:
        P2_BASE.rmdir()                                    # only when no other session's dir is left
    except OSError:
        pass


def contest_root() -> Path:
    """The per-session root of test_train03's module-level contest P2 dirs (under V4, outside P2_ROOT's per-test
    cleanup)."""
    return Path(__file__).resolve().parent / "_p2contest" / session_key()


def value_of(hp_id: str) -> float:
    return round(0.20 + 0.01 * (TRIALS[hp_id]["index"] - 1), 6)


def make_runner(delta: float = 0.0):
    """A stand-in runner writing values shifted by `delta` (a forged-evidence fixture uses two runners)."""
    def run(cmd, **kw):
        return _run(cmd, delta)
    return run


def runner(cmd, **kw):
    return _run(cmd, 0.0)


def _run(cmd, delta):
    hp = cmd[cmd.index("--hp-id") + 1]
    seed = int(cmd[cmd.index("--seed") + 1])
    rd = Path(cmd[cmd.index("--run-dir") + 1])
    rd.mkdir(parents=True, exist_ok=True)
    t = TRIALS[hp]
    v = round(value_of(hp) + delta, 6)
    ck = round(v + 0.001, 6) if t["family"] == "SASREC" else v
    summ = {"status": "DONE_TUNING", "endpoint": {"value": v}, "ckpt_avg": {"value": ck, "outcome": "CKPT_AVG_EVALUATED"},
            "shared_numel": NP[t["model_variant"]], "values_at_marks": {"6.0": v}, "peak_lr": t["peak_lr"],
            "model_variant": t["model_variant"], "seed": seed}
    (rd / "RUN_SUMMARY.json").write_text(json.dumps(summ), encoding="utf-8")
    row = {"metric": "mrr", "k": 20, "aggregation": "macro", "target_denominator": "E2E", "value": v, "efe": 6.0,
           "selection_rule": "CONTROLLED_BUDGET", "checkpoint_variant": "ENDPOINT", "hp_id": hp, "seed": seed,
           "run_label": "HP_SEARCH", "method": "C"}                         # the trainer's endpoint row (TUNING)
    (rd / "RESULT_ROWS.json").write_text(json.dumps({"run_id": f"{hp}_s{seed}", "rows": [row]}), encoding="utf-8")
    return argparse.Namespace(returncode=0)


def make_p2(p2_dir: Path, families=("GRU", "SASREC"), run=runner) -> Path:
    p2_dir = under_v4(p2_dir)
    for fam in families:
        for stage in (1, 2):
            a = argparse.Namespace(family=fam, stage=stage, gpu=0, out_dir=Path(p2_dir), history_policy="USER_HISTORY_L50",
                                   hp_registry=None, theta0_variants=None, dry_run=False)
            assert D.run_stage(a, runner=run, say=lambda *_: None) == 0
    return Path(p2_dir)


def register(tmp_path: Path, families=("GRU", "SASREC")) -> tuple:
    p2 = make_p2(Path(tmp_path) / "p2", families)                     # under V4 (under_v4)
    out = Path(tmp_path) / "TUNED_RECIPES_02B.json"
    return out, D.register_02b(p2, out, families=families)


def rehash_stage(path: Path, obj: dict) -> None:
    """Rewrite a stage file with a (forged) body and a matching self-hash."""
    obj = dict(obj)
    obj["self_sha256"] = D._canon_sha(obj)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
