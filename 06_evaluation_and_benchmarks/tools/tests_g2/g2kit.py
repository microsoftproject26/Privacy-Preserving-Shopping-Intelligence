"""G2-01 synthetic contest kit: a real TRAIN-05 registry / three-part G1B (p2kit + p2_driver.register_02b, as in
code/tests_train/test_train05.py), unseal-valid public run files (test_train05._fake_runs) moved into the
p3_driver layout <runs_root>/<FAMILY>/<recipe_id>_s<seed>, and sealed contents written the way the trainer writes them
(sealed/RESULT_ROWS.json, sealed/RUN_SUMMARY.json, sealed/evals/eval_<mark>.ranks_int32.npy) with row values computed
from the rank files by v4eval.metrics over a synthetic EvalManifest."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import p2kit
from test_train05 import _fake_runs
from v4eval.manifest import EvalManifest
from v4eval.metrics import credits_from_ranks, population_block
from v4train import p2_driver as D, p3_driver as P3, tuned_recipes as TR

from conftest import CODE, WORK

PRIMARY = "MAIN_VALIDATION_5_100_131072"
N_USERS, PER, K, N_CALIB = 400, 3, 100, 100
PB_MARK = 5.5
METRICS = (("mrr", 20), ("ndcg", 10), ("hr", 10))


# ------------------------------------------------------------------------------------------------ registrations
def _one_size_runner(keep: str):
    def run(cmd, **kw):
        result = p2kit.runner(cmd, **kw)
        hp_id = cmd[cmd.index("--hp-id") + 1]
        if keep not in hp_id:                                    # every other size level is invalid (< 0.5 x best)
            rd = Path(cmd[cmd.index("--run-dir") + 1])
            s = json.loads((rd / "RUN_SUMMARY.json").read_text())
            s["endpoint"]["value"] = 0.01
            s["values_at_marks"]["6.0"] = 0.01
            (rd / "RUN_SUMMARY.json").write_text(json.dumps(s))
            rows = json.loads((rd / "RESULT_ROWS.json").read_text())
            rows["rows"][0]["value"] = 0.01
            (rd / "RESULT_ROWS.json").write_text(json.dumps(rows))
        return result
    return run


def _stage(p2: Path, fam: str, stage: int, run) -> int:
    a = argparse.Namespace(family=fam, stage=stage, gpu=0, out_dir=p2, history_policy="USER_HISTORY_L50",
                           hp_registry=None, theta0_variants=None, dry_run=False)
    return D.run_stage(a, runner=run, say=lambda *_: None)


def register(base: Path, *, sasrec_no_c2: bool = False) -> tuple:
    """(tuned_path, gate_path): parts 1 (GRU C1), 1R (GRU C2), 2 (SASREC C1 [+ C2 | no_c2])."""
    p2 = WORK / ("p2_noc2" if sasrec_no_c2 else "p2_full")         # under V4: 02b stores V4-relative evidence
    if sasrec_no_c2:
        p2kit.make_p2(p2, families=("GRU",))
        assert _stage(p2, "SASREC", 1, p2kit.runner) == 0
        assert _stage(p2, "SASREC", 2, _one_size_runner("D384_B2")) == 3        # FABLE_TRIGGER: one valid size
    else:
        p2kit.make_p2(p2)
    base.mkdir(parents=True, exist_ok=True)
    out = base / "TUNED_RECIPES_02B.json"
    w1 = D.register_02b(p2, out, families=("GRU",))
    w2 = D.register_02b(p2, out, families=("GRU",), c2_only=True)
    w3 = D.register_02b(p2, out, families=("SASREC",), with_c2=True)
    gate = base / "G1B_APPROVED.json"
    parts = [{"part": part, "family": fam, "self_sha256": w["self_sha256"]}
             for part, fam, w in ((1, "GRU", w1), ("1R", "GRU", w2), (2, "SASREC", w3))]
    gate.write_text(json.dumps(TR.g1b_obj(parts)), encoding="utf-8")
    return out, gate


# ------------------------------------------------------------------------------------------------ manifest
CALIB = "CALIB_VALIDATION_5_100_32768"
CALIB_MAIN_USERS = np.arange(150, 150 + N_CALIB, dtype=np.int64)   # the CALIB users, in MAIN's key space
REGSHA = "e" * 64
ENTRY = {"n_decisions": N_USERS * PER, "content_sha256": "c" * 64, "header_sha256": "d" * 64}
SYN_COUNTS = {"main_users": N_USERS, "calib_users": N_CALIB, "disjoint_users": N_USERS - N_CALIB}


def calib_manifest(m, users=CALIB_MAIN_USERS, key_of=None):
    """CALIB_VALIDATION as its own manifest: the MAIN rows of `users`, with its OWN dense eval_user_key (reversed
    order, so a key-space isin would pick MAIN users 0..99 instead)."""
    rows = np.isin(m.eval_user_key, users)
    key_of = key_of or (lambda k: (users.max() - k))
    return EvalManifest(header={"manifest_id": CALIB}, decision_id=m.decision_id[rows].copy(),
                        eval_user_key=key_of(m.eval_user_key[rows]).astype(np.int64), eval_mask=m.eval_mask[rows],
                        target_class=m.target_class[rows], target_oov=m.target_oov[rows],
                        censor_class=m.censor_class[rows], catalogue_K=m.catalogue_K)


def make_manifest(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = N_USERS * PER
    keys = np.repeat(np.arange(N_USERS, dtype=np.int64), PER)       # dense per-manifest key 0..N_USERS-1
    eval_mask = rng.random(n) >= 0.05
    target = rng.integers(0, K, n).astype(np.int64)
    oov = eval_mask & (rng.random(n) < 0.05)
    target[oov | ~eval_mask] = -1
    censor = np.array(["NONE" if m else "CEN_NO_NEXT" for m in eval_mask], dtype=object)
    m = EvalManifest(header={"manifest_id": PRIMARY}, decision_id=np.arange(1, n + 1, dtype=np.int64) * 7,
                     eval_user_key=keys, eval_mask=eval_mask, target_class=target, target_oov=oov,
                     censor_class=censor, catalogue_K=K)
    base = rng.integers(1, 30, n).astype(np.int64)
    return m, calib_manifest(m), base


def theta0(rid: str, seed: int) -> str:
    import hashlib
    return hashlib.sha256(f"theta0|{rid}|{seed}".encode()).hexdigest()


def public_start(rd: Path, fam: str, rid: str, seed: int) -> None:
    """The public RUN_START fields the trainer records (key_base, theta0, torch/device, registry bindings)."""
    p = rd / "RUN_START.json"
    s = json.loads(p.read_text())
    h = theta0(rid, seed)
    s.update(theta0_init_sha256=h, init_sha256=h, torch="2.5.1+cu121", device="cuda:0",
             key_base={"eval_manifest_id": PRIMARY, "manifest_registry_sha256": REGSHA,
                       "initialization_block": f"I_META_RANDOM|{fam}|seed={seed}|theta0_sha256={h}"},
             manifest_registry={"path": "REGISTRY_V2.json", "file_sha256": REGSHA,
                                "bindings": {"eval": dict(ENTRY, manifest_id=PRIMARY, bound_by=["content"])}})
    p.write_text(json.dumps(s))


def _idx(frac: float, seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n) < frac


def _improve(r: np.ndarray, where: np.ndarray) -> np.ndarray:
    r = r.copy()
    r[where] = np.maximum(1, r[where] - 1)
    return r


def profile(gru: str = "c2_better", families_equal: bool = False):
    """Rank profiles. GRU C1 vs C2: c2_better | c1_better | tie_c1_slightly_better | flip (C1 wins on the CALIB users
    only, C2 on the disjoint users). SASREC: C2 better. GRU = 2 x SASREC ranks (SASREC decisive) unless
    families_equal (then GRU ranks = SASREC ranks: T1a -> TIEBREAK_REQUIRED)."""
    def f(fam, rid, seed, rule, base, calib_rows, rankable):
        n = base.size
        r = base.copy() if (fam == "SASREC" or families_equal) else base * 2
        c2 = rid.endswith("_C2")
        mode = gru if (fam == "GRU" and not families_equal) else "c2_better"
        if mode == "c2_better" and c2:
            r = _improve(r, _idx(0.10, 11, n))
        elif mode == "c1_better" and not c2:
            r = _improve(r, _idx(0.10, 11, n))
        elif mode == "tie_c1_slightly_better" and not c2:
            one = np.zeros(n, dtype=bool)
            one[np.flatnonzero(rankable & (r >= 3) & (r <= 20))[0]] = True   # one row, one rank step: << 0.0005
            r = _improve(r, one)
        elif mode == "flip":
            r = _improve(r, _idx(0.10, 11, n) & ~calib_rows) if c2 else _improve(r, calib_rows)
        if seed == 2027:
            r = np.where(_idx(0.05, 13, n), r + 1, r)
        if rule == "PRACTICAL_BEST":
            r = _improve(r, _idx(0.03, 17, n))
        r = r.astype(np.int64)
        r[~rankable] = -1
        return r
    return f


# ------------------------------------------------------------------------------------------------ contest
@dataclass
class Contest:
    runs_root: Path
    out: Path
    tuned: Path
    gate: Path
    reg: dict
    runs: dict
    manifest: object
    calib: object
    values: list = field(default_factory=list)
    registry: dict = field(default_factory=lambda: {
        "manifest_registry_path": "synthetic/REGISTRY_V2.json", "manifest_registry_file_sha256": REGSHA,
        "entries": {PRIMARY: dict(ENTRY), CALIB: {"n_decisions": 0}},
        "bindings": {PRIMARY: {"content_sha256": "c" * 64, "header_sha256": "d" * 64}}})

    def loader(self, lib):
        return self.manifest, self.calib, self.registry

    def argv(self, *extra):
        return ["--code-root", str(CODE), "--runs-root", str(self.runs_root), "--out", str(self.out), *extra]


def _rows(fam, rid, seed, rule, efe, ranks, m, qhash):
    blk = population_block(credits_from_ranks(ranks), m.eval_mask, m.eval_user_key)
    out = []
    for metric, k in METRICS:
        for agg in ("macro", "micro"):
            out.append({"family": fam, "recipe_id": rid, "seed": seed, "method": "C", "split": "VALIDATION",
                        "run_label": "PRODUCTION", "selection_rule": rule, "metric": metric, "k": k,
                        "aggregation": agg, "target_denominator": "E2E", "efe": efe, "value_status": "OK",
                        "value": blk[agg][f"{metric}@{k}"], "eval_manifest_id": PRIMARY, "query_order_hash": qhash,
                        "checkpoint_id": f"{rid}_MAIN_s{seed}:{'best@5.5' if rule == 'PRACTICAL_BEST' else 'endpoint_6.0'}"})
    return out


def build_contest(tmp: Path, registration: tuple, prof=None) -> Contest:
    tuned, gate = registration
    reg = TR.load_tuned_recipes(tuned)
    pins = [p["self_sha256"] for p in TR.load_g1b(gate)["tuned_recipes_parts"]]
    staged = _fake_runs(tmp / "staging", reg, pins)
    runs_root = tmp / "runs_p3"
    m, calib, base = make_manifest()
    calib_rows = np.isin(m.eval_user_key, CALIB_MAIN_USERS)          # the true CALIB users (MAIN keys)
    qhash = m.hashes["eval_ordered_decision_id_sha256"]
    prof = prof or profile()
    runs, values = {}, []
    for (fam, rid, seed), rd in staged.items():
        dest = Path(P3.run_dir_of(runs_root, fam, rid, seed))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rd), str(dest))
        public_start(dest, fam, rid, seed)
        sealed = dest / "sealed"
        (sealed / "evals").mkdir(parents=True, exist_ok=True)
        pb = prof(fam, rid, seed, "PRACTICAL_BEST", base, calib_rows, m.rankable)
        end = prof(fam, rid, seed, "CONTROLLED_BUDGET", base, calib_rows, m.rankable)
        np.save(sealed / "evals" / f"eval_{PB_MARK:.1f}.ranks_int32.npy", pb.astype(np.int32))
        np.save(sealed / "evals" / "eval_6.0.ranks_int32.npy", end.astype(np.int32))
        rows = (_rows(fam, rid, seed, "PRACTICAL_BEST", PB_MARK, pb, m, qhash)
                + _rows(fam, rid, seed, "CONTROLLED_BUDGET", 6.0, end, m, qhash))
        (sealed / "RESULT_ROWS.json").write_text(json.dumps({"run_id": f"{rid}_MAIN_s{seed}", "rows": rows}))
        prim = {r["selection_rule"]: r["value"] for r in rows if r["metric"] == "mrr" and r["aggregation"] == "macro"}
        (sealed / "RUN_SUMMARY.json").write_text(json.dumps({
            "status": "DONE_MAIN", "still_improving": False, "family": fam, "seed": seed,
            "practical_best": {"mark": PB_MARK, "value": prim["PRACTICAL_BEST"], "efe": PB_MARK},
            "controlled_budget": {"mark": 6.0, "value": prim["CONTROLLED_BUDGET"], "efe": 6.0}}))
        values += [r["value"] for r in rows]
        runs[(fam, rid, seed)] = dest
    shutil.rmtree(tmp / "staging", ignore_errors=True)
    return Contest(runs_root, tmp / "g2_out", tuned, gate, reg, runs, m, calib, values)
