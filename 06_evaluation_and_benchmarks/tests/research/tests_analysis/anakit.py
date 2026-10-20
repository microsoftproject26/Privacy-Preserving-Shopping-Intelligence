"""ANA-01 / ANA-02 synthetic block fixtures: tiny K, 2 seeds, three roles (MAIN / UNSEEN_HOLDOUT / TRUE_COLD), every arm
the report knows, rows computed from the rank files by v4eval.metrics exactly as the frozen evaluator writes them, carrying
the full E9.1 comparison key, the pinned-device status (06D §1) and realistic checkpoint ids (A-F10).

Known answers built in: C_SMALL ranks = 2 x C_BIG ranks with every C_BIG rank <= 10, so each MRR@20 credit halves and
R(C_SMALL / C_BIG) = 50 exactly (per seed, per resample, both rules); FT_C ranks = C_BIG ranks, so FT_C - C_SMALL =
C_BIG / 2 and no user is harmed; C_SMALL_INT8 (ONNX_CPU sub-block, rows-free, -1 outside its users subset) = C_SMALL on
the first N_SUBSET MAIN users in selection_key_v2 order (R vs the A40 C_SMALL = 100, DEVICE_MIXED); the optional
C_SMALL_FP32 ONNX entry = the same ranks (K16 = 100, CONFIRMATORY). split="TEST" builds the same block on the registered
RETRO_TEST role ids with gate-bound headers (make_test_gate)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import retention_report as R
from census_v2.core import selection_key_v2
from v4base import strata as ST
from v4eval.manifest import EvalManifest
from v4eval.metrics import credits_from_ranks, population_block

K = 40
PER = 5
DEV = "NVIDIA_A40-FP32-DET"
PIN = "p" * 64
EVSHA = "e" * 64
KEY_EXTRA = {"candidate_order_hash": "c" * 64, "label_policy_hash": "l" * 64, "data_protocol_hash": "d" * 64,
             "feature_contract_hash": "f" * 64, "precision_id": "FP32", "objective_id": "FULL_SOFTMAX_CE",
             "eval_score_block": 158486, "eval_stream": "E2E_ROWS"}
ROLES = {"MAIN": ("MAIN_VALIDATION_5_100_131072", "MAIN_TRAIN_5_100_131072", 48, 1),
         "UNSEEN_HOLDOUT": ("UNEW_HOLDOUT_VALIDATION_32768", "UNEW_HOLDOUT_SUPPORT_32768", 30, 2),
         "TRUE_COLD": ("TRUE_COLD_VALIDATION", None, 16, 3)}
TEST_IDS = {r: R.ROLE_MAP["TEST"][r][0] for r in ROLES}
SUPPORT_CYCLE = (0, 5, 7, 10, 11, 15, 20, 21, 30, 50, 51, 80, 100, 3, 150)
N_SUBSET = 24
MARKS = tuple(0.5 * i for i in range(1, 13))
FA_ROUNDS = 120
FA_PER_CLIENT = 2.0e9
FA_N_CLIENTS = 1000
FA_DROPPED = {2026: 250, 2027: 100}          # dropped visits: 2026 as the aggregate count, 2027 as the log's id lists
FA_VISIT_BYTES = 165680392
FA_DOWN_BYTES = 82840192
DEVICE_CELL = "P2_F0.1"
TRAIN_OF = {"C_BIG": "MAIN_TRAIN_5_100_131072", "C_SMALL": "MAIN_TRAIN_5_100_131072", "C_FULL": "FULL_TRAIN_5_100_TIEBREAK"}
# FIX-DELTA-03 M1: PRE / C_FULL rows are v4int.reeval rows of the registered central source (GATE-B B1 row labels)
SOURCE_LABELS = {a: {"central_source": a, "source_train_manifest_id": t, "source_eval_manifest_id": e,
                     "source_manifest_role": r}
                 for a, t, e, r in (("PRE", "PRETRAIN_TRAIN_5_100_NONMAIN", "UNEW_VAL_VALIDATION_32768", "PRETRAIN"),
                                    ("C_FULL", "FULL_TRAIN_5_100_TIEBREAK", "MAIN_VALIDATION_5_100_131072", "C_FULL"))}
# FIX-DELTA-03 M2: the theta_pre identity a warm-start FL run records (launch_fl arm_spec.init_from, INIT_IDENTITY_KEYS)
INIT_FROM = {"kind": "PRETRAIN_FOR_FL", "run_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026",
             "run_label": "PRETRAIN_FOR_FL", "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN",
             "fl_init_kind": "endpoint_6.0", "checkpoint_sha256": "a" * 64, "checkpoint_bytes": 1234,
             "state_digest": "b" * 64, "model_variant": "SASREC_D64_B2", "family": "SASREC", "seed": 2026}
ORT = {"onnxruntime_version": "1.24.1", "intra_op_num_threads": 1, "graph_optimization_level": "ORT_ENABLE_ALL"}


def hexkey(role: str, k: int) -> str:
    return hashlib.sha256(f"user|{role}|{k}".encode()).hexdigest()


def subset_users_v2(role: str = "MAIN", n: int = N_SUBSET):
    """The first n users (all MAIN users have E2E rows) in selection_key_v2 order -> (sorted keys, set sha256)."""
    keyed = sorted((selection_key_v2(bytes.fromhex(hexkey(role, k))), k) for k in range(ROLES[role][2]))
    chosen = [k for _s, k in keyed[:n]]
    sha = hashlib.sha256("\n".join(sorted(hexkey(role, k) for k in chosen)).encode("ascii")).hexdigest()
    return np.asarray(sorted(chosen), dtype=np.int64), sha


SUBSET_USERS, SUBSET_SHA = subset_users_v2()


def make_manifest(role: str, split: str = "VALIDATION", gate=None) -> EvalManifest:
    mid, _tid, n_users, seed = ROLES[role]
    if split == "TEST":
        mid = TEST_IDS[role]
    rng = np.random.default_rng(100 + seed)
    n = n_users * PER
    keys = np.repeat(np.arange(n_users, dtype=np.int64), PER)
    eval_mask = rng.random(n) >= 0.1
    eval_mask[::PER] = True                                      # every user has an E2E row
    target = rng.integers(0, K, n).astype(np.int64)
    oov = eval_mask & (rng.random(n) < 0.05)
    oov[::PER] = False
    target[oov | ~eval_mask] = -1
    censor = np.array(["NONE" if m else "CEN_NO_NEXT" for m in eval_mask], dtype=object)
    header = {"manifest_id": mid, "catalogue_K": K}
    if gate is not None:
        header.update(all_methods_lock_sha256=gate["all_methods_lock_sha256"],
                      test_exposure_register_entry_id=gate["test_exposure_register_entry_id"])
    return EvalManifest(header=header, decision_id=(np.arange(1, n + 1, dtype=np.int64) * 7 + seed * 1_000_000),
                        eval_user_key=keys, eval_mask=eval_mask, target_class=target, target_oov=oov,
                        censor_class=censor, catalogue_K=K)


def _mask(frac, seed, n):
    return np.random.default_rng(seed).random(n) < frac


def arm_ranks(arm: str, seed: int, rule: str, base: np.ndarray, rankable: np.ndarray) -> np.ndarray:
    n = base.size
    c = base.copy()
    if seed == 2027:
        c = np.where(_mask(0.2, 13, n), c + 1, c)
    if rule == "CONTROLLED_BUDGET":
        c = np.where(_mask(0.1, 17, n), c + 1, c)                  # C_BIG ranks stay <= 10 (base <= 8)
    add = {"C_BIG": 0, "FP": 1, "FA": 2, "FA_1024": 3, "DP8": 5, "DP4": 4, "DP1": 8, "FT_FA": 1, "LO": 10,
           "PF": 0, "FA_Q8": 2, "C_FULL": 0, "FT_C": 0, "POP": 15, "FA_WARM": 1, "PRE": 4,
           "FA_WARM_FROZEN": 3,                                        # GATE-B B3 (S8)
           "LAST": 6, "RULE": 9, "FT_PRE": 2, "DP8_FROZEN": 6, "GRU_D128": 1,   # ANA-04 (06L-C C3, C7, C11, C17)
           "FT_FA_WARM": 0, "FT_FA_WARM_FROZEN": 2,                     # ANA-05 (06M_B B4)
           "FP_WARM": 2, "PF_WARM": 1, "FA_Q8_WARM": 2, "FA_1024_WARM": 3, "DP8_WARM": 6, "DP1_WARM": 9,   # ANA-06
           "FP_WARM_FROZEN": 3, "PF_WARM_FROZEN": 2, "FA_Q8_WARM_FROZEN": 3, "FA_1024_WARM_FROZEN": 4,
           "DP1_WARM_FROZEN": 9}
    if arm in ("C_SMALL", "C_SMALL_INT8", "C_SMALL_FP32"):
        r = 2 * c
    elif arm == "C_FULL":
        r = np.maximum(1, c - 1)
    elif arm == "PF":
        r = np.where(_mask(0.5, 19, n), c + 1, c)
    elif arm == "FA_Q8":
        r = np.where(_mask(0.3, 23, n), c + 3, c + 2)
    else:
        r = c + add[arm]
    r = r.astype(np.int64)
    r[~rankable] = -1
    return r


def ckpt_id(arm, seed, rule, best=3.0):
    if rule == "CONTROLLED_BUDGET":
        return f"{arm}_s{seed}:endpoint_6.0"
    if rule == "PRACTICAL_BEST":
        return f"{arm}_s{seed}:best@{best:.1f}"
    return None


def rows_for(arm, seed, rule, ranks, m, *, device=DEV, split="VALIDATION", evsha=EVSHA, qhash=None, efe=None,
             carry_rule=True, extra=None, cid="AUTO", role="MAIN"):
    blk = population_block(credits_from_ranks(ranks), m.eval_mask, m.eval_user_key)
    qh = qhash or m.hashes["eval_ordered_decision_id_sha256"]
    if efe is None and rule in R.RULES:
        efe = 6.0 if rule == "CONTROLLED_BUDGET" else 3.0
    if cid == "AUTO":
        cid = ckpt_id(arm, seed, rule, efe if (efe is not None and rule == "PRACTICAL_BEST") else 3.0)
    out = []
    for metric, k in (("mrr", 20), ("ndcg", 10), ("hr", 10), ("mrr", 10)):
        for agg in ("macro", "micro"):
            r = {"method": arm, "seed": seed, "split": split, "selection_rule": rule if carry_rule else None,
                 "metric": metric, "k": k, "aggregation": agg, "target_denominator": "E2E", "efe": efe,
                 "value_status": "OK", "value": blk[agg][f"{metric}@{k}"], "eval_manifest_id": m.manifest_id,
                 "query_order_hash": qh, "eval_device_class": device, "evaluator_code_sha256": evsha,
                 "checkpoint_id": cid, "status": R.PINNED_STATUS if device == DEV else "MEASURED",
                 "device_pin_sha256": PIN if device == DEV else None, "population_hash": f"pop:{m.manifest_id}",
                 "topology_hash": "t" * 64, "initialization_block": f"init:{arm}", "eval_batch": 512,
                 **KEY_EXTRA}
            if role == "MAIN" and arm in TRAIN_OF:
                r["train_manifest_id"] = TRAIN_OF[arm]
            r.update(SOURCE_LABELS.get(arm, {}))
            if seed in R.DESCRIPTIVE_SEEDS.get(arm, ()):                  # ANA-05: v4int.reeval's s2027 labels
                r.update(comparison_status="DESCRIPTIVE", source_seed_status="DESCRIPTIVE",
                         source_seed_rule="06D §8 NICE seed; 06E §2 / DL 176")
            r.update(extra or {})
            out.append(r)
    return out


def jdump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")


def canon(obj: dict) -> dict:
    obj = dict(obj)
    obj["self_sha256"] = R._canon_compact(obj)
    return obj


V3_FILE = R.V4 / "company" / "gates" / "ANA_CONTRASTS_v3.json"


def contrasts_obj(version: int = 2, **over):
    """The registered contrasts JSON of format version 1 (K1-K16, dry check only), 2 (K1-K19; company/gates/
    ANA_CONTRASTS_v2.json) or 3 (= the real company/gates/ANA_CONTRASTS_v3.json of 06L-C §C4, K1-K21 + K9b, K9c, K11b,
    K20b, K20c + S8-S7), built from the script's registered text and the registry's own hierarchy / seeds (written out
    here, not derived from the script's tables, so a drift shows in a test)."""
    if version == 3:
        obj = json.loads(V3_FILE.read_text(encoding="utf-8"))
        obj.pop("self_sha256", None)
        obj.update(over)
        return obj
    n = {1: 16, 2: 19}[version]
    ids = [f"K{i}" for i in range(1, n + 1)]
    secondary = ["K1", "K3", "K8", "K12"] + (["K17"] if n >= 19 else []) + (["K20", "K21"] if n >= 21 else [])
    obj = {"format": f"PPSI_V4_ANA_CONTRASTS_{version}",
           "confirmatory_contrasts": [{"id": k, "contrast": c, "question": q} for k, c, q in R.REGISTERED_CONTRASTS
                                      if k in ids],
           "confirmatory_rule_map": dict(R.CONFIRMATORY_RULE), "default_confirmatory_rule": "PRACTICAL_BEST",
           "hierarchy": {"primary": ["K2"], "secondary": secondary,
                         "tertiary": [k for k in ids if k not in secondary and k != "K2"]},
           "multiplicity_statement": f"{n} pre-registered contrasts; (test fixture)"}
    if version >= 2:
        obj["registered_seeds"] = {k: [2026] for k in ("K13", "K15", "K16", "K17", "K18", "K19") if k in ids}
    obj.update(over)
    return obj


@dataclass
class Block:
    root: Path
    manifests: dict
    base: dict
    contrasts: Path
    ctx: dict = field(default_factory=dict)
    ranks: dict = field(default_factory=dict)
    split: str = "VALIDATION"
    test_args: list = field(default_factory=list)

    @property
    def index_path(self) -> Path:
        return self.root / R.INDEX_NAME

    def index(self) -> dict:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def write_index(self, idx) -> None:
        self.index_path.write_text(json.dumps(idx, indent=1, sort_keys=True), encoding="utf-8")

    def loader(self, role, spec):
        if role == "CALIBRATION":                                   # 06L-C C14: absent unless add_calib() was called
            return self.ctx.get(role)
        return self.ctx[role]

    def argv(self, out: Path, *extra, split=None):
        split = split or self.split
        return ["--block", str(self.root), "--split", split, "--contrasts", str(self.contrasts), "--out", str(out),
                *(self.test_args if split == "TEST" else []), *extra]


MAIN_ARMS = {"C_BIG": (2026, 2027), "C_SMALL": (2026, 2027), "FP": (2026, 2027), "PF": (2026, 2027),
             "FA_1024": (2026, 2027), "DP8": (2026, 2027), "DP4": (2026,), "DP1": (2026,), "FA_Q8": (2026, 2027),
             "C_FULL": (2026,)}
DEVICE_ARMS = {"FT_C": (2026, 2027), "FT_FA": (2026, 2027), "LO": (2026, 2027)}
PANEL_ARMS = {"UNSEEN_HOLDOUT": ("C_BIG", "C_SMALL", "FA", "FT_C", "LO"), "TRUE_COLD": ("C_BIG", "C_SMALL", "FA", "FT_C")}


def _fl_run(root: Path, rel: str, arm: str, seed: int, m, base, rankable, entries: list, split: str) -> None:
    """An FL run dir: 12 marks (5.5 = PRACTICAL_BEST, 6.0 = endpoint), RESULT_ROWS, RUN_SUMMARY, metrics.jsonl."""
    rd = root / rel
    pb = arm_ranks(arm, seed, "PRACTICAL_BEST", base, rankable)
    cb = arm_ranks(arm, seed, "CONTROLLED_BUDGET", base, rankable)
    lines = []
    for i, mark in enumerate(MARKS, start=1):
        if mark == 6.0:
            rk = cb
        elif mark == 5.5:
            rk = pb
        else:
            rk = np.where(rankable, pb + int(round((6.0 - mark) * 2)), -1).astype(np.int64)
        tag = f"{mark:.1f}"
        (rd / "evals").mkdir(parents=True, exist_ok=True)
        np.save(rd / "evals" / f"eval_{tag}.ranks_int32.npy", rk.astype(np.int32))
        jdump(rd / "evals" / f"eval_{tag}.rows.json",
              {"mark": mark, "efe": mark, "rows": rows_for(arm, seed, None, rk, m, carry_rule=False, efe=mark, cid=None,
                                                           split=split, extra={"schedule_mark": mark})})
        lines.append(json.dumps({"kind": "eval", "mark": mark, "efe": mark, "round": 10 * i,
                                 "rows_file": f"evals/eval_{tag}.rows.json"}))
    (rd / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_id = f"FL_{arm}_SMALL_s{seed}"
    ident = ({"fl_run_id": run_id, "arm_spec_sha256": warm_spec_sha(arm, seed)}     # launch_fl fl_rows labels (ANA-06:
             if R.warm_stage_of(arm) else {})                                        # REV-06O-ANA05 MINOR 1)
    jdump(rd / "RESULT_ROWS.json", {"arm": arm, "rows": rows_for(arm, seed, "PRACTICAL_BEST", pb, m, efe=5.5,
                                                                 cid=f"{run_id}:best@5.5", split=split,
                                                                 extra={"schedule_mark": 5.5, **ident})
                                    + rows_for(arm, seed, "CONTROLLED_BUDGET", cb, m, efe=6.0,
                                               cid=f"{run_id}:endpoint_6.0", split=split,
                                               extra={"schedule_mark": 6.0, **ident})})
    dropped = FA_DROPPED[seed]
    part = {"rounds": FA_ROUNDS, "sampled": 12000, "dropped": dropped, "survived": 12000 - dropped, "dropout_p": 0.1}
    summ_extra = {"participation": part, "plan": {"n_clients": FA_N_CLIENTS}}
    if seed == 2027:                                                  # the log form: dropped ids per round
        summ_extra["participation_log"] = [{"round": r, "n_sampled": 100, "dropped": [f"c{r}_{j}" for j in range(10)],
                                            "n_survived": 90, "noop": False} for r in range(10)]
    if arm in R.WARM_LINE_ARMS and R.WARM_LINE_ARMS[arm][2] == "FA_WARM_FROZEN":   # ANA-06: FL-16 S8 bytes record
        summ_extra.update(frozen_item_tables={"keys": list(FROZEN_KEYS)})
    jdump(rd / "RUN_SUMMARY.json", {"status": "DONE_MAIN", "arm": R.WARM_LINE_ARMS.get(arm, (arm,))[0],
                                    "rounds": FA_ROUNDS, "counters": {"efe": 6.0},
                                    "best_half_efe": 11, **summ_extra,
                                    "bytes": {"per_visit": {"send_plus_receive_bytes": FA_VISIT_BYTES,
                                                            "upload_bytes": 82840200, "download_bytes": FA_DOWN_BYTES,
                                                            "shared_numel": 20710048,
                                                            **({"frozen_bytes_once": FROZEN_ONCE}
                                                               if "frozen_item_tables" in summ_extra else {})},
                                              "per_client": {"mean": FA_PER_CLIENT}, "fleet": {"total": 2.6e14},
                                              **({"per_client_incl_frozen_once": {"mean": FA_PER_CLIENT + 500.0}}
                                                 if "frozen_item_tables" in summ_extra else {})},
                                    "arm_spec": (warm_line_spec(arm) if arm in R.WARM_LINE_ARMS else
                                                 {"cfg": {"lr_peak_local": 0.000711, "dp": None},
                                                  **({"init_from": dict(INIT_FROM)} if arm in R.WARM_INIT_ARMS
                                                     else {})}),
                                    **({"run_id": run_id, "arm_spec_sha256": warm_spec_sha(arm, seed)}   # ANA-05 / -06
                                       if R.warm_stage_of(arm) else {})})
    for rule, rk_rel in (("PRACTICAL_BEST", f"{rel}/evals/eval_5.5.ranks_int32.npy"),
                         ("CONTROLLED_BUDGET", f"{rel}/evals/eval_6.0.ranks_int32.npy")):
        e = {"arm": arm, "seed": seed, "rule": rule, "role": "MAIN", "rows": f"{rel}/RESULT_ROWS.json", "ranks": rk_rel}
        if rule == "PRACTICAL_BEST":
            e["run_dir"] = rel
        if R.warm_stage_of(arm):                                           # ANA-05 / -06: the LAUNCH 2-D binding
            e["warm_selection"] = warm_binding(arm)
        entries.append(e)


FROZEN_KEYS = ("item_embed.weight", "output_embed", "output_bias")
FROZEN_ONCE = 1000


def warm_line_spec(arm: str, z=None) -> dict:
    """ANA-06: a launch_fl arm_spec of an FL-16 warm-line arm (the cold arm's knobs, theta_pre, S8 frozen keys)."""
    lab, base, variant, _eps = R.WARM_LINE_ARMS[arm]
    pois = base in R.POISSON_BASE
    cfg = {"method": R.METHOD_CODE_OF[base], "seed": 2026, "lr_peak_local": 0.000711,
           "mu": 0.01 if base == "FP" else 0.0, "pf": {"lr": 0.000711, "lam": 1e-4, "clip": 1.0} if base == "PF" else None,
           "dp": ({"clip_norm": 1.0, "noise_multiplier": (z if z is not None else R.DP_Z[arm]) if base.startswith("DP")
                   else 0.0, "denominator": 1024} if pois else None)}
    spec = {"arm": lab, "cfg": cfg, "plan": {"group_size": 1024 if pois else 64, "sampling": "poisson" if pois else "sweep"},
            "role": "MAIN", "init_from": dict(INIT_FROM),
            "upload_codec": {"name": "Q8_SR_PER_TENSOR"} if base == "FA_Q8" else None}
    if variant == "FA_WARM_FROZEN":
        spec["frozen_item_tables"] = list(FROZEN_KEYS)
    return spec


def warm_variant_of(arm: str) -> str:
    return arm if arm in R.WARM_SELECTION_STAGE else R.WARM_LINE_ARMS[arm][2]


def warm_spec_sha(arm: str, seed: int) -> str:
    """ANA-05: a synthetic RUN_SUMMARY arm_spec_sha256 of a warm run (the FT_FA_WARM base binding)."""
    return hashlib.sha256(f"spec|{arm}|{seed}".encode()).hexdigest()


def warm_selection_obj(arm: str, winner=(1.0, 1.0)) -> dict:
    """ANA-05: a PPSI_V4_FL_WARM_SELECTION_2 file (lr_screen select-warm shape, launch_fl canon self_sha256)."""
    arm = warm_variant_of(arm)                                         # ANA-06: a warm-line arm -> its variant
    st = R.WARM_SELECTION_STAGE[arm]
    f, sf = winner
    obj = {"format": R.WARM_SELECTION_FORMAT_2D, "stage": st, "arm": arm, "winner_factor": f,
           "winner_server_lr_factor": sf, "winner": {"warm_lr_factor": f, "warm_server_lr_factor": sf,
                                                     "label": f"f={f}|sf={sf}", "server_lr": 0.01 * sf},
           "no_measurable_gain": winner == (1.0, 1.0), "init_from": dict(INIT_FROM),
           "tuning_selection": {"sha256": "5" * 64, "stage": "S4"}, "values": {"f=1|sf=1": 0.1}}
    return canon(obj)


def warm_binding(arm: str, winner=(1.0, 1.0)) -> dict:
    """ANA-05: the LAUNCH warm_selection of a MAIN warm run bound to warm_selection_obj(arm) (launch_fl
    bind_warm_selection fields)."""
    obj = warm_selection_obj(arm, winner)
    st = R.warm_stage_of(arm)
    text = json.dumps(obj, indent=1, sort_keys=True) + "\n"
    return {"path": f"/srv/contracts/fl/FL_WARM_SELECTION_{st}.json", "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "self_sha256": obj["self_sha256"], "stage": st, "format": obj["format"], "winner_factor": winner[0],
            "winner_server_lr_factor": winner[1], "winner_server_lr": 0.01 * winner[1], "server_optimizer": "fedadam"}


def write_warm_selection(path: Path, arm: str, winner=(1.0, 1.0)) -> Path:
    """The file whose sha256 warm_binding(arm, winner) names (same bytes: indent 1, sorted keys, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(warm_selection_obj(arm, winner), indent=1, sort_keys=True) + "\n").encode())
    return path


def bench_result(tags: dict) -> dict:
    """A v4dev.bench BENCH_RESULT.json (DEV-04 format: `retention` keyed by tag)."""
    ret = {}
    for tag, sha in tags.items():
        ret[tag] = {"ranks_path": f"/server/ONNX/evals/{tag}.ranks_int32.npy", "ranks_sha256": sha,
                    "model_sha256": "m" * 64, "manifest_n": ROLES["MAIN"][2] * PER, "n_rows_in_npz": 99,
                    "n_oov_skipped": 1, "n_rows_scored": 98, "rows_npz_sha256": "z" * 64, "rows_meta_sha256": "y" * 64,
                    **ORT}
    return {"family": "SASREC", "variant": "SASREC_D64_B2", "onnxruntime": dict(ORT, inter_op_num_threads=1),
            "intra_op_num_threads": 1, "retention": ret, "checkpoint_sha256": "k" * 64}


def onnx_ranks(base, m) -> np.ndarray:
    rk = arm_ranks("C_SMALL_INT8", 2026, "PRACTICAL_BEST", base, m.rankable)
    rk[~np.isin(m.eval_user_key, SUBSET_USERS)] = -1                 # bench: -1 for every row it did not score
    return rk


def write_onnx(root: Path, base, m, tags=("onnx_int8",)) -> list:
    """ONNX_CPU entries (rows-free): ranks + BENCH_RESULT.json + rows.meta.json (users subset, v2 order)."""
    rk = onnx_ranks(base, m)
    shas = {}
    for tag in tags:
        p = root / "ONNX" / "evals" / f"{tag}.ranks_int32.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, rk.astype(np.int32))
        shas[tag] = R._sha_file(p)
    jdump(root / "ONNX" / "BENCH_RESULT.json", bench_result(shas))
    jdump(root / "ONNX" / "rows.meta.json", {"manifest_id": m.manifest_id, "first_users": N_SUBSET,
                                             "first_users_rule": "first N E2E users in selection_key_v2 order",
                                             "first_users_set_sha256": SUBSET_SHA, "n_users": N_SUBSET})
    arm_of = {"onnx_int8": "C_SMALL_INT8", "onnx_fp32": "C_SMALL_FP32"}
    return [{"arm": arm_of[t], "seed": 2026, "rule": "PRACTICAL_BEST", "role": "MAIN", "rows": None,
             "ranks": f"ONNX/evals/{t}.ranks_int32.npy", "bench_result": "ONNX/BENCH_RESULT.json",
             "comparison_block": "ONNX_CPU",
             "users_subset": {"rule": "SELECTION_KEY_V2_FIRST", "n": N_SUBSET, "set_sha256": SUBSET_SHA},
             "meta": {"model_bytes": 21_000_000 if t == "onnx_int8" else 82_840_192, "param_count": 20_710_048}}
            for t in tags]


def add_warm_start(blk: Block, *, pre_cid: str = "PRE_s2026:endpoint_6.0", pre_efe: float = 6.0,
                   pre_rules=("PRACTICAL_BEST", "CONTROLLED_BUDGET"), pre_extra=None, fa_warm_seeds=(2026, 2027)) -> None:
    """Add the 06K arms to a built MAIN block as plain rows entries: FA_WARM (ranks = C_BIG + 1, both seeds, both rules)
    and PRE (ONE theta_pre, seed 2026, ranks = C_BIG + 4, the 6.0 endpoint under either rule label)."""
    idx = blk.index()
    m, base = blk.manifests["MAIN"], blk.base["MAIN"]
    rsplit = "RETRO_TEST" if blk.split == "TEST" else "VALIDATION"
    for arm, seeds in (("FA_WARM", tuple(fa_warm_seeds)), ("PRE", (2026,))):
        for seed in seeds:
            rows, rules = [], (pre_rules if arm == "PRE" else R.RULES)
            for rule in rules:
                rk = arm_ranks(arm, 2026 if arm == "PRE" else seed, "PRACTICAL_BEST" if arm == "PRE" else rule, base,
                               m.rankable)
                blk.ranks[("MAIN", arm, seed, rule)] = rk
                rel = f"MAIN/{arm}_s{seed}/ranks_{rule}.npy"
                (blk.root / rel).parent.mkdir(parents=True, exist_ok=True)
                np.save(blk.root / rel, rk.astype(np.int32))
                kw = ({"cid": pre_cid, "efe": pre_efe, "extra": pre_extra} if arm == "PRE" else
                      {"extra": warm_reeval_row_labels(arm, seed)})       # REV-ANA-05 F1: v4int.reeval FL row labels
                rows += rows_for(arm, seed, rule, rk, m, split=rsplit, **kw)
                e = {"arm": arm, "seed": seed, "rule": rule, "role": "MAIN",
                     "rows": f"MAIN/{arm}_s{seed}/RESULT_ROWS.json", "ranks": rel}
                if arm == "FA_WARM":                                     # the index's REEVAL warm binding (F1)
                    e.update(warm_selection=warm_binding(arm), warm_run=warm_run(arm, seed))
                idx["entries"].append(e)
            jdump(blk.root / "MAIN" / f"{arm}_s{seed}" / "RESULT_ROWS.json", {"rows": rows})
    if blk.split != "TEST":
        ensure_warm_records(blk.root, idx)
    blk.write_index(idx)


def warm_reeval_row_labels(arm: str, seed: int) -> dict:
    """REV-ANA-05 F1: the run identity v4int.reeval puts on an FL checkpoint's rows (fl_run_id, arm_spec_sha256)."""
    return {"fl_run_id": f"FL_{arm}_SMALL_s{seed}", "arm_spec_sha256": warm_spec_sha(arm, seed)}


def warm_run(arm: str, seed: int, factors=(1.0, 1.0)) -> dict:
    """REV-ANA-05 F1: tools/ana_block_index.reeval_warm_binding's warm_run (the FL run's LAUNCH identity)."""
    return {"run_id": f"FL_{arm}_SMALL_s{seed}", "arm_spec_sha256": warm_spec_sha(arm, seed), "seed": seed,
            "warm_lr_factor": factors[0], "warm_server_lr_factor": factors[1]}


def ensure_warm_records(root: Path, idx: dict) -> None:
    """REV-ANA-05 F3: a VALIDATION block with FA_WARM / FA_WARM_FROZEN entries carries the registered 2-D warm
    selection file of each such stage in records (written here unless the index already names one)."""
    for arm, st in R.WARM_SELECTION_STAGE.items():
        key = R.WARM_SELECTION_RECORD[st]
        if key in idx.setdefault("records", {}) or not any(R.warm_stage_of(e["arm"]) == st for e in idx["entries"]):
            continue
        write_warm_selection(root / "records" / f"FL_WARM_SELECTION_{st}.json", arm)
        idx["records"][key] = f"records/FL_WARM_SELECTION_{st}.json"


def add_onnx_fp32(blk: Block) -> None:
    """Add the ONNX FP32 entry (same rows / ranks as INT8) to a built block: K16 becomes computable."""
    idx = blk.index()
    idx["entries"] = [e for e in idx["entries"] if e.get("comparison_block") != "ONNX_CPU"]
    idx["entries"] += write_onnx(blk.root, blk.base["MAIN"], blk.manifests["MAIN"], tags=("onnx_fp32", "onnx_int8"))
    blk.write_index(idx)


# ------------------------------------------------------------------------------------------------ ANA-04 (06L-C)
BASE_DEV = "CPU-FP32"                       # a baseline's CPU rows: only eval_device_class / precision / objective differ
BASE_EXTRA = {"precision_id": "BASELINE_NO_PRECISION", "objective_id": "BASELINE_NO_OBJECTIVE", "run_label": "BASELINE"}
PRE_RUN = "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026"
REPEAT_FRAC = 0.4


def last_ties(ranks: np.ndarray, *, seed: int = 29, arm: str = "LAST") -> np.ndarray:
    """A v4base.run n_tied_int32 array: -1 where not ranked; LAST: 0 (target in the window) on ~40% of the ranked rows,
    else the sentinel tie count; other baselines: 0 on every ranked row."""
    t = np.where(_mask(REPEAT_FRAC, seed, ranks.size), 0, K - 5) if arm == "LAST" else np.zeros(ranks.size)
    t = t.astype(np.int64)
    t[ranks < 1] = -1
    return t


def add_baselines_ranked(blk: Block, arms=("LAST", "RULE"), *, role: str = "MAIN", ties: bool = True,
                         extra_rows=None) -> None:
    """BASELINE_CPU entries (06L-C C3): v4base.run-style rows (CPU device class, v4base placeholders) + rank files (+ the
    per-row tie count) for each baseline arm; RULE_FREE, seed-free, not historical."""
    idx = blk.index()
    m, base = blk.manifests[role], blk.base[role]
    rsplit = "RETRO_TEST" if blk.split == "TEST" else "VALIDATION"
    for arm in arms:
        rk = arm_ranks(arm, 2026, "PRACTICAL_BEST", base, m.rankable)
        rel = f"{role}/BASE_{arm}"
        (blk.root / rel).mkdir(parents=True, exist_ok=True)
        np.save(blk.root / rel / "ranks_int32.npy", rk.astype(np.int32))
        e = {"arm": arm, "seed": None, "rule": "RULE_FREE", "role": role, "rows": f"{rel}/rows.json",
             "ranks": f"{rel}/ranks_int32.npy", "comparison_block": "BASELINE_CPU"}
        if ties:
            np.save(blk.root / rel / "n_tied_int32.npy", last_ties(rk, arm=arm).astype(np.int32))
            e["row_ties"] = f"{rel}/n_tied_int32.npy"
        jdump(blk.root / rel / "rows.json", rows_for(arm, None, None, rk, m, device=BASE_DEV, carry_rule=False, cid=None,
                                                    split=rsplit, extra=dict(BASE_EXTRA, **(extra_rows or {})),
                                                    role=role))
        blk.ranks[(role, arm, None, "RULE_FREE")] = rk
        idx["entries"].append(e)
    blk.write_index(idx)


def add_plain(blk: Block, arm: str, seeds, rules=R.RULES, *, role: str = "MAIN", method=None, entry_extra=None,
              device_cell: bool = False) -> None:
    """Plain rows + ranks entries (the REEVAL / device shape) of `arm` on `role`; `method` overrides the rows' method
    (FT_PRE rows carry launch_device's FT_C); device_cell adds the selected-cell fields."""
    idx = blk.index()
    m, base = blk.manifests[role], blk.base[role]
    rsplit = "RETRO_TEST" if blk.split == "TEST" else "VALIDATION"
    for seed in seeds:
        rows = []
        for rule in rules:
            rk = arm_ranks(arm, seed, rule, base, m.rankable)
            blk.ranks[(role, arm, seed, rule)] = rk
            rel = f"{role}/{arm}_s{seed}/ranks_{rule}.npy"
            (blk.root / rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(blk.root / rel, rk.astype(np.int32))
            extra = {"method": method} if method else {}
            if device_cell:
                extra.update(device_cell=DEVICE_CELL, device_selected=True, efe=None)
                rows += rows_for(arm, seed, rule, rk, m, carry_rule=False, cid=None, split=rsplit, role=role,
                                 extra=extra)
            else:
                rows += rows_for(arm, seed, rule, rk, m, split=rsplit, role=role, extra=extra)
            e = {"arm": arm, "seed": seed, "rule": rule, "role": role, "rows": f"{role}/{arm}_s{seed}/RESULT_ROWS.json",
                 "ranks": rel}
            if device_cell:
                e["row_filter"] = {"device_cell": DEVICE_CELL}
            e.update(entry_extra or {})
            idx["entries"].append(e)
        jdump(blk.root / role / f"{arm}_s{seed}" / "RESULT_ROWS.json", {"rows": rows})
    blk.write_index(idx)


PRE_INIT_RECORD_SHA = "c" * 64                  # DEV-FTPRE: the PRETRAIN_FOR_FL_INIT.json file sha256 (synthetic)
PRE_INIT = {"checkpoint_sha256": INIT_FROM["checkpoint_sha256"], "init_record_sha256": PRE_INIT_RECORD_SHA,
            "run_id": INIT_FROM["run_id"]}      # the PRE entry's REEVAL binding to that record (index pre_init)


def add_ft_pre(blk: Block, seeds=(2026, 2027), start_run_id: str = PRE_RUN, method: str = "FT_PRE") -> None:
    """FT_PRE (06L-C C7): launch_device --arm FT_PRE rows (method FT_PRE; FT_C accepted) from the theta_pre base. DEV-FTPRE:
    the index records the LAUNCH start run id, start.init_from (start_init) and the init record sha256, and the block's PRE
    entries carry pre_init (the PRE REEVAL binding to the same PRETRAIN_FOR_FL_INIT.json)."""
    add_plain(blk, "FT_PRE", seeds, ("PRACTICAL_BEST",), method=method, device_cell=True,
              entry_extra={"start_run_id": start_run_id, "start_init": dict(INIT_FROM),
                           "start_init_record_sha256": PRE_INIT_RECORD_SHA})
    idx = blk.index()
    for e in idx["entries"]:
        if e["arm"] == "PRE":
            e["pre_init"] = dict(PRE_INIT)
    blk.write_index(idx)


def add_role(blk: Block, role: str, n_users: int = 20, seed: int = 4) -> None:
    """Another registered role (e.g. UNSEEN_VAL) with its own synthetic manifest and base ranks."""
    ev, tr = R.ROLE_MAP["TEST" if blk.split == "TEST" else "VALIDATION"][role]
    rng = np.random.default_rng(100 + seed)
    n = n_users * PER
    keys = np.repeat(np.arange(n_users, dtype=np.int64), PER)
    eval_mask = rng.random(n) >= 0.1
    eval_mask[::PER] = True
    target = rng.integers(0, K, n).astype(np.int64)
    target[~eval_mask] = -1
    censor = np.array(["NONE" if x else "CEN_NO_NEXT" for x in eval_mask], dtype=object)
    m = EvalManifest(header={"manifest_id": ev, "catalogue_K": K}, decision_id=np.arange(1, n + 1, dtype=np.int64) * 11
                     + seed * 1_000_000, eval_user_key=keys, eval_mask=eval_mask, target_class=target,
                     target_oov=np.zeros(n, dtype=bool), censor_class=censor, catalogue_K=K)
    blk.manifests[role] = m
    blk.base[role] = np.random.default_rng(500 + seed).integers(1, 9, m.n).astype(np.int64)
    hexes = np.asarray([hexkey(role, k) for k in range(n_users)], dtype=object)
    blk.ctx[role] = R.RoleCtx(role, m, hexes, None, {"eval_manifest_id": ev, "synthetic": True})
    idx = blk.index()
    idx["roles"][role] = {"eval_manifest_id": ev, **({"train_manifest_id": tr} if tr else {})}
    blk.write_index(idx)


CALIB_USERS, CALIB_OFFSET = 16, 20


def add_calib(blk: Block, n: int = CALIB_USERS, offset: int = CALIB_OFFSET) -> None:
    """A synthetic CALIBRATION context (06L-C C14): its users are MAIN users offset .. offset+n-1 (by client_key_hex; its
    own eval_user_key is 0 .. n-1, so a join by eval_user_key would pick the wrong users), each with one E2E row;
    MAIN \\ CALIB = the other MAIN users."""
    m = EvalManifest(header={"manifest_id": R.CALIB_EVAL_ID, "catalogue_K": K},
                     decision_id=np.arange(1, n + 1, dtype=np.int64) * 13, eval_user_key=np.arange(n, dtype=np.int64),
                     eval_mask=np.ones(n, dtype=bool), target_class=np.ones(n, dtype=np.int64),
                     target_oov=np.zeros(n, dtype=bool), censor_class=np.array(["NONE"] * n, dtype=object),
                     catalogue_K=K)
    hexes = np.asarray([hexkey("MAIN", offset + k) for k in range(n)], dtype=object)
    blk.ctx["CALIBRATION"] = R.RoleCtx("CALIBRATION", m, hexes, None, {"eval_manifest_id": R.CALIB_EVAL_ID,
                                                                        "synthetic": True})


def build(tmp: Path, *, include_records: bool = True, split: str = "VALIDATION", gate=None) -> Block:
    root = tmp / "block"
    root.mkdir(parents=True)
    rsplit = "RETRO_TEST" if split == "TEST" else "VALIDATION"
    manifests = {r: make_manifest(r, split, gate) for r in ROLES}
    base, ranks, ctx = {}, {}, {}
    for role, m in manifests.items():
        rng = np.random.default_rng(500 + ROLES[role][3])
        base[role] = rng.integers(1, 9, m.n).astype(np.int64)            # 1..8: 2x stays within @20
        n_users = ROLES[role][2]
        hexes = np.asarray([hexkey(role, k) for k in range(n_users)], dtype=object)
        sup = None
        if ROLES[role][1]:
            sup = {hexes[k]: SUPPORT_CYCLE[k % len(SUPPORT_CYCLE)] for k in range(n_users)}
        ctx[role] = R.RoleCtx(role, m, hexes, sup, {"eval_manifest_id": m.manifest_id, "synthetic": True})
    entries = []

    def add(role, arm, seed, rules, *, carry_rule=True, device=False, extra_entry=None):
        m = manifests[role]
        rows = []
        for rule in rules:
            rk = arm_ranks(arm, seed, rule, base[role], m.rankable)
            ranks[(role, arm, seed, rule)] = rk
            rel_r = f"{role}/{arm}_s{seed}/ranks_{rule}.npy"
            (root / rel_r).parent.mkdir(parents=True, exist_ok=True)
            np.save(root / rel_r, rk.astype(np.int32))
            if device:
                rows += rows_for(arm, seed, rule, rk, m, carry_rule=False, cid=None, split=rsplit, role=role,
                                 extra={"device_cell": DEVICE_CELL, "device_selected": True, "efe": None})
                other = arm_ranks("LO", seed, rule, base[role], m.rankable)          # a non-selected grid cell
                rows += rows_for(arm, seed, rule, other, m, carry_rule=False, cid=None, split=rsplit, role=role,
                                 extra={"device_cell": "P1_F0.3", "device_selected": False, "efe": None})
            else:
                rows += rows_for(arm, seed, rule, rk, m, carry_rule=carry_rule, split=rsplit, role=role)
            e = {"arm": arm, "seed": seed, "rule": rule, "role": role, "rows": f"{role}/{arm}_s{seed}/RESULT_ROWS.json",
                 "ranks": rel_r}
            if device:
                e["row_filter"] = {"device_cell": DEVICE_CELL}
            e.update(extra_entry or {})
            entries.append(e)
        jdump(root / role / f"{arm}_s{seed}" / "RESULT_ROWS.json", {"rows": rows})

    for arm, seeds in MAIN_ARMS.items():
        for seed in seeds:
            meta = {}
            if arm == "C_BIG":
                meta = {"meta": {"param_count": 80_000_000, "model_bytes": 320_000_000}}
            if arm == "C_SMALL":
                meta = {"meta": {"param_count": 20_710_048, "model_bytes": 82_840_192}}
            if arm in ("DP8", "DP4", "DP1"):
                meta = {"meta": {"dp_z": {"DP8": 0.5642, "DP4": 0.83, "DP1": 1.3094}[arm]}}
            add("MAIN", arm, seed, R.RULES, extra_entry=meta)
    for seed in (2026, 2027):
        _fl_run(root, f"MAIN/FL_FA_s{seed}", "FA", seed, manifests["MAIN"], base["MAIN"], manifests["MAIN"].rankable,
                entries, rsplit)
        m = manifests["MAIN"]
        for rule in R.RULES:
            ranks[("MAIN", "FA", seed, rule)] = arm_ranks("FA", seed, rule, base["MAIN"], m.rankable)
    for arm, seeds in DEVICE_ARMS.items():
        for seed in seeds:
            add("MAIN", arm, seed, ("PRACTICAL_BEST",), device=True)
    m = manifests["MAIN"]
    entries += write_onnx(root, base["MAIN"], m)                      # C_SMALL_INT8 (ONNX_CPU, rows-free)
    rk = arm_ranks("POP", 2026, "PRACTICAL_BEST", base["MAIN"], m.rankable)
    jdump(root / "MAIN" / "POP_rows.json", rows_for("POP", None, None, rk, m, carry_rule=False, cid=None, split=rsplit))
    entries.append({"arm": "POP", "seed": None, "rule": "RULE_FREE", "role": "MAIN", "rows": "MAIN/POP_rows.json"})
    for role, arms in PANEL_ARMS.items():
        for arm in arms:
            for seed in (2026, 2027):
                add(role, arm, seed, ("PRACTICAL_BEST",), device=arm in ("FT_C", "LO"))
    ids = TEST_IDS if split == "TEST" else {r: v[0] for r, v in ROLES.items()}
    idx = {"format": R.INDEX_FORMAT, "split": split, "eval_device_class": DEV,
           "roles": {r: ({"eval_manifest_id": ids[r], "train_manifest_id": v[1]} if v[1] else
                         {"eval_manifest_id": ids[r]}) for r, v in ROLES.items()},
           "entries": entries, "records": {}}
    if include_records:
        idx["records"] = write_records(root, rsplit)
    contrasts = tmp / "contrasts.json"
    jdump(contrasts, contrasts_obj())
    blk = Block(root, manifests, base, contrasts, ctx, ranks, split)
    blk.write_index(idx)
    return blk


def trunc_rows(split: str = "VALIDATION") -> list:
    """DATA-07-shaped rows built with v4int.launch_trunc.trunc_labels (read-only import): Study A C_BIG / C_SMALL on
    WARM_TRUNCATION (FIXED_20 + FIXED_50, FULL once per FIXED sub-panel at its largest h) and COLD_H (ALL_AVAILABLE,
    FULL per h), seeds 2026 / 2027; Study B LO with a selected and a non-selected cell."""
    from v4int.launch_trunc import PANELS, trunc_labels
    key = {"eval_device_class": DEV, "split": split, "evaluator_code_sha256": EVSHA, "metric": "mrr", "k": 20,
           "aggregation": "macro", "target_denominator": "E2E", "value_status": "OK", "selection_rule": None}
    out = []

    def row(arm, seed, study, label, ptype, h, full, v, **extra):
        r = dict(key, method=arm, seed=seed, value=v,
                 **trunc_labels(study=study, panel_id=PANELS[label], label=label, panel_type=ptype, h=h, full=full))
        r.update(extra)
        out.append(r)
    for seed in (2026, 2027):
        for ptype, hs, vals in (("FIXED_20", (0, 1, 5, 10, 20), (0.10, 0.15, 0.17, 0.185, 0.195)),
                                ("FIXED_50", (0, 5, 10, 20, 50), (0.05, 0.06, 0.07, 0.08, 0.2))):
            row("C_BIG", seed, "A", "WARM_TRUNCATION", ptype, max(hs), True, 0.2)
            row("C_SMALL", seed, "A", "WARM_TRUNCATION", ptype, max(hs), True, 0.1)
            for h, v in zip(hs, vals):
                row("C_SMALL", seed, "A", "WARM_TRUNCATION", ptype, h, False, v)
        for h, v in zip(R.TRUNC_H, (0.10, 0.15, 0.17, 0.185, 0.195, 0.2)):
            row("C_BIG", seed, "A", "COLD_H", "ALL_AVAILABLE", h, True, 0.2 + 0.001 * h)
            row("C_SMALL", seed, "A", "COLD_H", "ALL_AVAILABLE", h, True, 0.1)
            row("C_SMALL", seed, "A", "COLD_H", "ALL_AVAILABLE", h, False, v)
            row("LO", seed, "B", "COLD_H", "ALL_AVAILABLE", h, False, 0.05, device_cell="P1_F0.1",
                device_selected=True)
            row("LO", seed, "B", "COLD_H", "ALL_AVAILABLE", h, False, 0.9, device_cell="P4_F0.1",
                device_selected=False)
    return out


def write_records(root: Path, split: str = "VALIDATION") -> dict:
    stats = {}
    for k in range(ROLES["MAIN"][2]):
        if k % 12 == 0:
            continue                                           # absent -> NO_TRAIN stratum
        stats[hexkey("MAIN", k)] = {"n_train": SUPPORT_CYCLE[k % len(SUPPORT_CYCLE)] + 1,
                                    "n_distinct_main_categories": k % 7, "top_brand_share": (k % 10) / 10.0}
    ST.write_registry(root / "records" / "STRATA_CUTPOINTS_MAIN.json", stats, source_manifest_id="MAIN_TRAIN_5_100_131072")
    jdump(root / "records" / "S_RECORD.json", canon({"format": "PPSI_V4_DP_S_RECORD_1", "S": 1.2345, "n_norms": 20480,
                                                     "rule": "median", "seed": 2026, "label": "S_CALIBRATION",
                                                     "disclosure": "non-private"}))
    jdump(root / "records" / "SEL_FT_C.json", canon({"format": "PPSI_V4_DEVICE_SELECTION_1", "arm": "FT_C", "seed": 2026,
                                                     "role": "UNSEEN_VAL", "cell": "P2_F0.1",
                                                     "values_primary": {"P1_F0.1": 0.1, "P2_F0.1": 0.2},
                                                     "rule": "max; ties -> canonical order"}))
    jdump(root / "records" / "LR_SELECTION.json", canon({"format": "PPSI_V4_LR_SELECTION_1", "selected_m": 1.0,
                                                         "values": {"0.5": 0.11, "1.0": 0.12, "2.0": 0.10}}))
    jdump(root / "records" / "DP_WOR.json", canon({"records": [{"z": 0.5642, "epsilon_poisson": 7.99825,
                                                                "epsilon_wor": 6.5}, {"z": 1.3094, "epsilon_poisson": 0.99985,
                                                                                      "epsilon_wor": 0.8}],
                                                   "eps4_z": 0.83}))
    jdump(root / "records" / "TRUNC.json", {"rows": trunc_rows(split)})
    return {"STRATA": "records/STRATA_CUTPOINTS_MAIN.json", "S_RECORD": "records/S_RECORD.json",
            "SELECTION": ["records/SEL_FT_C.json"], "LR_SELECTION": "records/LR_SELECTION.json",
            "DP_WOR": "records/DP_WOR.json",
            "TRUNC_ROWS": [{"rows": "records/TRUNC.json", "rule": "PRACTICAL_BEST"}]}


def macro(ranks, m, metric="mrr", k=20) -> float:
    return population_block(credits_from_ranks(ranks, (k,)), m.eval_mask, m.eval_user_key)["macro"][f"{metric}@{k}"]


# ------------------------------------------------------------------------------------------------ the TEST world
def make_test_gate(tmp: Path) -> dict:
    """ALL_METHODS_LOCK (= the contrasts JSON + rule map), a hash-chained TEST_EXPOSURE_REGISTER with a MANIFEST_BUILD
    RETRO_TEST entry binding the lock (tools/test_exposure_register.py canon rule), and a REGISTRY_V2_TEST_EXT.json built
    from the real contracts/manifest_registry/REGISTRY_V2.json (+ the TEST roles / RETRO_TEST manifests)."""
    import test_exposure_register as TER
    from v4data08.d8_registry import with_self
    tmp.mkdir(parents=True, exist_ok=True)
    lock = tmp / "ALL_METHODS_LOCK.json"
    jdump(lock, contrasts_obj(3, document="ALL_METHODS_LOCK", status="LOCKED (synthetic)"))   # M5: TEST = v3 only
    lock_sha = R._sha_file(lock)
    entries, prev = [], None
    for i, (purpose, lsha) in enumerate((("HISTORICAL_PRIOR_EXPOSURE", None), ("MANIFEST_BUILD", lock_sha)), start=1):
        e = {"entry_id": i, "utc": "2026-09-29T00:00:0%dZ" % i, "actor_role": "Ops", "source": "synthetic",
             "split": "RETRO_TEST", "access_purpose": purpose, "observations": "NONE", "observations_description": "none",
             "decision": "synthetic", "all_methods_lock_sha256": lsha, "manifest_header_sha256": None,
             "model_checkpoint_sha256": None, "code_sha256": None, "receipt_path": None, "prev_entry_sha256": prev}
        e["entry_sha256"] = hashlib.sha256(TER.canon(e)).hexdigest()
        prev = e["entry_sha256"]
        entries.append(e)
    reg = tmp / "TEST_EXPOSURE_REGISTER.json"
    jdump(reg, {"register_id": "TEST_EXPOSURE_REGISTER", "spec_version": "MANIFEST_SPEC_V4 1.2.0", "entries": entries})
    base = json.loads(R.REGISTRY_V2_PATH.read_text(encoding="utf-8"))
    ext = {k: v for k, v in base.items() if k != "self_sha256"}
    ext = json.loads(json.dumps(ext))
    for role, (ev, tr) in R.ROLE_MAP["TEST"].items():
        ext["roles"][f"{role}_TEST"] = {"train": tr, "eval": ev} if tr else {"eval": ev}
        ext["manifests"][ev] = {"kind": "EVAL", "status": f"{role}_TEST", "tables": {}}
    ext["extends"] = {"path": "contracts/manifest_registry/REGISTRY_V2.json",
                      "file_sha256": R._sha_file(R.REGISTRY_V2_PATH), "self_sha256": base["self_sha256"]}
    ext["extension_test_gate"] = {"all_methods_lock_sha256": lock_sha, "test_exposure_register_entry_id": 2}
    ext = with_self(ext)
    ext_p = tmp / "REGISTRY_V2_TEST_EXT.json"
    jdump(ext_p, ext)
    return {"lock": lock, "register": reg, "ext": ext_p, "all_methods_lock_sha256": lock_sha,
            "test_exposure_register_entry_id": 2}


def validation_result_v3(tmp: Path) -> Path:
    """FIX-DELTA-03 M5: a TEST run takes the v3 registry only, so its VALIDATION_RESULT (same script / library /
    contrasts content) is the default VALIDATION block run with the v3 file."""
    blk = build(tmp)
    jdump(blk.contrasts, contrasts_obj(3))
    out = tmp / "out_v3"
    lines = []
    rc = R.main(blk.argv(out), context_loader=blk.loader, say=lines.append)
    assert rc == 0, "\n".join(lines)
    return out / R.RESULT_NAME


def build_test(tmp: Path, val_result: Path) -> Block:
    """The TEST block (RETRO_TEST roles, gate-bound headers) + records.VALIDATION_RESULT + the gate CLI args."""
    gate = make_test_gate(tmp / "gate")
    blk = build(tmp, split="TEST", gate=gate)
    (blk.root / "records").mkdir(exist_ok=True)
    (blk.root / "records" / "VAL_RESULT.json").write_bytes(Path(val_result).read_bytes())
    idx = blk.index()
    idx["records"]["VALIDATION_RESULT"] = "records/VAL_RESULT.json"
    blk.write_index(idx)
    blk.contrasts = gate["lock"]
    blk.test_args = ["--all-methods-lock", str(gate["lock"]), "--test-exposure-register", str(gate["register"]),
                     "--registry-ext", str(gate["ext"])]
    blk.gate = gate
    return blk
