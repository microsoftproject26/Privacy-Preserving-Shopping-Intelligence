"""Private, transient successful-upload capture. No raw updates survive analysis.

Only the input item table supports an input-history row attack. Encoder deltas
exclude the input/output catalogue tables and output bias. Cohort deltas use all
successful uploads with the actual n_consumed weights, including unsampled users.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

INPUT_KEY = "item_embed.weight"
EXCLUDED_ENCODER = {INPUT_KEY, "output_embed", "output_bias"}


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def declare_population(plan, cfg, *, rounds=50, sample_size=512, seed=481731, holdout_rounds=15):
    """Predeclare from deterministic participation/dropout, without reading uploads.

    Cohorts and users are disjoint across attack splits. We refuse a population
    with repeat users in this window instead of silently leaking across splits.
    """
    from fl.runtime import round_survivors
    if rounds < 2 or not 0 < holdout_rounds < rounds or sample_size < 2:
        raise ValueError("invalid audit population size/split")
    cohorts = []
    seen = set()
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    held = set(int(x) for x in rng.choice(rounds, holdout_rounds, replace=False))
    available = {"train": [], "holdout": []}
    for r in range(rounds):
        sampled, survived, dropped = round_survivors(plan, cfg, r)
        if seen.intersection(sampled):
            raise ValueError("audit window repeats clients across cohorts; user-disjoint split cannot be guaranteed")
        seen.update(sampled)
        split = "holdout" if r in held else "train"
        cohorts.append({"round": r, "split": split, "planned_keys": sampled,
                        "observed_keys": survived, "dropped_keys": dropped})
        available[split].extend((r, k) for k in survived)
    n_holdout = int(round(sample_size * holdout_rounds / rounds))
    counts = {"train": sample_size - n_holdout, "holdout": n_holdout}
    selected = []
    for split in ("train", "holdout"):
        population = available[split]
        if len(population) < counts[split]:
            raise ValueError(f"not enough observed {split} users for predeclared sample")
        for i in sorted(int(x) for x in rng.choice(len(population), counts[split], replace=False)):
            r, key = population[i]
            selected.append({"key": key, "round": r, "split": split, "cohort_id": r})
    body = {"format": "PPSI_E3_POPULATION_1", "label": "POST_HOC_EXPLORATORY", "role": "CALIBRATION",
            "seed": int(seed), "rounds": int(rounds), "sample_size": sample_size,
            "plan_digest": plan.digest(), "config_digest": cfg.digest(), "cohorts": cohorts,
            "selected": sorted(selected, key=lambda x: (x["round"], x["key"])),
            "split_counts": counts, "selection_rule": "uniform within deterministic surviving-cohort split"}
    return dict(body, self_sha256=canonical_sha(body))


def check_population(population):
    body = {k: v for k, v in population.items() if k != "self_sha256"}
    if population.get("self_sha256") != canonical_sha(body):
        raise ValueError("audit population self-hash mismatch")
    if population.get("role") != "CALIBRATION":
        raise ValueError("raw capture is CALIBRATION only")
    cohort_list = population["cohorts"]
    rounds = [c["round"] for c in cohort_list]
    if rounds != list(range(population["rounds"])):
        raise ValueError("audit cohorts must be contiguous unique rounds starting at0")
    all_planned = set()
    for c in cohort_list:
        p, o, d = c["planned_keys"], c["observed_keys"], c["dropped_keys"]
        if any(len(ks) != len(set(ks)) for ks in (p,o,d)):
            raise ValueError("audit cohort contains duplicate keys")
        if set(o) & set(d) or set(o) | set(d) != set(p):
            raise ValueError("audit observed/dropped keys must partition the planned cohort")
        if [k for k in p if k in set(o)] != o or [k for k in p if k in set(d)] != d:
            raise ValueError("audit surviving/dropped order differs from the planned cohort")
        if all_planned.intersection(p) or c["split"] not in ("train", "holdout"):
            raise ValueError("audit planned users repeat across cohorts or split is invalid")
        all_planned.update(p)
    ids = [(x["round"], x["key"]) for x in population["selected"]]
    if len(ids) != population["sample_size"] or len(set(ids)) != len(ids):
        raise ValueError("audit selected population has wrong size or duplicate observations")
    keys = [x["key"] for x in population["selected"]]
    if len(set(keys)) != len(keys):
        raise ValueError("audit selected users are not disjoint")
    actual_counts = {s: sum(x["split"] == s for x in population["selected"]) for s in ("train", "holdout")}
    if population.get("split_counts") != actual_counts or not all(actual_counts.values()):
        raise ValueError("audit split counts do not match the selected population")
    cohorts = {x["round"]: x for x in population["cohorts"]}
    for x in population["selected"]:
        c = cohorts[x["round"]]
        if x["key"] not in c["observed_keys"] or x["split"] != c["split"]:
            raise ValueError("selected observation is outside the declared successful population/split")


def sparse_input(delta):
    """All nonzero input rows, with explicit padding removal. No threshold tuning."""
    if delta.ndim != 2:
        raise ValueError("input item delta is not a 2-D table")
    norms = torch.linalg.vector_norm(delta.to(torch.float64), dim=1)
    idx = torch.nonzero(norms > 0, as_tuple=False).flatten()
    idx = idx[idx != 0]
    return idx.to(torch.int64), norms[idx].to(torch.float32)


class UploadCapture:
    """Observer protocol consumed by fl.runtime/fl.server, private lifetime owner.

    truth_fn(key) must derive labels and input items only from the verified CALIB
    TRAIN provider. secagg_fn may validate a predeclared feature projection and
    returns a receipt; it never replaces the real aggregation trajectory.
    """
    def __init__(self, *, parent, population, truth_fn: Callable, manifest, role="CALIBRATION",
                 secagg_fn=None, secagg_projector=None):
        if role != "CALIBRATION":
            raise ValueError("raw capture is CALIBRATION only")
        check_population(population)
        self.population = population
        self.role = role
        self.truth_fn = truth_fn
        self.secagg_fn = secagg_fn
        self.secagg_projector = secagg_projector
        if secagg_fn is not None and secagg_projector is None:
            raise ValueError("SecAgg simulation requires a predeclared raw-coordinate projector")
        self.encoder_keys = tuple(k for k in manifest.shared_keys if k not in EXCLUDED_ENCODER)
        self.input_status = "SHARED" if INPUT_KEY in manifest.shared_keys else "FROZEN_NOT_UPLOADED"
        # A tied output alias makes zero input rows invalid as an input-only support attack.
        if any(getattr(e, "alias_of", None) == INPUT_KEY for e in manifest.entries.values()):
            self.input_status = "TIED_OUTPUT_NOT_APPLICABLE"
        self.delta_keys = tuple(([INPUT_KEY] if INPUT_KEY in manifest.shared_keys else []) + list(self.encoder_keys))
        self.selected = {(x["round"], x["key"]): x for x in population["selected"]}
        self.cohorts = {x["round"]: x for x in population["cohorts"]}
        self.seen = set()
        self.complete_rounds = set()
        self.record_paths = []
        self.cohort_paths = []
        self.bound = False
        self.closed = False
        self.active = None
        self.aggregate = None
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="E3_PRIVATE_UPLOADS_", dir=parent))
        self.directory.chmod(0o700)
        self.receipt_path = parent / "CAPTURE_DELETION_RECEIPT.json"

    def bind_run(self, run):
        if self.bound or run.state.cursor != 0 or run.server.round != 0:
            raise ValueError("raw capture refuses repeated binding/resume")
        if run.cfg.dp is not None or run.cfg.method != "FA" or run.store is not None or run.pool is not None:
            raise ValueError("raw capture refuses DP, personal state and pools")
        if run.cfg.digest() != self.population["config_digest"] or run.plan.digest() != self.population["plan_digest"]:
            raise ValueError("raw capture run differs from predeclared population")
        self.bound = True

    def begin_round(self, *, round_idx, cohort_keys):
        if not self.bound or self.closed or self.active is not None or round_idx in self.complete_rounds:
            raise ValueError("invalid/duplicate capture round")
        c = self.cohorts.get(round_idx)
        if c is None or list(cohort_keys) != c["observed_keys"]:
            raise ValueError("successful cohort differs from the predeclared dropout plan")
        self.active = round_idx
        self.round_seen = set()
        self.round_weight = 0
        self.aggregate = None
        self.encoder_contributions = []

    def observe_upload(self, *, round_idx, position, key, n_consumed, n_valid, attempts, delta):
        observation = (round_idx, key)
        if self.active != round_idx or observation in self.seen or key not in self.cohorts[round_idx]["observed_keys"]:
            raise ValueError("invalid or duplicate successful upload observation")
        if tuple(delta) != self.delta_keys or n_consumed < 0 or n_valid < 0 or attempts < 1:
            raise ValueError("capture key set/counters invalid")
        if any(t.requires_grad or t.device.type != "cpu" or not bool(torch.isfinite(t).all()) for t in delta.values()):
            raise ValueError("capture requires detached finite CPU delta copies")
        self.seen.add(observation)
        self.round_seen.add(key)
        self.round_weight += int(n_consumed)
        if self.aggregate is None:
            self.aggregate = OrderedDict((k, torch.zeros_like(t, dtype=torch.float64)) for k, t in delta.items())
        for k, t in delta.items():
            self.aggregate[k].add_(t.to(torch.float64), alpha=int(n_consumed))
        encoder = torch.cat([delta[k].reshape(-1) for k in self.encoder_keys])
        # Bounded predeclared coordinates; simulator handles these weighted values only.
        if self.secagg_fn is not None:
            projected = self.secagg_projector(delta, encoder)
            self.encoder_contributions.append((key, torch.as_tensor(projected).detach().cpu().clone(), int(n_consumed)))
        if observation in self.selected:
            ids, norms = sparse_input(delta[INPUT_KEY]) if self.input_status == "SHARED" else (
                torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.float32))
            selected = self.selected[observation]
            rec = dict(selected, n_consumed=int(n_consumed), n_valid=int(n_valid), attempts=int(attempts),
                       position=int(position), input_row_ids=ids, input_row_norms=norms,
                       encoder_delta=encoder.clone(), input_table_status=self.input_status, truth=self.truth_fn(key))
            p = self.directory / f"individual_{round_idx:03d}_{position:03d}.pt"
            torch.save(rec, p)
            self.record_paths.append(p)

    def end_round(self, *, round_idx, report):
        if self.active != round_idx or self.round_seen != set(self.cohorts[round_idx]["observed_keys"]):
            raise ValueError("capture round missing accepted uploads")
        if int(report.weight) != self.round_weight or self.round_weight <= 0:
            raise ValueError("capture aggregate weight differs from real FedAvg")
        averaged = OrderedDict((k, t / self.round_weight) for k, t in self.aggregate.items())
        ids, norms = sparse_input(averaged[INPUT_KEY]) if self.input_status == "SHARED" else (
            torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.float32))
        c = self.cohorts[round_idx]
        rec = dict(c, n_planned=len(c["planned_keys"]), n_observed=len(c["observed_keys"]),
                   weight=self.round_weight, input_row_ids=ids, input_row_norms=norms,
                   encoder_delta=torch.cat([averaged[k].reshape(-1) for k in self.encoder_keys]).to(torch.float32),
                   input_table_status=self.input_status, aggregation="ideal_FP64_n_consumed_weighted_mean_of_FP32_deltas",
                   aggregation_limit="unquantized_raw_delta_mean; differs_from_FP32_full_weight_server_release")
        rec["secagg"] = self.secagg_fn(c, self.encoder_contributions) if self.secagg_fn is not None else {
            "status": "NOT_RUN", "production_crypto": False}
        p = self.directory / f"cohort_{round_idx:03d}.pt"
        torch.save(rec, p)
        self.cohort_paths.append(p)
        self.complete_rounds.add(round_idx)
        self.active = None
        self.aggregate = None
        self.encoder_contributions = []

    def verify_complete(self):
        if len(self.record_paths) != self.population["sample_size"] or self.complete_rounds != set(self.cohorts):
            raise ValueError("capture population incomplete")
        if self.active is not None:
            raise ValueError("capture ended during a round")
        return {"n_individual": len(self.record_paths), "n_cohorts": len(self.cohort_paths),
                "population_sha256": self.population["self_sha256"], "input_table_status": self.input_status,
                "encoder_keys": list(self.encoder_keys)}

    def read_records(self):
        for p in self.record_paths:
            yield torch.load(p, map_location="cpu", weights_only=True)

    def read_cohorts(self):
        for p in self.cohort_paths:
            yield torch.load(p, map_location="cpu", weights_only=True)

    def cleanup(self, *, outcome="FAILED_OR_INCOMPLETE"):
        if self.closed:
            return
        n_files = sum(1 for p in self.directory.rglob("*") if p.is_file())
        n_bytes = sum(p.stat().st_size for p in self.directory.rglob("*") if p.is_file())
        shutil.rmtree(self.directory)
        self.aggregate = None
        self.encoder_contributions = []
        self.closed = True
        write_json(self.receipt_path, {"format": "PPSI_E3_CAPTURE_DELETION_1", "outcome": outcome,
                                     "deleted_files": n_files, "deleted_bytes": n_bytes,
                                     "remaining_raw_capture_paths": 0 if not self.directory.exists() else 1,
                                     "private_directory_removed": not self.directory.exists(),
                                     "population_sha256": self.population["self_sha256"]})

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        self.cleanup(outcome="ANALYSIS_COMPLETED" if typ is None else "FAILED_OR_INCOMPLETE")
        return False
