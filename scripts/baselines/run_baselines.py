"""Execute the preregistered T2 LR/LightGBM and T1 session-kNN baselines.

No TEST, no new TaskExamples, no change to the existing experiment schemas.
Private fitted objects/predictions stay under artifacts/baselines/. Public results
use ExperimentResult v1, with an explicit non-final-centralized-baseline role.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import time
import warnings
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from ppsi.baselines.classical import FEATURES, feature_matrix, make_pipeline, positive_probabilities
from ppsi.baselines.session_knn import SessionKNN
from ppsi.baselines.t2_metrics import evaluate_t2
from ppsi.evaluation.t1 import (
    evaluate_t1_ranks,
    history_bucket,
    metric_records_from_t1_summary,
    rank_targets_from_scores,
)
from ppsi.training.identity import file_sha256
from ppsi.training.result import build_experiment_result, make_metric_record
from scripts.experiments.results import validate_result_for_reporting
from scripts.experiments.schemas import validate_experiment_config

CONFIG = "config/baselines/s2_pr_04_05.v1.json"
PUBLIC = "docs/evidence/s2-pr-04-05"
PRIVATE = "artifacts/baselines/s2-pr-04-05"
BRANCH = "s2-pr-04-05-classical-session-baselines"
SOURCE_FILES = (
    CONFIG,
    "config/baselines/t2_feature_manifest.v1.json",
    "docs/decisions/s2-pr-04-05-baselines.md",
    "scripts/baselines/run_baselines.py",
    "scripts/baselines/prepare_baseline_views.py",
    "scripts/baselines/supervise_baselines.py",
    "ppsi/baselines/classical.py",
    "ppsi/baselines/session_knn.py",
    "ppsi/baselines/t2_metrics.py",
    "ppsi/evaluation/t1.py",
    "ppsi/training/identity.py",
    "ppsi/training/result.py",
    "scripts/experiments/schemas.py",
    "scripts/experiments/results.py",
    "uv.lock",
)


def now():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def raw_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(rel, data, *, replace=False):
    path = ROOT / rel
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"preserving existing artifact: {rel}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def ref(rel, schema, logical=None):
    path = ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(rel)
    return {
        "schema": "artifact_ref_v1",
        "version": "1",
        "logical_id": logical or schema,
        "artifact_schema": schema,
        "artifact_version": "1",
        "uri": str(rel).replace("\\", "/"),
        "sha256": file_sha256(path),
    }


def source_snapshot():
    return {
        "schema": "baseline_source_snapshot_v1",
        "version": "1",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "execution_provenance": "WORKING_TREE_BASELINE_RUN_WITH_EXPLICIT_SOURCE_HASHES",
        "files_sha256": {rel: file_sha256(ROOT / rel) for rel in SOURCE_FILES},
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "lightgbm", "polars", "pyarrow", "torch")
        },
    }


def preflight():
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != BRANCH:
        raise RuntimeError(f"wrong branch: {branch}; human must select {BRANCH}")
    config = read(CONFIG)
    if config["test_allowed"] or config["seed"] != 13 or config["evaluation_split"] != "VALIDATION":
        raise ValueError("baseline plan changed outside its frozen scope")
    views = read(config["data_views"])
    if views["status"] != "PASS" or views["test_consumed"]:
        raise ValueError("prepared data is not accepted")
    for section in ("inputs", "outputs"):
        for rel, item in views[section].items():
            if "sealed_test" in Path(rel).parts or raw_sha(ROOT / rel) != item["sha256"]:
                raise ValueError(f"prepared data identity mismatch: {rel}")
    prep = views["preparation_source_ref"]
    actual = (
        file_sha256(ROOT / prep["uri"])
        if prep["hash_convention"] == "canonical_text"
        else raw_sha(ROOT / prep["uri"])
    )
    if actual != prep["sha256"]:
        raise ValueError("the preparation implementation changed after creating the views")
    for rel, item in views["upstream_config_refs"].items():
        actual = (
            file_sha256(ROOT / rel)
            if item["hash_convention"] == "canonical_text"
            else raw_sha(ROOT / rel)
        )
        if actual != item["sha256"]:
            raise ValueError(f"upstream protocol changed: {rel}")
    return config, views


def choose_candidate(records, tolerance):
    """Prespecified tie rule: earliest candidate within tolerance of the maximum."""
    if not records or any(not np.isfinite(r["selection_value"]) for r in records):
        raise ValueError("a complete finite candidate search is required")
    maximum = max(r["selection_value"] for r in records)
    return next(r for r in records if maximum - r["selection_value"] <= tolerance)


def t2_summary(frame, predictions):
    result = evaluate_t2(
        frame["label_value"].to_numpy(),
        predictions,
        frame["client"].to_numpy(),
        frame["row_ordinal"].to_numpy(),
    )
    counts = frame["train_history_count"].to_numpy()
    buckets = np.asarray([history_bucket(int(x)) for x in counts])
    result["history_buckets"] = {}
    for bucket in ("BELOW_10_RETAINED_C1", "10_19", "20_49", "50_99", "100_plus"):
        mask = buckets == bucket
        result["history_buckets"][bucket] = (
            evaluate_t2(
                frame["label_value"].to_numpy()[mask],
                predictions[mask],
                frame["client"].to_numpy()[mask],
                frame["row_ordinal"].to_numpy()[mask],
            )
            if mask.any()
            else {"status": "ZERO_SUPPORT"}
        )
    return result


def t2_records(summary):
    result = []
    for metric, key, support in (
        ("t2.purchase.pr_auc.micro", "pr_auc_micro", "observed_decisions"),
        (
            "t2.purchase.client_recall_at_3.macro",
            "client_recall_at_3_macro",
            "positive_client_support",
        ),
    ):
        if summary[key] is not None:
            result.append(
                make_metric_record(
                    metric_id=metric,
                    task="T2",
                    cohort="C1",
                    value=float(summary[key]),
                    direction="MAXIMIZE",
                    unit="FRACTION",
                    support=int(summary[support]),
                )
            )
    return result


def publish_trial(
    batch,
    tag,
    task,
    model,
    params,
    summary,
    metrics,
    model_path,
    started,
    snapshot_ref,
    *,
    repeat_of=None,
):
    directory = f"{PUBLIC}/{batch}/{tag}"
    model_config = f"{directory}/model_config.v1.json"
    write(
        model_config,
        {
            "schema": "baseline_model_config_v1",
            "version": "1",
            "model": model,
            "parameters": params,
            "task": task,
            "seed": 13,
            "plan_ref": ref(CONFIG, "classical_session_baseline_plan_v1"),
            "source_snapshot_ref": snapshot_ref,
            "result_role": "VALIDATION_BASELINE_NOT_FINAL_R1",
        },
    )
    init_path = f"{directory}/unfitted_state.v1.json"
    write(
        init_path,
        {
            "schema": "baseline_unfitted_state_v1",
            "version": "1",
            "model": model,
            "seed": 13,
            "preprocessor": "UNFITTED",
            "trained_parameters_loaded": False,
            "state_kind": "UNFITTED_ESTIMATOR_OR_EMPTY_SESSION_INDEX",
            "not_a_neural_common_initialization": True,
        },
    )
    summary_path = f"{directory}/metrics.v1.json"
    write(summary_path, summary)
    views = read("docs/evidence/s2-pr-04-05/data_views.v1.json")
    data_key = (
        f"data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_{task.lower()}_v1.proposed.parquet"
    )
    viewref = ref("docs/evidence/s2-pr-04-05/data_views.v1.json", "baseline_data_views_v1")
    protocol_ref = ref(
        "docs/evidence/s1-ds-05-06/data_protocol_v1.proposed.json", "data_protocol_v1"
    )
    evaluator = "ppsi/baselines/t2_metrics.py" if task == "T2" else "ppsi/evaluation/t1.py"
    cfg = {
        "schema": "experiment_config_v1",
        "version": "1",
        "config_id": f"{batch}_{tag}",
        "regime": "R1",
        "tasks": [task],
        "training_cohort": "C1",
        "evaluation_cohorts": ["C1"],
        "seed": 13,
        "source_dataset_ref": viewref,
        "canonical_data_contract_ref": protocol_ref,
        "cohort_manifest_ref": ref(
            "data/protocol/INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet",
            "cohort_manifest_v1",
        ),
        "split_manifest_ref": protocol_ref,
        "task_examples_manifest_ref": viewref,
        "evaluation_manifest_ref": ref(data_key, "task_example_v1"),
        "representation_ref": ref(
            "config/baselines/t2_feature_manifest.v1.json", "baseline_feature_manifest_v1"
        )
        if task == "T2"
        else ref(CONFIG, "classical_session_baseline_plan_v1"),
        "model_config_ref": ref(model_config, "baseline_model_config_v1"),
        "objective_config_ref": ref(CONFIG, "classical_session_baseline_plan_v1"),
        "shared_trainer_core_ref": ref(
            "scripts/baselines/run_baselines.py", "baseline_fit_entrypoint_v1"
        ),
        "evaluation_protocol_ref": ref(
            "docs/decisions/s2-pr-04-05-baselines.md", "baseline_scope_decision_v1"
        ),
        "evaluator_ref": ref(evaluator, "python_source_v1"),
        "environment_lock_ref": ref("uv.lock", "uv_lock_v1"),
        "initialization": {
            "kind": "COMMON_INITIALIZATION",
            "common_initialization_ref": ref(init_path, "baseline_unfitted_state_v1"),
        },
        "regime_config": {
            "orchestration_type": "CENTRALIZED",
            "result_role": "VALIDATION_BASELINE_NOT_FINAL_R1",
            "evaluation_split": "VALIDATION",
            "model": model,
            "repeat_of": repeat_of,
            "final_r1_denominator": False,
            "quality_retention_claim": False,
            "neural_trainer_used": False,
            "fitted_on": "TRAIN_ONLY",
            "initialization_semantics": "explicit unfitted classical state, not neural weights",
        },
    }
    validate_experiment_config(cfg)
    cfg_path = f"{directory}/experiment_config.v1.json"
    write(cfg_path, cfg)
    result = build_experiment_result(
        experiment_config=cfg,
        config_ref=ref(cfg_path, "experiment_config_v1"),
        git_sha=read(snapshot_ref["uri"])["git_head"],
        state="SUCCEEDED",
        attempt=1,
        started_at_utc=started,
        ended_at_utc=now(),
        metrics=metrics,
        artifacts=[
            ref(summary_path, "baseline_metrics_v1"),
            ref(model_path, "private_fitted_baseline_v1"),
            snapshot_ref,
        ],
        system_measurements={
            "schema": "system_measurement_reference_set_v1",
            "version": "1",
            "status": "NOT_APPLICABLE",
            "null_reason": "baseline quality task; no communication or federated performance claim",
        },
    )
    result_path = f"{directory}/experiment_result.v1.json"
    validate_result_for_reporting(result, source=Path(result_path).name)
    write(result_path, result)
    return {
        "id": tag,
        "model": model,
        "parameters": params,
        "result_ref": ref(result_path, "experiment_result_v1"),
        "metrics_ref": ref(summary_path, "baseline_metrics_v1"),
        "model_ref": ref(model_path, "private_fitted_baseline_v1"),
        "evaluation_source_sha256": views["inputs"][data_key]["sha256"],
    }


def run_classical(batch, config, snapshot_ref):
    destination = ROOT / f"{PRIVATE}/{batch}/classical"
    destination.mkdir(parents=True, exist_ok=False)
    train = pl.read_parquet(ROOT / f"{PRIVATE}/prepared/t2_train.parquet").filter(
        pl.col("task_mask")
    )
    valid = pl.read_parquet(ROOT / f"{PRIVATE}/prepared/t2_validation.parquet").filter(
        pl.col("task_mask")
    )
    if (train.height, valid.height) != (2291753, 322087):
        raise ValueError("frozen observed T2 support changed")
    xtrain = feature_matrix({name: train[name].to_numpy() for name in FEATURES})
    xvalid = feature_matrix({name: valid[name].to_numpy() for name in FEATURES})
    ytrain = train["label_value"].to_numpy().astype(np.int8)
    if not np.array_equal(np.unique(ytrain), [0, 1]):
        raise ValueError("T2 TRAIN is degenerate")
    del train
    gc.collect()
    records, winners = [], []

    def fit_one(candidate, tag, repeat_of=None):
        started = now()
        pipeline = make_pipeline(
            candidate["model"], candidate["params"], seed=config["seed"], threads=config["threads"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with threadpool_limits(limits=config["threads"]):
                pipeline.fit(xtrain, ytrain)
                chunk = config["prediction_chunk_size"]
                predictions = np.concatenate(
                    [
                        positive_probabilities(pipeline, xvalid[i : i + chunk])
                        for i in range(0, len(xvalid), chunk)
                    ]
                )
        model_path = f"{PRIVATE}/{batch}/classical/{tag}.joblib"
        joblib.dump(pipeline, ROOT / model_path, compress=3)
        # Real serialization readback, not just file existence.
        loaded = joblib.load(ROOT / model_path)
        np.testing.assert_allclose(
            positive_probabilities(loaded, xvalid[:1024]), predictions[:1024], atol=1e-12, rtol=0
        )
        np.save(destination / f"{tag}.probabilities.npy", predictions, allow_pickle=False)
        summary = t2_summary(valid, predictions)
        if summary["pr_auc_micro"] is None:
            raise ValueError("selection metric has no support")
        record = publish_trial(
            batch,
            tag,
            "T2",
            candidate["model"],
            candidate["params"],
            summary,
            t2_records(summary),
            model_path,
            started,
            snapshot_ref,
            repeat_of=repeat_of,
        )
        record["selection_value"] = summary["pr_auc_micro"]
        print(f"T2 {tag}: AP={record['selection_value']:.8f}", flush=True)
        del loaded, pipeline
        gc.collect()
        return record, predictions

    for candidate in config["classical_candidates"]:
        record, _ = fit_one(candidate, candidate["id"])
        records.append(record)
        write(
            f"{PUBLIC}/{batch}/classical_progress.v1.json",
            {"status": "IN_PROGRESS", "trials": records},
            replace=True,
        )
    for family in ("logistic_regression", "lightgbm"):
        selected = choose_candidate(
            [r for r in records if r["model"] == family], config["selection_tie_tolerance"]
        )
        candidate = next(c for c in config["classical_candidates"] if c["id"] == selected["id"])
        repeat, predictions = fit_one(candidate, selected["id"] + "_repeat", selected["id"])
        original = np.load(destination / f"{selected['id']}.probabilities.npy", allow_pickle=False)
        diff = float(np.max(np.abs(predictions - original)))
        if diff > config["winner_repeat_prediction_atol"]:
            raise ValueError(f"winner rerun not reproducible: {family}, {diff}")
        winners.append({"selected": selected, "repeat": repeat, "max_abs_probability_diff": diff})
    report = {
        "schema": "classical_baseline_summary_v1",
        "version": "1",
        "status": "PASS",
        "batch": batch,
        "task": "T2",
        "training_observed_rows": len(ytrain),
        "validation_observed_rows": valid.height,
        "trials": records,
        "winners": winners,
        "source_snapshot_ref": snapshot_ref,
        "selection_uses_validation": True,
        "test_consumed": False,
        "limitations": [
            "validation-selected baselines, not held-out TEST performance",
            "not the final neural R1 denominator",
            "no claim that probabilities are calibrated",
            "all four raw-data/task-label sources remain unchanged",
        ],
    }
    write(f"{PUBLIC}/{batch}/classical_summary.v1.json", report)
    write(
        f"{PUBLIC}/{batch}/classical_progress.v1.json",
        {"status": "COMPLETE", "trials": records},
        replace=True,
    )


def run_session(batch, config, snapshot_ref):
    settings = config["session_knn"]
    destination = ROOT / f"{PRIVATE}/{batch}/session"
    destination.mkdir(parents=True, exist_ok=False)
    sessions = pl.read_parquet(ROOT / f"{PRIVATE}/prepared/train_sessions.parquet")
    queries = pl.read_parquet(ROOT / f"{PRIVATE}/prepared/t1_validation.parquet")
    if (sessions.height, queries.height) != (788317, 438185):
        raise ValueError("frozen session or T1 evaluation count changed")
    index = SessionKNN(category_count=588, candidate_limit=settings["candidate_limit"])
    started = now()
    index.fit(
        sessions["session_key"].to_list(),
        sessions["end_time_ns"].to_list(),
        sessions["items"].to_list(),
        sessions["categories"].to_list(),
        np.load(ROOT / f"{PRIVATE}/prepared/t1_popularity.npy", allow_pickle=False),
        split="TRAIN",
    )
    model_path = f"{PRIVATE}/{batch}/session/train_index.joblib"
    joblib.dump(index, ROOT / model_path, compress=3)
    del sessions
    gc.collect()
    query_items = queries["query_items"].to_list()
    targets = queries["label_value"].to_numpy().astype(np.int64)
    ks = settings["ks"]
    # A label-blind feasibility probe only; it never changes k, membership or features.
    count = min(settings["feasibility_queries"], len(query_items))
    begin = time.perf_counter()
    for items in query_items[:count]:
        index.score_many_k(items, ks)
    projected = (time.perf_counter() - begin) / count * len(query_items)
    write(
        f"{PUBLIC}/{batch}/session_feasibility.v1.json",
        {
            "probe_queries": count,
            "projected_seconds": projected,
            "measurement_kind": "ESTIMATE_NOT_FULL_RUNTIME",
        },
    )
    if projected > settings["max_projected_evaluation_seconds"]:
        raise RuntimeError("BLOCKED_RUNTIME: do not shrink frozen evaluation membership")
    ranks = {k: np.empty(len(query_items), dtype=np.int64) for k in ks}
    cache = OrderedDict()
    readback_scores = []
    backoff_count = 0
    import torch

    torch.set_num_threads(config["threads"])
    rows_per_batch = settings["rank_batch_rows"]
    for begin in range(0, len(query_items), rows_per_batch):
        end = min(len(query_items), begin + rows_per_batch)
        scores = {k: np.empty((end - begin, 588), dtype=np.float64) for k in ks}
        for offset, items in enumerate(query_items[begin:end]):
            key = tuple(sorted(set(items)))
            if key in cache:
                predictions, diagnostic = cache.pop(key)
            else:
                predictions, diagnostic = index.score_many_k(items, ks)
            cache[key] = (predictions, diagnostic)
            if len(cache) > settings["cache_entries"]:
                cache.popitem(last=False)
            backoff_count += int(diagnostic["backoff"])
            for k in ks:
                scores[k][offset] = predictions[k]
            if begin + offset < settings["readback_query_count"]:
                readback_scores.append({k: v.copy() for k, v in predictions.items()})
        for k in ks:
            ranks[k][begin:end] = (
                rank_targets_from_scores(scores[k], targets[begin:end]).cpu().numpy()
            )
        if begin % (rows_per_batch * 100) == 0:
            print(f"T1 evaluated {end}/{len(query_items)}", flush=True)
    records = []
    for k in ks:
        # Combine ranks first, then aggregate ONCE; never average batch macros.
        summary = evaluate_t1_ranks(
            ranks[k],
            queries["category_changed"].to_list(),
            queries["client"].to_list(),
            queries["train_history_count"].to_list(),
        )
        summary_dict = summary.to_dict()
        np.save(destination / f"k{k}.ranks.npy", ranks[k], allow_pickle=False)
        record = publish_trial(
            batch,
            f"sknn_k{k}",
            "T1",
            "recent_session_cosine_category_knn",
            {**settings, "k": k},
            summary_dict,
            metric_records_from_t1_summary(summary),
            model_path,
            started,
            snapshot_ref,
        )
        record["selection_value"] = summary_dict["slices"]["next_distinct"]["mrr_at_20_macro"]
        records.append(record)
    winner = choose_candidate(records, config["selection_tie_tolerance"])
    del index, cache
    gc.collect()
    loaded = joblib.load(ROOT / model_path)
    for items, expected in zip(query_items[: len(readback_scores)], readback_scores, strict=True):
        actual, _ = loaded.score_many_k(items, ks)
        for k in ks:
            np.testing.assert_array_equal(actual[k], expected[k])
    report = {
        "schema": "session_knn_summary_v1",
        "version": "1",
        "status": "PASS",
        "batch": batch,
        "task": "T1",
        "training_sessions": 788317,
        "validation_decisions": queries.height,
        "index_ref": ref(model_path, "private_session_index_v1"),
        "trials": records,
        "winner": winner,
        "source_snapshot_ref": snapshot_ref,
        "no_neighbor_queries": backoff_count,
        "index_readback_reproduced_queries": len(readback_scores),
        "test_consumed": False,
        "validation_indexed": False,
        "limitations": [
            "category-projected session-kNN adaptation, not an exact reproduction of an item scorer",
            "recent 2000-session retrieval pool, not exhaustive nearest neighbors over all TRAIN sessions",
            "validation-selected baseline, not final held-out TEST performance",
        ],
    }
    write(f"{PUBLIC}/{batch}/session_summary.v1.json", report)


def verify_batch(batch):
    directory = ROOT / PUBLIC / batch
    required = [
        directory / "source_snapshot.v1.json",
        directory / "classical_summary.v1.json",
        directory / "session_summary.v1.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path.name)
    for name in ("classical_summary.v1.json", "session_summary.v1.json"):
        if read(f"{PUBLIC}/{batch}/{name}").get("status") != "PASS":
            raise ValueError(f"batch is incomplete: {name}")
    for rel, sha in read(f"{PUBLIC}/{batch}/source_snapshot.v1.json")["files_sha256"].items():
        if file_sha256(ROOT / rel) != sha:
            raise ValueError(f"execution source changed: {rel}")
    results, checked = set(), set()

    def visit(data):
        if isinstance(data, dict):
            if "uri" in data and "sha256" in data:
                pair = (data["uri"], data["sha256"])
                if pair not in checked:
                    if file_sha256(ROOT / pair[0]) != pair[1]:
                        raise ValueError(f"artifact identity mismatch: {pair[0]}")
                    checked.add(pair)
                if data.get("artifact_schema") == "experiment_result_v1":
                    results.add(data["uri"])
            for value in data.values():
                visit(value)
        elif isinstance(data, list):
            for value in data:
                visit(value)

    public_files = [p for p in directory.rglob("*.json") if p.name != "verification.v1.json"]
    for name in ("S2_PR_04_T2_Classical_Baselines.ipynb", "S2_PR_05_T1_Session_KNN.ipynb"):
        notebook = ROOT / "notebooks" / name
        if not notebook.is_file():
            raise FileNotFoundError(name)
        public_files.append(notebook)
    for path in public_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        visit(data)
    if len(results) != 12:  # 7 search fits, 2 selected reruns, 3 kNN settings
        raise ValueError(f"expected 12 full result records, got {len(results)}")
    for rel in sorted(results):
        record = read(rel)
        validate_result_for_reporting(record, source=Path(rel).name)
        cfg_ref = record["config_ref"]
        if file_sha256(ROOT / cfg_ref["uri"]) != cfg_ref["sha256"]:
            raise ValueError("result/config reference mismatch")
        visit(record)
        public_files.append(ROOT / rel)
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        if re.search(
            r"client-v1-[0-9a-f]{64}|\"(?:selected_client_ids|client_id|user_id)\"\s*:", text
        ):
            raise ValueError(f"private identity leaked: {path.name}")
    report = {
        "status": "PASS",
        "result_count": len(results),
        "checked_references": len(checked),
        "public_files_scanned": len(public_files),
        "result_paths": sorted(results),
        "no_test_claim_scope": "commands in this batch use TRAIN/VALIDATION only; no historical-access assertion",
    }
    write(f"{PUBLIC}/{batch}/verification.v1.json", report, replace=True)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=("preflight", "classical", "session", "verify")
    )
    parser.add_argument("--batch", default="batch-001")
    args = parser.parse_args()
    if re.fullmatch(r"batch-[0-9]{3}", args.batch) is None:
        raise ValueError("batch must have form batch-001; new batches preserve earlier attempts")
    config, _ = preflight()
    if args.stage == "preflight":
        print("BASELINE_PREFLIGHT_PASS")
        return
    if args.stage == "verify":
        verify_batch(args.batch)
        return
    snapshot_path = f"{PUBLIC}/{args.batch}/source_snapshot.v1.json"
    write(snapshot_path, source_snapshot())
    snapshot_ref = ref(snapshot_path, "baseline_source_snapshot_v1")
    try:
        (run_classical if args.stage == "classical" else run_session)(
            args.batch, config, snapshot_ref
        )
    except Exception as exc:
        # Do not serialize raw error strings: they can contain private identities.
        write(
            f"{PUBLIC}/{args.batch}/{args.stage}_failure.v1.json",
            {
                "status": "FAILED",
                "stage": args.stage,
                "exception_type": type(exc).__name__,
                "action": "inspect private execution log; preserve batch, never silently retry",
            },
        )
        raise


if __name__ == "__main__":
    main()
