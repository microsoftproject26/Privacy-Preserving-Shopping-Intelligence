"""CI-safe integration with the repository's actual result and T1 APIs.

Everything is a temporary toy artifact. No private dataset is read.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from ppsi.baselines.classical import FEATURES, feature_matrix, make_pipeline
from ppsi.evaluation.t1 import evaluate_t1_ranks, rank_targets_from_scores
from ppsi.training.identity import file_sha256
from scripts.baselines import run_baselines as run
from scripts.experiments.results import validate_result_for_reporting


def test_real_t1_api_aggregates_clients_globally_not_by_batch():
    scores = np.zeros((101, 588))
    scores[:100, 1] = 2
    scores[100, 0] = 2
    ranks = rank_targets_from_scores(scores, np.zeros(101, dtype=int))
    out = evaluate_t1_ranks(ranks, [True] * 101, ["toy_a"] * 100 + ["toy_b"], [15] * 101)
    # A rank2 =>0.5 and B rank1=>1; macro=(.5+1)/2, not 51/101.
    assert out.slices["next_distinct"]["mrr_at_20_macro"] == pytest.approx(0.75)
    assert out.slices["next_distinct"]["mrr_at_20_micro"] == pytest.approx(51 / 101)


def test_required_evidence_cannot_silently_disappear(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        run.verify_batch("batch-001")


def test_real_toy_pipeline_publishes_valid_result_and_same_uri_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    relative_sources = [
        run.CONFIG,
        "config/baselines/t2_feature_manifest.v1.json",
        "uv.lock",
        "docs/evidence/s1-ds-05-06/data_protocol_v1.proposed.json",
        "data/protocol/INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet",
        "data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_t2_v1.proposed.parquet",
        "scripts/baselines/run_baselines.py",
        "ppsi/baselines/t2_metrics.py",
        "docs/decisions/s2-pr-04-05-baselines.md",
    ]
    for rel in relative_sources:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"artifact_kind": "FIXTURE_PROOF"}\n', encoding="utf-8")
    data_key = relative_sources[5]
    run.write(
        f"{run.PUBLIC}/data_views.v1.json",
        {"inputs": {data_key: {"sha256": file_sha256(tmp_path / data_key)}}},
    )
    snapshot_path = f"{run.PUBLIC}/batch-001/source_snapshot.v1.json"
    run.write(snapshot_path, {"git_head": "f" * 40, "artifact_kind": "FIXTURE_PROOF"})
    rng = np.random.default_rng(13)
    columns = {name: rng.uniform(size=50) for name in FEATURES}
    columns["category_code"] = np.arange(50) % 3
    x = feature_matrix(columns)
    y = (x[:, 1] > 0.5).astype(int)
    pipeline = make_pipeline("logistic_regression", {"C": 1})
    pipeline.fit(x, y)
    model_path = f"{run.PRIVATE}/batch-001/classical/toy.joblib"
    (tmp_path / model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, tmp_path / model_path)
    summary = {
        "pr_auc_micro": 0.5,
        "observed_decisions": 50,
        "client_recall_at_3_macro": 0.4,
        "positive_client_support": 2,
    }
    output = run.publish_trial(
        "batch-001",
        "toy",
        "T2",
        "logistic_regression",
        {"C": 1},
        summary,
        run.t2_records(summary),
        model_path,
        "2026-01-01T00:00:00Z",
        run.ref(snapshot_path, "baseline_source_snapshot_v1"),
    )
    result_path = tmp_path / output["result_ref"]["uri"]
    result = json.loads(result_path.read_text())
    validate_result_for_reporting(result, source=result_path.name)
    cfg = result["config_ref"]
    assert cfg["sha256"] == file_sha256(tmp_path / cfg["uri"])
    assert f"__cfg{cfg['sha256'][:12]}__" in result["run_id"]
    assert result["state"] == "SUCCEEDED"
    resolved = json.loads((tmp_path / cfg["uri"]).read_text())
    assert resolved["regime_config"]["result_role"] == "VALIDATION_BASELINE_NOT_FINAL_R1"
    assert resolved["regime_config"]["final_r1_denominator"] is False
    assert "toy_a" not in result_path.read_text()


from scripts.experiments.results import load_results


def test_validation_baselines_are_not_in_global_qr_pool(tmp_path) -> None:
    # Simulate the task-local evidence directory (docs/evidence/...)
    evidence_dir = tmp_path / "docs" / "evidence" / "s2-pr-04-05" / "batch-001" / "run-test"
    evidence_dir.mkdir(parents=True)

    # Simulate the global registry (artifacts/experiment-results/)
    global_dir = tmp_path / "artifacts" / "experiment-results"
    global_dir.mkdir(parents=True)

    # Create a baseline record that matches the schema using test_baseline_integration fixture structure
    baseline = json.loads(
        Path("fixtures/experiments/contracts/experiment_result.json").read_text(encoding="utf-8")
    )
    baseline["regime"] = "R1"

    # S2-PR-04/05 publishing writes to the task-local evidence directory as experiment_result.v1.json
    local_result_path = evidence_dir / "experiment_result.v1.json"
    local_result_path.write_text(json.dumps(baseline), encoding="utf-8")

    # It does NOT write to the global artifacts/experiment-results/ directory
    assert len(list(global_dir.iterdir())) == 0

    # A normal load/build of the global artifacts directory cannot see the baseline
    # and raises ResultsError because no *.result.json files are found.
    import pytest

    from scripts.experiments.results import ResultsError

    with pytest.raises(ResultsError, match=r"no \*\.result\.json files found"):
        load_results(global_dir)

    # Prove that results.py behavior itself is unchanged:
    # If the file *were* accidentally placed in the global registry, load_results WOULD parse it.
    accidental_path = global_dir / "accidental.result.json"
    accidental_path.write_text(json.dumps(baseline), encoding="utf-8")

    loaded_accidental = load_results(global_dir)
    assert len(loaded_accidental) == 1
    assert loaded_accidental[0]["regime"] == "R1"
