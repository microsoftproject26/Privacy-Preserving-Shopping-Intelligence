"""Tests for consistency between JSON evidence files and final reports/addenda."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "docs/evidence/mvp"
SUBMISSION_DIR = ROOT / "artifacts/mvp/submission"


@pytest.fixture
def closeout_json():
    json_path = EVIDENCE_DIR / "final-t2-t3-closeout.json"
    assert json_path.exists(), f"Missing {json_path}"
    return json.loads(json_path.read_text(encoding="utf-8"))


def test_t1_canonical_invariants(closeout_json):
    t1 = closeout_json["t1"]
    assert t1["status"] == "CANONICAL_SCOPED_MVP"
    assert t1["run_id"] == "mvp-t1-003"
    assert pytest.approx(t1["r1"], abs=1e-6) == 0.11268339009749091
    assert pytest.approx(t1["r2a"], abs=1e-6) == 0.09959066403653290
    assert pytest.approx(t1["delta"], abs=1e-6) == -0.013092726060958015
    assert pytest.approx(t1["quality_retention"], abs=1e-3) == 0.8838
    assert t1["validation_decisions"] == 438185
    assert t1["headline_decisions"] == 77457
    assert t1["headline_clients"] == 16096


def test_t3_final_component_invariants(closeout_json):
    t3 = closeout_json["t3"]
    assert t3["status"] == "NOT_APPLICABLE_SHARED_FIXED_COMPONENT"
    assert t3["component"] == "cooccurrence_then_popularity"
    assert pytest.approx(t3["macro_ndcg_at_5"], abs=1e-4) == 0.2707
    assert pytest.approx(t3["micro_ndcg_at_5"], abs=1e-4) == 0.2046
    assert t3["evaluable_queries"] == 18814
    assert t3["clients"] == 4591
    assert pytest.approx(t3["candidate_recall"], abs=1e-4) == 0.8159
    assert t3["corrected_candidate_rows"] == 10258299
    assert t3["anchors_total"] == 107484
    assert t3["reported_sha256"] == "8892029ED48AEA056F6357811A53E3D74F5F9AC7C5572057741477D256B854C3"


def test_t2_evidence_consistency(closeout_json):
    t2 = closeout_json["t2"]
    assert "r1" in t2 and t2["r1"] > 0
    assert "r2a" in t2 and t2["r2a"] > 0
    assert pytest.approx(t2["delta"], abs=1e-4) == (t2["r2a"] - t2["r1"])
    assert pytest.approx(t2["quality_retention"], abs=1e-4) == (t2["r2a"] / t2["r1"])


@pytest.mark.skipif(
    not (SUBMISSION_DIR / "FINAL_MVP_RESULTS.json").exists(),
    reason=(
        "The submission bundle lives under artifacts/, which .gitignore excludes on purpose: "
        "it is a local deliverable, not repository content, so a clean checkout never has it. "
        "When the bundle is present this test still checks it strictly."
    ),
)
def test_submission_files_exist_and_consistent(closeout_json):
    results_json = SUBMISSION_DIR / "FINAL_MVP_RESULTS.json"
    assert results_json.exists()
    sub_data = json.loads(results_json.read_text(encoding="utf-8"))
    assert sub_data == closeout_json

    addendum_path = SUBMISSION_DIR / "FINAL_MVP_T2_T3_ADDENDUM.md"
    assert addendum_path.exists()
    content = addendum_path.read_text(encoding="utf-8")
    assert "0.2707" in content
    assert "0.1127" in content or "0.11268" in content or f"{closeout_json['t1']['r1']:.4f}" in content
    assert f"{closeout_json['t2']['r1']:.4f}" in content
