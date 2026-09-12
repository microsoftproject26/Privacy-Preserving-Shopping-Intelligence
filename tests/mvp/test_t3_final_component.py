"""Tests for T3 final component evidence and reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
T3_EVIDENCE_DIR = ROOT / "docs/evidence/mvp/t3-final-component-001"


@pytest.fixture
def t3_evidence():
    assert T3_EVIDENCE_DIR.exists(), "T3 evidence directory does not exist"
    return {
        "preflight": json.loads((T3_EVIDENCE_DIR / "preflight.json").read_text(encoding="utf-8")),
        "retrieval": json.loads((T3_EVIDENCE_DIR / "retrieval_component.json").read_text(encoding="utf-8")),
        "reranker": json.loads((T3_EVIDENCE_DIR / "reranker_rejection.json").read_text(encoding="utf-8")),
        "artifact": json.loads((T3_EVIDENCE_DIR / "artifact_verification.json").read_text(encoding="utf-8")),
        "contract": json.loads((T3_EVIDENCE_DIR / "system_contract.json").read_text(encoding="utf-8")),
    }


def test_t3_retrieval_metrics(t3_evidence):
    retrieval = t3_evidence["retrieval"]
    assert retrieval["component"] == "cooccurrence_then_popularity"
    assert pytest.approx(retrieval["macro_ndcg_at_5"], abs=1e-4) == 0.2707
    assert pytest.approx(retrieval["micro_ndcg_at_5"], abs=1e-4) == 0.2046
    assert retrieval["evaluable_queries"] == 18814
    assert retrieval["clients"] == 4591
    assert pytest.approx(retrieval["candidate_recall"], abs=1e-4) == 0.8159


def test_t3_reranker_rejection(t3_evidence):
    reranker = t3_evidence["reranker"]
    assert pytest.approx(reranker["listwise"]["macro_ndcg_at_5"], abs=1e-4) == 0.2505
    assert reranker["listwise"]["beats_baseline"] is False
    assert pytest.approx(reranker["pointwise"]["macro_ndcg_at_5"], abs=1e-4) == 0.2186
    assert reranker["pointwise"]["beats_baseline"] is False
    assert reranker["query_aware"]["beats_acceptance_bar"] is False


def test_t3_system_contract(t3_evidence):
    contract = t3_evidence["contract"]
    assert contract["component_type"] == "SHARED_FIXED_RETRIEVAL"
    assert contract["trainable_under_r1"] is False
    assert contract["trainable_under_r2a"] is False
    assert contract["same_component_in_both_variants"] is True
    assert contract["quality_retention"] is None
    assert contract["quality_retention_status"] == "NOT_APPLICABLE_SHARED_FIXED_COMPONENT"


def test_t3_candidate_artifact_provenance(t3_evidence):
    artifact = t3_evidence["artifact"]
    assert artifact["rows"] == 10258299
    assert artifact["anchors_total"] == 107484
    assert artifact["anchors_train"] == 102282
    assert artifact["reported_sha256"] == "8892029ED48AEA056F6357811A53E3D74F5F9AC7C5572057741477D256B854C3"
    assert artifact["frozen_lists_reproduce_exactly"] is True
    assert artifact["validation_baseline_unchanged"] is True
