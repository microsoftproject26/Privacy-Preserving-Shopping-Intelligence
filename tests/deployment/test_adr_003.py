from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ADR = REPO / "docs" / "adr" / "ADR-003-deployment-measurement-readiness.md"
FINAL = REPO / "docs" / "evidence" / "s2-se-08" / "final_export.v1.json"
INT8 = REPO / "docs" / "evidence" / "s2-se-02" / "int8_benchmark.v1.json"
BYTES = REPO / "docs" / "evidence" / "s2-se-05" / "communication_bytes.v1.json"
COLLISIONS = REPO / "docs" / "evidence" / "s2-se-03" / "product_hash_collisions.v1.json"


@pytest.fixture(scope="module")
def adr() -> str:
    return ADR.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def final_export() -> dict:
    return json.loads(FINAL.read_text(encoding="utf-8"))


def test_the_decision_matches_what_the_measurements_support(adr, final_export) -> None:
    # An ADR that says FP32 while the evidence says INT8, or the reverse, is worse than
    # no ADR. The decision has to follow from the file it cites.
    quantization = final_export["quantization"]
    accepted = quantization["meets_size_rule"] and quantization["meets_speed_rule"]

    assert final_export["recommended_artifact"] == ("INT8" if accepted else "FP32")
    assert "FP32 is the Phase 1 deployment artifact" in adr
    assert "INT8 is not accepted" in adr
    assert not accepted


def test_every_cited_evidence_file_exists(adr) -> None:
    for path in (FINAL, INT8, BYTES, COLLISIONS):
        cited = path.relative_to(REPO).as_posix()
        assert cited in adr, f"the ADR should cite {cited}"
        assert path.is_file()


def test_the_quoted_sizes_and_candidate_match_the_evidence(adr, final_export) -> None:
    assert f"{final_export['fp32']['serialized_bytes']:,}" in adr
    assert f"{final_export['quantization']['int8']['serialized_bytes']:,}" in adr
    assert final_export["candidate"]["deployment_candidate_checkpoint_id"] in adr


def test_the_quoted_parity_matches_the_evidence(adr, final_export) -> None:
    for value in final_export["parity"]["max_absolute_difference"].values():
        assert f"{value:.2e}" in adr or value == 0.0
    assert final_export["parity"]["within_tolerance"] is True


def test_the_adr_does_not_present_the_quality_condition_as_passed(adr) -> None:
    assert "Unanswered" in adr
    assert "unanswered, not passed" in adr
    assert "not a metric delta" in adr


def test_the_adr_states_the_weights_caveat(adr, final_export) -> None:
    # The committed evidence was produced without trained weights. An ADR that quoted
    # these numbers without saying so would overstate what was verified.
    assert final_export["candidate"]["weights_loaded"] is False
    flat = " ".join(adr.split())
    assert "shipped architecture and not the shipped model" in flat


def test_the_adr_repeats_no_privacy_claim_the_code_does_not_support(adr) -> None:
    # Whitespace-insensitive: the ADR is hard-wrapped, so a phrase can straddle a line
    # break. The assertion is about the sentence being present, not about where it wraps.
    flat = " ".join(adr.lower().split())

    assert "is not physical device isolation" in flat
    assert "differential privacy and secure aggregation are not implemented" in flat
    assert "hashed or namespaced identifiers are not anonymous" in flat
