from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ppsi.deployment.onnx_export import INPUT_NAMES, OUTPUT_NAMES

REPO = Path(__file__).resolve().parents[2]
DEMO_SPEC = REPO / "docs" / "demo-spec.md"
THREAT_MODEL = REPO / "docs" / "security" / "threat-model.md"
SMOKE_FIXTURE = REPO / "fixtures" / "smoke_sample.parquet"


@pytest.fixture(scope="module")
def demo_spec() -> str:
    return DEMO_SPEC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def threat_model() -> str:
    return THREAT_MODEL.read_text(encoding="utf-8")


def test_every_documented_output_is_an_actual_output(demo_spec) -> None:
    for name in OUTPUT_NAMES:
        assert f"`{name}`" in demo_spec


def test_every_input_family_the_graph_takes_is_described(demo_spec) -> None:
    for name in INPUT_NAMES:
        stem = name.split("_")[0]
        assert stem in demo_spec, f"{name} is an input to the graph but the demo spec never mentions it"


def test_the_documented_smoke_fixture_counts_match_the_file(threat_model) -> None:
    # The public-repository section names exact counts. A fixture that is regenerated
    # without updating them would leave the threat model quietly wrong about what is
    # published, which is the one thing this section exists to get right.
    frame = pl.read_parquet(SMOKE_FIXTURE, columns=["user_id", "session_id"])

    assert f"{frame.height:,}" in threat_model
    assert f"{frame.get_column('user_id').n_unique():,}" in threat_model
    assert f"{frame.get_column('session_id').n_unique():,}" in threat_model


def test_the_threat_model_states_the_repository_is_public(threat_model) -> None:
    assert "The repository is public." in threat_model
    assert "Public-repository boundary" in threat_model


def test_no_claim_the_implementation_does_not_support(demo_spec) -> None:
    lowered = demo_spec.lower()
    for forbidden in ("differential privacy is implemented", "secure aggregation is implemented"):
        assert forbidden not in lowered
    assert "not physical device isolation" in lowered


def test_the_time_gap_cap_in_the_spec_matches_the_code(demo_spec) -> None:
    from ppsi.features.price_time import TIME_GAP_CAP_SECONDS

    assert str(int(TIME_GAP_CAP_SECONDS)) in demo_spec
