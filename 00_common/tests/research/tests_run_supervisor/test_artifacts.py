"""Expected-artifact validation: missing/empty artifacts fail (a phase that exits 0 but writes
nothing must fail), a NaN/Infinity literal in a JSON result is refused (T45), and an artifact path
that would escape the workspace is refused (T44).
"""
from __future__ import annotations

import json

import pytest

from v4sup.artifacts import parse_strict_json, validate_artifacts
from v4sup.registry import ExpectedArtifact


def test_valid_json_artifact_passes(workspace):
    (workspace / "out.json").write_text(json.dumps({"metric": 0.5}))
    ok, checks = validate_artifacts(workspace, (ExpectedArtifact(path="out.json"),))
    assert ok
    assert checks[0].ok


def test_missing_artifact_fails(workspace):
    """The core of 'a fake phase that exits 0 but writes nothing must fail'."""
    ok, checks = validate_artifacts(workspace, (ExpectedArtifact(path="out.json"),))
    assert not ok
    assert "missing" in checks[0].reason


def test_empty_artifact_fails(workspace):
    (workspace / "out.json").write_text("")
    ok, checks = validate_artifacts(workspace, (ExpectedArtifact(path="out.json"),))
    assert not ok
    assert "empty" in checks[0].reason


def test_non_json_text_fails_for_json_artifact(workspace):
    (workspace / "out.json").write_text("not json at all")
    ok, checks = validate_artifacts(workspace, (ExpectedArtifact(path="out.json"),))
    assert not ok


def test_nan_literal_refused():
    """T45: invalid JSON/NaN serialization cannot leave a success marker. Python's json module is
    permissive by default and happily parses the bare word NaN; parse_strict_json must not."""
    with pytest.raises(ValueError):
        parse_strict_json('{"metric": NaN}')


def test_infinity_literal_refused():
    with pytest.raises(ValueError):
        parse_strict_json('{"metric": Infinity}')
    with pytest.raises(ValueError):
        parse_strict_json('{"metric": -Infinity}')


def test_ordinary_json_still_parses():
    assert parse_strict_json('{"metric": 0.5, "n": 3}') == {"metric": 0.5, "n": 3}


def test_nan_artifact_marks_phase_not_done(workspace):
    """A phase's result.json containing a literal NaN must never validate, so the phase can never
    be reported DONE -- this is what T45 requires at the artifact-validation boundary."""
    (workspace / "result.json").write_text('{"macro_mrr": NaN}')
    ok, checks = validate_artifacts(workspace, (ExpectedArtifact(path="result.json"),))
    assert not ok
    assert "strict JSON" in checks[0].reason


def test_finite_but_unusual_numbers_still_pass(workspace):
    (workspace / "result.json").write_text(json.dumps({"macro_mrr": 0.0, "big": 1e300}))
    ok, checks = validate_artifacts(workspace, (ExpectedArtifact(path="result.json"),))
    assert ok


def test_non_json_artifact_skips_strict_parse(workspace):
    (workspace / "checkpoint.bin").write_bytes(b"\x00\x01binary-not-json")
    ok, checks = validate_artifacts(workspace, (ExpectedArtifact(path="checkpoint.bin", json_strict=False),))
    assert ok


def test_artifact_path_escape_refused(workspace):
    """T44 at the artifact-validation boundary: an expected_artifacts path that would resolve
    outside the workspace is refused, not silently followed."""
    ok, checks = validate_artifacts(workspace, (ExpectedArtifact(path="../outside.json"),))
    assert not ok
    assert "workspace escape" in checks[0].reason


def test_min_bytes_enforced(workspace):
    (workspace / "out.json").write_text("{}")
    ok, checks = validate_artifacts(workspace, (ExpectedArtifact(path="out.json", min_bytes=1000),))
    assert not ok
    assert "truncated" in checks[0].reason or "empty" in checks[0].reason


def test_multiple_artifacts_all_must_pass(workspace):
    (workspace / "a.json").write_text(json.dumps({"x": 1}))
    # b.json intentionally absent
    ok, checks = validate_artifacts(workspace, (ExpectedArtifact(path="a.json"), ExpectedArtifact(path="b.json")))
    assert not ok
    by_path = {c.path: c for c in checks}
    assert by_path["a.json"].ok
    assert not by_path["b.json"].ok
