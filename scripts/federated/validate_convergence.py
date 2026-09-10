"""S2-PR-02 convergence fixture validation CLI.

Executes the fixed JSON oracle cases, the numeric boundary cases and the
hand-worked byte oracle, then writes deterministic public evidence.

This validator proves arithmetic only. It never authenticates a scientific R1
value and never measures Flower traffic: both stay explicitly pending.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ppsi.federated.convergence import (
    analyze_curve,
    sum_measured_communication,
)
from ppsi.training.identity import file_sha256

_EVIDENCE_PATH = Path("docs/evidence/s2-pr-02/convergence_validation.v1.json")
_FIXTURE_PATH = Path("fixtures/federated/convergence_cases.v1.json")
_MODULE_PATH = Path("ppsi/federated/convergence.py")


class ConvergenceValidationError(RuntimeError):
    """Raised when a fixed oracle case does not reproduce its recorded answer."""


def _curve(rounds: list[int], values: list[Any]) -> list[dict[str, Any]]:
    return [{"server_round": r, "value": v} for r, v in zip(rounds, values, strict=True)]


def run_fixture_cases(fixture: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay every fixed case and compare against its recorded expectation."""
    results: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        analysis = analyze_curve(
            _curve(case["rounds"], case["values"]),
            cadence_rounds=case.get("cadence_rounds", policy["evaluation_cadence_rounds"]),
            self_fraction=policy["self_fraction"],
            consecutive=policy["self_consecutive_evaluations"],
            target_fraction=policy["target_fraction"],
            r1_value=case.get("r1_value"),
        )
        self_result = analysis["self_convergence"]
        target_result = analysis["r1_target"]
        actual = {
            "self_status": self_result["status"],
            "first_round": self_result["first_qualifying_round"],
            "confirmation_round": self_result["confirmation_round"],
            "target_status": target_result["status"],
            "target_round": target_result["crossing_round"],
        }
        expected = {
            "self_status": case["expected_self_status"],
            "first_round": case["expected_first_round"],
            "confirmation_round": case["expected_confirmation_round"],
            "target_status": case["expected_target_status"],
            "target_round": case["expected_target_round"],
        }
        if actual != expected:
            raise ConvergenceValidationError(
                f"case {case['case_id']!r} expected {expected} but produced {actual}"
            )
        results.append({"case_id": case["case_id"], "match": True, **actual})
    return results


def run_byte_oracle(fixture: dict[str, Any]) -> dict[str, Any]:
    """Replay the hand-worked byte oracle; this is not measured Flower traffic."""
    oracle = fixture["byte_oracle"]
    run_id = "convergence-fixture-run"
    config_sha256 = "a" * 64
    records = [
        {
            "server_round": row["round"],
            "upload_bytes": row["upload"],
            "download_bytes": row["download"],
            "run_id": run_id,
            "config_sha256": config_sha256,
            "measurement_basis": "MEASURED_APPLICATION_MODEL_PAYLOAD",
        }
        for row in oracle["rounds"]
    ]
    summed = sum_measured_communication(
        records,
        through_round=oracle["through_round"],
        expected_run_id=run_id,
        expected_config_sha256=config_sha256,
    )
    expected = {
        "status": "AVAILABLE",
        "upload_bytes": oracle["expected_upload"],
        "download_bytes": oracle["expected_download"],
        "total_bytes": oracle["expected_total"],
    }
    actual = {key: summed[key] for key in expected}
    if actual != expected:
        raise ConvergenceValidationError(f"byte oracle expected {expected} but produced {actual}")
    return {
        "match": True,
        "artifact_kind": "FIXTURE_PROOF",
        "actual_measured_flower_traffic": bool(oracle["actual_measured_flower_traffic"]),
        **actual,
    }


def run_boundary_cases(policy: dict[str, Any]) -> list[dict[str, Any]]:
    """Assert the numeric edges that separate this policy from a looser one."""
    checks: list[dict[str, Any]] = []

    def record(check_id: str, passed: bool) -> None:
        if not passed:
            raise ConvergenceValidationError(f"boundary case {check_id!r} failed")
        checks.append({"check_id": check_id, "match": True})

    fraction = policy["self_fraction"]
    consecutive = policy["self_consecutive_evaluations"]

    # Exactly-at-threshold observations count; the comparison is inclusive.
    inclusive = analyze_curve(
        _curve([1, 2, 3], [fraction, fraction, 1.0]),
        self_fraction=fraction,
        consecutive=consecutive,
    )
    record("inclusive_threshold_counts", inclusive["self_convergence"]["status"] == "REACHED")

    # One observation below threshold restarts the streak.
    broken = analyze_curve(
        _curve([1, 2, 3, 4], [1.0, 0.5, 1.0, 1.0]),
        self_fraction=fraction,
        consecutive=consecutive,
    )
    record("below_threshold_breaks_streak", broken["self_convergence"]["status"] == "NOT_REACHED")

    # The target rule needs one crossing only, never the sustained self rule.
    single = analyze_curve(_curve([1, 2], [0.95, 0.10]), r1_value=1.0)
    record("target_needs_single_observation", single["r1_target"]["crossing_round"] == 1)

    # A missing round breaks the streak without being interpolated away.
    gapped = analyze_curve(_curve([1, 3, 4, 5], [1.0, 1.0, 1.0, 1.0]))
    record("cadence_gap_breaks_streak", gapped["cadence_gap_pairs"] == [[1, 3]])

    # Missing measurements are pending, never zero bytes.
    pending = sum_measured_communication(
        None, through_round=3, expected_run_id="r", expected_config_sha256="a" * 64
    )
    record(
        "missing_bytes_are_pending_not_zero",
        pending["status"] == "PENDING_COMMUNICATION" and pending["total_bytes"] is None,
    )

    # No crossing yet means there is no byte boundary to sum through.
    no_boundary = sum_measured_communication(
        None, through_round=None, expected_run_id="r", expected_config_sha256="a" * 64
    )
    record("absent_boundary_is_pending_crossing", no_boundary["status"] == "PENDING_CROSSING")

    return checks


def build_evidence(repo_root: Path, policy_path: Path) -> dict[str, Any]:
    """Run every fixed check and assemble deterministic evidence."""
    policy = json.loads((repo_root / policy_path).read_text(encoding="utf-8"))
    if policy.get("schema") != "convergence_policy_v1":
        raise ConvergenceValidationError(f"unexpected policy schema: {policy.get('schema')!r}")
    fixture = json.loads((repo_root / _FIXTURE_PATH).read_text(encoding="utf-8"))
    if fixture.get("schema") != "convergence_fixture_v1":
        raise ConvergenceValidationError(f"unexpected fixture schema: {fixture.get('schema')!r}")

    cases = run_fixture_cases(fixture, policy)
    boundary = run_boundary_cases(policy)
    byte_oracle = run_byte_oracle(fixture)

    return {
        "schema": "convergence_validation_v1",
        "version": "1",
        "status": "PASS",
        "artifact_kind": "FIXTURE_PROOF",
        "cases": cases,
        "case_count": len(cases),
        "boundary_checks": boundary,
        "boundary_check_count": len(boundary),
        "byte_oracle": byte_oracle,
        "policy_ref": {
            "uri": policy_path.as_posix(),
            "sha256": file_sha256(repo_root / policy_path),
        },
        "fixture_ref": {
            "uri": _FIXTURE_PATH.as_posix(),
            "sha256": file_sha256(repo_root / _FIXTURE_PATH),
        },
        "module_ref": {
            "uri": _MODULE_PATH.as_posix(),
            "sha256": file_sha256(repo_root / _MODULE_PATH),
        },
        "metric_id": policy["metric_id"],
        "evaluation_split": policy["split"],
        "real_r1_status": "PENDING_R1",
        "actual_communication_status": "PENDING_50_INSTRUMENTATION",
        "limitations": [
            "Fixture arithmetic only; no model was trained and no curve was measured.",
            "Byte totals are a hand-worked oracle, not measured Flower wire traffic.",
            "Self convergence is retrospective and must not drive online early stopping.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate S2-PR-02 convergence primitives.")
    parser.add_argument(
        "--config",
        default="config/federated/convergence_policy.v1.json",
        help="Path to the convergence policy config, relative to the repository root.",
    )
    args = parser.parse_args()

    evidence = build_evidence(_REPO_ROOT, Path(args.config))
    out_path = _REPO_ROOT / _EVIDENCE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    out_path.write_text(payload, encoding="utf-8", newline="\n")

    print("=== S2-PR-02 Convergence Validation ===")
    print(f"Status:          {evidence['status']}")
    print(f"Fixture cases:   {evidence['case_count']} matched")
    print(f"Boundary checks: {evidence['boundary_check_count']} matched")
    print(f"Byte oracle:     total {evidence['byte_oracle']['total_bytes']} (FIXTURE_PROOF)")
    print(f"Real R1:         {evidence['real_r1_status']}")
    print(f"Actual bytes:    {evidence['actual_communication_status']}")
    print(f"Evidence:        {_EVIDENCE_PATH.as_posix()}")
    print(f"Evidence SHA256: {file_sha256(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
