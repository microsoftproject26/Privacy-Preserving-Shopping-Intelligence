"""Build ClientManifest v1 and run smoke sampling.

Usage::

    uv run --locked python scripts/federated/build_client_manifest.py \\
      --cohort-manifest data/protocol/INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet \\
      --output data/protocol/INTERNAL_DO_NOT_UPLOAD_client_manifest_v1.parquet \\
      --summary docs/evidence/s1-pr-06/client_partition_summary.v1.json

Optional flags::

    --events-parquet data/raw/processed_raw_parquet_v1.parquet
    --excluded-sessions data/protocol/INTERNAL_DO_NOT_UPLOAD_excluded_sessions_v1.proposed.parquet
    --trace data/protocol/INTERNAL_DO_NOT_UPLOAD_client_sampling_trace_v1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import polars as pl

from ppsi.federated.clients import (
    CLIENT_IDENTITY_VERSION,
    build_client_manifest,
    compute_distribution_stats,
    manifest_content_sha256,
)
from ppsi.federated.sampling import (
    DEFAULT_SAMPLER_VERSION,
    build_trace,
    sample_clients,
)

# ---------------------------------------------------------------------------
# Smoke sampling defaults
# ---------------------------------------------------------------------------

_SMOKE_SEED = 13
_SMOKE_ROUNDS = 3
_SMOKE_CLIENTS_PER_ROUND = 10


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_public_summary(
    manifest: pl.DataFrame,
    manifest_content_hash: str,
    manifest_file_hash: str,
    cohort_manifest_hash: str,
    cohort_manifest_path: str,
    traces: list[dict],
    events_used: bool,
) -> dict:
    """Build the public-safe aggregate summary."""
    summary: dict = {
        "schema": "client_partition_summary_v1",
        "version": "1",
        "data_scope": "official_phase1_25_percent_user_slice",
        "additional_sampling_applied": False,
        "client_identity_version": CLIENT_IDENTITY_VERSION,
        "sampler_version": DEFAULT_SAMPLER_VERSION,
        "source_cohort_manifest": cohort_manifest_path,
        "source_cohort_manifest_sha256": cohort_manifest_hash,
        "client_manifest_content_sha256": manifest_content_hash,
        "client_manifest_file_sha256": manifest_file_hash,
        "client_manifest_sha256": manifest_content_hash,
        "client_count": len(manifest),
        "events_parquet_used": events_used,
        "event_statistics_status": (
            "measured" if events_used else "not_measured_raw_events_unavailable"
        ),
        "task_example_counts_status": "not_measured",
    }

    # Distribution stats (only if event counts are present)
    if "train_event_count" in manifest.columns:
        non_null = manifest["train_event_count"].drop_nulls()
        if len(non_null) > 0:
            stats = compute_distribution_stats(manifest["train_event_count"])
            summary["train_event_distribution"] = stats.to_dict()

    if "validation_event_count" in manifest.columns:
        non_null = manifest["validation_event_count"].drop_nulls()
        if len(non_null) > 0:
            stats = compute_distribution_stats(manifest["validation_event_count"])
            summary["validation_event_distribution"] = stats.to_dict()

    # Sampling smoke digests (no raw IDs)
    summary["smoke_sampling"] = {
        "experiment_seed": _SMOKE_SEED,
        "rounds": _SMOKE_ROUNDS,
        "clients_per_round": _SMOKE_CLIENTS_PER_ROUND,
        "round_digests": [
            {"round_index": t["round_index"], "selected_digest": t["selected_digest"]}
            for t in traces
        ],
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ClientManifest v1 and run smoke sampling")
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        required=True,
        help="Path to the upstream cohort manifest parquet",
    )
    parser.add_argument(
        "--events-parquet",
        type=Path,
        default=None,
        help="Optional path to raw events parquet for per-split statistics",
    )
    parser.add_argument(
        "--excluded-sessions",
        type=Path,
        default=None,
        help="Optional path to excluded sessions parquet (REQUIRED if --events-parquet is set)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the client manifest parquet",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Path to write sampling trace JSONL (internal)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to write public aggregate summary JSON",
    )
    args = parser.parse_args()

    # Validate inputs exist
    if not args.cohort_manifest.exists():
        print(f"ERROR: Cohort manifest not found: {args.cohort_manifest}", file=sys.stderr)
        sys.exit(1)
    if args.events_parquet is not None:
        if not args.events_parquet.exists():
            print(f"ERROR: Events parquet not found: {args.events_parquet}", file=sys.stderr)
            sys.exit(1)
        if args.excluded_sessions is None:
            print(
                "ERROR: --excluded-sessions is REQUIRED when --events-parquet is provided. "
                "Boundary-crossing sessions must be excluded per protocol.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.excluded_sessions.exists():
            print(
                f"ERROR: Excluded sessions file not found: {args.excluded_sessions}. "
                "This file is required when generating raw-event statistics.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Build the manifest
    print("Building ClientManifest v1...")
    manifest = build_client_manifest(
        str(args.cohort_manifest),
        events_parquet_path=str(args.events_parquet) if args.events_parquet else None,
        excluded_sessions_path=(str(args.excluded_sessions) if args.excluded_sessions else None),
    )

    # Write manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(args.output)
    m_content_hash = manifest_content_sha256(manifest)
    m_file_hash = _file_sha256(args.output)
    print(f"  Manifest written: {args.output}")
    print(f"  Client count: {len(manifest)}")
    print(f"  Manifest logical content SHA-256: {m_content_hash}")
    print(f"  Manifest physical artifact SHA-256: {m_file_hash}")

    # Run smoke sampling
    print(
        f"\nRunning smoke sampling (seed={_SMOKE_SEED}, "
        f"rounds={_SMOKE_ROUNDS}, clients/round={_SMOKE_CLIENTS_PER_ROUND})..."
    )
    eligible_ids = manifest["client_id"].to_list()
    traces = []
    for r in range(_SMOKE_ROUNDS):
        result = sample_clients(
            eligible_ids,
            experiment_seed=_SMOKE_SEED,
            round_index=r,
            clients_per_round=_SMOKE_CLIENTS_PER_ROUND,
        )
        trace = build_trace(
            result,
            sampler_version=DEFAULT_SAMPLER_VERSION,
            experiment_seed=_SMOKE_SEED,
            round_index=r,
            eligible_pool_count=len(eligible_ids),
            clients_per_round=_SMOKE_CLIENTS_PER_ROUND,
        )
        traces.append(trace.to_dict())
        unique_in_round = len(set(result.selected_client_ids))
        print(
            f"  Round {r}: {unique_in_round} unique clients, "
            f"digest={result.selected_digest[:16]}..."
        )

    # Write trace
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with open(args.trace, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n" for t in traces)
        print(f"\n  Trace written: {args.trace}")

    # Build and write public summary
    cohort_hash = _file_sha256(args.cohort_manifest)
    summary = _build_public_summary(
        manifest=manifest,
        manifest_content_hash=m_content_hash,
        manifest_file_hash=m_file_hash,
        cohort_manifest_hash=cohort_hash,
        cohort_manifest_path=args.cohort_manifest.as_posix(),
        traces=traces,
        events_used=args.events_parquet is not None,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary written: {args.summary}")

    print("\n=== S1-PR-06 build complete ===")
    print("  data_scope: official_phase1_25_percent_user_slice")
    print("  additional_sampling_applied: false")
    print("  sealed_test: NOT accessed")


if __name__ == "__main__":
    main()
