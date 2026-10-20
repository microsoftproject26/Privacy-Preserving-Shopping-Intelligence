"""CLI entry point.

  python -B -m v4sup.cli run --workspace WS --registry REGISTRY.json --status WS/state/status.json \
      --event-log WS/logs/events.jsonl --logs-dir WS/logs [--poll-interval-s 5] [--max-rounds N]

  python -B -m v4sup.cli probe --status WS/state/status.json    # read-only, one JSON line
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .probe import probe_line
from .registry import load_registry
from .runner import Supervisor, SupervisorConfig


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(prog="v4sup")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run every registered phase until each is terminal")
    p_run.add_argument("--workspace", required=True)
    p_run.add_argument("--registry", required=True)
    p_run.add_argument("--status", required=True)
    p_run.add_argument("--event-log", required=True)
    p_run.add_argument("--logs-dir", required=True)
    p_run.add_argument("--poll-interval-s", type=float, default=5.0)
    p_run.add_argument("--max-rounds", type=int, default=100000)

    p_probe = sub.add_parser("probe", help="read-only one-line JSON status")
    p_probe.add_argument("--status", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "probe":
        print(probe_line(Path(args.status)))
        return 0

    registry = load_registry(Path(args.registry))
    cfg = SupervisorConfig(workspace_root=Path(args.workspace), registry=registry, status_path=Path(args.status),
                           event_log_path=Path(args.event_log), logs_dir=Path(args.logs_dir),
                           poll_interval_s=args.poll_interval_s)
    sup = Supervisor(cfg)
    sup.run_all(max_rounds=args.max_rounds)
    print(probe_line(Path(args.status)))
    any_failed = any(sup.phase_state(pid) == "FAILED" for pid in registry)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
