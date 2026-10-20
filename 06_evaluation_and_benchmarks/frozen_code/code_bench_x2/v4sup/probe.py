"""Read-only status probe for the lead (package 10 S4: "a read-only probe CLI for the lead, a
one-line JSON"). Never writes anything -- it only reads the status file the runner already wrote.
"""
from __future__ import annotations

import json
from pathlib import Path

from .status import read_status


def probe_line(status_path: "Path | str") -> str:
    s = read_status(status_path) or {}
    phases = s.get("phases", {}) or {}
    counts: "dict[str, int]" = {}
    for p in phases.values():
        st = p.get("state", "PENDING")
        counts[st] = counts.get(st, 0) + 1
    if not phases:
        overall = "NOT_STARTED"
    elif counts.get("FAILED"):
        overall = "FAILED"
    elif all(p.get("state") == "DONE" for p in phases.values()):
        overall = "DONE"
    else:
        overall = "IN_PROGRESS"
    out = {"overall": overall, "heartbeat_utc": s.get("heartbeat_utc"), "updated_utc": s.get("updated_utc"),
           "counts": counts, "phases": {k: v.get("state") for k, v in phases.items()}}
    return json.dumps(out, sort_keys=True)
