"""20k-update go/no-go watcher for ONE run (owner: Opus). Same rule as tools/v3_gate_watch.py
(registry v3 go_no_go_20k + amendment 1), used for the owner-approved third run A1_RICH256 seed 2027.
On NO-GO it stops ONLY that run's tmux session and writes review/GATE20K_NO_GO_<run>.json.
Usage: python gate20k_watch.py <run_id> <seed>
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_gate_watch as g  # noqa: E402

S = g.S


def main() -> int:
    run_id, seed = sys.argv[1], int(sys.argv[2])
    run = S / "runs" / f"{run_id}_s{seed}"
    out = S / "review" / f"GATE20K_{run_id}_s{seed}.json"
    while True:
        aggs, _ = g.load(run)
        if any(a["update"] >= 20_000 for a in aggs):
            break
        time.sleep(60)
    res = g.evaluate(aggs)
    res["run"] = f"{run_id}_s{seed}"
    res["rule"] = g.__doc__.split("Runs detached.")[1].split("Writes")[0].strip()
    res["decided_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(res, indent=1))
    if not res["go"]:
        subprocess.run(["tmux", "kill-session", "-t", f"sasrec-{run_id}-s{seed}"], check=False)
        (S / "review" / f"GATE20K_NO_GO_{run_id}_s{seed}.json").write_text(json.dumps(
            {"utc": res["decided_utc"], "checks": res["checks"], "action": "stopped this run only; excluded"}, indent=1))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
