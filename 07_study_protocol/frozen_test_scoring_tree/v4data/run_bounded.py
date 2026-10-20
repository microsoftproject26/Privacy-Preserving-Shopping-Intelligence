"""v4data.run_bounded (owner: Data Lead, DATA-04) - run one command under a hard peak-RSS guard (pattern of
feature_opt_v1/code/run_bounded.py, reproduced here so the server needs no foreign code tree).

Usage: python -m v4data.run_bounded --max-rss-mib 8192 --log <file.jsonl> -- <cmd...>
Polls VmRSS of the child AND its descendants every 0.5 s (Linux /proc). Over the limit, only the child's process
group is SIGKILLed (never anything else) and the runner exits 97. One JSON line (rc, peak RSS, wall time) is
appended to --log.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time


def _children(pid: int) -> list[int]:
    out = []
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as fh:
            kids = [int(x) for x in fh.read().split()]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return out
    for k in kids:
        out.append(k)
        out.extend(_children(k))
    return out


def _rss_mib(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rss-mib", type=float, default=8192.0)
    ap.add_argument("--log", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    t0 = time.time()
    p = subprocess.Popen(cmd, start_new_session=True)
    peak, killed = 0.0, False
    while p.poll() is None:
        tot = _rss_mib(p.pid) + sum(_rss_mib(c) for c in _children(p.pid))
        peak = max(peak, tot)
        if peak > a.max_rss_mib:
            os.killpg(p.pid, signal.SIGKILL)
            killed = True
            p.wait()
            break
        time.sleep(0.5)
    rc = 97 if killed else p.returncode
    rec = {"cmd": cmd, "rc": rc, "killed_rss_guard": killed, "peak_rss_mib": round(peak, 1),
           "limit_mib": a.max_rss_mib, "wall_s": round(time.time() - t0, 1),
           "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(a.log, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return rc


if __name__ == "__main__":
    sys.exit(main())
