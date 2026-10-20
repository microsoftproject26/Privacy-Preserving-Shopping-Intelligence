"""run_bounded.py (owner: Opus) — run one study command under a hard RSS guard while the campaign is active.

Usage: run_bounded.py --max-rss-mib 600 --log <file> -- <cmd...>
Polls the child's VmRSS every 0.5 s. If it goes over the limit, the child is SIGKILLed (the child
only, never anything else) and the runner exits with code 97. Peak RSS, wall time and exit code are
written as one JSON line to --log.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def rss_mib(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rss-mib", type=float, default=600.0)
    ap.add_argument("--log", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    t0 = time.time()
    p = subprocess.Popen(cmd)
    peak, killed = 0.0, False
    while p.poll() is None:
        peak = max(peak, rss_mib(p.pid))
        if peak > a.max_rss_mib:
            os.kill(p.pid, signal.SIGKILL)
            killed = True
            p.wait()
            break
        time.sleep(0.5)
    rc = 97 if killed else p.returncode
    rec = {"cmd": cmd, "rc": rc, "killed_rss_guard": killed, "peak_rss_mib": round(peak, 1),
           "limit_mib": a.max_rss_mib, "wall_s": round(time.time() - t0, 1),
           "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(a.log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return rc


if __name__ == "__main__":
    sys.exit(main())
