"""census_v2.finalize (owner: Data Lead, CENSUS-02) - completes CENSUS02_RECEIPT.json after the guarded run.

Adds the RSS-guard record (peak RSS of the census process tree, rc, wall time) and the job's `date -u` start/end
stamps (written by job.sh from `date -u`, never hand-typed), then recomputes receipt_sha256 with the same rule
(sha256 of the canonical JSON without the receipt_sha256 field). Nothing else in the receipt is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from v4data.common import atomic_write_bytes, canonical_json_bytes, json_bytes_strict, sha256_bytes  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rss-log", required=True)
    ap.add_argument("--job-start", required=True)
    ap.add_argument("--job-end", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    rp = out / "CENSUS02_RECEIPT.json"
    rec = json.loads(rp.read_text(encoding="utf-8"))
    body = {k: v for k, v in rec.items() if k != "receipt_sha256"}
    if sha256_bytes(canonical_json_bytes(body)) != rec.get("receipt_sha256"):
        raise SystemExit("receipt self-hash does not verify before finalisation")
    lines = [json.loads(x) for x in Path(a.rss_log).read_text(encoding="utf-8").splitlines() if x.strip()]
    guard = lines[-1]
    rec["resources"]["rss_guard"] = {"peak_rss_mib_process_tree": guard["peak_rss_mib"], "limit_mib": guard["limit_mib"],
                                     "rc": guard["rc"], "killed_rss_guard": guard["killed_rss_guard"],
                                     "wall_s": guard["wall_s"], "guard": "v4data.run_bounded (0.5 s VmRSS polling)"}
    rec["utc"]["job_start_date_u"] = Path(a.job_start).read_text(encoding="utf-8").strip()
    rec["utc"]["job_end_date_u"] = Path(a.job_end).read_text(encoding="utf-8").strip()
    rec.pop("receipt_sha256", None)
    rec["receipt_sha256"] = sha256_bytes(canonical_json_bytes(rec))
    atomic_write_bytes(out, "CENSUS02_RECEIPT.json", json_bytes_strict(rec))
    print(json.dumps({"receipt_sha256": rec["receipt_sha256"], "peak_rss_mib": guard["peak_rss_mib"],
                      "rc": guard["rc"], "status": rec["status"]}))
    return 0 if guard["rc"] == 0 and rec["status"] == "PASS" else 4


if __name__ == "__main__":
    sys.exit(main())
