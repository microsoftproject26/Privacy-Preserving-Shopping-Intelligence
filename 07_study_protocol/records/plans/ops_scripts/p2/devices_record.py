"""devices_record.py - builds the DESCRIPTIVE DEVICES record (PREG2 §8; P2_SITTING_RUNBOOK §8.6 open item 3) from the per-run
ppsi.v4.device_bench_p2.v1 JSON files: per device and (model, arm), the run count and the median / min / max over sessions of
the warm p50 and p95 latency and the cold-start total, plus the model bytes and sha256. Latency only; no accuracy value.
  python3 devices_record.py --device D1=<dir> --device D2=<dir> --device SERVER_EDGE=<dir> --out <DEVICES.json>
stdlib only; the record's self_sha256 follows the utf-8 compact rule that retention_report._self_ok accepts."""
import argparse
import glob
import hashlib
import json
import os
import statistics
import sys

SCHEMA = "ppsi.v4.device_bench_p2.v1"
FORMAT = "PPSI_V4_P2_DEVICES_RECORD_1"


def load_runs(d):
    runs, bad = [], []
    for f in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
        try:
            o = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(o, dict) or o.get("schema") != SCHEMA:
            continue
        m = (o.get("models") or {}).get(o.get("model"))
        if not m or "warm" not in m or m.get("error"):
            bad.append(os.path.basename(f))
            continue
        runs.append((o, m, os.path.basename(f)))
    return runs, bad


def summarise(name, d):
    runs, bad = load_runs(d)
    out = {"device": name, "n_runs": len(runs), "n_failed_or_invalid": len(bad), "cells": {}}
    if runs:
        ident = runs[0][0].get("identity") or {}
        out["user_agent"] = ident.get("user_agent")
        out["model_description"] = ident.get("model_description")
        out["ort_runtime"] = (runs[0][0].get("ort") or {}).get("runtime")
    by = {}
    for o, m, f in runs:
        ua = (o.get("identity") or {}).get("user_agent", "")
        key = f"{o['model']}|{o['arm']}" + (f"|{ua.split('Edg/')[-1]}" if name == "SERVER_EDGE" and "Edg/" in ua else "")
        by.setdefault(key, []).append(m)
    for key, ms in sorted(by.items()):
        p50 = [x["warm"]["p50_ms"] for x in ms]; p95 = [x["warm"]["p95_ms"] for x in ms]
        cold = [x["cold_start"]["total_ms"] for x in ms]
        out["cells"][key] = {"n": len(ms), "warm_p50_ms_median": statistics.median(p50), "warm_p50_ms_min": min(p50),
                             "warm_p50_ms_max": max(p50), "warm_p95_ms_median": statistics.median(p95),
                             "cold_start_ms_median": statistics.median(cold), "model_bytes": ms[0].get("bytes"),
                             "model_sha256": ms[0].get("sha256")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="append", required=True, help="NAME=<dir of per-run JSON files>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f"refused: {a.out} exists (write-once)", file=sys.stderr)
        return 2
    devs = [summarise(*x.split("=", 1)) for x in a.device]
    rec = {"format": FORMAT, "label": "DESCRIPTIVE (PREG2 §8; latency, cold start and bytes only; no accuracy)",
           "n_devices": len(devs), "n_runs_total": sum(d["n_runs"] for d in devs),
           "n_failed_total": sum(d["n_failed_or_invalid"] for d in devs), "devices": devs}
    body = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    rec["self_sha256"] = hashlib.sha256(body).hexdigest()
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, ensure_ascii=False)
    print(f"DEVICES record written: {a.out} devices={len(devs)} runs={rec['n_runs_total']} failed={rec['n_failed_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
