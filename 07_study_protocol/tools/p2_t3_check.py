"""tools/p2_t3_check.py — PREG2 §5.2 test P2-T3: does the Phase-2 code (launch_fl --arm P2_W, the incumbent recipe
FedAdam 1e-3 / client SGD-m 0.1 / frozen tables) reproduce run S8Tc (FA_WARM_FROZEN CALIB, S4 point, server-LR factor
0.1) bit-identically on its first 64 rounds? Writes the receipt v4int.p2_select.check_p2t3_receipt accepts (P2-01/02,
REV-P2-01_02 R1 / m2).

  python -B tools/p2_t3_check.py --p2-run-dir <P2_W incumbent smoke run, --smoke-rounds 65> \
         (--s8tc-run-dir <S8Tc run dir> | --s8tc-search-root <R>/S7S8_T) --out <NEW P2T3_RECEIPT.json> [--code-root <V4>/code]

Value-free: S8Tc's RESULT_ROWS.json / RUN_SUMMARY.json are HASHED (bytes), never parsed; the comparison reads ARM_SPEC.json,
LAUNCH_*.json (clients, history policy, GPU name, CALIB manifests), STATUS.json, the P2 smoke's SMOKE_SUMMARY.json
(rounds, server digest) and the first 65 lines of both TRAIN_LOSS.jsonl (training-loss monitoring lines, not an
evaluation). Line r of TRAIN_LOSS is a function of theta_r, so 65 byte-equal lines = theta_0 .. theta_64 reproduced, i.e.
all 64 server updates of the registered "first 64 rounds" ("TRAIN_LOSS-exact" evidence; theta_64 itself is not stored
by S8Tc, whose latest.pt was removed at completion).
Checks (all must hold for verdict BIT_IDENTICAL, else FAIL — a FAIL receipt is written too): S8Tc is the S8 f1/sf0.1
cell and DONE_MAIN; the P2 run is P2_W with the incumbent label, a valid p2 spec (launch_fl.check_p2_spec) and a 65-round
SMOKE_DONE_NOT_EVIDENCE; the two arm specs agree on v4int.p2_select's INCUMBENT_COMPARED fields (cfg, plan, max_rounds,
frozen_item_tables, init_from, exposure_budget, role); same evaluation device record, clients (n, decisions, manifest
hash), history policy, GPU name and CALIB manifests; 65 equal TRAIN_LOSS lines. Receipt keys required by p2_select:
test "P2-T3", verdict, rounds 64, s8tc_run_id (= S8Tc's run id; ARM_SPEC run_id, which execute writes from the same run
id as RESULT_ROWS), result_rows_sha256, run_summary_sha256. The receipt is written exclusively (O_EXCL), only after the
comparison completed; a missing input refuses with exit 2 and writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import ppsi_paths

V4 = ppsi_paths.PROJECT_ROOT.resolve()   # the ppsi_v4 workspace
FORMAT = "PPSI_V4_P2T3_RECEIPT_1"
INCUMBENT_LABEL = "fedadam|eta=0.001|c=0.1"
ROUNDS = 64                                 # the registered "first 64 rounds" (= p2_select P2T3_RECEIPT_KEYS rounds)
LINES = ROUNDS + 1                          # TRAIN_LOSS lines 0..64 cover theta_0..theta_64 (all 64 updates)
INCUMBENT_COMPARED = ("cfg", "plan", "max_rounds", "frozen_item_tables", "init_from", "exposure_budget", "role")


class P2T3Refused(RuntimeError):
    pass


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha(p: Path) -> str:
    return _sha_bytes(Path(p).read_bytes())


def _json(p: Path, what: str) -> dict:
    try:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise P2T3Refused(f"{what} {p} unreadable ({e})") from e
    if not isinstance(obj, dict):
        raise P2T3Refused(f"{what} {p} is not a JSON object")
    return obj


def _canon(obj) -> str:
    body = {k: v for k, v in obj.items() if k != "self_sha256"}
    return _sha_bytes(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def is_s8tc_spec(spec: dict) -> bool:
    """S8's f1/sf0.1 cell: FA_WARM_FROZEN, CALIBRATION, warm_server_lr_factor 0.1, local factor 1 (key absent)."""
    return (spec.get("arm") == "FA_WARM_FROZEN" and spec.get("role") == "CALIBRATION"
            and spec.get("warm_server_lr_factor") == 0.1 and "warm_lr_factor" not in spec
            and bool(spec.get("frozen_item_tables")))


def find_s8tc(root: Path) -> Path:
    """The unique run dir under `root` whose ARM_SPEC is the S8 f1/sf0.1 cell (refused if 0 or > 1)."""
    hits = []
    for p in sorted(Path(root).glob("*/ARM_SPEC.json")):
        try:
            spec = json.loads(p.read_text(encoding="utf-8")).get("arm_spec") or {}
        except (OSError, ValueError):
            continue
        if is_s8tc_spec(spec):
            hits.append(p.parent)
    if len(hits) != 1:
        raise P2T3Refused(f"S8Tc not unique under {root}: {[str(h) for h in hits]}")
    return hits[0]


def _launch(run_dir: Path) -> dict:
    ls = sorted(Path(run_dir).glob("LAUNCH*.json"))
    if not ls:
        raise P2T3Refused(f"{run_dir}: no LAUNCH_*.json")
    return _json(ls[0], "LAUNCH")


def _lines(p: Path, n: int) -> list:
    if not Path(p).is_file():
        raise P2T3Refused(f"{p} is absent")
    return [x for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()][:n]


def check(s8tc_dir: Path, p2_dir: Path) -> dict:
    """The P2-T3 receipt (see the module docstring). Refuses (P2T3Refused) when an input file is missing."""
    s8tc_dir, p2_dir = Path(s8tc_dir), Path(p2_dir)
    for f in ("ARM_SPEC.json", "RESULT_ROWS.json", "RUN_SUMMARY.json", "STATUS.json", "TRAIN_LOSS.jsonl"):
        if not (s8tc_dir / f).is_file():
            raise P2T3Refused(f"S8Tc {s8tc_dir} lacks {f}")
    for f in ("ARM_SPEC.json", "STATUS.json", "SMOKE_SUMMARY.json", "TRAIN_LOSS.jsonl"):
        if not (p2_dir / f).is_file():
            raise P2T3Refused(f"P2 run {p2_dir} lacks {f}")
    a, b = _json(s8tc_dir / "ARM_SPEC.json", "ARM_SPEC"), _json(p2_dir / "ARM_SPEC.json", "ARM_SPEC")
    sa, sb = a.get("arm_spec") or {}, b.get("arm_spec") or {}
    la, lb = _launch(s8tc_dir), _launch(p2_dir)
    st_a, st_b = _json(s8tc_dir / "STATUS.json", "STATUS"), _json(p2_dir / "STATUS.json", "STATUS")
    smoke = _json(p2_dir / "SMOKE_SUMMARY.json", "SMOKE_SUMMARY")
    ta, tb = _lines(s8tc_dir / "TRAIN_LOSS.jsonl", LINES), _lines(p2_dir / "TRAIN_LOSS.jsonl", LINES)
    try:
        import v4int.launch_fl as F
        F.check_p2_spec(sb)
        p2_spec_ok = True
    except Exception:  # noqa: BLE001 — any refusal / import failure is a failed check
        p2_spec_ok = False
    role_rec = (lambda la_: ((la_.get("fl05") or {}).get("role_record") or {}).get("manifests"))
    checks = {
        "s8tc_is_s8_f1_sf0.1": is_s8tc_spec(sa) and a.get("arm") == "FA_WARM_FROZEN",
        "s8tc_done_main": st_a.get("status") == "DONE_MAIN",
        "p2_is_incumbent": b.get("arm") == "P2_W" and sb.get("p2_label") == INCUMBENT_LABEL
                           and sb.get("role") == "CALIBRATION",
        "p2_spec_valid": p2_spec_ok,
        "p2_smoke_65_rounds": st_b.get("status") == "SMOKE_DONE_NOT_EVIDENCE" and smoke.get("rounds") == LINES,
        **{f"spec_{k}": k in sa and k in sb and sa[k] == sb[k] for k in INCUMBENT_COMPARED},
        "device_record": a.get("device") is not None and a.get("device") == b.get("device"),
        "clients": la.get("clients") is not None and la.get("clients") == lb.get("clients"),
        "history_policy": la.get("history_policy") is not None and la.get("history_policy") == lb.get("history_policy"),
        "gpu_name": (la.get("device") or {}).get("name") is not None and
                    (la.get("device") or {}).get("name") == (lb.get("device") or {}).get("name"),
        "calib_manifests": role_rec(la) is not None and role_rec(la) == role_rec(lb),
        f"train_loss_{LINES}_lines_equal": len(ta) == len(tb) == LINES and ta == tb,
    }
    rec = {"format": FORMAT, "test": "P2-T3", "verdict": "BIT_IDENTICAL" if all(checks.values()) else "FAIL",
           "rounds": ROUNDS, "train_loss_lines_compared": LINES,
           "evidence": ("TRAIN_LOSS-exact: lines 0..64 (functions of theta_0..theta_64) byte-equal, plus the configuration "
                        "checks; theta_64 itself is not stored by S8Tc"),
           "s8tc_run_id": a.get("run_id"), "s8tc_run_dir": str(s8tc_dir),
           "result_rows_sha256": _sha(s8tc_dir / "RESULT_ROWS.json"),
           "run_summary_sha256": _sha(s8tc_dir / "RUN_SUMMARY.json"),
           "s8tc_arm_spec_sha256": a.get("arm_spec_sha256"), "p2_run_dir": str(p2_dir), "p2_run_id": b.get("run_id"),
           "p2_arm_spec_sha256": b.get("arm_spec_sha256"), "p2_smoke_server_digest": smoke.get("server_digest"),
           "train_loss_sha256": {"s8tc": _sha_bytes("\n".join(ta).encode()), "p2": _sha_bytes("\n".join(tb).encode())},
           "checks": checks, "tool_sha256": _sha(Path(__file__)),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "authority": "PREG2 §5.2 (P2-T3); PHASE2_IMPLEMENTATION_PLAN P2-02; REV-P2-01_02 R1 / m2"}
    rec["self_sha256"] = _canon(rec)
    return rec


def write_exclusive(path: Path, rec: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(Path(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, sort_keys=True, allow_nan=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PREG2 §5.2 P2-T3 receipt (incumbent bit-identity vs S8Tc)")
    ap.add_argument("--p2-run-dir", required=True, type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--s8tc-run-dir", type=Path)
    g.add_argument("--s8tc-search-root", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--code-root", type=Path, default=V4 / "code")
    a = ap.parse_args(argv)
    sys.dont_write_bytecode = True
    if str(a.code_root) not in sys.path:
        sys.path.insert(0, str(a.code_root))
    if a.out.exists():
        print(f"REFUSED: {a.out} exists (the receipt is written once)", file=sys.stderr)
        return 2
    try:
        s8 = a.s8tc_run_dir if a.s8tc_run_dir is not None else find_s8tc(a.s8tc_search_root)
        rec = check(s8, a.p2_run_dir)
    except P2T3Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    write_exclusive(a.out, rec)
    print(f"[p2_t3_check] {a.out}: {rec['verdict']}")
    return 0 if rec["verdict"] == "BIT_IDENTICAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
