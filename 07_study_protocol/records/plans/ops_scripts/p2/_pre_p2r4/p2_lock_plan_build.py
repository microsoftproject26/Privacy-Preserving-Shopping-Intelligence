"""p2_lock_plan_build.py (P2-SITTING) - builds the P2_METHODS_LOCK plan, the input of `tools/p2_lock.py lock --plan`
(format PPSI_V4_P2_LOCK_PLAN_1, read from that tool, never re-typed), from the finished Phase-2 MAIN run dirs on the
server, and prints the Phase-1 LOCKED_EVALUATION register entry of the T0a sitting. Stdlib only; value-free.

  python -B p2_lock_plan_build.py plan --out <new P2_LOCK_PLAN.json> [--dry-run] [--root <ppsi_v4 root>]
         [--declare-not-run ARM:SEED=REASON | ARM:*=REASON ...] [--ft-without-main-run] [--p2-lock-tool <tools/p2_lock.py>]
  python -B p2_lock_plan_build.py p1-entry [--actor-role "Operations Director"] [--p2-lock-tool ...]
         -> stdout: the entry JSON for `python -B tools/test_exposure_register.py append --entry <file>` (tool unchanged)

Binding spec: contracts/PREG2.md (REGISTERED, DL 411) §6.3 S1 / S2 / S4 / S5, §9.5 (registered seeds), §2.1 (Part B never
delays the sitting), §13 items 5-6; DL 467 (Part B reduced), DL 469 / DL 475 (E1 validity reading, the S1 free fields
`e1_validity_reading` and `part_b_not_run_analyses`, P2_SCRIPT_FREEZE ruling condition iii); DL 478 (futility: RUN-2
skipped, W* = A*). The plan layout, the registered seeds, the finished-status words and the default Phase-1 reused list
are taken from tools/p2_lock.py itself (REGISTERED_SEEDS, FINISHED, DEFAULT_PHASE1_REUSED, PLAN_FORMAT).

RUN DIRS (the RUN-1 / RUN-3 lane layout under <root>/runs_p2):
  P2_WSTAR s2026 / s2027              MAIN/P2_WSTAR_SMALL_s<S>
  P2_WSTAR_FA1024 / _DP8 {2026, 2027} MAIN/<arm>_SMALL_s<S>        P2_WSTAR_DP1 {2026}: MAIN/P2_WSTAR_DP1_SMALL_s2026
  P2_C1_FT_CENTRAL_FULL / _FROZEN     C1/P2_C1_FT_CENTRAL_<FULL|FROZEN>_MAIN_s<S>
  P2_C2_CFULL_SMALL s2026             C2/P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026
  FT_P2_WSTAR s2026 / s2027           MAIN/FT_P2_WSTAR_s<S> (its MAIN device run; sealed on the W* base)
SELECTION FILES: contracts/fl/P2_WARM_SELECTION_FINAL.json, _A.json, P2_C1_SELECTION.json (required) and S_RECORD_P2.json
of runs_p2/MAIN/P2_WSTAR_S_CAL_SMALL_s2026 (required when a DP-chain arm is sealed). CODE TREE: <root>/code_p2r3 (the
scoring tree; DL 477). Paths are written absolute (the lock records them; laneP2_SIT.sh scores from them).
WHAT IS READ (S8): per run dir only the "status" key of STATUS.json (and "still_improving" for C2); from
P2_WARM_SELECTION_FINAL only "dropped.units" (the §5.9 drops, which must be declared not run). No RUN_SUMMARY, row,
metric or checkpoint byte is opened here (p2_lock hashes the checkpoints).
DECLARED NOT RUN: P2_WSTAR_DPFTRL8 and P2_WSTAR_SECAGG (every seed; DL 467 wording); the arms of FINAL dropped.units
(DP1, C2); and only what Ops passes with --declare-not-run (each with its own DL row). A registered (arm, seed) whose run
is missing or unfinished is a REFUSAL unless Ops declares it (S4: never declared silently); a finished run that is
declared not run is a refusal too.
OUTPUT: the plan JSON (indent 1, sort_keys, LF), written once (O_EXCL); --dry-run prints it and writes nothing. stdout:
paths, sha256, counts. Exit: 0 ok, 2 refused (nothing written).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[3]                                   # <root>/company/plans/ops_scripts/p2/ -> <root>
REFUSED = 2
CODE_TREE = "code_p2r3"
DL467_REASON = ("declared not run: Part B reduced to fit the schedule and budget (owner decision, DL 467); Part B never "
                "delays the sitting (PREG2 §2.1, §9.3 E-S4)")
DROP_REASON = "declared not run: dropped under PREG2 §5.9 (P2_WARM_SELECTION_FINAL dropped.units {unit})"
PART_B_ALWAYS_DNR = ("P2_WSTAR_DPFTRL8", "P2_WSTAR_SECAGG")      # DL 467
DROP_UNIT_ARM = {"DP1": "P2_WSTAR_DP1", "C2": "P2_C2_CFULL_SMALL"}   # v4int/p2_select.py DROP_ARMS (arm-level units)
E1_VALIDITY_READING = (
    "PREG2 §7.1 (a) validity gate, registered reading: PRE's RAW mean cross-entropy member / non-member AUC 95 % CI must "
    "include 0.5 (theta_pre, trained on neither population, cannot separate them); the calibrated score s(u) of PRE is "
    "identically 0, so the literal reading would be vacuous. v4int/audit_e1_mia.py records it as validity.score = "
    "raw_mean_ce and writes audit_status VALID / AUDIT_INVALID with eps_lb_reported. Confirmed by the P2_SCRIPT_FREEZE gate "
    "ruling (DL 475). Disclosed beside E1 with the DL 469 note: at 5,000 half-B non-members the eps_LB ceiling is about "
    "7.2 < 8, so P2_WSTAR_DP8 cannot FAIL by construction.")
PART_B_NOT_RUN = ["E2", "E3", "E5_LDP"]                          # P2_SCRIPT_FREEZE ruling (DL 475)
PART_B_REASON = ("E2 canary / one-run audit, E3 update leakage and E5 LDP feasibility are not run: Part B reduced to fit "
                 "the schedule and budget (owner decision, DL 467); Part B never delays the sitting (PREG2 §2.1). "
                 "E1 and E4 run as DESCRIPTIVE (tertiary) analyses; the DP-FTRL and SECAGG arms are in declared_not_run.")
SEED_RE = re.compile(r"^(?P<arm>[A-Z][A-Z0-9_]*):(?P<seed>\d{4}|\*)=(?P<reason>.+)$")


class PlanRefused(RuntimeError):
    pass


def load_p2_lock(path: Path):
    path = Path(path)
    if not path.is_file():
        raise PlanRefused(f"{path} (tools/p2_lock.py) does not exist")
    spec = importlib.util.spec_from_file_location("p2_lock_for_plan", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _json(p: Path) -> dict:
    try:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PlanRefused(f"{p}: not readable JSON ({e})") from None
    if not isinstance(obj, dict):
        raise PlanRefused(f"{p}: not a JSON object")
    return obj


def run_dir(root: Path, arm: str, seed: int) -> Path:
    runs = root / "runs_p2"
    if arm == "P2_WSTAR":
        return runs / "MAIN" / f"P2_WSTAR_SMALL_s{seed}"
    if arm in ("P2_WSTAR_FA1024", "P2_WSTAR_DP8", "P2_WSTAR_DP1", "P2_WSTAR_DPFTRL8", "P2_WSTAR_SECAGG"):
        return runs / "MAIN" / f"{arm}_SMALL_s{seed}"
    if arm.startswith("P2_C1_FT_CENTRAL_"):
        return runs / "C1" / f"{arm}_MAIN_s{seed}"
    if arm == "P2_C2_CFULL_SMALL":
        return runs / "C2" / f"P2_C2_CFULL_SMALL_SASREC_D64_B2_s{seed}"
    if arm == "FT_P2_WSTAR":
        return runs / "MAIN" / f"FT_P2_WSTAR_s{seed}"
    raise PlanRefused(f"no run-dir layout for {arm}")


def run_kind(PL, arm: str) -> str:
    if arm in PL.FL_RUN_ARMS:
        return "FL"
    if arm in PL.C1_ARMS:
        return "C1"
    if arm == PL.C2_ARM:
        return "C2"
    return "DEVICE"


def run_state(PL, rd: Path, kind: str) -> str:
    """FINISHED | ABSENT | UNFINISHED:<status> - the status key of STATUS.json only (+ C2 still_improving)."""
    sp = rd / "STATUS.json"
    if not rd.is_dir() or not sp.is_file():
        return "ABSENT"
    st = _json(sp)
    s = st.get("status")
    if s not in PL.FINISHED[kind]:
        return f"UNFINISHED:{s}"
    if kind == "C2" and s == "DONE_MAIN" and st.get("still_improving") is True:
        return "UNFINISHED:DONE_MAIN_STILL_IMPROVING"
    return "FINISHED"


def parse_dnr(items, PL) -> list:
    out = []
    for raw in items or []:
        m = SEED_RE.match(str(raw))
        if not m:
            raise PlanRefused(f"--declare-not-run {raw!r} is not ARM:SEED=REASON or ARM:*=REASON")
        arm, seed = m.group("arm"), None if m.group("seed") == "*" else int(m.group("seed"))
        if arm not in PL.REGISTERED_SEEDS or (seed is not None and seed not in PL.REGISTERED_SEEDS[arm]):
            raise PlanRefused(f"--declare-not-run {raw!r}: not a registered (arm, seed) of PREG2 §9.5")
        out.append({"arm": arm, "seed": seed, "reason": m.group("reason").strip()})
    return out


def build_plan(root: Path, PL, *, extra_dnr=(), ft_without_main_run: bool = False) -> dict:
    root = Path(root)
    if not hasattr(PL, "FREE_S1_FIELDS") or not hasattr(PL, "check_free_fields"):
        raise PlanRefused("tools/p2_lock.py does not carry the S1 free fields (s1_free_fields): deploy the P2-SITTING "
                          "version before building the plan (DL 475 condition iii)")
    sel_dir = root / "contracts" / "fl"
    sel = {"P2_WARM_SELECTION_FINAL": sel_dir / "P2_WARM_SELECTION_FINAL.json",
           "P2_WARM_SELECTION_A": sel_dir / "P2_WARM_SELECTION_A.json",
           "P2_C1_SELECTION": sel_dir / "P2_C1_SELECTION.json"}
    problems = [f"selection file {k} {p} does not exist" for k, p in sel.items() if not p.is_file()]
    if problems:
        raise PlanRefused("; ".join(problems))
    fin = _json(sel["P2_WARM_SELECTION_FINAL"])
    units = list(((fin.get("dropped") or {}).get("units")) or [])
    dnr = [{"arm": a, "seed": None, "reason": DL467_REASON} for a in PART_B_ALWAYS_DNR]
    for u in units:
        if u in DROP_UNIT_ARM:
            dnr.append({"arm": DROP_UNIT_ARM[u], "seed": None, "reason": DROP_REASON.format(unit=u)})
    have = {(d["arm"], d["seed"]) for d in dnr}
    for d in extra_dnr:
        if (d["arm"], d["seed"]) in have or (d["arm"], None) in have:
            raise PlanRefused(f"--declare-not-run {d['arm']}:{d['seed'] or '*'} is already declared not run")
        dnr.append(d)
        have.add((d["arm"], d["seed"]))

    def declared(arm, seed):
        return any(d["arm"] == arm and d["seed"] in (None, seed) for d in dnr)

    sealed, problems = [], []
    for arm, seeds in PL.REGISTERED_SEEDS.items():
        kind = run_kind(PL, arm)
        for s in seeds:
            rd = run_dir(root, arm, s)
            state = run_state(PL, rd, kind)
            if declared(arm, s):
                if state == "FINISHED":
                    problems.append(f"{arm} s{s} is declared not run but its run {rd} is finished (S4: a sealed "
                                    "checkpoint exists; remove the declaration or record why in a DL row and edit here)")
                continue
            if kind == "DEVICE":
                if declared(PL.DEVICE_BASE_ARM, s):
                    problems.append(f"{arm} s{s}: its base P2_WSTAR s{s} is declared not run; declare {arm}:{s} too")
                    continue
                if state == "FINISHED":
                    sealed.append({"arm": arm, "seed": s, "run_dir": rd.as_posix()})
                elif state == "ABSENT" and ft_without_main_run:
                    sealed.append({"arm": arm, "seed": s})                     # sealed on the W* base only
                else:
                    problems.append(f"{arm} s{s}: MAIN device run {rd} is {state} (finish it, or pass "
                                    "--ft-without-main-run when it was never started, or declare it not run)")
                continue
            if state != "FINISHED":
                problems.append(f"{arm} s{s}: run {rd} is {state} (S4: wait for it, or declare it not run with "
                                f"--declare-not-run {arm}:{s}=<reason> and a DL row)")
                continue
            sealed.append({"arm": arm, "seed": s, "run_dir": rd.as_posix()})
    srec = root / "runs_p2" / "MAIN" / "P2_WSTAR_S_CAL_SMALL_s2026" / "S_RECORD_P2.json"
    dp_sealed = sorted({i["arm"] for i in sealed if i["arm"] in ("P2_WSTAR_FA1024", "P2_WSTAR_DP8", "P2_WSTAR_DP1")})
    if srec.is_file():
        sel["S_RECORD_P2"] = srec
    elif dp_sealed:
        problems.append(f"{srec} is missing while {dp_sealed} are sealed (their runs are bound to it)")
    tree = root / CODE_TREE
    if not tree.is_dir():
        problems.append(f"the scoring tree {tree} does not exist (DL 477)")
    if problems:
        raise PlanRefused("; ".join(problems))
    plan = {"format": PL.PLAN_FORMAT,
            "sealed": sealed,
            "declared_not_run": dnr,
            "selection_files": {k: Path(v).as_posix() for k, v in sel.items()},
            "code_tree": {"name": CODE_TREE, "path": tree.as_posix()},
            "phase1_reused": [{"arm": a, "seed": s, "rule": r} for a, s, r in PL.DEFAULT_PHASE1_REUSED],
            "s1_free_fields": {"e1_validity_reading": E1_VALIDITY_READING,
                               "part_b_not_run_analyses": list(PART_B_NOT_RUN),
                               "part_b_not_run_reason": PART_B_REASON}}
    PL.check_free_fields(plan["s1_free_fields"])
    PL.check_cover({(i["arm"], i["seed"], "x") for i in sealed}, PL.check_dnr(dnr))   # S4 cover, as the lock checks it
    return plan


def dumps(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_new(path: Path, data: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o444)
    except FileExistsError:
        raise PlanRefused(f"{path} already exists: a plan is written once (O_EXCL)") from None
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(data).hexdigest()


def p1_entry(PL, actor_role: str = "Operations Director") -> dict:
    """The Phase-1 LOCKED_EVALUATION / RETRO_TEST entry (PREG2 §6.3 S2, §13 item 6): all_methods_lock_sha256 = the
    registered ALL_METHODS_LOCK file sha; code_sha256 = its code_ds3r7 aggregate (as register entry 5); exactly the
    register tool's FIELDS (the tool is loaded by path after its pin check)."""
    aml = Path(PL.AML_PATH)
    if not aml.is_file() or _sha(aml) != PL.AML_SHA256:
        raise PlanRefused(f"{aml} is not the registered ALL_METHODS_LOCK ({PL.AML_SHA256[:12]}...)")
    lock = (_json(aml).get("all_methods_lock") or {})
    code = ((lock.get("code_trees") or {}).get("code_ds3r7") or {}).get("aggregate_sha256")
    if not (isinstance(code, str) and re.fullmatch(r"[0-9a-f]{64}", code)):
        raise PlanRefused("ALL_METHODS_LOCK has no code_trees.code_ds3r7.aggregate_sha256")
    entry = {"actor_role": actor_role, "source": "company/gates/ALL_METHODS_LOCK.json", "split": "RETRO_TEST",
             "access_purpose": "LOCKED_EVALUATION", "observations": "NONE",
             "observations_description": (f"LOCKED_EVALUATION for ALL_METHODS_LOCK (Phase-1 methods lock, file sha256 "
                                          f"{PL.AML_SHA256[:16]}, PREG2 S1 / S2, T0a sitting): appended before the first "
                                          "TEST scoring of any arm; no TEST row read, no model scored, no metric"),
             "decision": "NO_FEEDBACK", "all_methods_lock_sha256": PL.AML_SHA256, "manifest_header_sha256": None,
             "model_checkpoint_sha256": None, "code_sha256": code, "receipt_path": "company/gates/ALL_METHODS_LOCK.json"}
    fields = tuple(PL.register_tool().FIELDS)
    if tuple(sorted(entry)) != tuple(sorted(fields)):
        raise PlanRefused(f"entry keys {sorted(entry)} are not the register tool's FIELDS {sorted(fields)}")
    return entry


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="PPSI V4 P2_METHODS_LOCK plan builder (P2-SITTING)")
    sp = ap.add_subparsers(dest="op", required=True)
    a = sp.add_parser("plan")
    a.add_argument("--out", type=Path, default=None)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    a.add_argument("--declare-not-run", nargs="*", default=[])
    a.add_argument("--ft-without-main-run", action="store_true")
    a.add_argument("--p2-lock-tool", type=Path, default=None)
    e = sp.add_parser("p1-entry")
    e.add_argument("--actor-role", default="Operations Director")
    e.add_argument("--p2-lock-tool", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv=None, say=print) -> int:
    args = parse_args(argv)
    try:
        PL = load_p2_lock(args.p2_lock_tool or (DEFAULT_ROOT / "tools" / "p2_lock.py"))
        if args.op == "p1-entry":
            say(json.dumps(p1_entry(PL, args.actor_role), indent=1, sort_keys=True))
            return 0
        if args.out is None and not args.dry_run:
            raise PlanRefused("--out is required (or --dry-run)")
        try:
            plan = build_plan(args.root, PL, extra_dnr=parse_dnr(args.declare_not_run, PL),
                              ft_without_main_run=args.ft_without_main_run)
        except PL.LockRefused as e:
            raise PlanRefused(str(e)) from None
        data = dumps(plan)
        n = f"{len(plan['sealed'])} sealed (arm, seed), {len(plan['declared_not_run'])} declared not run, " \
            f"{len(plan['selection_files'])} selection files, code tree {plan['code_tree']['name']}"
        if args.dry_run:
            say(data.decode("utf-8").rstrip("\n"))
            say(f"P2_LOCK_PLAN DRY RUN: {n}; nothing written")
            return 0
        sha = write_new(args.out, data)
        say(f"P2_LOCK_PLAN: written {args.out} (sha256 {sha}); {n}")
        return 0
    except PlanRefused as e:
        print(f"P2_LOCK_PLAN REFUSED: {e}", file=sys.stderr)
        return REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
