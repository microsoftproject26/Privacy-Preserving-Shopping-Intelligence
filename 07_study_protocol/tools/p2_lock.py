"""tools/p2_lock.py (P2-DEVLOCK piece B = the P2-00 lock tool; owner: Operations Director) - writes the Phase-2 methods
lock company/gates/P2_METHODS_LOCK.json (and the script freeze) WITHOUT opening any metric value, and prints the
LOCKED_EVALUATION register entry that binds it. Stdlib only; every output O_EXCL; a refusal writes nothing (exit 2).

  python -B tools/p2_lock.py lock --plan <P2_LOCK_PLAN.json> [--out company/gates/P2_METHODS_LOCK.json] [--dry-run]
  python -B tools/p2_lock.py freeze [--out company/gates/P2_SCRIPT_FREEZE.json] [--dry-run]
  python -B tools/p2_lock.py register-entry [--lock company/gates/P2_METHODS_LOCK.json] [--actor-role ...]
         -> stdout: the entry JSON for `python -B tools/test_exposure_register.py append --entry <file>` (tool unchanged)
  python -B tools/p2_lock.py verify [--lock company/gates/P2_METHODS_LOCK.json]

Binding spec: contracts/PREG2.md (REGISTERED, DL 411, sha256 be0561ab...) §6.3 S1 / S2 / S4 / S5, §13 items 4-6;
company/plans/PHASE2_IMPLEMENTATION_PLAN.md P2-00; company/handoffs/P2-12.md §3 = the layout tools/p2_report.py
(check_p2_lock / check_p2_freeze / check_locked_evaluations) and v4int/reeval_p2.py (p2_lock_arm, p2_test_gate) read.
Contrasts of record: company/gates/P2_CONTRASTS_v3.json (self e98580564b..., DL 427; PREG2 S1 still names v1).

THE PLAN (written by Ops; value-free; paths absolute or relative to the ppsi_v4 root):
  {"format": "PPSI_V4_P2_LOCK_PLAN_1",
   "sealed": [{"arm": <P2 arm>, "seed": 2026|2027, "run_dir": <MAIN run dir>,
               "expect_sha256": {"PRACTICAL_BEST": <hex>, "CONTROLLED_BUDGET": <hex>} (optional pins, e.g. from a DL row)},
              {"arm": "FT_P2_WSTAR", "seed": S, "run_dir": <its MAIN device run dir> (optional), "expect_base_sha256": ...}],
   "declared_not_run": [{"arm": <P2 arm>, "seed": <int>|null (= every seed), "reason": <06D §8 wording>}],
   "selection_files": {"P2_WARM_SELECTION_FINAL": <path>, "P2_WARM_SELECTION_A": <path>, "P2_C1_SELECTION": <path>,
                       "S_RECORD_P2": <path>},
   "code_tree": {"name": "code_p2r1", "path": <the frozen code tree dir>},
   "phase1_reused": [{"arm", "seed", "rule", "checkpoint_sha256" (optional pin)}]  (optional; default below),
   "s1_free_fields": {"e1_validity_reading": <text>, "part_b_not_run_analyses": [<analysis id>], "part_b_not_run_reason":
                      <text>}  (optional; copied verbatim into the lock's top level: the S1 free fields the P2_SCRIPT_FREEZE
                      ruling binds, DL 475 condition iii)}

WHAT IS READ (identity and hashes only; S8): per sealed run the value-free records - FL: STATUS.json, ARM_SPEC.json,
LAUNCH_<k>.json; central: RUN_START.json, STATUS.json; device: STATUS.json, LAUNCH_<k>.json - and the checkpoint FILE
BYTES for sha256 (never torch-loaded). RUN_SUMMARY / RESULT_ROWS / metrics / evals are never opened. The selection files
are CALIB selection records (format / stage / self-sha checked). The Phase-1 ALL_METHODS_LOCK (pinned 04a3cd19...) gives
the reused checkpoints' sha256 (S5).
CHECKS (any failure refuses): PREG2 file sha = the registered pin; P2_CONTRASTS_v3 format / draft false / document /
utf8-compact self-sha = the pin; a sealed run is finished (FL and C1: STATUS DONE_MAIN; C2, which has PRE's C3
continuation: DONE_MAIN with still_improving not true, or DONE_CONTINUATION; the device run: DONE) and is the arm's MAIN
run (FL: STATUS / ARM_SPEC / every LAUNCH leg agree on arm + run id + arm spec sha, PRODUCTION, role MAIN, study PREG2,
cfg seed, init_from = theta_pre; P2_WSTAR: run id P2_WSTAR_SMALL_s<seed> and every leg's p2_selection sha256 / label =
the P2_WARM_SELECTION_FINAL file / winner the lock records. C1: RUN_START p2_c1 identity arm / variant / role MAIN /
theta_pre and its bound P2_C1_SELECTION sha = the recorded file. C2: run id prefix P2_C2_CFULL_SMALL_, run label, seed);
both rule checkpoints exist (best.pt = PRACTICAL_BEST, endpoint_6.0.pt = CONTROLLED_BUDGET) and match any expect pin;
FT_P2_WSTAR seals (seed, PRACTICAL_BEST) with base_checkpoint_sha256 = the sealed P2_WSTAR best.pt of the same seed (and
its MAIN device run, when given, started from exactly that base); every registered (arm, seed) of PREG2 §9.5 is sealed
or declared not run, never both (S4); the reused Phase-1 entries are bound to ALL_METHODS_LOCK (FT_PRE, a device arm:
null).
LOCK (format PPSI_V4_P2_METHODS_LOCK_1, draft false; self_sha256 by the utf8-compact sort_keys rule): preg2_sha256,
p2_contrasts_self_sha256, bootstrap_seed 20261004, sealed_arms, phase1_reused, declared_not_run,
selection_files_sha256, the code-tree digest (+ the selection tool v4int/p2_select.py sha, PREG2 §5.6), phase1_lock,
plan / tool shas, and the plan's s1_free_fields (when given). It binds CHECKPOINTS only: TEST rows do not exist at lock time (P2-12 §3).
Code-tree digest rule: every regular file under the tree (directories __pycache__ / .pytest_cache and *.pyc skipped,
symlinks refused), {posix relative path: sha256}, aggregate = sha256 of that map as compact sort_keys JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Optional

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
TASK = "P2-DEVLOCK tools/p2_lock.py (the P2-00 lock tool)"
AUTHORITY = ("contracts/PREG2.md (REGISTERED, DL 411) §6.3 S1 / S2 / S4 / S5, §13; company/plans/"
             "PHASE2_IMPLEMENTATION_PLAN.md P2-00; company/handoffs/P2-12.md §3 (the layout p2_report / reeval_p2 read)")
LOCK_FORMAT = "PPSI_V4_P2_METHODS_LOCK_1"
FREEZE_FORMAT = "PPSI_V4_P2_SCRIPT_FREEZE_1"
PLAN_FORMAT = "PPSI_V4_P2_LOCK_PLAN_1"
PREG2_PATH = V4 / "contracts" / "PREG2.md"
PREG2_SHA256 = "be0561abe5ceec450d3a75bef49853a3c480d3847f75559dd94e9e0545df18a8"
CONTRASTS_PATH = V4 / "company" / "gates" / "P2_CONTRASTS_v3.json"
CONTRASTS_FORMAT = "PPSI_V4_P2_CONTRASTS_1"
CONTRASTS_DOCUMENT = "P2_CONTRASTS_v3"
CONTRASTS_SELF_SHA256 = "e98580564b618f411db7e7e328fbcedb6e1134f4be5b65806b34e12229373799"
AML_PATH = V4 / "company" / "gates" / "ALL_METHODS_LOCK.json"
AML_SHA256 = "04a3cd196149e2acf2ab5bb004f58c997199aa52e32f3c11c9ae647255beed23"        # the registered Phase-1 lock
REGISTER_TOOL = HERE / "test_exposure_register.py"
REGISTER_TOOL_SHA256 = "7d59e828f0d8514541ac9cd68393053e064f9c70df3e3655e9c4f0a34eae2622"   # frozen (PREG2 §13 item 6)
RR_SHA256 = "9a3b2c2a918ef2baacc5299426941f1eb94c366f5bb6aa7c29b4bfdbca0afe58"          # tools/retention_report.py
ABI_SHA256 = "6e7043bbe42967b0de313206496de38fb0cf20cc85aaa34945dfdd19863cb662"         # tools/ana_block_index.py
FREEZE_FILES = ("tools/p2_report.py", "tools/p2_block_index.py", "tools/retention_report.py", "tools/ana_block_index.py")
FREEZE_PINS = {"tools/retention_report.py": RR_SHA256, "tools/ana_block_index.py": ABI_SHA256}
BOOTSTRAP_SEED = 20261004                                                               # PREG2 F2 / §9.2
DEFAULT_LOCK = V4 / "company" / "gates" / "P2_METHODS_LOCK.json"
DEFAULT_FREEZE = V4 / "company" / "gates" / "P2_SCRIPT_FREEZE.json"
PRE_CHECKPOINT_SHA256 = "cf08588c11f019cc760f129ba7ba6b23806297660e0170f9938436a5cb88cf78"   # theta_pre
PB, CB = "PRACTICAL_BEST", "CONTROLLED_BUDGET"
RULES = (PB, CB)
RULE_FILES = {PB: "best.pt", CB: "endpoint_6.0.pt"}
AML_RULE_KEY = {PB: "PB", CB: "CB"}
REGISTERED_SEEDS = {                                     # PREG2 §9.5 (= tools/p2_report P2_ARMS / P2_SINGLE_SEED)
    "P2_WSTAR": (2026, 2027), "P2_WSTAR_FA1024": (2026, 2027), "P2_WSTAR_DP8": (2026, 2027), "P2_WSTAR_DP1": (2026,),
    "P2_WSTAR_DPFTRL8": (2026,), "P2_WSTAR_SECAGG": (2026,), "P2_C1_FT_CENTRAL_FULL": (2026, 2027),
    "P2_C1_FT_CENTRAL_FROZEN": (2026, 2027), "P2_C2_CFULL_SMALL": (2026,), "FT_P2_WSTAR": (2026, 2027)}
P2_ARMS = tuple(REGISTERED_SEEDS)
FL_RUN_ARMS = {"P2_WSTAR": ("P2_W",), "P2_WSTAR_FA1024": ("P2_WSTAR_FA1024",),                 # STATUS / LAUNCH arm
               "P2_WSTAR_DP8": ("P2_WSTAR_DP8",), "P2_WSTAR_DP1": ("P2_WSTAR_DP1",),       # launch_fl writes the label
               "P2_WSTAR_DPFTRL8": ("P2_W", "DP"), "P2_WSTAR_SECAGG": ("P2_W",)}           # declared not run (DL 467)
DP_CHAIN_RUN_ID = "{arm}_SMALL_s{seed}"                                                     # = laneP2_RUN3 run dirs
C1_ARMS = {"P2_C1_FT_CENTRAL_FULL": "FULL", "P2_C1_FT_CENTRAL_FROZEN": "FROZEN"}
C2_ARM, C2_RUN_ID_PREFIX = "P2_C2_CFULL_SMALL", "P2_C2_CFULL_SMALL_"
DEVICE_ARM, DEVICE_BASE_ARM, DEVICE_BASE_FILE = "FT_P2_WSTAR", "P2_WSTAR", "best.pt"   # Q5 base_best_only
WSTAR_RUN_ID = "P2_WSTAR_SMALL_s{seed}"
FINISHED = {"FL": ("DONE_MAIN",), "C1": ("DONE_MAIN",), "C2": ("DONE_MAIN", "DONE_CONTINUATION"), "DEVICE": ("DONE",)}
PHASE1_REUSED_ARMS = ("PRE", "FA_WARM_FROZEN", "FT_PRE", "C_FULL")                       # PREG2 §6.3 S5
PHASE1_DEVICE_ARMS = ("FT_PRE",)
DEFAULT_PHASE1_REUSED = (("PRE", 2026, PB), ("PRE", 2026, CB), ("FA_WARM_FROZEN", 2026, PB), ("FA_WARM_FROZEN", 2026, CB),
                         ("FA_WARM_FROZEN", 2027, PB), ("FA_WARM_FROZEN", 2027, CB), ("FT_PRE", 2026, PB),
                         ("FT_PRE", 2027, PB), ("C_FULL", 2026, PB), ("C_FULL", 2026, CB))   # Q1/Q2/Q4/Q5/Q7/Q9 + CB
SELECTION_RECORDS = ("P2_WARM_SELECTION_FINAL", "P2_WARM_SELECTION_A", "P2_C1_SELECTION", "S_RECORD_P2")
FINAL_FORMAT = "PPSI_V4_P2_WARM_SELECTION_FINAL_1"
SELECTION_TOOL_REL = "v4int/p2_select.py"
CODE_TREE_SKIP_DIRS = {"__pycache__", ".pytest_cache"}
FREE_S1_FIELDS = ("e1_validity_reading", "part_b_not_run_analyses", "part_b_not_run_reason")   # DL 475 condition iii
PART_B_ANALYSES = ("E1", "E2", "E3", "E4", "E5_DPFTRL", "E5_LDP")                           # PREG2 §7 analysis ids
SELF_KEY = "self_sha256"
REFUSED = 2


class LockRefused(RuntimeError):
    """A check failed: nothing is written."""


# ------------------------------------------------------------------------------------------------ helpers
def _sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _strict_json(p) -> dict:
    p = Path(p)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"),
                         parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except (OSError, ValueError) as e:
        raise LockRefused(f"{p}: not readable strict JSON ({e})") from None
    if not isinstance(obj, dict):
        raise LockRefused(f"{p}: not a JSON object")
    return obj


def _hex64(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _body(obj: Mapping) -> dict:
    return {k: v for k, v in obj.items() if k != SELF_KEY}


def utf8_self(obj: Mapping) -> str:
    """The rule company/gates/*.json are written with (= tools/p2_report._utf8_compact_self)."""
    return hashlib.sha256(json.dumps(_body(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def self_ok(obj) -> str:
    """retention_report._self_ok's three accepted rules (compact ASCII, indent-1, utf8 compact)."""
    if not isinstance(obj, Mapping) or SELF_KEY not in obj:
        return "NO_SELF_SHA256"
    b = _body(obj)
    compact = hashlib.sha256(json.dumps(b, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    indent = hashlib.sha256((json.dumps(b, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
                            .encode("utf-8")).hexdigest()
    return "VERIFIED" if obj[SELF_KEY] in (compact, indent, utf8_self(obj)) else "MISMATCH"


def with_self(obj: dict) -> dict:
    obj = json.loads(json.dumps(obj, allow_nan=False))
    obj[SELF_KEY] = utf8_self(obj)
    return obj


def dumps(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, data: bytes) -> str:
    """O_EXCL, binary (LF kept), fsync; returns the file sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    except FileExistsError:
        raise LockRefused(f"{path} already exists: a lock is written once (O_EXCL), never overwritten") from None
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(data).hexdigest()


def _p(s) -> Path:
    q = Path(str(s))
    return q if q.is_absolute() else V4 / q


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------ registered inputs
def check_registered() -> dict:
    """PREG2 file sha = the pin; P2_CONTRASTS_v3 final, its utf8-compact self-sha = the pin."""
    if not Path(PREG2_PATH).is_file() or _sha_file(PREG2_PATH) != PREG2_SHA256:
        raise LockRefused(f"{PREG2_PATH} is absent or its sha256 is not the registered {PREG2_SHA256[:12]}... (DL 411)")
    c = _strict_json(CONTRASTS_PATH)
    if c.get("format") != CONTRASTS_FORMAT or c.get("draft") is not False or c.get("document") != CONTRASTS_DOCUMENT:
        raise LockRefused(f"{CONTRASTS_PATH}: not the final {CONTRASTS_DOCUMENT} ({c.get('format')!r}, draft "
                          f"{c.get('draft')!r}, {c.get('document')!r})")
    if c.get(SELF_KEY) != CONTRASTS_SELF_SHA256 or utf8_self(c) != CONTRASTS_SELF_SHA256:
        raise LockRefused(f"{CONTRASTS_PATH}: self_sha256 is not the registered {CONTRASTS_SELF_SHA256[:12]}... (DL 427)")
    return {"preg2_file_sha256": PREG2_SHA256, "p2_contrasts_file_sha256": _sha_file(CONTRASTS_PATH)}


def code_tree_digest(path) -> dict:
    root = _p(path)
    if not root.is_dir():
        raise LockRefused(f"code tree {root} is not a directory")

    def boom(e):
        raise LockRefused(f"code tree {root}: unreadable ({e})")

    files = {}
    for dp, dns, fns in os.walk(root, onerror=boom):
        dns[:] = sorted(d for d in dns if d not in CODE_TREE_SKIP_DIRS)
        for fn in sorted(fns):
            q = Path(dp) / fn
            if q.is_symlink():
                raise LockRefused(f"code tree {root}: symlink {q} (a frozen tree holds regular files only)")
            if fn.endswith(".pyc"):
                continue
            files[q.relative_to(root).as_posix()] = _sha_file(q)
    if not files:
        raise LockRefused(f"code tree {root} is empty")
    agg = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    sel = files.get(SELECTION_TOOL_REL)
    if sel is None:
        raise LockRefused(f"code tree {root} has no {SELECTION_TOOL_REL} (the selection tool's sha is in the lock, "
                          "PREG2 §5.6)")
    return {"aggregate_sha256": agg, "n_files": len(files), "selection_tool": {"path": SELECTION_TOOL_REL, "sha256": sel},
            "rule": "sha256(compact sort_keys JSON of {posix relpath: file sha256}); __pycache__ / .pytest_cache / *.pyc "
                    "skipped, symlinks refused"}


def load_selection_files(spec: Mapping) -> dict:
    if not isinstance(spec, Mapping) or any(k not in SELECTION_RECORDS for k in spec):
        raise LockRefused(f"selection_files keys must be among {SELECTION_RECORDS}")
    out = {}
    for key, path in spec.items():
        p = _p(path)
        if not p.is_file():
            raise LockRefused(f"selection file {key} {p} does not exist")
        rec = {"path": str(path), "sha256": _sha_file(p)}
        if key == "P2_WARM_SELECTION_FINAL":
            obj = _strict_json(p)
            if obj.get("format") != FINAL_FORMAT or obj.get("stage") != "FINAL" or self_ok(obj) != "VERIFIED":
                raise LockRefused(f"{p} is not a self-verified {FINAL_FORMAT} (stage FINAL) record")
            rec.update(format=FINAL_FORMAT, self_sha256=obj[SELF_KEY], winner=obj.get("winner"))
        out[key] = rec
    return out


# ------------------------------------------------------------------------------------------------ sealing
def _entry(arm, seed, rule, ckpt: Path, run_id, kind, expect) -> dict:
    if not ckpt.is_file():
        raise LockRefused(f"{arm} s{seed} {rule}: checkpoint {ckpt} does not exist")
    sha = _sha_file(ckpt)
    want = (expect or {}).get(rule)
    if want is not None and str(want).lower() != sha:
        raise LockRefused(f"{arm} s{seed} {rule}: checkpoint sha256 {sha} differs from the expected {want}")
    return {"arm": arm, "seed": seed, "rule": rule, "checkpoint_sha256": sha, "checkpoint": ckpt.name,
            "run_id": run_id, "source_kind": kind}


def _status(rd: Path, kind: str, arm: str, seed: int, run_id: Optional[str] = None) -> dict:
    st = _strict_json(rd / "STATUS.json")
    if st.get("status") not in FINISHED[kind]:
        raise LockRefused(f"{arm} s{seed}: {rd} STATUS {st.get('status')!r} is not {' / '.join(FINISHED[kind])} (S4: a "
                          "run unfinished at lock time is declared not run)")
    if run_id is not None and st.get("run_id") not in (None, run_id):
        raise LockRefused(f"{arm} s{seed}: STATUS run id {st.get('run_id')!r} is not {run_id!r}")
    if kind == "C2" and st.get("status") == "DONE_MAIN" and st.get("still_improving") is True:
        raise LockRefused(f"{arm} s{seed}: DONE_MAIN with STILL_IMPROVING - its C3 continuation is not finished")
    return st


def seal_fl(arm: str, seed: int, rd: Path, expect, sel: Mapping) -> list:
    st = _status(rd, "FL", arm, seed)
    rid, run_arm = st.get("run_id"), st.get("arm")
    if not rid or run_arm not in FL_RUN_ARMS[arm]:
        raise LockRefused(f"{arm} s{seed}: STATUS arm {run_arm!r} / run id {rid!r} is not a run of {arm} "
                          f"{FL_RUN_ARMS[arm]}")
    side = _strict_json(rd / "ARM_SPEC.json")
    spec = side.get("arm_spec") or {}
    if side.get("run_id") != rid or side.get("arm") != run_arm:
        raise LockRefused(f"{arm} s{seed}: ARM_SPEC run id / arm differ from STATUS")
    if spec.get("study") != "PREG2" or spec.get("role", "MAIN") != "MAIN" or (spec.get("cfg") or {}).get("seed") != seed:
        raise LockRefused(f"{arm} s{seed}: arm spec study {spec.get('study')!r} / role {spec.get('role')!r} / seed "
                          f"{(spec.get('cfg') or {}).get('seed')!r} is not a PREG2 MAIN run of seed {seed}")
    if (spec.get("init_from") or {}).get("checkpoint_sha256") != PRE_CHECKPOINT_SHA256:
        raise LockRefused(f"{arm} s{seed}: init_from is not theta_pre ({PRE_CHECKPOINT_SHA256[:12]}...)")
    launches = sorted(rd.glob("LAUNCH_*.json"))
    if not launches:
        raise LockRefused(f"{arm} s{seed}: {rd} has no LAUNCH_<k>.json")
    legs = [_strict_json(q) for q in launches]
    for q, L in zip(launches, legs):
        if (L.get("run_id"), L.get("arm"), L.get("arm_spec_sha256")) != (rid, run_arm, side.get("arm_spec_sha256")) or \
                L.get("label") != "PRODUCTION" or (L.get("fl05") or {}).get("role", "MAIN") != "MAIN":
            raise LockRefused(f"{arm} s{seed}: {q.name} is not a PRODUCTION MAIN leg of {rid} (arm / run id / arm spec)")
    if arm in ("P2_WSTAR_FA1024", "P2_WSTAR_DP8", "P2_WSTAR_DP1") and rid != DP_CHAIN_RUN_ID.format(arm=arm, seed=seed):
        raise LockRefused(f"{arm} s{seed}: run id {rid!r} is not {DP_CHAIN_RUN_ID.format(arm=arm, seed=seed)!r}")
    if arm == "P2_WSTAR":
        if rid != WSTAR_RUN_ID.format(seed=seed):
            raise LockRefused(f"{arm} s{seed}: run id {rid!r} is not {WSTAR_RUN_ID.format(seed=seed)} (the MAIN run bound "
                              "to the selection)")
        fin = sel.get("P2_WARM_SELECTION_FINAL")
        if fin is None:
            raise LockRefused("a sealed P2_WSTAR needs selection_files.P2_WARM_SELECTION_FINAL (its run is bound to it)")
        for q, L in zip(launches, legs):
            b = L.get("p2_selection") or {}
            if b.get("sha256") != fin["sha256"] or b.get("label") != fin.get("winner"):
                raise LockRefused(f"{arm} s{seed}: {q.name} p2_selection (sha256 {str(b.get('sha256'))[:12]}..., label "
                                  f"{b.get('label')!r}) is not the recorded P2_WARM_SELECTION_FINAL "
                                  f"({fin['sha256'][:12]}..., winner {fin.get('winner')!r})")
    ck = rd / "ckpt" / rid
    return [_entry(arm, seed, r, ck / RULE_FILES[r], rid, "FL", expect) for r in RULES]


def _central_ckpt_dir(rd: Path) -> Path:
    return rd / "sealed" / "checkpoints" if (rd / "sealed" / "checkpoints").is_dir() else rd / "checkpoints"


def seal_central(arm: str, seed: int, rd: Path, expect, sel: Mapping) -> list:
    start = _strict_json(rd / "RUN_START.json")                         # config and hashes only (no outcome)
    rid = start.get("run_id")
    if (start.get("config") or {}).get("seed") != seed:
        raise LockRefused(f"{arm} s{seed}: RUN_START config seed {(start.get('config') or {}).get('seed')!r}")
    if arm in C1_ARMS:
        kind = "C1"
        p2 = start.get("p2_c1") or {}
        ident = p2.get("identity") or {}
        want = {"arm": arm, "variant": C1_ARMS[arm], "role": "MAIN", "init_checkpoint_sha256": PRE_CHECKPOINT_SHA256}
        bad = {k: ident.get(k) for k, v in want.items() if ident.get(k) != v}
        if bad:
            raise LockRefused(f"{arm} s{seed}: RUN_START p2_c1 identity {bad} is not a MAIN C1 run of {arm} from theta_pre")
        c1 = sel.get("P2_C1_SELECTION")
        if c1 is None:
            raise LockRefused("a sealed C1 arm needs selection_files.P2_C1_SELECTION (its run is bound to it)")
        if (p2.get("selection") or {}).get("sha256") != c1["sha256"]:
            raise LockRefused(f"{arm} s{seed}: the run's bound P2_C1_SELECTION sha256 "
                              f"{str((p2.get('selection') or {}).get('sha256'))[:12]}... is not the recorded file's "
                              f"{c1['sha256'][:12]}...")
    else:
        kind = "C2"
        if not str(rid or "").startswith(C2_RUN_ID_PREFIX) or (start.get("key_base") or {}).get("run_label") != C2_ARM:
            raise LockRefused(f"{arm} s{seed}: run id {rid!r} / run label is not the C2 run ({C2_RUN_ID_PREFIX}*)")
    _status(rd, kind, arm, seed, rid)
    ck = _central_ckpt_dir(rd)
    return [_entry(arm, seed, r, ck / RULE_FILES[r], rid, kind, expect) for r in RULES]


def seal_device(item: Mapping, seed: int, sealed: Mapping) -> dict:
    w = sealed.get((DEVICE_BASE_ARM, seed, PB))
    if w is None:
        raise LockRefused(f"{DEVICE_ARM} s{seed}: its base, the P2_WSTAR s{seed} best.pt, is not sealed (Q5 "
                          "base_best_only; declare FT_P2_WSTAR not run instead)")
    base = w["checkpoint_sha256"]
    want = item.get("expect_base_sha256")
    if want is not None and str(want).lower() != base:
        raise LockRefused(f"{DEVICE_ARM} s{seed}: the sealed W* best.pt {base} differs from the expected base {want}")
    main_run = None
    if item.get("run_dir") is not None:
        rd = _p(item["run_dir"])
        st = _status(rd, "DEVICE", DEVICE_ARM, seed)
        launches = sorted(rd.glob("LAUNCH_*.json"))
        if not launches:
            raise LockRefused(f"{DEVICE_ARM} s{seed}: {rd} has no LAUNCH_<k>.json")
        for q in launches:
            s = _strict_json(q).get("start") or {}
            if (s.get("kind"), s.get("seed"), s.get("checkpoint"), s.get("file_sha256")) != \
                    (DEVICE_BASE_ARM, seed, DEVICE_BASE_FILE, base):
                raise LockRefused(f"{DEVICE_ARM} s{seed}: {q.name} start {dict(s)} is not the sealed P2_WSTAR s{seed} "
                                  "best.pt")
        main_run = {"run_dir": str(item["run_dir"]), "status": st.get("status"),
                    "launch_files_sha256": {q.name: _sha_file(q) for q in launches}}
    return {"arm": DEVICE_ARM, "seed": seed, "rule": PB, "base_checkpoint_sha256": base, "base_run_id": w["run_id"],
            "base_checkpoint": DEVICE_BASE_FILE, "source_kind": "DEVICE", "main_run": main_run}


def phase1_reused(items, aml_path: Path) -> tuple:
    if _sha_file(aml_path) != AML_SHA256:
        raise LockRefused(f"{aml_path} is not the registered Phase-1 ALL_METHODS_LOCK ({AML_SHA256[:12]}...)")
    cks = (_strict_json(aml_path).get("all_methods_lock") or {}).get("checkpoints") or {}
    raw = [dict(arm=a, seed=s, rule=r) for a, s, r in DEFAULT_PHASE1_REUSED] if items is None else list(items)
    out, seen = [], set()
    for it in raw:
        a, s, r = it.get("arm"), it.get("seed"), it.get("rule")
        if a not in PHASE1_REUSED_ARMS or s not in (2026, 2027) or r not in RULES or (a, s, r) in seen:
            raise LockRefused(f"phase1_reused {(a, s, r)} is not a registered (arm, seed, rule) of S5 or is listed twice")
        seen.add((a, s, r))
        if a in PHASE1_DEVICE_ARMS:
            if r != PB:
                raise LockRefused(f"phase1_reused {a} is a device arm: PRACTICAL_BEST only")
            ck = None
        else:
            rec = (cks.get(f"{a}_s{s}") or {}).get(AML_RULE_KEY[r]) or {}
            ck = rec.get("sha")
            if not _hex64(ck):
                raise LockRefused(f"phase1_reused {(a, s, r)}: no checkpoint sha256 in ALL_METHODS_LOCK")
        pin = it.get("checkpoint_sha256")
        if pin is not None and pin != ck:
            raise LockRefused(f"phase1_reused {(a, s, r)}: checkpoint_sha256 {pin} is not ALL_METHODS_LOCK's {ck}")
        out.append({"arm": a, "seed": s, "rule": r, "checkpoint_sha256": ck})
    return out, {"path": "company/gates/ALL_METHODS_LOCK.json", "file_sha256": AML_SHA256,
                 "retention_report_sha256": RR_SHA256,
                 "rule": "S5: the reused Phase-1 rows are produced once by the Phase-1 plan; checkpoints bound here, rows "
                         "by the sitting's Phase-1 RETENTION_RESULT (tools/p2_report.check_phase1_binding)"}


def check_dnr(items) -> list:
    out, seen = [], set()
    for i, it in enumerate(items or []):
        if not isinstance(it, Mapping) or it.get("arm") not in P2_ARMS or not str(it.get("reason") or "").strip():
            raise LockRefused(f"declared_not_run[{i}] needs a P2 arm and a reason (06D §8 wording)")
        s = it.get("seed")
        if s is not None and s not in REGISTERED_SEEDS[it["arm"]]:
            raise LockRefused(f"declared_not_run[{i}]: {it['arm']} seed {s} is not registered (PREG2 §9.5)")
        if (it["arm"], s) in seen:
            raise LockRefused(f"declared_not_run lists ({it['arm']}, {s}) twice")
        seen.add((it["arm"], s))
        out.append({"arm": it["arm"], "seed": s, "reason": str(it["reason"])})
    return out


def check_free_fields(spec) -> dict:
    """The S1 free fields (PREG2 §6.3 S1; P2_SCRIPT_FREEZE ruling, DL 475): text fields non-empty, the analysis list a
    non-empty list of distinct PREG2 §7 analysis ids. Any other key refuses."""
    if spec is None:
        return {}
    if not isinstance(spec, Mapping) or any(k not in FREE_S1_FIELDS for k in spec):
        raise LockRefused(f"s1_free_fields keys must be among {FREE_S1_FIELDS}")
    out = {}
    for k, v in spec.items():
        if k == "part_b_not_run_analyses":
            if not isinstance(v, list) or not v or len(set(map(str, v))) != len(v) or \
                    any(x not in PART_B_ANALYSES for x in v):
                raise LockRefused(f"s1_free_fields.{k} must be a non-empty list of distinct ids among {PART_B_ANALYSES}")
            out[k] = [str(x) for x in v]
        else:
            if not isinstance(v, str) or not v.strip():
                raise LockRefused(f"s1_free_fields.{k} must be a non-empty text")
            out[k] = v
    return out


def check_cover(sealed_keys, dnr) -> None:
    """S4: every registered (arm, seed) is sealed or declared not run, never both."""
    have = {(a, s) for a, s, _r in sealed_keys}
    for d in dnr:
        hit = [k for k in have if k[0] == d["arm"] and (d["seed"] is None or k[1] == d["seed"])]
        if hit:
            raise LockRefused(f"{d['arm']} seed {d['seed']} is both sealed and declared not run")
    for arm, seeds in REGISTERED_SEEDS.items():
        for s in seeds:
            if (arm, s) not in have and not any(d["arm"] == arm and d["seed"] in (None, s) for d in dnr):
                raise LockRefused(f"{arm} seed {s} is neither sealed nor declared not run (S4: an arm without a sealed "
                                  "checkpoint at lock time is declared not run)")


# ------------------------------------------------------------------------------------------------ the lock
def build_lock(plan: Mapping, *, plan_sha256: Optional[str] = None, aml_path: Path = None) -> dict:
    if plan.get("format") != PLAN_FORMAT:
        raise LockRefused(f"the plan is {plan.get('format')!r}, not {PLAN_FORMAT}")
    free = check_free_fields(plan.get("s1_free_fields"))
    reg = check_registered()
    sel = load_selection_files(plan.get("selection_files") or {})
    tree = plan.get("code_tree") or {}
    if not tree.get("name") or not tree.get("path"):
        raise LockRefused("the plan needs code_tree {name, path} (S1: the code-tree digest)")
    ct = dict(code_tree_digest(tree["path"]), name=str(tree["name"]))
    items, seen = list(plan.get("sealed") or []), set()
    for it in items:
        a, s = it.get("arm"), it.get("seed")
        if a not in P2_ARMS or s not in REGISTERED_SEEDS.get(a, ()) or (a, s) in seen:
            raise LockRefused(f"sealed ({a}, {s}) is not a registered P2 (arm, seed) of PREG2 §9.5 or is listed twice")
        seen.add((a, s))
        if a != DEVICE_ARM and not it.get("run_dir"):
            raise LockRefused(f"sealed ({a}, {s}) needs run_dir")
    sealed = {}
    for it in items:
        a, s = it["arm"], it["seed"]
        if a == DEVICE_ARM:
            continue
        rd = _p(it["run_dir"])
        es = seal_fl(a, s, rd, it.get("expect_sha256"), sel) if a in FL_RUN_ARMS else \
            seal_central(a, s, rd, it.get("expect_sha256"), sel)
        for e in es:
            sealed[(e["arm"], e["seed"], e["rule"])] = dict(e, run_dir=str(it["run_dir"]))
    for it in items:
        if it["arm"] == DEVICE_ARM:
            e = seal_device(it, it["seed"], sealed)
            sealed[(e["arm"], e["seed"], e["rule"])] = e
    dnr = check_dnr(plan.get("declared_not_run"))
    check_cover(sealed, dnr)
    reused, p1 = phase1_reused(plan.get("phase1_reused"), Path(aml_path or AML_PATH))
    lock = {"format": LOCK_FORMAT, "draft": False, "document": "P2_METHODS_LOCK", "study": "PREG2", "task": TASK,
            "authority": AUTHORITY, "written_utc": _utc(), "preg2_sha256": PREG2_SHA256,
            "p2_contrasts_self_sha256": CONTRASTS_SELF_SHA256, "p2_contrasts_document": CONTRASTS_DOCUMENT,
            "p2_contrasts_file_sha256": reg["p2_contrasts_file_sha256"], "bootstrap_seed": BOOTSTRAP_SEED,
            "sealed_arms": [sealed[k] for k in sorted(sealed, key=lambda k: (k[0], k[1], k[2]))],
            "phase1_reused": reused, "phase1_lock": p1, "declared_not_run": dnr,
            "selection_files_sha256": {k: v["sha256"] for k, v in sel.items()}, "selection_files": sel,
            "selection_tool": ct["selection_tool"], "code_tree": ct, "plan_sha256": plan_sha256,
            "lock_tool_sha256": _sha_file(Path(__file__)),
            "binds": "checkpoints only (TEST rows do not exist at lock time, P2-12 §3); read: STATUS / ARM_SPEC / LAUNCH / "
                     "RUN_START records and checkpoint file bytes - no metric value was opened (PREG2 §6.3 S8)",
            "self_sha256_rule": "utf8 compact sort_keys (json.dumps(body, sort_keys=True, separators=(',', ':'), "
                                "ensure_ascii=False).encode('utf-8'))"}
    lock.update(free)
    lock = with_self(lock)
    verify_lock(lock)
    return lock


def verify_lock(obj: Mapping) -> dict:
    """The lock's own invariants (the readers re-check their part: p2_report.check_p2_lock, reeval_p2.p2_lock_arm)."""
    if not isinstance(obj, Mapping) or obj.get("format") != LOCK_FORMAT or obj.get("draft") is not False:
        raise LockRefused(f"not a {LOCK_FORMAT} lock with draft false")
    if obj.get(SELF_KEY) != utf8_self(obj):
        raise LockRefused("self_sha256 does not verify under the utf8-compact rule")
    for k, want in (("preg2_sha256", PREG2_SHA256), ("p2_contrasts_self_sha256", CONTRASTS_SELF_SHA256),
                    ("bootstrap_seed", BOOTSTRAP_SEED)):
        if obj.get(k) != want:
            raise LockRefused(f"{k} {obj.get(k)!r} is not {want!r}")
    keys = []
    for i, e in enumerate(obj.get("sealed_arms") or []):
        k = (e.get("arm"), e.get("seed"), e.get("rule"))
        if k[0] not in P2_ARMS or k[1] not in REGISTERED_SEEDS.get(k[0], ()) or k[2] not in RULES or k in keys:
            raise LockRefused(f"sealed_arms[{i}] {k} is not a registered (arm, seed, rule) or is listed twice")
        ok = (k[2] == PB and _hex64(e.get("base_checkpoint_sha256"))) if k[0] == DEVICE_ARM else \
            _hex64(e.get("checkpoint_sha256"))
        if not ok:
            raise LockRefused(f"sealed_arms[{i}] {k} has no checkpoint / base sha256")
        keys.append(k)
    check_cover(keys, check_dnr(obj.get("declared_not_run")))
    check_free_fields({k: obj[k] for k in FREE_S1_FIELDS if k in obj} or None)
    sel = obj.get("selection_files_sha256") or {}
    if any(k not in SELECTION_RECORDS or not _hex64(v) for k, v in sel.items()):
        raise LockRefused("selection_files_sha256 must map selection record keys to sha256 values")
    return {"n_sealed": len(keys), "n_declared_not_run": len(obj.get("declared_not_run") or []),
            "n_phase1_reused": len(obj.get("phase1_reused") or []), "self_sha256": obj[SELF_KEY]}


def write_lock(plan_path, out: Path = DEFAULT_LOCK, *, dry_run: bool = False, aml_path: Path = None,
               say=print) -> dict:
    plan_path = Path(plan_path)
    plan = _strict_json(plan_path)
    if Path(out).exists() and not dry_run:
        raise LockRefused(f"{out} already exists: a lock is written once (O_EXCL), never overwritten")
    lock = build_lock(plan, plan_sha256=_sha_file(plan_path), aml_path=aml_path)
    summ = verify_lock(lock)
    if dry_run:
        say(f"P2_LOCK DRY RUN: {summ['n_sealed']} sealed, {summ['n_declared_not_run']} declared not run, "
            f"{summ['n_phase1_reused']} Phase-1 reused; nothing written")
        return dict(summ, written=None)
    sha = _write_new(Path(out), dumps(lock))
    say(f"P2_LOCK: written {out} (file sha256 {sha}; self_sha256 {lock[SELF_KEY]}); {summ['n_sealed']} sealed, "
        f"{summ['n_declared_not_run']} declared not run")
    return dict(summ, written=str(out), file_sha256=sha)


# ------------------------------------------------------------------------------------------------ freeze / register
def write_freeze(out: Path = DEFAULT_FREEZE, *, dry_run: bool = False, say=print) -> dict:
    """P2_SCRIPT_FREEZE (PREG2 §6.3 S2; P2-12 §3): the four running files; retention_report / ana_block_index must be
    the frozen Phase-1 shas."""
    files = {}
    for rel in FREEZE_FILES:
        p = V4 / rel
        if not p.is_file():
            raise LockRefused(f"{rel} does not exist")
        files[rel] = _sha_file(p)
    bad = {k: files[k] for k, v in FREEZE_PINS.items() if files[k] != v}
    if bad:
        raise LockRefused(f"the freeze refuses {sorted(bad)}: not the frozen Phase-1 sha256 "
                          f"{ {k: v[:8] for k, v in FREEZE_PINS.items()} }")
    obj = with_self({"format": FREEZE_FORMAT, "document": "P2_SCRIPT_FREEZE", "task": TASK, "authority": AUTHORITY,
                     "written_utc": _utc(), "files_sha256": files, "preg2_sha256": PREG2_SHA256})
    if dry_run:
        say(f"P2_FREEZE DRY RUN: {len(files)} files; nothing written")
        return {"files_sha256": files, "written": None}
    sha = _write_new(Path(out), dumps(obj))
    say(f"P2_FREEZE: written {out} (file sha256 {sha})")
    return {"files_sha256": files, "written": str(out), "file_sha256": sha}


def register_tool():
    if _sha_file(REGISTER_TOOL) != REGISTER_TOOL_SHA256:
        raise LockRefused(f"{REGISTER_TOOL} is not the frozen register tool ({REGISTER_TOOL_SHA256[:8]}...)")
    spec = importlib.util.spec_from_file_location("p2_lock_register_tool", REGISTER_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def register_entry(lock_path: Path = DEFAULT_LOCK, *, actor_role: str = "Operations Director",
                   receipt_path: Optional[str] = None) -> dict:
    """The LOCKED_EVALUATION / RETRO_TEST entry whose all_methods_lock_sha256 is the P2 lock FILE sha256 and whose
    observations_description names P2_METHODS_LOCK (PREG2 §6.3 S2, §13 item 6); exactly the register tool's FIELDS."""
    lock_path = Path(lock_path)
    obj = _strict_json(lock_path)
    verify_lock(obj)
    fsha = _sha_file(lock_path)
    try:
        src = lock_path.resolve().relative_to(V4.resolve()).as_posix()
    except ValueError:
        src = str(lock_path)
    entry = {"actor_role": actor_role, "source": src, "split": "RETRO_TEST", "access_purpose": "LOCKED_EVALUATION",
             "observations": "NONE",
             "observations_description": (f"LOCKED_EVALUATION for P2_METHODS_LOCK (Phase-2 methods lock, self_sha256 "
                                          f"{obj[SELF_KEY][:16]}, PREG2 S1 / S2): appended before the first TEST "
                                          "scoring of any arm; no TEST row read, no model scored, no metric"),
             "decision": "NO_FEEDBACK", "all_methods_lock_sha256": fsha, "manifest_header_sha256": None,
             "model_checkpoint_sha256": None, "code_sha256": (obj.get("code_tree") or {}).get("aggregate_sha256"),
             "receipt_path": receipt_path if receipt_path is not None else src}
    fields = tuple(register_tool().FIELDS)
    if tuple(sorted(entry)) != tuple(sorted(fields)):
        raise LockRefused(f"entry keys {sorted(entry)} are not the register tool's FIELDS {sorted(fields)}")
    return entry


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="PPSI V4 Phase-2 lock tool (P2-00)")
    sp = ap.add_subparsers(dest="op", required=True)
    a = sp.add_parser("lock")
    a.add_argument("--plan", required=True, type=Path)
    a.add_argument("--out", type=Path, default=DEFAULT_LOCK)
    a.add_argument("--dry-run", action="store_true")
    f = sp.add_parser("freeze")
    f.add_argument("--out", type=Path, default=DEFAULT_FREEZE)
    f.add_argument("--dry-run", action="store_true")
    r = sp.add_parser("register-entry")
    r.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    r.add_argument("--actor-role", default="Operations Director")
    r.add_argument("--receipt-path", default=None)
    v = sp.add_parser("verify")
    v.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return ap.parse_args(argv)


def main(argv=None, *, say=print) -> int:
    args = parse_args(argv)
    try:
        if args.op == "lock":
            write_lock(args.plan, args.out, dry_run=args.dry_run, say=say)
        elif args.op == "freeze":
            write_freeze(args.out, dry_run=args.dry_run, say=say)
        elif args.op == "register-entry":
            say(json.dumps(register_entry(args.lock, actor_role=args.actor_role, receipt_path=args.receipt_path),
                           indent=1, sort_keys=True))
        else:
            say(json.dumps(verify_lock(_strict_json(args.lock)), sort_keys=True))
        return 0
    except LockRefused as e:
        print(f"P2_LOCK REFUSED: {e}", file=sys.stderr)
        return REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
