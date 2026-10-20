"""v4data09.d9_gate (owner: Data engineer, DATA-09) - the TEST gate of MANIFEST_SPEC_V4 global_rules.test_gate /
EVALUATION_CONTRACT_V4 E10.1 ("the builder and the loader both refuse otherwise").

  check_test_gate(lock_path, register_path) -> GateRecord     refuses (TestGateRefused) unless ALL of these hold:
    1. company/gates/ALL_METHODS_LOCK.json exists (a regular file);
    2. the TEST_EXPOSURE_REGISTER is strict JSON and its hash chain verifies - the verification is the existing Ops tool
       tools/test_exposure_register.py `verify` (loaded by path, not copied);
    3. an entry with access_purpose == "MANIFEST_BUILD", split == "RETRO_TEST" carries all_methods_lock_sha256 == the
       sha256 of the lock file bytes. If several entries qualify, the FIRST (lowest entry_id) is the binding one, so a
       later append (LOCKED_EVALUATION, RESTART_INCOMPLETE, ...) never changes the id written into the manifest headers.
  The returned record carries the two MANIFEST_SPEC M07 header_extra values (`all_methods_lock_sha256`,
  `test_exposure_register_entry_id`) plus the entry / register hashes for receipts.
  revalidate(gate) re-runs the whole check on the record's own paths and refuses unless the result is identical (the
  lock and the binding entry unchanged). d9_read.read_with_test calls it before every scan, so a GateRecord cannot be
  forged or outlive its lock.
Nothing here reads the dataset. Every builder and loader entry point calls check_test_gate FIRST (before any file of the
dataset, a TEST manifest or the TEST registry extension is opened).

CLI (server step 1 of the look, read-only):  python -B -m v4data09.d9_gate [--lock P] [--register P]
  prints the GateRecord as JSON and exits 0, or prints the refusal and exits 2.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from v4data.common import TestGateError  # noqa: E402

DEFAULT_LOCK = V4 / "company" / "gates" / "ALL_METHODS_LOCK.json"
DEFAULT_REGISTER = V4 / "company" / "registers" / "TEST_EXPOSURE_REGISTER.json"
REGISTER_TOOL = V4 / "tools" / "test_exposure_register.py"
PURPOSE = "MANIFEST_BUILD"
SPLIT = "RETRO_TEST"


class TestGateRefused(TestGateError):
    """The TEST gate is not satisfied: nothing TEST-related may be built, read or resolved."""


@dataclass(frozen=True)
class GateRecord:
    lock_path: str
    lock_sha256: str                    # = all_methods_lock_sha256 (MANIFEST_SPEC M07 header_extra)
    register_path: str
    register_file_sha256: str
    entry_id: int                       # = test_exposure_register_entry_id (MANIFEST_SPEC M07 header_extra)
    entry_sha256: str
    entry_utc: str
    n_register_entries: int
    checked_utc: str

    def header_fields(self) -> dict:
        return {"all_methods_lock_sha256": self.lock_sha256, "test_exposure_register_entry_id": self.entry_id}

    def binding(self) -> dict:
        """The time-independent identity of the gate (what a header / the registry extension records)."""
        return {"all_methods_lock_sha256": self.lock_sha256, "test_exposure_register_entry_id": self.entry_id,
                "test_exposure_register_entry_sha256": self.entry_sha256,
                "rule": "v4data09.d9_gate.check_test_gate: ALL_METHODS_LOCK.json file sha256 == the FIRST "
                        "TEST_EXPOSURE_REGISTER entry with access_purpose MANIFEST_BUILD and split RETRO_TEST; the "
                        "register hash chain verifies (tools/test_exposure_register.py verify)"}

    def as_json(self) -> dict:
        return asdict(self)


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _register_tool(tool_path: Path = REGISTER_TOOL):
    """The Ops tool module (its `verify` is THE chain verification), loaded by path from ppsi_v4/tools."""
    if not Path(tool_path).is_file():
        raise TestGateRefused(f"the TEST_EXPOSURE_REGISTER tool {tool_path} is missing: the chain cannot be verified")
    spec = importlib.util.spec_from_file_location("_ppsi_test_exposure_register_tool", str(tool_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strict_json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"),
                          parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except (OSError, ValueError) as e:
        raise TestGateRefused(f"{p}: not readable strict JSON ({e})") from None


def check_test_gate(lock_path: Optional[Path] = None, register_path: Optional[Path] = None, *,
                    tool_path: Path = REGISTER_TOOL) -> GateRecord:
    lock_path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK
    register_path = Path(register_path) if register_path is not None else DEFAULT_REGISTER
    if not lock_path.is_file():
        raise TestGateRefused(f"ALL_METHODS_LOCK {lock_path} does not exist: every TEST manifest build, TEST row read "
                              "and TEST role resolution is refused (E10.1; MANIFEST_SPEC global_rules.test_gate)")
    lock_sha = _sha_file(lock_path)
    if not register_path.is_file():
        raise TestGateRefused(f"TEST_EXPOSURE_REGISTER {register_path} does not exist")
    reg_file_sha = _sha_file(register_path)
    reg = _strict_json(register_path)
    entries = reg.get("entries") if isinstance(reg, dict) else None
    if not isinstance(entries, list) or not entries:
        raise TestGateRefused(f"{register_path}: no entries")
    tool = _register_tool(tool_path)
    try:
        errs = tool.verify(reg)
    except Exception as e:                              # a malformed entry breaks the chain verification itself
        raise TestGateRefused(f"{register_path}: the hash chain cannot be verified ({type(e).__name__}: {e})") from None
    if errs:
        raise TestGateRefused(f"{register_path}: the hash chain does not verify: {errs}")
    hits = [e for e in entries if isinstance(e, dict) and e.get("access_purpose") == PURPOSE
            and e.get("split") == SPLIT and e.get("all_methods_lock_sha256") == lock_sha]
    if not hits:
        purposes = sorted({str(e.get("access_purpose")) for e in entries if isinstance(e, dict)})
        raise TestGateRefused(f"no TEST_EXPOSURE_REGISTER entry with access_purpose {PURPOSE}, split {SPLIT} and "
                              f"all_methods_lock_sha256 == sha256(ALL_METHODS_LOCK) {lock_sha} (purposes present: "
                              f"{purposes})")
    e = min(hits, key=lambda x: int(x["entry_id"]))
    return GateRecord(lock_path=str(lock_path), lock_sha256=lock_sha, register_path=str(register_path),
                      register_file_sha256=reg_file_sha, entry_id=int(e["entry_id"]), entry_sha256=e["entry_sha256"],
                      entry_utc=str(e.get("utc")), n_register_entries=len(entries),
                      checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def revalidate(gate) -> GateRecord:
    """Re-run the gate on the record's own paths; the lock sha, the binding entry id and its sha must be unchanged."""
    if not isinstance(gate, GateRecord):
        raise TestGateRefused(f"a GateRecord from v4data09.d9_gate.check_test_gate is required (got {type(gate).__name__})")
    now = check_test_gate(Path(gate.lock_path), Path(gate.register_path))
    if (now.lock_sha256, now.entry_id, now.entry_sha256) != (gate.lock_sha256, gate.entry_id, gate.entry_sha256):
        raise TestGateRefused("the TEST gate changed since the GateRecord was issued (lock or binding entry differs)")
    return now


def header_matches_gate(header: dict, gate: GateRecord) -> bool:
    """A TEST manifest header is served only under the gate it was built under."""
    return (header.get("all_methods_lock_sha256") == gate.lock_sha256
            and header.get("test_exposure_register_entry_id") == gate.entry_id)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DATA-09 TEST gate check (read-only)")
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--register", type=Path, default=DEFAULT_REGISTER)
    a = ap.parse_args(argv)
    try:
        g = check_test_gate(a.lock, a.register)
    except TestGateError as e:
        print(f"TEST GATE REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps({"gate": "OPEN", **g.as_json()}, indent=1, sort_keys=True))
    return 0


__all__ = ["GateRecord", "TestGateRefused", "check_test_gate", "revalidate", "header_matches_gate", "DEFAULT_LOCK",
           "DEFAULT_REGISTER", "PURPOSE", "SPLIT"]

if __name__ == "__main__":
    raise SystemExit(main())
