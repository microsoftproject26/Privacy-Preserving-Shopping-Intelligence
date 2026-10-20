"""T1: fill the BIND / LOOK register drafts from the FINAL exploratory lock (nothing is appended here).

  python -B tools_x2/t1_fill_entries.py
writes company/gates/REGISTER_ENTRY_EXP_BENCH_X_BIND.json and _LOOK.json (write-once) with
all_methods_lock_sha256 = sha256 of company/gates/EXP_BENCH_X_TEST_LOCK.json (the FILE) and code_sha256 = the lock's
code.final_x_sha256 (which must equal tools_x2/final_x.py now). Ops appends them with tools/test_exposure_register.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(os.path.dirname(HERE))
G = os.path.join(V4, "company", "gates")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    lp = os.path.join(G, "EXP_BENCH_X_TEST_LOCK.json")
    lock = json.load(open(lp))
    if lock.get("draft") or lock.get("pending"):
        print("REFUSED: the lock is a draft", file=sys.stderr)
        return 2
    fx = sha(os.path.join(HERE, "final_x.py"))
    if fx != lock["code"]["final_x_sha256"]:
        print(f"REFUSED: final_x.py {fx[:16]} != the lock's {lock['code']['final_x_sha256'][:16]}", file=sys.stderr)
        return 2
    for n in ("BIND", "LOOK"):
        d = json.load(open(os.path.join(G, f"REGISTER_ENTRY_DRAFT_EXP_BENCH_X_TEST_LOCK_{n}.json")))
        d["all_methods_lock_sha256"] = sha(lp)
        d["code_sha256"] = fx
        if re.search(r"\d\.\d|\d%", d["observations_description"]):
            print("REFUSED: a decimal in the description", file=sys.stderr)
            return 2
        out = os.path.join(G, f"REGISTER_ENTRY_EXP_BENCH_X_{n}.json")
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            os.write(fd, (json.dumps(d, indent=1, sort_keys=True) + "\n").encode())
        finally:
            os.close(fd)
        print(f"{n}: {out} sha256 {sha(out)} (lock file {sha(lp)[:16]}, final_x {fx[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
