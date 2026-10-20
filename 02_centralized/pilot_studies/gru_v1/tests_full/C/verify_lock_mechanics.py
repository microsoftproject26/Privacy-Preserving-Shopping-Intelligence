"""Ad hoc mechanics verification for lock_and_eval.py -- NOT one of Sonnet C's
four owned deliverable files; a throwaway script run once from tests_full/C/
to exercise lock refusal, claim exclusivity + crash recovery, sha256 mismatch
refusal, and the tie -> F0 rule, per the Sonnet C task brief. Safe to delete
afterwards.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

WS = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1")
CODE = WS / "code"
SCRATCH = WS / "tests_full" / "C" / "lock_mechanics"
PY = "/home/jovyan/venvs/ppsi-gpu/bin/python"

F0_RUN = WS / "experiments" / "gru" / "T-M_F0_item_only"
F4_RUN = WS / "experiments" / "gru" / "T-M_F4_rich_side_information"


def run(args, expect_ok):
    r = subprocess.run([PY, str(CODE / "lock_and_eval.py")] + args, cwd=str(CODE),
                        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin",
                        "CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3",
                        "HOME": "/home/jovyan"})
    ok = (r.returncode == 0)
    status = "OK" if ok == expect_ok else "**UNEXPECTED**"
    print(f"[{status}] rc={r.returncode} expect_ok={expect_ok} args={args}")
    if status == "**UNEXPECTED**" or "--help" in args:
        print("STDOUT tail:", r.stdout[-1500:])
        print("STDERR tail:", r.stderr[-1500:])
    return r


def reset():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)


def section(title):
    print("\n=== " + title + " ===")


# ---------------------------------------------------------------------------
section("1. lock: basic success + lock refusal on 2nd invocation")
reset()
locks_dir = SCRATCH / "locks1"
run(["lock", "--f0-run", str(F0_RUN), "--f4-run", str(F4_RUN), "--locks-dir", str(locks_dir)],
    expect_ok=True)
lock_manifest = json.loads((locks_dir / "MODEL_LOCK.json").read_text())
print("selected_arm:", lock_manifest["selected_arm"],
      "(expected F4_rich_side_information_h256, since F4's pilot metric 0.10116 > F0's 0.10079)")
assert lock_manifest["selected_arm"] == "F4_rich_side_information_h256"

run(["lock", "--f0-run", str(F0_RUN), "--f4-run", str(F4_RUN), "--locks-dir", str(locks_dir)],
    expect_ok=False)  # MODEL_LOCK.json already exists -> must refuse

# ---------------------------------------------------------------------------
section("2. tie -> F0: synthetic equal-metric run dirs")
tie_scratch = SCRATCH / "tie_runs"
tie_scratch.mkdir(parents=True)
for name, run_dir_src in (("f0", F0_RUN), ("f4", F4_RUN)):
    dst = tie_scratch / name
    (dst / "checkpoints").mkdir(parents=True)
    shutil.copy(run_dir_src / "checkpoints" / "best.pt", dst / "checkpoints" / "best.pt")
    (dst / "summary.json").write_text(json.dumps({
        "best_metric": 0.123456, "update_number": 100, "pass_index": 1,
    }))
locks_dir_tie = SCRATCH / "locks_tie"
run(["lock", "--f0-run", str(tie_scratch / "f0"), "--f4-run", str(tie_scratch / "f4"),
     "--locks-dir", str(locks_dir_tie)], expect_ok=True)
tie_manifest = json.loads((locks_dir_tie / "MODEL_LOCK.json").read_text())
print("tie selected_arm:", tie_manifest["selected_arm"], "(expected F0_item_only_h256)")
assert tie_manifest["selected_arm"] == "F0_item_only_h256"

# ---------------------------------------------------------------------------
section("3. confirm --dry-run-split VALIDATION_SELECT: claim exclusivity + crash recovery")
locks_dir2 = SCRATCH / "locks2"
run(["lock", "--f0-run", str(F0_RUN), "--f4-run", str(F4_RUN), "--locks-dir", str(locks_dir2)],
    expect_ok=True)

# simulate a stale claim (a previous process that claimed but crashed before VC_DONE.json)
(locks_dir2 / "VC_CLAIM.json").write_text(json.dumps({
    "claimed_at_utc": "2020-01-01T00:00:00+00:00", "pid": 999999, "host": "stale-host",
    "incidents": [],
}))
run(["confirm", "--locks-dir", str(locks_dir2), "--out", str(SCRATCH / "vc_out.json"),
     "--dry-run-split", "VALIDATION_SELECT"], expect_ok=False)  # must refuse: stale claim, no --rerun flag

r = run(["confirm", "--locks-dir", str(locks_dir2), "--out", str(SCRATCH / "vc_out.json"),
         "--dry-run-split", "VALIDATION_SELECT", "--rerun-after-crash", "simulated crash for mechanics test"],
        expect_ok=True)
claim_after = json.loads((locks_dir2 / "VC_CLAIM.json").read_text())
print("incidents recorded:", claim_after.get("incidents"))
assert len(claim_after.get("incidents", [])) == 1

# a second confirm now must refuse because VC_DONE.json exists
run(["confirm", "--locks-dir", str(locks_dir2), "--out", str(SCRATCH / "vc_out2.json"),
     "--dry-run-split", "VALIDATION_SELECT"], expect_ok=False)

# ---------------------------------------------------------------------------
section("4. sha256 mismatch refusal")
locks_dir3 = SCRATCH / "locks3"
run(["lock", "--f0-run", str(F0_RUN), "--f4-run", str(F4_RUN), "--locks-dir", str(locks_dir3)],
    expect_ok=True)
manifest3 = json.loads((locks_dir3 / "MODEL_LOCK.json").read_text())
for info in manifest3["arms"].values():
    info["best_checkpoint_sha256"] = "0" * 64  # tamper
(locks_dir3 / "MODEL_LOCK.json").write_text(json.dumps(manifest3))
run(["confirm", "--locks-dir", str(locks_dir3), "--out", str(SCRATCH / "vc_out3.json"),
     "--dry-run-split", "VALIDATION_SELECT"], expect_ok=False)  # must refuse: sha256 mismatch

# ---------------------------------------------------------------------------
section("5. full confirm -> test dry-run chain (real evaluation numbers)")
locks_dir4 = SCRATCH / "locks4"
run(["lock", "--f0-run", str(F0_RUN), "--f4-run", str(F4_RUN), "--locks-dir", str(locks_dir4)],
    expect_ok=True)
run(["confirm", "--locks-dir", str(locks_dir4), "--out", str(SCRATCH / "vc_out4.json"),
     "--dry-run-split", "VALIDATION_SELECT"], expect_ok=True)
vc_out = json.loads((SCRATCH / "vc_out4.json").read_text())
print("VC dry-run selected arm macro_mrr@20:",
      vc_out[vc_out["selected_arm"]]["end_to_end"]["macro"]["mrr@20"])
print("VC dry-run TRANS_full macro_mrr@20:", vc_out["TRANS_full"]["end_to_end"]["macro"]["mrr@20"])
print("VC dry-run POP_full macro_mrr@20:", vc_out["POP_full"]["end_to_end"]["macro"]["mrr@20"])
print("VC dry-run oov:", vc_out["oov"])

run(["test", "--locks-dir", str(locks_dir4), "--out", str(SCRATCH / "test_out4.json"),
     "--dry-run-split", "VALIDATION_SELECT"], expect_ok=True)
test_out = json.loads((SCRATCH / "test_out4.json").read_text())
print("TEST dry-run bootstraps:", json.dumps(test_out["bootstraps"], indent=2))

# a second test invocation must refuse (TEST_DONE.json exists)
run(["test", "--locks-dir", str(locks_dir4), "--out", str(SCRATCH / "test_out4b.json"),
     "--dry-run-split", "VALIDATION_SELECT"], expect_ok=False)

print("\n=== ALL MECHANICS CHECKS COMPLETED ===")
