"""Gate C attested evidence run (owner: Opus). Answers Fable Gate C blocker 1.

Records the sha256 of every production file BEFORE and AFTER, runs the full pytest suite
(all independent test files) and the Opus integration checks (tiny A0/A1 end-to-end,
crash at update 37 + resume with bitwise comparison, A0/A1 permutation equality,
lock_retro mechanics), all against exactly those hashes, and writes
tests/evidence/GATE_C_ATTESTED_EVIDENCE.json. Synthetic data only for training; the data
adapter tests read the frozen data read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

S = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1")
CODE = S / "code"
EV = S / "tests" / "evidence"
PY = os.path.expanduser("~/venvs/ppsi-gpu/bin/python")
SUITE = ["tests/test_attention.py", "tests/test_head_objective.py", "tests/test_schedule_stopping.py",
         "tests/test_eval_and_precision.py", "tests/test_rich.py", "tests/test_trainer_exposure_resume.py",
         "tests/test_data_adapter.py", "tests/test_runtime.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prod_hashes() -> dict:
    out = {p.name: sha(p) for p in sorted(CODE.glob("*.py"))}
    out["supervise_run.sh"] = sha(CODE / "supervise_run.sh")
    for p in sorted((CODE / "reuse").rglob("*.py")):
        out["reuse/" + p.relative_to(CODE / "reuse").as_posix()] = sha(p)
    for p in sorted((CODE / "tests").rglob("*.py")):
        out["tests/" + p.relative_to(CODE / "tests").as_posix()] = sha(p)
    out["config/run_registry.locked.json"] = sha(S / "config" / "run_registry.locked.json")
    out["review/TF32_PARITY_PREREG.md"] = sha(S / "review" / "TF32_PARITY_PREREG.md")
    return out


def run(cmd, env=None, timeout=3000) -> dict:
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(CODE), env=env, capture_output=True, text=True, timeout=timeout)
    return {"cmd": " ".join(map(str, cmd)), "rc": r.returncode, "seconds": round(time.time() - t0, 1),
            "tail": (r.stdout + r.stderr)[-1500:]}


def main() -> int:
    ev = {"utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hashes_start": prod_hashes()}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    junit = EV / "GATE_C_SUITE_junit.xml"
    ev["suite"] = run([PY, "-m", "pytest", *SUITE, "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"], env=env)
    root = ET.parse(junit).getroot()
    cases = []
    for tc in root.iter("testcase"):
        outcome = "passed"
        for child in tc:
            if child.tag in ("failure", "error"):
                outcome = child.tag
            elif child.tag == "skipped":
                outcome = "skipped"
        cases.append({"file": tc.get("classname"), "name": tc.get("name"), "outcome": outcome,
                      "time": float(tc.get("time") or 0)})
    ev["suite_cases"] = cases
    ev["suite_counts"] = {k: sum(1 for c in cases if c["outcome"] == k) for k in ("passed", "failure", "error", "skipped")}

    # Opus integration (synthetic): tiny A0/A1, crash + resume, mechanics
    T = S / "tests" / "opus_tiny"
    for d in ("a0", "a1", "a0crash"):
        shutil.rmtree(T / d, ignore_errors=True)
    T.mkdir(parents=True, exist_ok=True)
    genv = dict(env, CUDA_VISIBLE_DEVICES="1")
    base = [PY, "train_sasrec.py", "--seed", "2026", "--test-tiny", "--tiny-planned-epochs", "4", "--precision-mode", "tf32"]
    integ = {}
    integ["a0"] = run(base + ["--run-id", "A0_ID256", "--run-dir", str(T / "a0")], env=genv)
    integ["a1"] = run(base + ["--run-id", "A1_RICH256", "--run-dir", str(T / "a1")], env=genv)
    integ["crash"] = run(base + ["--run-id", "A0_ID256", "--run-dir", str(T / "a0crash"), "--stop-after-updates", "37"], env=genv)
    integ["resume"] = run(base + ["--run-id", "A0_ID256", "--run-dir", str(T / "a0crash"), "--resume"], env=genv)
    import torch
    s0 = json.loads((T / "a0" / "summary.json").read_text())
    s1 = json.loads((T / "a1" / "summary.json").read_text())
    sc = json.loads((T / "a0crash" / "summary.json").read_text())
    wa = torch.load(T / "a0" / "checkpoints" / "best_weights.pt", map_location="cpu", weights_only=False)["state_dict"]
    wb = torch.load(T / "a0crash" / "checkpoints" / "best_weights.pt", map_location="cpu", weights_only=False)["state_dict"]
    integ["checks"] = {
        "a0_complete": s0["final_status"] == "COMPLETE", "a1_complete": s1["final_status"] == "COMPLETE",
        "crash_rc_is_99": integ["crash"]["rc"] == 99, "resume_complete": sc["final_status"] == "COMPLETE",
        "resume_count": sc["resume_count"],
        "resume_best_weights_bitwise_identical": all(torch.equal(wa[k], wb[k]) for k in wa),
        "resume_metrics_identical": [r["full_vs"] and r["full_vs"]["macro_mrr20"] for r in s0["epoch_records"]]
        == [r["full_vs"] and r["full_vs"]["macro_mrr20"] for r in sc["epoch_records"]],
        "a0_a1_perm_digests_equal": s0["perm_sha256_by_epoch"] == s1["perm_sha256_by_epoch"],
        "phases_a0": [r["updates_by_phase"] for r in s0["epoch_records"]],
        "eval_epochs": s0["eval_epochs"], "precision": [s0["precision_training"], s0["precision_eval"]],
        "canary": s0["config"]["canary"],
    }
    M = S / "tests" / "opus_mech"
    shutil.rmtree(M, ignore_errors=True)
    M.mkdir(parents=True)
    models = {}
    for rid, d in (("A0_ID256", "a0"), ("A1_RICH256", "a1")):
        s = json.loads((T / d / "summary.json").read_text())
        models[f"{rid}_s2026"] = {"run_id": rid, "seed": 2026, "roles": ["anchor"], "status": "COMPLETE",
                                  "run_dir": str(T / d), "group": s["group"],
                                  "best_weights": str(T / d / "checkpoints" / "best_weights.pt"),
                                  "best_weights_sha256": sha(T / d / "checkpoints" / "best_weights.pt"),
                                  "registry_entry": s["config"]["registry_entry"]}
    (M / "test_lock.json").write_text(json.dumps({"selected_configuration": "A0_ID256", "primary_seed": 2026,
                                                  "seeds": [2026], "models": models, "pair_check": {"comparable": True},
                                                  "synthetic": {"n_train": 4096, "n_vs": 1024, "K": 997, "seed": 0,
                                                                "task": "mapping"}}))
    integ["mechanics"] = run([PY, "lock_retro.py", "eval", "--mechanics-split", "VALIDATION_SELECT", "--out-dir", str(M),
                              "--lock", str(M / "test_lock.json")], env=genv)
    mr = json.loads((M / "MECHANICS_RESULTS.json").read_text())
    integ["checks"]["mechanics_fresh_process_equals_trainer_best"] = (
        mr["models"]["A0_ID256_s2026"]["VALIDATION_SELECT"]["end_to_end"]["macro"]["mrr@20"] == s0["best_metric"])
    integ["checks"]["mechanics_bootstrap_present"] = bool(mr["bootstraps"])
    ev["integration"] = integ
    ev["hashes_end"] = prod_hashes()
    ev["hashes_unchanged_during_run"] = ev["hashes_start"] == ev["hashes_end"]
    ev["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ok = (ev["suite"]["rc"] == 0 and ev["suite_counts"]["failure"] == 0 and ev["suite_counts"]["error"] == 0
          and all(v for k, v in integ["checks"].items() if isinstance(v, bool)) and ev["hashes_unchanged_during_run"])
    ev["overall_pass"] = ok
    (EV / "GATE_C_ATTESTED_EVIDENCE.json").write_text(json.dumps(ev, indent=1, default=str))
    print("overall_pass", ok, ev["suite_counts"], {k: v for k, v in integ["checks"].items() if isinstance(v, bool)})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
