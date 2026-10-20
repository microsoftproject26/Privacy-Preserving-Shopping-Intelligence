"""Attested tiny evidence for continue_cooldown.py (owner: Opus). Synthetic data only.

1. source: tiny 4-epoch run (TF32 train / strict eval), COMPLETE.
2. cooldown: continue to epoch 5 in <source>_cool5; checks u == 5U, epoch-5 phase = all floor,
   first/last LR of epoch 5 = cool_lr(4U+1) / 1e-6, full-VS eval at epoch 5, best over {3,4,5},
   epoch-5 permutation digest == default_rng(seed+4), source dir untouched (hashes equal).
3. crash + resume inside epoch 5 -> bitwise-identical final weights and metrics vs (2).
4. production firewall: a non-tiny invocation refuses without GATE_C_COOLDOWN_PASS.json.
Writes tests/evidence/COOLDOWN_ATTESTED_EVIDENCE.json with the sha256 of every file involved.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

S = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1")
CODE = S / "code"
T = S / "tests" / "opus_cool"
PY = os.path.expanduser("~/venvs/ppsi-gpu/bin/python")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_hashes(d: Path) -> dict:
    return {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob("*")) if p.is_file()}


def run(cmd, env) -> dict:
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(CODE), env=env, capture_output=True, text=True, timeout=1800)
    return {"cmd": " ".join(map(str, cmd)), "rc": r.returncode, "seconds": round(time.time() - t0, 1),
            "tail": (r.stdout + r.stderr)[-1200:]}


def main() -> int:
    sys.path[:0] = [str(CODE), str(CODE / "reuse"), str(CODE / "reuse" / "vendor")]
    import numpy as np
    import torch
    ev = {"utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "code_sha256": {f: sha(CODE / f) for f in ("continue_cooldown.py", "train_sasrec.py", "sasrec_model.py",
                                                     "sasrec_schedule.py", "precision.py", "ckpt_io.py")}}
    shutil.rmtree(T, ignore_errors=True)
    T.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    tiny = ["--seed", "2026", "--test-tiny", "--tiny-planned-epochs", "4", "--precision-mode", "tf32"]
    src, cool, src2, cool2 = T / "src", T / "src_cool5", T / "b", T / "b_cool5"
    steps = {}
    steps["source"] = run([PY, "train_sasrec.py", "--run-id", "A1_RICH256", "--run-dir", str(src), *tiny], env)
    src_hash_before = tree_hashes(src)
    steps["cooldown"] = run([PY, "continue_cooldown.py", "--source-run-dir", str(src), "--run-id", "A1_RICH256",
                             "--run-dir", str(cool), "--log-every-update", *tiny], env)
    src_hash_after = tree_hashes(src)
    # second copy of the source for the crash/resume comparison
    shutil.copytree(src, src2)
    steps["cooldown_crash"] = run([PY, "continue_cooldown.py", "--source-run-dir", str(src2), "--run-id", "A1_RICH256",
                                   "--run-dir", str(cool2), "--stop-after-updates", "71", *tiny], env)
    steps["cooldown_resume"] = run([PY, "continue_cooldown.py", "--source-run-dir", str(src2), "--run-id", "A1_RICH256",
                                    "--run-dir", str(cool2), "--resume", *tiny], env)
    steps["firewall_negative"] = run([PY, "continue_cooldown.py", "--source-run-dir", str(src), "--run-id", "A1_RICH256",
                                      "--seed", "2026", "--run-dir", str(cool)],
                                     dict(env, CUDA_VISIBLE_DEVICES="1"))
    steps["guard_negative"] = run([PY, "continue_cooldown.py", "--source-run-dir", str(src), "--run-id", "A1_RICH256",
                                   "--seed", "2026", "--run-dir", str(cool), "--stop-after-updates", "5"], env)
    s = json.loads((cool / "summary.json").read_text())
    s2 = json.loads((cool2 / "summary.json").read_text())
    U = s["config"]["schedule"]["U"]
    dbg = [json.loads(l) for l in (cool / "metrics.jsonl").read_text().splitlines() if '"update_debug"' in l]
    lrs = {d["update"]: d["lr"] for d in dbg}
    import continue_cooldown as cc
    rec5 = next(r for r in s["epoch_records"] if r["epoch"] == 5)
    wa = torch.load(cool / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    wb = torch.load(cool2 / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    checks = {
        "source_complete": steps["source"]["rc"] == 0,
        "cooldown_complete": steps["cooldown"]["rc"] == 0 and s["final_status"] == "COMPLETE",
        "epochs_5_updates_5U": s["epochs_completed"] == 5 and s["updates"] == 5 * U,
        "epoch5_all_floor_phase": rec5["updates_by_phase"]["floor"] == U,
        "epoch5_first_lr_matches": abs(lrs[4 * U + 1] - cc.cool_lr(4 * U + 1, U, 1e-4)) < 1e-15,
        "epoch5_last_lr_is_1e-6": abs(lrs[5 * U] - 1e-6) < 1e-15,
        "epoch5_evaluated": rec5["full_vs"] is not None and s["eval_epochs"][-1] == 5,
        "best_over_3_4_5": s["best_epoch"] in (3, 4, 5) and s["best_metric"] == max(
            r["full_vs"]["macro_mrr20"] for r in s["epoch_records"] if r.get("full_vs")),
        "epoch5_perm_digest": rec5["perm_sha256"] == hashlib.sha256(
            np.ascontiguousarray(np.random.default_rng(2026 + 4).permutation(s["config"]["n_train"]), dtype=np.int64).tobytes()).hexdigest(),
        "source_untouched": src_hash_before == src_hash_after,
        "crash_rc_99": steps["cooldown_crash"]["rc"] == 99,
        "resume_complete": steps["cooldown_resume"]["rc"] == 0 and s2["final_status"] == "COMPLETE",
        "resume_bitwise_identical_weights": all(torch.equal(wa[k], wb[k]) for k in wa),
        "resume_identical_metrics": [r["full_vs"] and r["full_vs"]["macro_mrr20"] for r in s["epoch_records"]]
        == [r["full_vs"] and r["full_vs"]["macro_mrr20"] for r in s2["epoch_records"]],
        "epoch5_weights_saved": (cool / "checkpoints" / "epoch5_weights.pt").exists() and s.get("epoch5_weights") is not None,
        "test_hook_refused_without_tiny": steps["guard_negative"]["rc"] == 5 and "test-only options" in steps["guard_negative"]["tail"],
        "firewall_refuses_without_pass": steps["firewall_negative"]["rc"] == 5 and "cooldown firewall" in steps["firewall_negative"]["tail"],
    }
    ev.update({"steps": steps, "checks": checks, "U": U, "lr_epoch5_first_last": [lrs[4 * U + 1], lrs[5 * U]],
               "epoch5_macro_mrr20": rec5["full_vs"]["macro_mrr20"], "best": [s["best_epoch"], s["best_metric"]],
               "overall_pass": all(checks.values()),
               "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (S / "tests" / "evidence" / "COOLDOWN_ATTESTED_EVIDENCE.json").write_text(json.dumps(ev, indent=1, default=str))
    print("overall_pass", ev["overall_pass"], checks)
    return 0 if ev["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
