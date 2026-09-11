"""Which checkpoints a downstream task may load, by name and by hash.

Every `.pt` in this project is loadable, and three of them should not be loaded. Nothing in
the file system says so, and a task that picks the most recent or the largest file gets a
wrong answer that looks entirely normal:

* **`t2_2.pt`** is the fine-tuned T2 encoder. It reaches the best T2 number in the project
  and it drives T1 from `0.3479` to `0.1791` - below T1's own model-free baseline. It is a
  legitimate T2 artifact and a disastrous shared encoder, and the file name says neither.
* **`t3_listwise.pt` / `t3_pointwise.pt`** are trained T3 rerankers that score below the
  frozen retrieval order they were meant to improve on, and they were trained by a head that
  could not see the query item at all. They are kept as evidence of a losing rung, not as
  models.

So the manifest is not bookkeeping. It is the only thing standing between `S2-DS-08`,
`S2-SE-01` or the federated lane and a checkpoint that produces plausible numbers about the
wrong thing.

Run it to regenerate `output/checkpoint_manifest.json` after any training run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "_SEAL"

# status -> (may a downstream task load it?, why)
POLICY = {
    "SHARED_ENCODER": (
        True,
        ("the T1 encoder. This is the only checkpoint any other task should start from until "
          "S2-DS-08 demonstrates a joint objective that holds both tasks.")),
    "TASK_HEAD": (
        True,
        ("a task-specific result. Load it to reproduce that task's number. Do NOT reuse its "
          "encoder for another task.")),
    "DIAGNOSTIC_UNDERPERFORMING": (
        False,
        ("kept as evidence of a losing rung. It scores below the model-free baseline it was "
          "meant to beat, so loading it downstream would be strictly worse than doing nothing.")),
    "DIAGNOSTIC_DESTRUCTIVE": (
        False,
        ("its encoder was fine-tuned for one task and measurably destroys another. Valid as "
          "that task's own result; never valid as a shared starting point.")),
    "MULTITASK": (
        True,
        ("the selected joint model - the only checkpoint that serves T1 and T2 at once. "
          "S2-SE-08 exports this; S2-PR-06 and S2-PR-09 federate it.")),
    "MULTITASK_ALTERNATE": (
        True,
        ("a joint model at a lower T2 weight. Reproducible evidence for the lambda ladder; "
          "use the selected one unless you are re-running that comparison.")),
    "ARCHITECTURE_REFERENCE": (
        True,
        ("the selected sequence core from S2-DS-05, kept so inference cost can be "
          "benchmarked without retraining.")),
    "ARCHITECTURE_ALTERNATE": (
        True,
        ("a losing core from S2-DS-05. Load it only to reproduce that comparison or to "
          "benchmark its inference cost - it is not the selected architecture.")),
    "COMMON_INITIALIZATION": (
        True,
        ("untrained shared weights for a seed. R1 and R2 must both start here or their "
          "measured gap is partly a different random start.")),
}

CLASSIFY = {
    "s2_ds_01_gru_t1_seed13.pt": ("SHARED_ENCODER", "S2-DS-01"),
    "s2_ds_01_gru_t1_seed42.pt": ("SHARED_ENCODER", "S2-DS-01"),
    "s2_ds_01_gru_t1_seed2026.pt": ("SHARED_ENCODER", "S2-DS-01"),
    "common_initialization_seed13.pt": ("COMMON_INITIALIZATION", "S2-DS-01"),
    "common_initialization_seed42.pt": ("COMMON_INITIALIZATION", "S2-DS-01"),
    "common_initialization_seed2026.pt": ("COMMON_INITIALIZATION", "S2-DS-01"),
    "t2_1.pt": ("TASK_HEAD", "S2-DS-06, frozen encoder - safe to reuse"),
    "t2_2.pt": ("DIAGNOSTIC_DESTRUCTIVE", "S2-DS-06, fine-tuned encoder"),
    "t2_2_trained_on_published_mask.pt": (
        "DIAGNOSTIC_DESTRUCTIVE", "S2-DS-06, pre-correction, kept for the label ablation"),
    "t3_listwise.pt": ("DIAGNOSTIC_UNDERPERFORMING", "S2-DS-07, query-blind reranker"),
    "t3_pointwise.pt": ("DIAGNOSTIC_UNDERPERFORMING", "S2-DS-07, query-blind reranker"),
    # S2-DS-08. The joint model is what S2-SE-08 exports and what S2-PR-06 and S2-PR-09
    # federate; it is the only checkpoint that holds T1 and T2 at once.
    "joint_lambda1.0.pt": ("MULTITASK", "S2-DS-08, the selected joint model"),
    "joint_lambda0.3.pt": ("MULTITASK_ALTERNATE", "S2-DS-08, a lower T2 weight"),
    "joint_lambda0.1.pt": ("MULTITASK_ALTERNATE", "S2-DS-08, a lower T2 weight"),
    # S2-DS-05. Kept so S2-SE-02 can benchmark inference cost without retraining. Only the
    # GRU is the selected architecture; the other three exist to make the comparison
    # reproducible.
    "t1_gru_seed13.pt": ("ARCHITECTURE_REFERENCE", "S2-DS-05, the selected core"),
    "t1_lstm_seed13.pt": ("ARCHITECTURE_ALTERNATE", "S2-DS-05, tied on quality, 27% slower"),
    "t1_tcn_seed13.pt": ("ARCHITECTURE_ALTERNATE", "S2-DS-05, -0.0079 against the GRU"),
    "t1_transformer_seed13.pt": ("ARCHITECTURE_ALTERNATE",
                                 "S2-DS-05, -0.0037 and undertrained at this budget"),
    "t2_separate_model.pt": ("DIAGNOSTIC_DESTRUCTIVE",
                             "S2-DS-ST1, a T2-only model with no shared encoder"),
}
SEED_PREFIXES = {
    "t2_finetuned_seed": ("DIAGNOSTIC_DESTRUCTIVE", "S2-DS-06 seed confirmation"),
}


def classify(name: str) -> tuple:
    if name in CLASSIFY:
        return CLASSIFY[name]
    for prefix, entry in SEED_PREFIXES.items():
        if name.startswith(prefix):
            return entry
    return ("UNCLASSIFIED", "not in the manifest - classify it before any task loads it")


def digest(path: Path) -> str:
    blake = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            blake.update(block)
    return blake.hexdigest()


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    entries = []
    for path in sorted(PROJECT.rglob("*.pt")):
        if ".venv" in path.parts or "site-packages" in path.parts:
            continue
        status, origin = classify(path.name)
        loadable, reason = POLICY.get(
            status, (False, "unclassified checkpoints are refused by default"))
        entries.append({
            "file": str(path.relative_to(PROJECT)).replace("\\", "/"),
            "name": path.name, "origin": origin, "status": status,
            "downstream_may_load": loadable, "reason": reason,
            "megabytes": round(path.stat().st_size / 1e6, 2),
            "sha256": digest(path),
        })

    allowed = [e for e in entries if e["downstream_may_load"]]
    refused = [e for e in entries if not e["downstream_may_load"]]
    print(f"  {len(entries)} checkpoints: {len(allowed)} loadable, {len(refused)} refused\n")
    for entry in entries:
        flag = "ok " if entry["downstream_may_load"] else "NO "
        print(f"  {flag} {entry['status']:<26} {entry['name']}")

    unclassified = [e for e in entries if e["status"] == "UNCLASSIFIED"]
    (OUTPUT / "checkpoint_manifest.json").write_text(json.dumps({
        "purpose": "which checkpoints a downstream task may load, by name and hash",
        "policy": {k: {"may_load": v[0], "reason": v[1]} for k, v in POLICY.items()},
        "checkpoints": entries,
        "loadable": len(allowed), "refused": len(refused),
        "unclassified": [e["name"] for e in unclassified],
    }, indent=2), encoding="utf-8")

    if unclassified:
        print(f"\n  {len(unclassified)} unclassified: "
              f"{', '.join(e['name'] for e in unclassified)}")
        print("  add them to CLASSIFY before any task loads them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
