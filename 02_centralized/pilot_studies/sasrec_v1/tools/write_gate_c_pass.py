"""Write review/GATE_C_PASS.json from Fable's saved Gate C verdict (owner: Opus).

Pins the exact training-code hashes the trainer firewall will enforce, plus the runtime
files (lock_retro, parity gate, queue runner, supervisor, report, benchmark) whose drift
lock_retro records at lock time. Refuses unless the saved verdict text says
APPROVE_FOR_REAL_TRAINING and the current hashes equal the attested evidence run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/code")
import paths  # noqa: E402
import train_sasrec as t  # noqa: E402

PINNED_RUNTIME = ("lock_retro.py", "tf32_parity_gate.py", "queue_runner.py", "supervise_run.sh",
                  "render_report.py", "bench_campaign.py", "resources.py")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    verdict_file = Path(sys.argv[1]) if len(sys.argv) > 1 else paths.S / "review" / "FABLE_GATE_C_RESPONSE.md"
    marker = sys.argv[2] if len(sys.argv) > 2 else "## Correction-only review"
    text = verdict_file.read_text(encoding="utf-8")
    final = text.split(marker)[-1]
    if "VERDICT: APPROVE_FOR_REAL_TRAINING" not in final.replace("**", ""):
        print("refusing: the final Gate C verdict is not APPROVE_FOR_REAL_TRAINING")
        return 2
    ev = json.loads((paths.S / "tests" / "evidence" / "GATE_C_ATTESTED_EVIDENCE.json").read_text())
    code = t.code_sha256_map()
    if not ev.get("overall_pass") or any(ev["hashes_start"].get(k) != v for k, v in code.items()):
        print("refusing: current code differs from the attested evidence run")
        return 2
    out = {"verdict": "APPROVE_FOR_REAL_TRAINING", "reviewer_model": "claude-fable-5-1",
           "verdict_file": str(verdict_file), "verdict_file_sha256": sha(verdict_file),
           "code_sha256": t.training_code_map(code), "all_code_sha256_at_approval": code,
           "pinned_runtime_sha256": {f: sha(paths.CODE / f) for f in PINNED_RUNTIME},
           "registry_sha256": t.load_registry()[1],
           "attested_evidence": str(paths.S / "tests" / "evidence" / "GATE_C_ATTESTED_EVIDENCE.json"),
           "attested_evidence_sha256": sha(paths.S / "tests" / "evidence" / "GATE_C_ATTESTED_EVIDENCE.json"),
           "utc": t.utc()}
    dst = paths.S / "review" / "GATE_C_PASS.json"
    fd = os.open(str(dst), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", dst, sha(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
