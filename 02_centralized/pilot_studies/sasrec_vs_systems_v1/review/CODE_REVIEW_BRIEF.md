# Code review (correction-style, required by AUTO_DECISION_GPU1.md) — before any GPU1 launch

Your plan is in `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_vs_systems_v1\review\AUTO_DECISION_GPU1.md`.
Review the actual code, not a summary:

- `...\sasrec_vs_systems_v1\code\vs_systems.py`, the only script that runs;
- `...\code\run_gpu1.sh`, the launcher;
- `...\code\test_vs_systems_local.py`, the local CPU mechanics test on synthetic data. Its result: all
  checks passed, including a negative control.

Reference only: the snapshot is `...\code\snap\` (22 files, byte-identical to the live campaign code;
see `SNAPSHOT_MANIFEST.json`). Its `reuse\eval_full.py` lines 264–350 contain `evaluate_model`, and
`ensembles.py` (sha dcd382fe…) is the approved final-stage version. The hashes are in
`...\review\REVIEW_HASHES.json`.

Your checklist from the plan:
- no import of claim/lock code;
- a VS-only split constant;
- the output root;
- TF32 flags off;
- the hook capture;
- sha verification of the inputs;
- the guard thread.

Also verify:
- abort thresholds;
- hard stop 03:00Z and no new stage after 02:00Z;
- the bootstrap is unconditional;
- member-rank consistency against the saved ranks, through the `decision_id_sha256` order check;
- nothing is written under `sasrec_v1`.

Output (≤ 600 tokens):
```
ROUTING: ...
VERDICT: APPROVE_FOR_GPU1 | FIX_REQUIRED
blockers (file:line) or "none"
```
