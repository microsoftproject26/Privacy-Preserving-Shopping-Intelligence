# AUTO_DECISION — idle-GPU1 plan (owner-delegated to Fable), 2026-09-25T21:10Z

- **Owner directive** (verbatim): «خلي fable يراجع الكلام ده ولو كويس ومش هيئاثر بالسلب خليه يخطط ليه ويعمله»
- **Proposal:** review/OWNER_PROPOSAL_VERBATIM.md
- **Brief:** review/BRIEF_GPU1_PLAN.md
- **Runtime routing** (from the transcript): ['claude-fable-5-1']

## Fable response (verbatim)

ROUTING: claude-fable-5-1

**1. VERDICT: GO_WITH_CHANGES.**
Reason [evidence]: S2 and S4a are fixed pre-registered systems (S95 L46–53); members of S2 = each run's best epoch, which is e5 for both A0_s2026 (0.178585 > e3 0.173762) and A1_s2026 (0.186298 > e3 0.181421), so the proposal names exactly the pre-registered members. The final stage evaluates only VC/TEST (`lock_retro_v3b.py` L36), selection is single-model max, so VS numbers for systems cannot change the lock or the headline. `campaign_v3b.py` launches on GPU1 only the two s2026 cooldowns (L4, L8), both already complete; s2027 cool5 is gated by `gpu_free("0")` (L243), so GPU1 use cannot block the orchestrator. Changes required [inference]: no rehearsal via `cmd_eval` (claim file, L225–234); bootstrap unconditional, not "only if better"; no unregistered systems; hooks-based member check instead of extra runs; tighter guards below.

**2. PLAN (ordered).**
- P0 (CPU, now): complementarity from `runs/*/ranks/epoch5.npz` (A0 e5, A1 e5), after verifying each npz sha256 against its summary.json [evidence: brief L31–32]. Save: both-hit@20, A1-only, A0-only, neither, oracle-union Recall@20, per-user macro MRR@20 difference. Also A1 e3/e4/e5 pairwise for context of S4a. No GPU.
- P1 (GPU1): S2 = `LogProbEnsemble([A0 cool5 e5, A1 cool5 e5])`, full VS (8,826,488 decisions), strict FP32 (TF32 off, process-local), full catalogue K=158,486. Register read-only forward hooks on the two members to capture per-member logits → ranks in the same pass (zero extra cost); compare to the saved epoch5.npz ranks. Exact/near-exact match validates the F4-store + item-only-kwargs path that the real final stage uses; this replaces the "rehearsal".
- P2 (GPU1): S4a = `average_state_dicts` of A1_s2026 e3 (best_weights), e4 (epoch4_weights), e5 (cool5 epoch5_weights) → full VS. Record the three input shas and the averaged state_dict sha.
- P3 (CPU): paired user-level bootstrap, 1,000 resamples, fixed seed, run for BOTH S2−A1e5 and S4a−A1e5 regardless of sign (conditional testing is a forking path). Descriptive only.
- Saved per system: ranks .npz (+sha), macro/micro @1/@5/@10/@20 metrics, decision count, dec/s runtime, member weight shas, code manifest, CUDA/torch versions, start/end UTC.
- Excluded: S4-style averaging for A0 (not pre-registered; "any other system" excluded, L58); recomputing single e5 VS numbers (exist); S1/S3/S4b (need s2027, out of scope); any d384/seed/feature work.

**3. ISOLATION and code.**
- Workspace: `.local_ai/sasrec_vs_systems_v1/` only. Nothing written under `sasrec_v1/code`, `sasrec_v1/review` or `sasrec_v1/runs` (ranks dirs are campaign artifacts with recorded shas) [inference].
- Byte-identical snapshot of `ensembles.py`, `sasrec_model.py`, `sasrec_data.py`, `precision.py`, `paths.py`, `train.py` (needed for VOCAB_SIZES/product_idx, L199–206) and the reuse evaluator into the workspace, with a sha256 manifest matching the pinned shas; plus `PYTHONDONTWRITEBYTECODE=1`. Snapshot preferred over live import so the run is reproducible and no bytecode lands in the pinned tree. The script must NOT import `lock_retro_v3b` or `base` (claim path).
- Store: one VS-only F4 `CompactStore`; A0 gets item kwargs only, as `LogProbEnsemble.forward` already enforces (ensembles.py L44–45). Never open VC/TEST/TRAIN.
- My correction-style review of the one script is REQUIRED before launch. Reviewer checklist: no claim/lock import; VS-only split constant; output root; TF32 flags off; hook capture; sha verification of inputs; guard thread present.

**4. GUARDS.**
- Canary: GPU0 dec/s averaged over 3 min before start; re-measure over 5 min after start; abort GPU1 job if drop > 10%. Re-baseline (3 min) whenever GPU0 changes phase (base run end → cool5 launch, ETA ~00:44Z), since the cooldown's dps differs.
- Disk: abort if free < 10 GiB; workspace budget ≤ 1 GiB (ranks only; never save logits).
- RAM: abort if cgroup usage > 56 GiB or the GPU1 process RSS > 12 GiB; dataloader workers ≤ 2; `nice 10`.
- `CUDA_VISIBLE_DEVICES=1`; VRAM check ≤ 12 GB.
- Hard stop: every GPU1 process ended and workspace closed by 03:00Z; nothing new launched after 02:00Z. Final stage (≈04:45Z, both GPUs) proceeds regardless of this work's completion. Registry hard stops (L62–63) unaffected.

**5. REPORTING RULES.**
- Label everything "VS-descriptive (selection split), post-hoc look"; increment the VS-look log in the thesis reference.
- Never a headline; never compared with GRU (L54); never used to add, drop, re-weight or re-order S1–S4b; the lock and `selection_v3b.json` are unchanged by design.
- Reportable system results are the final stage's VC/TEST numbers only; VS bootstrap CIs are descriptive, not confirmatory.
- The member-rank consistency check may be cited as a mechanics validation.

**6. SKIP.**
- Final-stage evaluator rehearsal (claim risk; throughput is obtained from P1 anyway).
- Conditional bootstrap ("only if better").
- A0 epoch-averaging, any new model/seed/feature work.

Files: `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_vs_systems_v1\review\BRIEF_GPU1_PLAN.md`, `...\OWNER_PROPOSAL_VERBATIM.md`, `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\code\lock_retro_v3b.py`, `...\code\ensembles.py`, `F:\Queens University\Microsoft Project\eda\.local_ai\thesis_reference\sources\line_numbered\S95_run_registry.v3_amendment_2.json.txt`, `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\tools\campaign_v3b.py`.
