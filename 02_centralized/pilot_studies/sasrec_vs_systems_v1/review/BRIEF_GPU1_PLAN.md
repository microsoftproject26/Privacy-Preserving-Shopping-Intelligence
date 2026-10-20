# Brief to Fable 5.1 — use of the idle GPU1 before the SASRec final stage (owner-delegated)

You are the owner's delegated decision authority for this. Your verdict is binding, and the lead
(Opus) will execute your plan. You are read-only: do not execute, edit or delegate. The first line of
your answer must be `ROUTING: <exact model id from your system prompt>`. Keep it to ≤ 1,200 tokens.

## The owner's request and the proposal
Read `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_vs_systems_v1\review\OWNER_PROPOSAL_VERBATIM.md`.
In short, the owner says: "review this; if it is good and has no negative effect, plan it and do it."

## Facts (measured at 2026-09-25T21:00Z unless stated otherwise)
- **GPU0:** A1_RICH256_s2027 base run, epoch 4, u=481,529/627,824, 3,136 dec/s. ETA 2026-09-26T00:44Z (an
  estimate).
- **Next on GPU0:** the orchestrator (`sasrec_v1/tools/campaign_v3b.py`) then launches the
  `A1_RICH256_s2027_cool5` cooldown, about 4 h.
- **Final stage:** `tools/final_stage.sh` runs select → lock → one VC/TEST evaluation on BOTH GPUs →
  finalize. Estimated start ≈ 04:45Z.
- **GPU1:** idle at 0% since A0 cool5 finished at 17:40Z.
- **Completed full-VS results (8,826,488 decisions):**
  - A1_s2026 e3 0.181421, e4 0.179761, e5 (cooldown) **0.186298**;
  - A0_s2026 e3 0.173762, e4 0.172820, e5 (cooldown) **0.178585**.

  Sources: `.local_ai\thesis_reference\sources\local\S183_*`, `S186_*`, `S135_*`, `S136_*`.
- **Weight files present on the server (bytes):**
  - A0 base: best_weights (e3) and epoch4_weights;
  - A0 cool5: best_weights and epoch5_weights;
  - A1 base: best_weights (e3) and epoch4_weights;
  - A1 cool5: best_weights and epoch5_weights.

  Each is about 332 MB. **Per-epoch full-VS ranks are already saved** as
  `runs/<run>/ranks/epoch{3,4,5}.npz`, with sha256 values in each summary.json. The complementarity
  analysis therefore needs NO GPU.
- **Resources:**
  - cgroup RAM 45.4 of 64 GiB (the page cache is included); trainer RSS 18.4 GB;
  - disk free 13.6 GiB (target ≥10, floor 8);
  - 2×A16 with 15.4 GB each; the s2027 trainer uses about 5.5 GB on GPU0.
- **Pre-registration (amendment 2)**, `.local_ai\thesis_reference\sources\line_numbered\S95_run_registry.v3_amendment_2.json.txt`
  L44–58:
  - the systems S1–S4b are fixed;
  - equal weights, with the mean of per-model log-softmax;
  - members = each run's best epoch by the pre-registered rule;
  - S4a = the uniform average of A1_s2026 epochs {3,4,5};
  - "any other system" is excluded;
  - systems are never compared with GRU as model-vs-model.
- **Final-stage code** (approved, armed), `.local_ai\sasrec_v1\code\lock_retro_v3b.py`:
  - `SPLITS = ("VALIDATION_CONFIRM","TEST")` (L36), so systems are NOT evaluated on VS by the plan;
  - `build_model` (L198–222) builds singles, `weight_average` and ensembles through
    `code\ensembles.py` (`LogProbEnsemble`, `average_state_dicts`);
  - `claim()` (L225–232) writes `locks/…CLAIM` and refuses a second look. **Any "rehearsal" through
    `cmd_eval` would therefore create a claim file and break the real final stage.**
  - The lock records `review/GATE_C*_PASS.json` shas and the pinned code shas under `code/` (L176–180).
- **Feature study:** its queue (tmux `fo-queue`) is QUEUED_RESOURCE until the campaign's
  FINAL_STAGE_DONE, and its code approval assumes that. It is not part of this question.

## The lead's pre-analysis (a hypothesis for you to accept or reject)
- **S2 and S4a on full VS** are pre-registered systems. Evaluating them on VS now is descriptive,
  because the final stage's headline rule is single-model and the systems are fixed. It changes no
  lock or selection. It does add VS looks, so it must be reported as VS-descriptive only, and must
  never be used to add, drop or re-weight a system.
- **Complementarity** comes from the existing rank files, on CPU.
- **"Rehearsal"** must not use `cmd_eval`/`claim`. If wanted, it should be a throughput measurement of
  `LogProbEnsemble` scoring on VS through a separate script.
- **Isolation:**
  - a new workspace, `.local_ai/sasrec_vs_systems_v1/` (nothing written under `sasrec_v1/code` or
    `review`);
  - either a byte-identical snapshot of `ensembles.py`, `sasrec_model.py`, `sasrec_data.py`,
    `precision.py`, `paths.py` and the reuse evaluator, or read-only imports of the live code with
    `PYTHONDONTWRITEBYTECODE=1`;
  - a VS-only `CompactStore` (F0 for A0 and F4 for A1, or F4 for both, with A0 receiving item kwargs
    only, as `LogProbEnsemble` does).
- **Guards:** a GPU0 throughput canary before and after 5 min, with an abort above a 10% drop; disk and
  RAM limits; a hard wall-clock stop well before the final stage (for example, all GPU1 work finished
  by 03:00Z); and CUDA_VISIBLE_DEVICES=1.

## Decide
1. **VERDICT:** GO_AS_PROPOSED / GO_WITH_CHANGES / NO_GO, with the reason. It must be
   no-negative-effect: on the running campaign, on its final stage and lock, and on scientific
   integrity (VS reuse, pre-registration).
2. **PLAN:** ordered steps. Include whether S4a, the S2 bootstrap and complementarity are in scope,
   and what exactly is computed and saved. Say whether to add anything, for example also S4-style
   averaging for A0 (it is NOT pre-registered, so it is likely excluded), or VS numbers for the
   individual e5 members (already exist).
3. **ISOLATION and code:** the snapshot or live-import choice; where outputs go; whether your code
   review is required before running. The lead proposes a short correction-style review of one small
   script.
4. **GUARDS:** the exact abort thresholds and the hard stop time.
5. **REPORTING RULES:** how these VS system numbers may and may not be used or claimed (the headline
   and the lock are unaffected).
6. **SKIP:** anything in the proposal to drop.
