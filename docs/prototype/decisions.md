# Phase 1 Decisions

## Phase 1 Scope

We will focus on completing the original technical stages W1–W4 during the current two-week Phase 1 sprint.

## Team Responsibilities

Each member will keep one main workstream during Phase 1.

- Eid: Data & Model
- Ahmed Abdelhamed: Federated Training
- Ahmed Sherif: System, Deployment & Security

There will be no role rotation during the current two-week Phase 1 sprint to reduce context switching and move faster.

## Parallel Work

The three workstreams should progress in parallel whenever possible.

Members should not wait for another workstream unless their current task has a real dependency.

If a task is blocked, the member should move to another ready task in the same workstream or help resolve the blocker.

## GitHub Workflow

Issue → Branch → Pull Request → Validation → Merge → Done

Pull Requests do not require another-member approval. The author may merge after the required
tests pass. Review remains optional for scientific protocol changes and shared cross-lane contracts.

## Blocked Tasks

If a task becomes blocked, the reason and dependency should be documented in the Issue.

The member should move to another ready task whenever possible instead of waiting.

## Team Coordination

- Short daily progress updates.
- Integration/review meeting every 2–3 days.
- Blockers should be reported as soon as they appear.
- Important technical decisions should be documented.

## Development Principle

We will follow the "Assemble, Don't Build" principle.

We will use established tools for standard infrastructure whenever possible and focus our implementation effort on the research-specific parts of the project.

Issue descriptions define the intended outcome, not mandatory internal architecture. Use the
smallest useful implementation that preserves shared contracts and can be extended later.


## S1-PR-03 — Experiment Contract Freeze

The project freezes the S1-PR-03 reproducibility contracts for experiment
configuration, result identity, common initialization, and R1/R2A comparison.

- Allowed seeds: `13`, `42`, `2026`.
- R1–R4 use the same seed-specific untrained CommonInitialization.
- R5 starts from a trained R1 checkpoint.
- Run IDs use the versioned `run-v1` grammar.
- Attempts start at 1; terminal attempts are immutable.
- ExperimentResult references SystemMeasurement records rather than duplicating
  measurement values.
- The detailed machine-readable freeze is recorded in
  `config/experiments/contract_freeze.v1.json`.

The technical freeze was approved for implementation under the project-owner
decision process. Changes use the standard team workflow: required validation,
then merge by the author. Peer review is optional.

## S1-PR-05 — Unified Trainer Contract Freeze

**Status:** Frozen by the merged S1-PR-05 implementation. Later changes use the
standard team workflow: required validation, then merge by the author. Peer
review is optional.

The project freezes the following shared trainer decisions:

- centralized and Flower execution use one `LocalTrainerCore`;
- reusable code lives in the flat `ppsi/training/` package;
- `Phase1Batch v1` uses named categorical channels, right padding, explicit
  masks and optional zero-width continuous candidate features;
- padding IDs are representation-declared; masks are authoritative;
- empty history uses `ZERO_HIDDEN`: semantic length zero produces an exact
  zero historical representation after history-only transformation;
- `t3_present=true` requires at least one valid candidate;
- missing task labels never become negatives;
- raw outputs are T1 logits, T2 logit and T3 candidate scores with no embedded
  activations or sorting;
- objectives are injected and identified separately from shared trainer code;
- no-contribution performs no backward, optimizer or scheduler step;
- exact CPU resume uses a deterministic sampler, explicit next-batch cursor,
  saved RNG state and `num_workers=0` in the P0 replay path;
- Checkpoint v1 uses atomic trusted PyTorch persistence, byte-level integrity
  SHA-256 and semantic compatibility identities;
- Flower update weighting is an injected policy; the #18 contributing-row
  policy is smoke-only and S1-PR-06 owns the final scientific policy;
- run-level output reuses `ExperimentResult v1`;
- `shared_trainer_core_sha256` is derived from a canonical manifest of the
  behavior-defining shared trainer files, with UTF-8/LF/no-BOM text
  normalization to avoid platform-specific line-ending drift.

This freeze does not select the final GRU architecture, final T1/T2/T3
objectives, final T3 candidate protocol, client sampling/weighting, FedAdam,
personalization, authoritative final initialization, or real REES46 training.
