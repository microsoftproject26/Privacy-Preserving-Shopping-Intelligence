# G0(b) — Opus response and scientific synthesis (Operations Director)

**Inputs:**
- `G0B_PROTOCOL_OPUS_POSITION.md`, written before Fable answered and withheld from it;
- `G0B_PROTOCOL_FABLE_POSITION.md`, verbatim.

## Agreement
- **F-1:** adopt, as a literal application of package 03 §6.
- **P3:** no early stop before 6.0 EFE. Fable made this binding and specified the 6→8 continuation rule.
- **P12:** move the 140 AB rows to CEN_AMBIGUOUS_NEXT_BARRIER.
- **P4–P6:** as proposed by EVAL-01, with Fable's C_ref and M04S detail.
- **FL-01 choices 1–4:** all adopted.

## Disagreement: P11, the cold-h panel
- **Opus:** candidate B, QA-01's panel of users with ≥ 50 VALIDATION events, split into COLD_START (2,617) and
  WARM_TRUNCATION (13,767). Reason: every h up to 50 needs ≥ 50 prior events.
- **Fable:** candidate A. M04 = the 16,384 smallest-key cold users (M03), plus a separate WARM_TRUNCATION panel = the
  16,384 smallest-key warm-cohort VALIDATION users (M02 prefix, no new salt).
- **Why the disagreement exists.** It is scientific: a trade-off between the estimator's validity and support at large
  h.
  - B selects users on their evaluation-period activity (≥ 50 VALIDATION rows), which conditions the panel on future
    behaviour. It is also 84% warm, so it measures truncation of trained history, not cold start.
  - A samples the target population (cold users) independently of outcomes, but gives little FIXED_50 support.
- **Opus response.** Fable's objection is correct. Selecting on evaluation-window activity is exactly the adaptive
  population choice the package forbids ("do not sample only future-active users", 04 §3), and my own position did
  not address it.
- **Decision: candidate A plus Fable's WARM_TRUNCATION panel.**
  - FIXED_50 support is reported honestly, and may be small.
  - ALL_AVAILABLE and FIXED_20 carry the cold-start study.
  - QA-01's panel receipt is kept as historical only.
- **Remaining uncertainty.** Few truly cold users have 50 prior events, so the h = 50 point for cold users may rest on
  few users. It is reported with its n.
- **Experiment needed?** No; this is a population-definition choice, not an empirical question.

## Other Fable findings, accepted
- **Disk guard.** The rule "refuse if free − peak < 10 GiB" would block the planned concurrent contest (12.81 − 3.4 =
  9.4). New rule: refuse if free − peak < 8 GiB floor + temp; pause scheduling below 10 GiB. That is a flaw in my plan;
  the plan is updated.
- **Lock-text fixes** C1(a)–(d) and C5 go to the contract owners as correction tasks. The C8 values are transcribed
  into the lock.
- **Before G1:** the GPU determinism receipt, OI-05 (model widths), FL-02 (participation plan and the s(EFE) hook), and
  the owner's conditional pre-approval of deleting `gru_v1/cache/`.
