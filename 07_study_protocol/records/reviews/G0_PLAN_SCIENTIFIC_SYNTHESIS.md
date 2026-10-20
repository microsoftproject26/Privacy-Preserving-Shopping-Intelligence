# G0 planning — scientific synthesis (Operations Director)

**Inputs:**
- `G0_PLAN_OPUS_POSITION.md`, written first and kept independent;
- `G0_PLAN_FABLE_POSITION.md`, verbatim; Fable did not see the Opus memo;
- `G0_PLAN_OPUS_RESPONSE.md`.

## 1. What both sides agree on
- **Population:** a 262,144-user salted-hash cohort for every method, including the central contest, with cold users
  outside the matched block.
- **Catalogue:** support ≥ 5 on the full TRAIN.
- **VALIDATION:** merged minimal-diff.
- **Features:** clean Rich per 03 §7, with the M1 valid-only refit, trained from scratch.
- **Precision:** strict FP32.
- **FL defaults:** 64 clients per round, 2 passes, AdamW reset per round, n-weighted averaging, μ grid {0.001, 0.01,
  0.1}, λ grid {0, 1e-4, 1e-3}, p = query dimension.
- **Device:** a declared SERVER-CPU proxy.
- **Selection:** device export feasibility never influences the family choice.

## 2. What we disagreed on
The central schedule, the calibration trigger and the local batch size. All three are resolved in Fable's favour; see
§5.

## 3. Why the disagreement existed
The Opus positions leaned on the continuity of executed recipes and on cost. Fable's leaned on how well the design
estimates the effect under the new population:
- a 10× smaller step count invalidates step-timed schedules;
- FedProx needs more than one local step to act.

The difference is **scientific (validity of the estimator), with a resource cost**: about 41% more FL time and 3.4
GPU-h of calibration.

## 4. Evidence
- PROJECT_MEASURED: the client distribution (median 6, p90 35); the step costs; the cohort step count (≈ 15.3k per
  EFE).
- Package rules: 08 §3 (the cohort step-count warning), 06 §4 (the proximal term is zero at round start) and 05 §E (C
  runs can be reused).

## 5. Final project decisions
All of Fable's DECISIONS and LOCK_VALUES are adopted. The seven blockers become Wave A tasks. The one interpretive point
(runs complete 6.0 EFE; patience applies only to a 6→8 continuation) awaits Fable's confirmation at the G0 protocol
approval. See ADR-002.

## 6. Why this serves the whole thesis question
The question is quality retained under a common contract. Choosing the contest population equal to the downstream
population, and a schedule shared by all central rows, keeps every C–M difference attributable to the paradigm rather
than to a data-regime or schedule mismatch.

## 7. Remaining uncertainty
- **Contest population:** a cohort-selected family might differ from an ALL-selected family. This is recorded as a
  limitation. A full-data central run is optional, descriptive and not planned.
- **Absolute scores:** they will be below the v1 full-data scores, because this is a protocol change, not a regression.

## 8. Experiment needed?
No additional experiment. The LR calibration is itself the bounded experiment that resolves the recipe question.
