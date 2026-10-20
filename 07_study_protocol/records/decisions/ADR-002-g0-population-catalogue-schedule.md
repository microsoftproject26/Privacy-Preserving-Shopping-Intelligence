# ADR-002 — G0 protocol decisions: population, catalogue, VALIDATION, features, precision, schedule, FL defaults

## Decision
Fable's G0 lock values (`reviews/G0_PLAN_FABLE_POSITION.md`, LOCK_VALUES) are adopted, pending the seven blocker
receipts and the G0 protocol-approval call:
- **Population:**
  - a 262,144-user salted-hash cohort of warm users (salt `PPSI_V4_POPULATION_20260926`), used for every method,
    including the central contest;
  - a calibration prefix of 65,536 users;
  - a cold panel of 381,295 users, split into COLD_NO_TRAIN_EVENTS and COLD_TRAIN_EVENTS_NO_LABEL;
  - a cold-h panel of 16,384 users.
- **Catalogue:** support ≥ 5 on the full TRAIN (K = 158,486).
- **Splits:** VALIDATION = [11-16, 11-23), minimal-diff, with subperiods reported.
- **Features:** clean Rich per 03 §7 (a)–(d), with the M1 valid-only refit and re-zeroing, trained from scratch.
- **Precision:** strict FP32, with the matmul and cuDNN TF32 flags off.
- **Central training:**
  - the common schedule 0.02 / 2.0 / 3.0 / 5.5 / 6.0 EFE, with both families run to 6 EFE;
  - LR calibration at {1.0, 0.5} × the executed peak, for 2 EFE with seed 2026;
  - evaluation every 0.5 EFE.
- **FL:**
  - 64 clients per round, 2 passes, 3 visits, local batch 16;
  - AdamW reset per round;
  - μ grid {0.001, 0.01, 0.1}; λ grid {0, 1e-4, 1e-3}.
- **LO:** 6 passes, batch 16, procedural.
- **Primary metric:** client-macro MRR@20, end-to-end, over 62,368 warm-cohort VALIDATION users.
- **Device:** a SERVER-CPU proxy.

## Question
How can the thesis comparison be finished, under a common contract, within measured resources?

## Options considered
- The population: ALL / 262,144 / 65,536, and whether the contest runs on the cohort or on ALL.
- The catalogue threshold: s = 1 / 2 / 5, and whether it is fitted on the full or on the cohort TRAIN.
- VALIDATION: minimal-diff or re-derived.
- The central schedule: executed per-family or common EFE.
- The local batch: 64 or 16.
- The LR calibration: conditional or unconditional.

## Project evidence (PROJECT_MEASURED)
- `g0/data_census/*`, `g0/COST_FORECAST.json` (INFERENCE from measured inputs), `g0/BARRIER_COUNT.json`,
  `g0/feature_map/*` and `g0/export_smoke/*`.
- The handoffs DATA-01, DATA-02, SYS-01, SYS-02 and EDGE-01.

## Literature evidence
- Package cards R14 (exhaustive candidates, no sampled metrics) and R26 (precision as part of the comparison key).
- `research/LITERATURE_REGISTER.json` (RES-01).
- These are cited by Fable; they support defaults only.

## Opus position
`reviews/G0_PLAN_OPUS_POSITION.md`.

## Fable position
`reviews/G0_PLAN_FABLE_POSITION.md`: APPROVE_WITH_FIXES.

## Points of agreement and disagreement; final synthesis
`reviews/G0_PLAN_SCIENTIFIC_SYNTHESIS.md`. Opus adopted Fable's choices on the schedule, the calibration and the local
batch.

## Rationale
- It keeps the contract common: the contest and the downstream methods use one population and one schedule.
- It finishes within measured resources. The downstream core is about 123 GPU-h for GRU or about 366 GPU-h for SASRec,
  plus calibration.
- The estimator stays valid: FedProx acts on multi-step clients.

## Consequences
- Absolute scores will be lower than the v1 full-data scores.
- A family selected on the cohort may differ from one selected on ALL; this is a LIMITATION.
- The FL cost rises by 41% for the local batch of 16.
- The GRU export needs an adapter, only if GRU wins (EDGE-02, deferred).

## Validation required
- The seven blocker receipts.
- Fable's G0 protocol approval of DATA_CONTRACT_CLEAN_V2, EVALUATION_CONTRACT_V4, CLIENT_CONTRACT_V1 and
  FEATURE_SCHEMA_CLEAN_V2.
- Confirmation of the stopping interpretation (`reviews/G0_PLAN_OPUS_RESPONSE.md`).

## Thesis relevance
These choices define the Methodology chapter's data, population and training protocol, and they bound the claims:
retrospective only, cohort-scale, and data locality without formal privacy.
