# ADR-001 — Adopt V4 (single central winner) and the company operating model

## Decision
Execute `PPSI_SINGLE_BACKBONE_V4_2026_09_26` as the current assignment under the owner's "research & engineering
company" operating model.
1. Compare GRU and SASRec centrally on one clean protocol.
2. Lock ONE family.
3. Carry only that family into C, FA, FP, PF, CPF and LO, plus cold-start and on-device feasibility.

## Question
What is the project's research question and working structure after the SASRec campaign and the feature study
closed?

## Options considered
1. **V2/V3 plans.** Both architectures in every regime, plus FedBN. Rejected by the owner: FedBN was removed, and a
   loser-backbone downstream is waste.
2. **V4 single-backbone plan.** Adopted.
3. **Treat the two closed campaigns as final.** Rejected. They answer a different question: centralized quality
   under the v1 protocol, not the retention of quality across paradigms.

## Project evidence (PROJECT_MEASURED)
- **SASRec campaign (v1 protocol, retrospective TEST):** A1_s2027 e5 scored 0.170475 against GRU F0 at 0.168695, a
  difference of +0.00178 with CI [+0.00154, +0.00202]. The ID-only A0 was below GRU F0 (−0.0022, descriptive).
  Thesis reference §5.6f; S217.
- **Feature study:** KEEP_CURRENT.
  - D3 channels gave +0.00026, below the gate.
  - Lean parity failed at 0.998683, under the 0.999 gate.
  - The M1 defect was confirmed on all of TRAIN.

  Thesis reference §6.4b; S236.
- **G0 cost benchmark:** every client visit pays for the full-K head and AdamW over all parameters (S274, S275).

## Literature evidence
The package research cards (`research/RESEARCH_REVIEW.md`, S269) are summarised there. They will be registered by
RES-01. The package itself states that these cards motivate defaults and do not establish them as facts
(LITERATURE_SUPPORTED vs PLAN).

## Opus position
Adopt. The scope is feasible only if the population and budget are decided from measured costs before any central
training. SYS-01 shows the full universe is the dominant cost risk (R1).

## Fable position
Not yet consulted. The gate is G0.

## Points of agreement / disagreement
None recorded yet. The G0 deliberation follows the packet.

## Consequences
- The historical v1 data and all earlier campaign artifacts stay immutable.
- New work goes in `.local_ai/ppsi_v4/`.
- Old VC/TEST results stay retrospective.
- There is no training before the Fable G0 approval and the data/protocol lock.

## Validation required
- Fable G0 on the evidence packet.
- The Clean V2 freeze with a before/after census.

## Thesis relevance
This defines the thesis question: how much recommendation quality is retained when moving from centralized to
federated, personalized federated, or local-only learning, under a common data, architecture, capacity and
evaluation contract.
