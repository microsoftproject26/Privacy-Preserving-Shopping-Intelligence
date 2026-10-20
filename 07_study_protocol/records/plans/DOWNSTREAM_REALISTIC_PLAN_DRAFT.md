# Downstream "realistic deployment" plan — Ops DRAFT for Fable (2026-09-27)

Owner's request (paraphrase, Arabic original in the session): simulate reality. (1) A big, high-accuracy model on the
server. (2) If the user's data leaves the device only as WEIGHTS to the main model, how much accuracy do we lose?
(3) A SMALL model on the device, pre-trained in general beforehand, then placed on the client's device and further
trained on that client's own data (personalization). The owner will approve or reject the plan.
Already registered: PREG1_AMENDMENT_03 (BIG = G2 winner; SMALL = device cap ≤ 170 MB per visit: GRU_D128 or
SASREC_D64_B2; all FL/LO arms at SMALL; acceptability threshold PENDING owner).

## Scenario map (PLAN)
| Scenario | Arm(s) | Real-world meaning | Where the data is |
|---|---|---|---|
| 1 Server ceiling | C_BIG (P3 winner runs) | today's cloud recommender | server |
| 1b Server, small | C_SMALL | same small model, server-trained | server |
| 2 Federated main model | FA (FedAvg), FP (FedProx) at SMALL | data stays on device; devices send weight updates; server aggregates | device |
| 3 General small model + on-device personalization | CPF: C_SMALL pre-trained on the server, then fine-tuned on each client's own history; PF: the FL model (FA/FP) then fine-tuned on each client | industry pattern: ship a general model, personalize on device | general part: server or FL; personal part: device |
| 3b Device alone | LO: small model trained only on the client's own data from scratch | lower bound, no collaboration | device |

Measured losses: C_BIG → C_SMALL (size), C_SMALL → FA/FP (federation), FA/FP → PF and C_SMALL → CPF
(personalization gain), LO (value of collaboration). Also on U_NEW (users never seen in training: a new install
personalizes from the general model) and the SPARSE/HEAVY panels.

## Realism knobs to register (Ops proposal; Fable to set values)
- Partial participation (64 clients per round, sampled), local work B16 × 2 passes (already G0/ADR-005), and a random
  client drop-out rate per round (e.g. 10%), with the aggregation weighted by the examples actually used.
- Personalization budget per client (e.g. k local epochs on its own support events, a small LR, early-stop-free) —
  identical for CPF, PF and LO.
- Report, beside accuracy: bytes per client per round and in total, local updates per client, model size, and the
  personalization gain by user stratum.
- Out of scope unless Fable says otherwise: differential privacy and secure aggregation (no accuracy effect for secure
  aggregation; DP would be a separate study).

## Optional (ask Fable)
- KD: a small general model DISTILLED from C_BIG (the big server model teaches the device model) as a third pre-training
  source for scenario 3 (KD_SMALL → personalized). Cost and value?

## Cost (INFERENCE)
Matrix at SMALL ≈ 58 GPU-h per Fable (DL 84) plus personalization evaluations; KD, if adopted, adds a teacher-scoring
pass plus one small training.
