# Demo Data Flow

What a local run of the Phase 1 model reads, computes, sends and keeps. Written against what is
implemented, not against what the design intends.

## Two flows, kept apart

Training and inference are different flows with different boundaries, and conflating them is how a
privacy claim becomes false.

### Inference, local

```text
session events  ->  Phase1Batch  ->  ONNX Runtime  ->  T1 / T2 / T3 raw outputs
```

Everything stays on the machine. Nothing is transmitted. No network call is made, and the exported
graph has no path to one.

### Training, federated simulation

```text
coordinator  --model payload-->  client   (download)
client       --model payload-->  coordinator   (upload)
```

Model payloads cross the boundary; session events do not. The measured payload for a round is
recorded in `docs/evidence/s2-se-05/communication_bytes.v1.json`.

## Inputs a local run reads

| Input | Shape | Source |
|---|---|---|
| `category_id`, `product_bucket`, `event_type_id`, `brand_bucket`, `price_band` | `[batch, history]` | the session, encoded |
| `history_gap` | `[batch, history, 1]` | `log1p(min(seconds, 86400))` |
| `lengths` | `[batch]` | events before and including the decision |
| `query_*` | `[batch]` | the item the decision is about |
| `candidate_ids`, `candidate_category_id`, `candidate_price_band`, `candidate_rank` | `[batch, candidates]` | the frozen retrieval list |

Product and brand identity arrive already hashed into fixed buckets. The raw product id is not an
input to the graph, and neither is any user identifier: nothing in the signature carries one.

## Outputs

| Output | Shape | Meaning |
|---|---|---|
| `t1_logits` | `[batch, categories]` | unnormalised scores over next category |
| `t2_logit` | `[batch]` | unnormalised purchase score for the query item |
| `t3_scores` | `[batch, candidates]` | unnormalised ranking scores |

Raw activations only. No softmax, no sigmoid, no sorting. Whoever consumes them owns those choices,
so a probability shown in a demo is the demo's claim, not the model's.

## What is local, transmitted, stored, retained

| Information | Inference | Federated training |
|---|---|---|
| Session events | local | local |
| Encoded batch | local | local |
| Model weights | local, read-only | transmitted both directions |
| Gradients or updates | none | transmitted as part of the model payload |
| Outputs | local | local |
| Opaque client id | not used | recorded in the communication log |
| Raw user id | never | never |

The communication log holds a round, an opaque client id, a direction and a byte count. It has no
field that can hold payload contents, and a raw numeric user identifier is refused rather than
stored.

## Behaviour at the edges

| Condition | Behaviour |
|---|---|
| Offline | Unchanged. Inference makes no network call. |
| Empty history | The session contribution is exactly zero, not the state of a padding step. |
| Padded positions | Inert. Rewriting them leaves every head unchanged. |
| Unknown product or brand | Hashed into a bucket like any other id. There is no separate unknown path, and a collision is indistinguishable from a match. |
| Unknown category | Carries the frozen out-of-vocabulary code. |
| Model file missing or corrupt | ONNX Runtime refuses to create the session and the run fails. There is no fallback model and no silent degradation. |
| Input names drifted | Refused by name before any inference runs. |
| Batch size, history length or candidate width changes | Supported. All three axes are dynamic and checked. |

## What this demo does not do

It does not log raw history, emit telemetry, or persist inputs. Nothing in the inference path writes
to disk.

## Privacy claims

The claims table in `docs/security/threat-model.md` governs. Nothing here adds to it.

In particular: Flower and Ray here are a simulation on one shared host. That demonstrates logical
message separation between clients. It is not physical device isolation, not process isolation
against a hostile co-tenant, and not evidence that a deployed system would have either. Differential
Privacy and Secure Aggregation are not implemented, and hashed or namespaced identifiers are not
anonymous.

## The repository is public

`github.com/microsoftproject26/Privacy-Preserving-Shopping-Intelligence` is a public repository. That
is a trust boundary, and anyone reading this document can read every artifact it describes. See the
public-repository section of the threat model for exactly what that includes.
