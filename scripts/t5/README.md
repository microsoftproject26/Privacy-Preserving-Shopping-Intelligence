# S2-DS-05 — is the GRU the right core, or the one we reached for first?

The board's wording is the task: *"we want the final architecture to be justified by
quality-efficiency trade-offs rather than chosen by habit."*

**Answer: keep the GRU — but the reason is cost, not quality.**

---

## Result

Four sequence cores, one frozen protocol. Same data, same split, same T1 evaluator, same
budget, same seed. Only the core between the input projection and the heads changes.

| core | slice macro MRR | vs GRU | parameters | train minutes | best epoch |
|---|---:|---:|---:|---:|---:|
| **GRU** | **0.3477** | — | 2,379,263 | **27.5** | 28 |
| LSTM | 0.3471 | −0.0006 | 2,412,287 | 34.8 | 23 |
| transformer | 0.3440 | −0.0037 | 2,420,863 | 46.0 | 28 |
| TCN | 0.3398 | −0.0079 | 2,329,727 | 28.9 | 23 |
| *transition table* | *0.3143* | | *—* | *—* | |

**The GRU and the LSTM are tied.** `0.0006` is inside T1's own three-seed spread of `0.0007`,
so on this evidence they are the same model — and the LSTM takes **27% longer** to reach it.

**The transformer and the TCN are genuinely behind**, by 5× and 11× that spread. Those are
differences, not noise.

---

## The gate

The GRU rung reproduced `0.3477` against the published `0.3479` — a drift of `0.0002`. Had
it not, this harness would be scoring something other than what `S2-DS-01` scored and the
ranking below would rank nothing.

---

## What this comparison is biased toward, stated plainly

**One budget, taken from the GRU's own selected configuration**: 30 epochs, batch 512,
`lr = 0.001`, cosine schedule, hidden 128, one layer. Every core got exactly that.

That favours the GRU, and the evidence is in the run itself. At epoch 18 the transformer's
train-minus-validation gap was still **negative** (`−0.0006`) while the GRU's had reached
`+0.0464` by epoch 25. The GRU was memorising; the transformer had not finished learning.

**So the honest claim is narrow:** *under the GRU's budget, the GRU is at least as good as
every alternative and cheaper than all but one.* It is **not** *"the GRU is the best
architecture for this problem"*. A transformer given its own schedule and warmup might close
`0.0037`; nobody has tried, and trying it now — after seeing it lose — would be tuning one
contestant after the race.

The board asked for a time-boxed comparison. This is one, and the box is disclosed.

---

## Why capacity was not equalised either

Each core got the same width and depth, and its parameter count fell where it fell. The
counts landed within 4% of each other (`2.33M` to `2.42M`), so nothing here wins by being
larger. Equalising capacity exactly would have meant tuning each architecture's width, and
then the comparison measures the tuning.

---

## What this hands downstream

| | |
|---|---|
| **`S2-ALL-CP-ARCH`** | the architecture checkpoint has its comparison table, and the GRU is confirmed rather than assumed |
| **`S2-SE-01`** (export) | nothing changes — the GRU stays, and it is the core that exports most simply |
| **`S2-SE-02`** (latency) | the training-time column is a proxy, not a substitute; CPU inference latency is still that task's to measure |

**The checkpoints for all four cores are kept** (`output/t1_<core>_seed13.pt`), so a later
task can benchmark inference cost without retraining. They carry their batch spec and the
frozen-artifact digests.

---

## The trap this task exists to avoid, and a defect it found

A pluggable core is easy to write badly in two specific ways, and both pass every shape and
dtype check:

* **A non-causal core.** A TCN with its padding on the wrong side, or a transformer without
  its mask, reads events *after* the decision. The metric then measures the future.
  `test_every_sequence_core_is_causal` overwrites everything past each decision position and
  requires the encoded vector not to move.
* **A renamed parameter.** The first version wrapped every core so they all returned a
  tensor. That turned `encoder.weight_ih_l0` into `encoder.module.weight_ih_l0` and **every
  one of the project's 15 published checkpoints stopped loading** — while all 31 tests
  passed. The checkpoint loader's contract guard caught it. The recurrent cores are now
  unwrapped and `test_the_recurrent_cores_keep_their_published_parameter_names` makes the
  name itself the contract.

---

## Scope

| | |
|---|---|
| cohort | `C1`, the 25% slice — 97,279 clients |
| TRAIN | 3,113,814 decisions |
| VALIDATION | 438,185 decisions |
| seed | 13 — **one**, see below |
| `TEST` | never used; the seal is measured in `_SEAL/test_seal_measured.json` |

**One seed, and that is a real limit.** The GRU-versus-LSTM gap of `0.0006` sits inside the
three-seed spread, so a second seed could reverse their order. It could not plausibly reverse
the TCN's `0.0079`. The conclusion that survives one seed is *"the GRU is not worse than any
alternative and is the cheapest of the two that tie"* — which is enough to keep it, and not
enough to claim it is best.
