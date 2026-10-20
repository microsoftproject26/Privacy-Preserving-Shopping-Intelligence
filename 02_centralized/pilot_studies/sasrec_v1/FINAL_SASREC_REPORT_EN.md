# SASRec campaign final report — `SASREC_PERFORMANCE_EXECUTION_V1` (English companion)

> **PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK.** VALIDATION_CONFIRM and TEST were inspected in the GRU project. Every
> VC/TEST number here is **retrospective**, not a new confirmatory test. No fresh confirmatory holdout exists.
>
> The required deliverable is `FINAL_SASREC_REPORT_AR.md`; this file carries the same numbers.

- Lock: `locks/SASREC_LOCK.json`, sha256 `004130c685676889030ef124c0482c152e6d2466807eac4c177e3ba5e8fa47d0`, written 2026-09-26T05:05:18Z before any VC/TEST row was loaded.
- One-look claim: `locks/RETRO_CLAIM.json` (2026-09-26T05:05:21Z); evaluation completed 2026-09-26T07:24:16Z; `retro_eval/RETRO_RESULTS.json` sha256 `1ac73dcda6d3bb8984fb5f48f3792ad648f4f7690cc23992560b01503f56cd6b`.

## 1. Summary
- **Headline single model:** `A1_RICH256_s2027` at epoch 5 (end of cooldown), selected on VS only by the pre-registered rule. VS macro MRR@20 = 0.186513.
- **TEST (retrospective):** macro MRR@20 = **0.170475** versus the locked GRU F0 0.168695 (difference +0.001780 [+0.001538, +0.002022]: higher, with the whole 95% CI above zero) and versus TRANS 0.147604 (difference +0.022871 [+0.022614, +0.023149]: higher, with the whole 95% CI above zero).
- **VALIDATION_CONFIRM (retrospective):** 0.171339 versus GRU F0 0.168988 (difference +0.002351; no bootstrap on VC).
- **Rich versus ID (seed 2026, both cooled down; one comparison):** TEST +0.003741 [+0.003514, +0.003955]: higher, with the whole 95% CI above zero. A single comparison, not general evidence of feature value.
- **Systems (separate; never model versus model against GRU):** the best system on TEST is S3 at 0.173224 (+0.002749 versus the headline; descriptive, no CI).
- **SOTA:** not claimed. No privacy mechanism tested. No causal claim.

### 1.1 Reading the results
- **(MEASURED)** The headline's TEST margin over GRU F0 is +0.001780 (+1.06% relative), about 10× the A1 two-seed TEST std (0.000177). Small, but statistically stable at the user level.
- **(MEASURED)** The margin shrank from VS (+0.012639) to VC (+0.002351) and TEST (+0.001780). The headline changed by -0.016038 from VS to TEST; GRU F0 by -0.005179. **(INFERENCE, untested)** Possible reasons: selection-split optimism (VS chose the epoch, the post-hoc cooldown and the seed), temporal distance from TRAIN, more OOV. None was isolated.
- **(MEASURED, descriptive, no CI)** The ID-only SASRec (A0) is **below** GRU F0: VC -0.001465, TEST -0.002211. The headline's advantage over GRU therefore comes with the extra feature channels; the architecture alone is not shown to beat GRU. This pair was not pre-registered and has no bootstrap.
- **(MEASURED)** Rich − ID is +0.003741 on TEST versus +0.007712 on VS (each at its best epoch).
- **(MEASURED)** The micro MRR@20 gain on TEST is near zero (+0.000417); micro R@20 +0.006212. The gain is in the per-user average and in top-20 retrieval. Warm +0.002445, unseen +0.000594.
- **(MEASURED, descriptive)** The best system, S3, is +0.002749 above the headline on TEST; the weight averages are below their single models: S4a -0.001619, S4b -0.001944, consistent with the VS-descriptive reading.

## 2. Task and protocol (MEASURED)
- Next-item prediction on the frozen REES46 dataset `rees46_nextitem_v1` (hash `860e5a1c…`), the same protocol as GRU.
- Full-catalogue ranking (K = 158,486); no sampling; no catalogue reduction.
- Primary metric user-macro MRR@20 over `eval_user_key`; OOV targets score zero (`end_to_end`); the `closed` denominator is shown separately.
- VC: 2,461,568 evaluable decisions, 373,953 users. TEST: 5,862,968 decisions, 746,662 users, 198,319 OOV targets.
- Fingerprint gate: 8 checks reproduce the archived GRU/TRANS/POP numbers exactly; all matched.

## 3. Campaign path
- **v2 failed (MEASURED):** tied head, input ×√d (×16), betas (0.9, 0.98), peak 5e-4. VS macro MRR@20: A0_ID256_s2026 e3 0.09133, e4 0.09009; A1_RICH256_s2026 e3 0.11791, e4 0.11681. Stopped by Fable as the owner's delegate and archived intact in `runs_v2/`. Interpretation (INFERENCE): common-mode drift of the tied table under Adam and dense CE, amplified ×16; A0 failed too, so the columns are not the cause.
- **v3 repair (a bundle of five, not isolated):** untied head; no √d plus input LayerNorm; betas (0.9, 0.999); recentred output table after every step; peak 1e-3, floor 1e-4, 4 epochs evaluated at 3 and 4.
- **Amendment 1:** the miscalibrated 20k criterion (≤ 3×init) was replaced by absolute caps, **after seeing 2.1–3.7k updates and before 20k**.
- **Amendment 2** (owner: "5 epochs", "best accuracy"): a fifth epoch **as a cooldown only** (LR = 1e-6 + (1e-4 − 1e-6)(1 − √t)), seed 2027, conditional cooldowns for A0 and s2027, systems S1–S4b reported separately.
- **Disclosure:** VS for epochs 3–4 had been seen before amendment 2, so the cooldown decision is **post-hoc**. VS was used only for selection; VC/TEST were used for no decision.
- **Precision:** TF32 training, strict FP32 evaluation. Pre-registered parity gate: loss difference 1.60e-05 (tolerance 0.0459), cohort difference 8.84e-05, grad-norm ratio 1.000124 → **TF32 ACCEPTABLE**.

## 4. VS results (selection split, full VS)
| Run | epoch | macro MRR@20 | micro MRR@20 | micro R@20 | macro R@20 |
|---|---|---|---|---|---|
| A0_ID256_s2026 | 3 | 0.173762 | 0.157588 | 0.447713 | 0.467502 |
| A0_ID256_s2026 | 4 | 0.172820 | 0.156587 | 0.446040 | 0.466152 |
| A0_ID256_s2026_cool5 | 5 | 0.178585 | 0.163117 | 0.454042 | 0.472578 |
| A1_RICH256_s2026 | 3 | 0.181421 | 0.164569 | 0.463627 | 0.484289 |
| A1_RICH256_s2026 | 4 | 0.179761 | 0.162887 | 0.461200 | 0.482829 |
| A1_RICH256_s2026_cool5 | 5 | 0.186298 | 0.170455 | 0.470434 | 0.489958 |
| A1_RICH256_s2027 | 3 | 0.181504 | 0.164840 | 0.464357 | 0.484863 |
| A1_RICH256_s2027 | 4 | 0.180630 | 0.163763 | 0.462194 | 0.483177 |
| A1_RICH256_s2027_cool5 | 5 | 0.186513 | 0.170701 | 0.470746 | 0.489614 |

- A1 two-seed gaps (matched epochs): e3 0.000083, e4 0.000869, e5 0.000216: a **range from two seeds, not a standard deviation**.

## 5. Selection and lock
- Rule: max full-VS macro MRR@20 over {A0_s2026, A1_s2026, A1_s2027} × epochs {3,4,5}; tie ≤ 1e-6 → micro R@20 → fewer params → registry order.
- Selected: `A1_RICH256_s2027`. The A1 seed gap at e5 (0.000216) is far below the cooldown gain and the Rich−ID gap; which seed won is not evidence.
- The lock holds 8 entries (3 singles + S1/S2/S3 + S4a/S4b); pair_check: both cooldowns (5 epochs) (comparable = True); dropped entries: none.

## 6. Retrospective VC and TEST results (PREVIOUSLY INSPECTED)
### 6.1 VALIDATION_CONFIRM (end_to_end)
| Model | macro MRR@20 | micro MRR@20 | micro R@20 | macro R@20 | macro NDCG@20 |
|---|---|---|---|---|---|
| A0_ID256_s2026 | 0.167523 | 0.150924 | 0.419138 | 0.438985 | 0.227831 |
| A1_RICH256_s2026 | 0.171278 | 0.154163 | 0.431176 | 0.452508 | 0.233748 |
| **A1_RICH256_s2027 (headline)** | 0.171339 | 0.154173 | 0.431504 | 0.452662 | 0.233836 |
| GRU_F0_locked | 0.168988 | 0.153140 | 0.424205 | 0.442041 | 0.229446 |
| GRU_F4_descriptive | 0.156429 | 0.139588 | 0.412435 | 0.434757 | 0.217930 |
| TRANS_full | 0.148740 | 0.128027 | 0.373900 | 0.410781 | 0.206916 |
| POP_full | 0.020168 | 0.012987 | 0.054264 | 0.080413 | 0.033256 |

### 6.2 TEST (end_to_end)
| Model | macro MRR@20 | micro MRR@20 | micro R@20 | macro R@20 | macro NDCG@20 |
|---|---|---|---|---|---|
| A0_ID256_s2026 | 0.166484 | 0.148456 | 0.410566 | 0.435926 | 0.226354 |
| A1_RICH256_s2026 | 0.170225 | 0.151451 | 0.422179 | 0.448659 | 0.232105 |
| **A1_RICH256_s2027 (headline)** | 0.170475 | 0.151677 | 0.422425 | 0.448530 | 0.232269 |
| GRU_F0_locked | 0.168695 | 0.151260 | 0.416213 | 0.439388 | 0.228657 |
| GRU_F4_descriptive | 0.156504 | 0.137933 | 0.405277 | 0.432728 | 0.217567 |
| TRANS_full | 0.147604 | 0.125881 | 0.365223 | 0.406487 | 0.205116 |
| POP_full | 0.021118 | 0.013237 | 0.055517 | 0.083133 | 0.034600 |

GRU_F4_descriptive is descriptive only (F4 < F0 in GRU is confounded by LR-step timing). POP_full is the popularity reference.

### 6.3 Closed denominator (OOV targets excluded)
| Model | closed macro MRR@20 | closed micro R@20 | closed n |
|---|---|---|---|
| A0_ID256_s2026 | 0.169623 | 0.424940 | 5,664,649 |
| A1_RICH256_s2026 | 0.173465 | 0.436960 | 5,664,649 |
| A1_RICH256_s2027 | 0.173728 | 0.437214 | 5,664,649 |
| GRU_F0_locked | 0.172055 | 0.430784 | 5,664,649 |
| GRU_F4_descriptive | 0.159465 | 0.419466 | 5,664,649 |
| TRANS_full | 0.150109 | 0.378010 | 5,664,649 |
| POP_full | 0.021387 | 0.057460 | 5,664,649 |

### 6.4 Paired user-level bootstrap (TEST, 1000 resamples, seed 2026)
| Paired difference (TEST, user-macro MRR@20) | diff | 95% CI | users | resamples |
|---|---|---|---|---|
| A1_RICH256_s2027_minus_GRU_F0_locked | +0.001780 | [+0.001538, +0.002022] | 746,662 | 1000 |
| A1_RICH256_s2027_minus_TRANS_full | +0.022871 | [+0.022614, +0.023149] | 746,662 | 1000 |
| A1_RICH256_s2026_minus_A0_ID256_s2026 | +0.003741 | [+0.003514, +0.003955] | 746,662 | 1000 |

- Conditional on these fitted models and a previously inspected period; the CIs do not capture training-seed variance.
- TEST seed variation (A1, two seeds): {"2026": 0.17022502422332764, "2027": 0.1704750955104828}, std(ddof=1) = 0.000177; two seeds only.

## 7. Systems S1–S4b (separate; never compared with GRU as model versus model)
### TEST
| System | definition | macro MRR@20 | micro MRR@20 | micro R@20 |
|---|---|---|---|---|
| S1 | A1 s2026 + A1 s2027 (log-prob mean) | 0.172115 | 0.153866 | 0.427231 |
| S2 | A0 s2026 + A1 s2026 (log-prob mean) | 0.172051 | 0.153830 | 0.424917 |
| S3 | A0 s2026 + A1 s2026 + A1 s2027 (log-prob mean) | 0.173224 | 0.155156 | 0.428307 |
| S4a | weight average A1 s2026 e3/e4/e5 | 0.168606 | 0.150080 | 0.422716 |
| S4b | weight average A1 s2027 e3/e4/e5 | 0.168531 | 0.150043 | 0.422775 |

### VALIDATION_CONFIRM
| System | definition | macro MRR@20 | micro MRR@20 | micro R@20 |
|---|---|---|---|---|
| S1 | A1 s2026 + A1 s2027 (log-prob mean) | 0.173156 | 0.156466 | 0.436545 |
| S2 | A0 s2026 + A1 s2026 (log-prob mean) | 0.173073 | 0.156471 | 0.433927 |
| S3 | A0 s2026 + A1 s2026 + A1 s2027 (log-prob mean) | 0.174282 | 0.157719 | 0.437560 |
| S4a | weight average A1 s2026 e3/e4/e5 | 0.169654 | 0.152655 | 0.431644 |
| S4b | weight average A1 s2027 e3/e4/e5 | 0.169536 | 0.152727 | 0.431768 |

- Weights fixed a priori (equal-weight log-softmax mean, or a uniform state_dict average). No system was added, dropped or re-weighted after any number was seen.
- The earlier VS-descriptive S2/S4a numbers (idle GPU1, before the lock): S2 ≈ A1 e5 (CI crosses 0); S4a < A1 e5 (CI below 0). Not used for selection.

## 8. User cohorts (warm / unseen)
| Model | split | warm n | warm macro MRR@20 | unseen n | unseen macro MRR@20 |
|---|---|---|---|---|---|
| A1_RICH256_s2027 | VALIDATION_CONFIRM | 1,782,654 | 0.169399 | 678,914 | 0.175351 |
| A1_RICH256_s2027 | TEST | 4,139,931 | 0.167479 | 1,723,037 | 0.175822 |
| GRU_F0_locked | VALIDATION_CONFIRM | 1,782,654 | 0.166302 | 678,914 | 0.174545 |
| GRU_F0_locked | TEST | 4,139,931 | 0.165034 | 1,723,037 | 0.175229 |

## 9. Resources
| Run | GPU | start | end | elapsed h | cumulative wall h | updates | params | peak alloc GiB | RSS max GiB |
|---|---|---|---|---|---|---|---|---|---|
| A0_ID256_s2026 | 0 | 2026-09-24T17:29:54Z | 2026-09-25T08:29:20Z | 14.99 | 15.83 | 627,824 | 82,897,430 | 3.12 | 7.26 |
| A0_ID256_s2026_cool5 | 1 | 2026-09-25T13:43:31Z | 2026-09-25T17:40:47Z | 3.95 | 20.20 | 784,780 | 82,897,430 | 3.12 | 7.17 |
| A1_RICH256_s2026 | 1 | 2026-09-24T17:30:51Z | 2026-09-25T08:52:55Z | 15.37 | 16.22 | 627,824 | 83,094,368 | 3.13 | 17.38 |
| A1_RICH256_s2026_cool5 | 1 | 2026-09-25T09:37:16Z | 2026-09-25T13:41:01Z | 4.06 | 20.70 | 784,780 | 83,094,368 | 3.13 | 17.11 |
| A1_RICH256_s2027 | 0 | 2026-09-25T09:23:22Z | 2026-09-26T00:48:01Z | 15.41 | 16.26 | 627,824 | 83,094,368 | 3.13 | 17.54 |
| A1_RICH256_s2027_cool5 | 0 | 2026-09-26T00:51:10Z | 2026-09-26T05:04:31Z | 4.22 | 20.91 | 784,780 | 83,094,368 | 3.13 | 17.61 |

- elapsed = this run's end − start; the cumulative wall time of a cooldown includes its source run. GPU0/GPU1 are on one Lobot server.

## 10. Disclosed errata
- Stale text fields in the locked registry, not fixed because editing breaks the resume hash: `tie_item_embeddings: true` (the model is untied), fallback text "5 epochs", `SASREC_INTERFACES.md` describes v2, a stale status field.
- Gate C on v2 attributed evidence to an old `train_sasrec.py` hash → regenerated (112/112).
- The final-stage review found a missing `model = build_model(...)` line that would have scored zero on every VC/TEST row → fixed before running.
- The cooldown could have deleted the source checkpoints → fixed before running.
- The automatic `render_report.py` was written for v2: it looks for `state/selection.json` and a v2 bootstrap key, and it did print "Not yet measured" (it built the key `A1_RICH256_s2026_minus_GRU_F0_locked` instead of `A1_RICH256_s2027_…`). Its output is preserved unmodified in `final/auto_render/`; this report supersedes it.
- An earlier working note gave the e5 seed gap as 0.000215 (truncated); the rounded value is 0.000216.

## 11. Realistic ceiling
- Fable's ≈ 0.20–0.22 macro MRR@20 is **INFERENCE from nearby literature, not a proven bound**. The headline VS value is 0.186513.

## 12. Limitations and prohibited claims
- No SOTA: no comparison with published numbers under other protocols; no same-protocol reproduction of SASRec papers.
- No privacy mechanism tested. No causality: the v3 repair is a bundle; Rich > ID is one comparison; A0 has one seed, A1 two.
- Systems are never compared with GRU as model versus model. VC/TEST are retrospective; no fresh confirmatory holdout.
