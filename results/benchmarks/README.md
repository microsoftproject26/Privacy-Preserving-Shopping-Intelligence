# Public benchmark results

The REES46 recipe was run unchanged on six public sequential-recommendation datasets: the central, federated and
on-device arms, with the same code (`ppsi.benchrun`). The goals were to check that the findings transfer, and to
compare with published numbers where the splits are identical. No new method was tuned for these datasets. Every file
is an aggregate table, and no per-user value is stored.

## Datasets and protocol

`datasets.csv` lists the six datasets.

* **Data.** Beauty, Sports, Toys and MovieLens-1M are the eSASRec release splits; our copies are byte-identical to
  the published files. Video Games and Baby Products are the official Amazon Reviews 2023 5-core
  `last_out_w_his` splits. They have no published reference, so they are compared ours against ours only.
* **Split.** Leave one out. The test target is each user's held-out last item. The inner-validation target is the
  last item of the training history, and model selection uses only that.
* **Ranking.** Full catalogue: every item is a candidate except the user's already-seen items. Ties go to the smaller
  item id. A target never seen in training counts as a miss.
* **Metrics.** NDCG@10 is primary. HR@10, NDCG@20, HR@20 and MRR@20 are secondary. With one target per user,
  HR@10 equals Recall@10.
* **Populations.** All test users. The 15 % server band and the other 85 % are also kept in the raw results.
* **Checkpoints.** `practical_best` is the best inner-validation NDCG@10 over the training marks; this is the eSASRec
  practice and the default in the tables. `controlled_budget` is the fixed endpoint.
* **Seeds.** 2026 and 2027. A third seed, 2028, was added where the first two differed by more than 0.002 on
  validation (MovieLens-1M `C_FULL`).
* **Single test look.** Every run and checkpoint was locked per dataset, and then the test split was scored exactly
  once.

Arms: the same as on REES46 (see [`../rees46/README.md`](../rees46/README.md)).

* `C_FULL` is the central model (d 256, 3 blocks) on all users.
* `PRE` is the device-size model (d 64, 2 blocks), pretrained centrally on a seeded 15 % band of users. That band
  plays the role of the users who consented to server training.
* `FA` is federated training from random weights over all users, as clients.
* `FA_WARM` and `FA_WARM_FROZEN` start from `PRE`, and `FA_WARM_FROZEN` keeps the item tables frozen.
  `warm_star = yes` marks the warm variant chosen on inner validation. The FedProx, personalised, 8-bit, DP and
  fine-tuning arms (`FP_*`, `PF_*`, `FA_Q8_*`, `FA_1024_*`, `DP8_*`, `FT_*`) are built on that variant.

## Confirmatory test results

| File | Content |
|---|---|
| `confirmatory_track_a.csv` | **Track A (headline):** the REES46 recipe unchanged, every arm, test metrics, NDCG@10 as % of `C_FULL` |
| `confirmatory_track_b.csv` | **Track B (secondary):** the best grid cell per arm on the four eSASRec datasets (budget in epochs of exposure for the central arms, maximum sequence length, dropout) |

Track A and Track B are never mixed in one claim.

* The central model wins on every dataset.
* With the registered recipe, the best federated arm keeps between 13 % (Toys) and 77 % (MovieLens-1M) of
  `C_FULL`'s NDCG@10.
* On the Amazon datasets the warm start from a 15 % band is weak: `PRE` alone is at 9-48 % of `C_FULL`.
* The main reason the federated arms fall so far behind, found after the test look, is the FedAdam setting
  `tau = 1e-3`, which froze the federated runs (see [`docs/fedadam-tau.md`](../../docs/fedadam-tau.md)). A short
  round budget and the weak warm start add to it. The exploratory re-runs below address all three.

## Comparison with published results

| File | Content |
|---|---|
| `published_table1_ndcg10.csv` | Table 1 of Spotify's RecSys 2026 capacity-probe paper (arXiv 2608.19833): full-catalogue NDCG@10 of MC, FMC, FMC+, SASRec+, eSASRec, SeqRules and PCTM on the four eSASRec splits |
| `published_reference.csv` | our `C_FULL` (Track A and the Track-B best) and our best federated arm against SASRec+, eSASRec and PCTM |

* **Source.** These are the only published numbers on byte-identical splits with the same full-ranking protocol.
  Our copy of the data and the protocol reproduces the deterministic MC, SeqRules and PCTM cells of that table to
  four decimals, in 12 of 12 cells. Most federated-recommendation papers report sampled metrics (for example, 100
  random negatives), which are not comparable with full ranking.
* **Claim rule.** The published cells are single runs, with a run-to-run spread of about 0.003 for SASRec+ and
  eSASRec. A difference is therefore stated only if it is larger than 0.005 NDCG@10 and every seed lies on the same
  side; otherwise the claim is `on par`.
* **Disclosure.** The published SASRec+ and eSASRec rank a catalogue built after the validation item is removed, so
  a few test targets (89 / 265 / 232 / 0 on Beauty / Sports / Toys / MovieLens-1M) are unreachable for them. We rank
  the full training catalogue.

Results:

* With the recipe unchanged (Track A), our central SASRec is below SASRec+ and eSASRec on the three Amazon sets.
  On MovieLens-1M it is above both (0.1951 against 0.1739 for eSASRec).
* With the Track-B budget and dropout, it is on par with eSASRec on Beauty and Toys, still below on Sports, and above
  on MovieLens-1M.
* PCTM, the strongest published model on these splits, stays ahead on the Amazon sets.
* Every federated arm of the registered recipe is far below the published central models.

## Exploratory fixed-recipe test (after the confirmatory test look)

> **Exploratory, after the confirmatory test look.** This recipe was chosen after the confirmatory results were
> known, on inner validation only. Its runs were locked before the one exploratory test look. It is reported beside
> the confirmatory tables and replaces none of them.

The fixed federated recipe changes four things:

* FedAdam `tau = 1e-6`;
* a 4x larger round budget (24 epochs of exposure);
* trainable item tables;
* server step and client learning rate picked on validation.

`C_SMALL` is the central device-size model (d 64) trained on the same users. It separates the cost of the smaller
model from the cost of federation.

| File | Content |
|---|---|
| `exploratory_fixed_fl_test.csv` | test NDCG@10 (mean of 2 seeds) of `C_FULL`, `C_SMALL`, the fixed cold and warm federated arms, `PRE` and the confirmatory `FA` / `FA_WARM`, with percentages of `C_FULL` |
| `exploratory_fixed_fl_contrasts.csv` | paired differences with 95 % bootstrap intervals and the claim rule (`descriptive` on the two Amazon 2023 sets) |

The fix works on the test split:

* The fixed warm federated arm keeps 75-92 % of `C_FULL`. With the registered recipe the federated arms kept 13-77 %.
* The smaller model costs little: `C_SMALL` keeps 90-110 % of `C_FULL`.
* The cost of federation at matched size, `FA_WARM_TAUFIX / C_SMALL`, is 78-86 %.
* On Toys the fixed warm arm is `on par` with `C_FULL` under the claim rule.

## Validation-only follow-ups (exploratory)

> **Exploratory, inner validation only.** These experiments were run after the confirmatory look, and no test value
> was read for them. They use the fixed federated recipe.

**Disjoint server band (`validation_disjoint_server_band.csv`).** `PRE` is trained on a band of users, 15 % or
50 %, that is disjoint from the federated clients. The federated arms then run only on the remaining users, and
every value is NDCG@10 on those users' inner validation. `c_small_on_fl_users` is a central model trained on the
federated users alone.

* On the three Amazon sets with a 50 % band, the warm start beats the cold federated run by 27-55 %. It also matches
  or beats the central model trained on the federated users alone.
* With a 15 % band, the gain is 5-8 %.
* On MovieLens-1M (6,040 users) the warm start does not beat cold, even at 50 %.

A real server-side data advantage is what makes a warm start pay off.

**Capacity (`validation_capacity.csv`).** The total gap to `C_FULL` splits into the cost of the smaller model
(`C_SMALL / C_FULL`, 94-111 %) and the cost of federation at matched size (`FA d64 / C_SMALL`, 81-93 %).
Federated training of the large model (d 256) is no better than at d 64. From random weights it is slightly worse
on all four datasets. From the pretrained model it is within +-5 % of d 64: a little better on Beauty and Toys, a
little worse on Sports and MovieLens-1M. Model size is not the bottleneck of the federated arms.
