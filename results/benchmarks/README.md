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
