# REES46 data for the deployment study: task, splits and populations

This page defines the data behind the final results in [`results.md`](results.md). The earlier phase-1 pipeline (raw
CSV to Parquet, canonical events, the first cohort and temporal split) is documented in the `s1-ds-*` pages. No user
id or user-level file is in this repository; see [`data-access.md`](data-access.md) for how to obtain the data.

## Source

* **Logs.** REES46 e-commerce behaviour logs of one multi-category online store, October and November 2019: views,
  cart additions and purchases with product, category, brand, price, user and session.
* **Frozen dataset.** The study uses one frozen, hash-verified version of the processed data. Any preprocessing
  change must produce a new version, never an edit of the frozen one.

## The task

* **Target.** The next *different* product the user views within the session.
* **Input.** The user's last 50 events across all earlier sessions, strictly before the decision. The same history
  rule applies to training, validation and test.
  * A per-event channel marks the events of the current session.
  * The locked feature schema has 24 input columns. It was chosen by a minimal-sufficient feature search on a
    separate group of users.
* **Catalogue.** Every product with at least 5 training-window occurrences, which gives 158,486 products. Every
  prediction ranks the whole catalogue.
* **Denominator.** All evaluation rows count. A target outside the catalogue is a miss.
* **Normalisation.** Feature normalisers are fitted on the training window. For the federated arms this corresponds
  to one round of federated statistics, and it is disclosed as such.

## Chronological splits

One global timeline for every user (half-open UTC windows), so no user's future can leak into another user's past:

| Split | Window | Use |
|---|---|---|
| Training | 2019-10-01 to 2019-11-16 | training; labels must be observed before the window ends |
| Validation | 2019-11-16 to 2019-11-23 | model selection and the validation read |
| Test | 2019-11-23 to 2019-11-30 | the single test look |
| Label grace | 2019-11-30 to 2019-12-01 | late targets only; no decisions |

The test week contains Black Friday (29 November 2019). The change of the central model from validation to test is
reported as a shift indicator.

## Populations

* **Ranking.** Each user gets a pseudonymous selection key (a salted hash of the user id). Users are ranked by this
  key, and every population below is a fixed, outcome-free range of that ranking.
* **Pool.** The pool is the users with 5 to 100 eligible training decisions: 1,446,986 users and 28.6 M decisions.
  Fewer than 5 leaves a simulated device with too little history, and very long histories are often automated
  traffic.

| Population | Definition | Users | Role |
|---|---|---|---|
| Federated users | key ranks 0 - 131,071 | 131,072 (2.59 M training decisions) | the clients; every central reference on "the same users" trains on them |
| Calibration users | key ranks 0 - 32,767, inside the federated users | 32,768 | the only users on which tuning values were read |
| Server users | key ranks 131,072 - 1,348,681 | 1,217,610 (24.1 M decisions) | assumed to consent to server training; they train the pretrained model `PRE` |
| Full-data population | federated users + server users | 1,348,682 | trains `C_FULL` |
| Unseen validation users | the next 32,768 keys | 32,768 | never trained on; used to select the on-device fine-tuning recipe |
| Unseen holdout users | the next 32,768 keys | 32,768 | never trained on; new-user panel |
| Feature-search users | the last 32,768 keys | 32,768 | feature selection only |

Panels outside the pool, for descriptive results:

* users with no training history (true cold start);
* a 16,384-user sample with 1-4 decisions;
* a 16,384-user sample with more than 100 decisions.

**Disjointness.**

* The federated users, server users, unseen panels and feature-search users are disjoint key ranges, so every
  "unseen" panel is disjoint from every training population.
* The calibration users are *not* disjoint from the federated users. For that reason the on-device recipes were
  selected on the unseen validation users, and every tuned arm also reports a calibration-disjoint diagnostic.

**Evaluated users.**

* Validation: 40,764 federated users with at least one evaluation row (519,714 decisions); 51,211 users appear in
  the validation manifest.
* Test: 26,254 federated users with 229,742 decisions.

## Membership-inference populations

The privacy audit ([`membership-inference-audit.md`](membership-inference-audit.md)) reads training-window rows only:

* members: 10,000 federated users, drawn with a fixed seed;
* non-members: 10,000 unseen holdout users.

No validation or test row is used.
