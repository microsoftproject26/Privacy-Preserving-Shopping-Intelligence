# Membership-inference audit (`ppsi.privacy`)

Federated learning keeps raw events on the device, but the trained model can still reveal *whose* events it was
trained on. The audit measures that directly. It asks: given a user's events and the final model, can an attacker
tell whether the user was in the training set?

## Protocol

1. **Groups.**
   * Members: users whose training-window events were used to train the audited model (10,000 in our audits).
   * Non-members: the same number of users from the same population who were never trained on.
   * Only training-window events are used, so no validation or test row is read.
2. **Losses.** For every user and every model, compute the mean cross-entropy over that user's events
   (`ppsi.privacy.user_losses.user_mean_losses`). The models are the audited ones plus a *reference* model that saw
   neither group, for example the server-pretrained starting point.
3. **Score.** Calibrated score = reference loss - audited loss. A model that memorised a user fits that user better
   than the reference does. The raw score (minus the audited loss) is reported beside it.
4. **Validity.** The reference model's raw-score AUC CI must include 0.5. If it does not, the two groups differ for
   some other reason, and no leakage number is reported.
5. **Metrics.**
   * ROC AUC, with a user-bootstrap 95 % CI (2,000 replicates).
   * Threshold transfer: users are split into halves, the threshold is set on half A's non-members at a 1 % (and
     0.1 %) false-positive rate, and it is evaluated on half B with exact Clopper-Pearson bounds.
   * Empirical epsilon lower bound: `ln(TPR_lower / FPR_upper)` at 1 % FPR.
6. **Verdicts.**
   * A non-DP model shows *material leakage* if the AUC CI lower bound is above 0.55, or if the TPR at 1 % FPR has a
     lower bound above 3 %.
   * A DP model *passes* if its empirical epsilon lower bound does not exceed its accountant epsilon.

## Usage

```python
from ppsi.privacy import membership_inference as mi
from ppsi.privacy.user_losses import user_mean_losses

ref = (user_mean_losses(pre, member_batches, n), user_mean_losses(pre, nonmember_batches, n))
targets = {name: (user_mean_losses(m, member_batches, n), user_mean_losses(m, nonmember_batches, n))
           for name, m in models.items()}
result = mi.audit(ref, targets, dp_epsilon={"DP8": 8.0}, split_seed=2026, boot_seed=2026)
print(mi.as_rows(result))
```

Each batch carries the adapter's input tensors, `target_class`, and `user`, the user's index in the selection.

## What we found on REES46

The table is `results/rees46/privacy_membership_inference.csv`.

* The confirmatory federated models showed no material leakage (AUC about 0.50), but only because they barely moved
  away from the pretrained model.
* A tau-fixed federated model trained from random weights leaked clearly: AUC 0.747, with a TPR of 14 % at 1 % FPR.
* The same model trained with DP at epsilon 8 was back at chance (AUC 0.502) and passed.
* Central fine-tuning on the users' raw events leaked as well (AUC 0.626).

## Tests

```powershell
uv run python -m pytest tests/privacy -q
```

The tests run on synthetic losses and the synthetic model family. They cover:

* AUC against brute force, with ties;
* Clopper-Pearson against closed forms;
* the threshold rule and the epsilon bound;
* leakage, no leakage and DP verdicts;
* the invalid-reference case;
* per-user losses against a direct computation.
