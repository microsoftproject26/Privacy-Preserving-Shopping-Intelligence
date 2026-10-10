"""Per-user mean losses of a `ppsi.fedsim` model adapter, the input of the membership-inference audit.

`user_mean_losses(adapter, batches, n_users)` runs the adapter in eval mode without gradients over batches of
decisions and returns, per user, the mean full-catalogue cross-entropy of that user's decisions, summed in float64.
Each batch is a mapping with the adapter's input tensors, `target_class` (a class index per row, all >= 0) and
`user` (the 0-based index of the row's user in the audited selection). Every user must have at least one decision.
Use each user's training-window events only: the audit asks whether those events were trained on.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .membership_inference import AuditError


def user_mean_losses(adapter, batches: Iterable[Mapping], n_users: int, *, device: str | torch.device = "cpu"):
    sums = np.zeros(int(n_users), dtype=np.float64)
    counts = np.zeros(int(n_users), dtype=np.int64)
    module = getattr(adapter, "module", None)
    was_training = bool(module.training) if module is not None else False
    if module is not None:
        module.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                target = torch.as_tensor(batch["target_class"]).long()
                if bool((target < 0).any()):
                    raise AuditError("every audited row needs a target class")
                users = np.asarray(batch["user"], dtype=np.int64)
                if users.shape != (target.shape[0],) or users.min() < 0 or users.max() >= n_users:
                    raise AuditError("`user` must give one in-range user index per row")
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
                          if k not in ("user",)}
                logits = adapter.logits(adapter.query(inputs))
                ce = F.cross_entropy(logits.float(), target.to(logits.device), reduction="none")
                ce = ce.detach().to("cpu", dtype=torch.float64).numpy()
                if not np.all(np.isfinite(ce)):
                    raise AuditError("a loss value is not finite")
                np.add.at(sums, users, ce)
                np.add.at(counts, users, 1)
    finally:
        if module is not None and was_training:
            module.train()
    if np.any(counts == 0):
        raise AuditError(f"{int(np.count_nonzero(counts == 0))} users have no scored decision")
    return sums / counts
