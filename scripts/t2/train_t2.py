"""The T2 purchase-likelihood head on the encoder S2-DS-01 trained.

**The decision:** the first time a product is viewed inside a session.
**The label:** whether that same product is purchased later in the same session.

This is where `Phase1Batch`'s *query* channels finally earn their place. T1 asks "what
category next" and reads only the history; T2 asks "will *this* product be bought", so the
model has to see the item the question is about.

Three things make T2 a different problem from T1, and each one has its own trap.

**1. Censoring.** 17.95% of VALIDATION decisions are censored: the user viewed a product at
the end of a session and whether they would have bought it was **never observable**. Their
`label_value` is null, not zero. Training them as negatives adds 22% more "no" on data where
the answer is unknown, and teaches the model that people do not buy - something the data
never said. `task_mask` is the whole defence, and nothing in a metric would flag its
absence.

**2. Imbalance.** 11,297 positives in 322,087 mature decisions - a base rate of **3.51%**.
A model that answers "never" scores **96.5% accuracy**. Accuracy is meaningless here, which
is why the protocol's headline is PR-AUC.

**3. Calibration.** Plain BCE produces probabilities that mean something. `pos_weight` lifts
ranking metrics while destroying that, so it is used only if validation evidence justifies
it, and if used, Brier and Log Loss are reported on a calibrated output with the calibration
fitted on validation alone.

The baselines to beat, from `task_headroom_v1`:

    prevalence, no signal          0.0351   PR-AUC
    TRAIN category purchase rate   0.0572
    TRAIN item popularity          0.0831   <- the number to beat
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

PROJECT = Path(__file__).resolve().parent.parent
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "ppsi" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    sys.path.insert(0, str(PROJECT / "Repo_S2DS01"))

from ppsi.models.batch_spec import phase1_batch_spec_v1  # noqa: E402
from ppsi.models.checkpoint import load_encoder as shared_load_encoder
from ppsi.models.session_gru import SessionGRU, SessionGRUConfig, build_model  # noqa: E402
from ppsi.training.batch import Phase1Batch  # noqa: E402

TASK = Path(__file__).resolve().parent
OUTPUT = TASK / "output"
OUTPUT.mkdir(exist_ok=True)

SPEC = phase1_batch_spec_v1()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Two sets, because the censoring correction in `labels.py` moves 70,467 VALIDATION rows
# from unscored to scored negatives, and every baseline moves with them. Reporting the
# model on corrected labels against a baseline measured on the published mask would be a
# comparison across two different populations - the exact deception this task documents.
#
# `smoothed item+category` is new and it is not ours: the review found it, and it beats the
# item-popularity rule we published as "best simple" by a wide margin. A baseline nobody
# tried is not a baseline. It is now the number to beat.
BASELINES_PUBLISHED_MASK = {
    "prevalence (no signal)": 0.0351,
    "TRAIN category purchase rate": 0.0572,
    "TRAIN item popularity": 0.0831,
    "smoothed item+category, alpha=100": 0.0974,
}
BASELINES_CORRECTED = {
    "prevalence (no signal)": 0.0288,
    "TRAIN item popularity": 0.0657,
    "smoothed item+category, alpha=100": 0.0757,
}
BASELINES = BASELINES_PUBLISHED_MASK          # retained for the older result JSONs
BEST_SIMPLE = 0.0974                          # published mask, the strongest simple rule
BEST_SIMPLE_CORRECTED = 0.0757
HALF_WIDTH = 0.0029

ARRAYS = ("category", "product", "event", "brand", "price_band", "gap", "lengths",
          "target", "query_category", "query_product", "query_brand", "query_price_band",
          "client")


def resolve_cache() -> Path:
    """The T2 window cache, wherever this is running."""
    for candidate in (Path(__file__).resolve().parent.parent / "S2-DS-01_GRU_T1_Model" / "cache_t2",
                      Path(__file__).resolve().parent.parent / "data" / "cache_t2",
                      Path("data/cache_t2")):
        if candidate.is_dir():
            return candidate
    raise SystemExit("cannot find the T2 window cache")


CACHE = resolve_cache()


@dataclass(slots=True)
class Split:
    """T2 windows, their labels, and the mask that decides which rows may train."""

    name: str
    data: dict
    label: np.ndarray
    mask: np.ndarray
    published: np.ndarray

    @classmethod
    def load(cls, split: str, *, corrected: bool = True) -> "Split":
        """`corrected` restores the terminal negatives upstream withheld - see `labels.py`.

        A censored label is null in the parquet and arrives here as NaN, and it must never
        silently become a zero. That rule has not changed. What changed is which rows are
        censored at all: upstream marks a decision censored when it is the last event of
        its **session**, but `S1-DS-05/06` already excluded every session that crosses a
        split boundary, so each surviving session is complete inside its own window and the
        answer to *"purchased later in this session?"* is observed. It is no.

        Those rows carry NaN in the cache purely because upstream withheld them, and their
        true label is exactly the zero `nan_to_num` produces - the upstream rule only ever
        censored rows whose label was already 0. So the correction is a mask change, not a
        label change, and `published` keeps the old mask so both numbers stay reportable.
        """
        prefix = f"t2_{split}"
        data = {key: np.load(CACHE / f"{prefix}_{key}.npy") for key in ARRAYS}
        label = np.load(CACHE / f"{prefix}_label_value.npy")
        mask = np.load(CACHE / f"{prefix}_task_mask.npy").astype(bool)

        withheld = ~mask
        assert np.isnan(label[withheld]).all(), (
            "a withheld row carries a label; the cache is not the contract labels.py "
            "describes and the correction must not be applied blind")
        assert not np.isnan(label[mask]).any(), "an observed row lost its label"

        label = np.nan_to_num(label, nan=0.0).astype("float32")
        return cls(split, data, label,
                   np.ones_like(mask) if corrected else mask, mask)

    def __len__(self) -> int:
        return len(self.mask)

    @property
    def trainable(self) -> np.ndarray:
        return np.flatnonzero(self.mask)


def to_batch(split: Split, rows: np.ndarray) -> Phase1Batch:
    take = lambda key, dtype: torch.from_numpy(  # noqa: E731
        np.ascontiguousarray(split.data[key][rows]).astype(dtype))
    lengths = take("lengths", "int64")
    size, width = len(rows), split.data["category"].shape[1]
    present = torch.from_numpy(split.mask[rows].copy())
    target = torch.from_numpy(split.label[rows].copy()).unsqueeze(-1)
    # The canonical filler for an absent target. Masked rows must carry it rather than a
    # stale value, so that a masking bug shows up as a contract violation.
    target[~present] = 0.0
    return Phase1Batch(
        history_categorical_ids={
            "category_id": take("category", "int64"),
            "product_bucket": take("product", "int64"),
            "event_type_id": take("event", "int64"),
            "brand_bucket": take("brand", "int64"),
            "price_band": take("price_band", "int64"),
        },
        history_continuous_features=take("gap", "float32").unsqueeze(-1),
        lengths=lengths,
        history_mask=torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1),
        query_categorical_ids={
            "query_category_id": take("query_category", "int64"),
            "query_product_bucket": take("query_product", "int64"),
            "query_brand_bucket": take("query_brand", "int64"),
            "query_price_band": take("query_price_band", "int64"),
        },
        query_continuous_features=torch.zeros(size, 0, dtype=torch.float32),
        candidate_ids=torch.zeros(size, 1, dtype=torch.int64),
        candidate_categorical_ids={
            "candidate_category_id": torch.full((size, 1), 589, dtype=torch.int64),
            "candidate_price_band": torch.full((size, 1), 5, dtype=torch.int64),
        },
        # Width 1, not 0: `S2-DS-07` added a retrieval-rank channel to the batch spec and
        # the canonical validator checks this shape against it. T2 has no candidates, so
        # the single slot is a masked zero - but it must exist, or the one spec has two
        # different meanings depending on which task built the batch.
        candidate_continuous_features=torch.zeros(
            size, 1, SPEC.candidate_continuous_dim, dtype=torch.float32),
        candidate_mask=torch.zeros(size, 1, dtype=torch.bool),
        t1_target=torch.zeros(size, dtype=torch.int64),
        t2_target=target,
        t3_gains=torch.zeros(size, 1, dtype=torch.float32),
        t1_present=torch.zeros(size, dtype=torch.bool),
        t2_present=present,
        t3_present=torch.zeros(size, dtype=torch.bool),
    )


def load_encoder(model: SessionGRU, checkpoint: Path) -> dict:
    """Start from the S2-DS-01 encoder rather than from scratch.

    The whole premise is one backbone serving three tasks. Training T2 from random weights
    would measure something else entirely - a separate model that happens to share an
    architecture.

    The rule itself lives in `ppsi.models.checkpoint`. It used to live here *and* in
    `S2-DS-07`, the two drifted apart, and this task's copy then refused the very
    checkpoint it is built on when the T3 candidate projection was widened.
    """
    return shared_load_encoder(model, checkpoint,
                               new_prefixes=("t2_head", "query_"))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """PR-AUC, delegated to scikit-learn.

    This was hand-written first, to avoid depending on a library's tie convention. That was
    the wrong instinct twice over: `scikit-learn==1.9.0` is already pinned in the
    repository, and a second implementation of a metric is exactly the defect that has cost
    this project three wrong numbers.

    The hand-written version agreed with sklearn to **six decimal places** on any input with
    distinct scores, and disagreed only in the degenerate all-tied case - 0.03540 against
    the true base rate 0.03507 - because a step-wise sum over an arbitrary ordering of ties
    is not the same as grouping them. A trained model produces distinct scores, so it never
    mattered in practice; it mattered in the gate, which scores a no-signal baseline where
    everything ties.
    """
    return float(average_precision_score(labels, scores))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def evaluate(model: SessionGRU, split: Split, *, batch_size: int = 2048) -> dict:
    """Score every *mature* validation decision. Censored rows are not scored at all."""
    model.eval()
    rows = split.trainable
    scores = np.zeros(len(rows), dtype="float64")
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            batch = to_batch(split, chunk).to(DEVICE)
            scores[start:start + len(chunk)] = (
                model(batch).t2_logit.squeeze(-1).float().cpu().numpy())

    labels = split.label[rows].astype("float64")
    probability = 1.0 / (1.0 + np.exp(-scores))

    def measure(keep: np.ndarray) -> dict:
        y, s, q = labels[keep], scores[keep], probability[keep]
        return {
            "pr_auc": round(average_precision(s, y), 4),
            "roc_auc": round(roc_auc(s, y), 4),
            "brier": round(float(((q - y) ** 2).mean()), 5),
            "log_loss": round(float(-(y * np.log(np.clip(q, 1e-7, 1))
                                      + (1 - y) * np.log(np.clip(1 - q, 1e-7, 1))
                                      ).mean()), 5),
            "scored": int(keep.sum()),
            "positives": int(y.sum()),
            "base_rate": round(float(y.mean()), 4),
            "mean_probability": round(float(q.mean()), 4),
        }

    result = measure(np.ones(len(rows), dtype=bool))
    # The same scores on the rows upstream would have kept, so the correction stays
    # auditable: a number that cannot be compared to the one it replaced is an assertion.
    on_published = split.published[rows]
    if not on_published.all():
        result["on_published_mask"] = measure(on_published)
    return result


def train(train_split: Split, validation: Split, *, config: SessionGRUConfig, seed: int,
          checkpoint: Path, freeze_encoder: bool, pos_weight: float | None,
          learning_rate: float, epochs: int, batch_size: int, label: str) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(seed, batch_spec=SPEC, config=config)
    provenance = load_encoder(model, checkpoint)
    model = model.to(DEVICE)

    if freeze_encoder:
        # Freezing makes negative transfer structurally impossible but limits the head.
        # Fine-tuning does the opposite. Both are measured rather than chosen on taste.
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(("t2_head", "query_"))
    trainable = [p for p in model.parameters() if p.requires_grad]

    optimiser = torch.optim.Adam(trainable, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    weight = (torch.tensor([pos_weight], device=DEVICE) if pos_weight else None)
    loss_function = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=weight)

    rows = train_split.trainable
    generator = np.random.default_rng(seed)
    print(f"  {label}  |  {sum(p.numel() for p in trainable):,} trainable  |  "
          f"{len(rows):,} of {len(train_split):,} rows contribute "
          f"({(1 - len(rows) / len(train_split)) * 100:.2f}% censored)")

    best, best_state, curve = -1.0, None, []
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        order = rows.copy()
        generator.shuffle(order)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = to_batch(train_split, np.sort(order[start:start + batch_size])).to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            # Both sides masked: a censored row contributes nothing, by construction.
            present = batch.t2_present
            loss = loss_function(batch_logit := model(batch).t2_logit[present],
                                 batch.t2_target[present]) / max(present.sum().item(), 1)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()
            total += loss.item() * len(batch_logit)
            seen += len(batch_logit)
        scheduler.step()

        measured = evaluate(model, validation)
        row = {"epoch": epoch, "train_loss": round(total / max(seen, 1), 5),
               **{k: measured[k] for k in ("pr_auc", "roc_auc", "brier", "log_loss")},
               "seconds": round(time.time() - started, 1)}
        curve.append(row)
        marker = ""
        if measured["pr_auc"] > best:
            best = measured["pr_auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"    epoch {epoch:>2}  loss {row['train_loss']:.4f}  "
              f"PR-AUC {row['pr_auc']:.4f}  ROC {row['roc_auc']:.4f}  "
              f"brier {row['brier']:.5f}  ({row['seconds']:.0f}s){marker}")

    model.load_state_dict(best_state)
    return model, curve, best, provenance
