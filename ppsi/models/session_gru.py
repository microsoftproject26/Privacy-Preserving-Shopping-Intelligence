"""The shared session encoder and its three heads.

One encoder reads the session once; three small heads ask it three different questions.
That is not a tidiness preference - the project's premise is a model that runs on a
phone, and three separate models would be three times the storage and battery.

The encoder satisfies `ppsi.training.protocol.Phase1Model`, so `LocalTrainerCore` and the
Flower adapters can drive it unchanged. It is not trained *through* the core, because the
core packs and sha256-hashes the whole state dict on every step and this model has a
1.2M-parameter embedding table; at 6,081 steps per epoch that cost dominates. The
contract is honoured, the loop is ours.

**Readout: gather, not pack.** `pack_padded_sequence` obstructs the ONNX export that
`S2-SE-01` has to produce. With right-padded history the two are mathematically
identical - padding sits strictly after position `lengths-1`, and a unidirectional GRU's
state there cannot be influenced by anything after it - so the export-friendly form costs
only a little wasted compute on padding. `ppsi.training.stub_model` already does this.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from ppsi.models.batch_spec import (
    CATEGORY_COUNT,
    HISTORY_CHANNELS,
    phase1_batch_spec_v1,
)
from ppsi.training.batch import Phase1Batch, Phase1BatchSpec
from ppsi.training.outputs import RawModelOutput
from ppsi.training.state import SharedStateSpec

# The three seeds S2-PR-09 runs the final R1 across, and the ones R2a must match.
ALLOWED_SEEDS = (13, 42, 2026)

# Per-channel embedding widths. Categories carry the most signal and get the most room;
# price band has six values and needs almost none.
DEFAULT_WIDTHS = {
    "category_id": 64,
    "product_bucket": 24,
    "event_type_id": 4,
    "brand_bucket": 16,
    "price_band": 4,
}


@dataclass(frozen=True, slots=True)
class SessionGRUConfig:
    """Everything the ablation ladder varies, and nothing it does not.

    `channels` is what makes the ladder possible: the batch always carries all five
    history channels, and the model chooses which to consume. The batch spec therefore
    never changes as features are added or removed, which is what lets `S2-SE-01` export
    against a stable contract while `S2-DS-05` is still exploring.
    """

    channels: tuple[str, ...] = HISTORY_CHANNELS
    use_gap: bool = True
    widths: dict = field(default_factory=lambda: dict(DEFAULT_WIDTHS))
    hidden: int = 128
    layers: int = 1
    dropout: float = 0.1
    # Width of the T3 cross-feature reranker's hidden layer. Deliberately small: it scores
    # 100 candidates per decision, so its cost is multiplied by the candidate width, and it
    # is asked to add a correction to a strong prior rather than to rank from nothing.
    t3_hidden: int = 64

    def __post_init__(self) -> None:
        unknown = set(self.channels) - set(HISTORY_CHANNELS)
        if unknown:
            raise ValueError(f"unknown history channels: {sorted(unknown)}")
        if not self.channels and not self.use_gap:
            raise ValueError("the encoder needs at least one input channel")
        if self.hidden <= 0 or self.layers <= 0:
            raise ValueError("hidden and layers must be positive")
        if self.t3_hidden <= 0:
            raise ValueError("t3_hidden must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class SessionGRU(nn.Module):
    """Session history to a T1 category distribution, a T2 purchase logit and T3 scores.

    Satisfies `Phase1Model`: exposes `category_count`, implements `shared_state_spec`,
    and takes a whole `Phase1Batch` in `forward`.
    """

    def __init__(
        self, *, batch_spec: Phase1BatchSpec | None = None, config: SessionGRUConfig | None = None
    ) -> None:
        super().__init__()
        self.batch_spec = batch_spec if batch_spec is not None else phase1_batch_spec_v1()
        self.config = config if config is not None else SessionGRUConfig()
        # A plain attribute, not a property: LocalTrainerCore reads it directly.
        self.category_count = CATEGORY_COUNT

        spec_by_name = {channel.name: channel for channel in self.batch_spec.history_categorical}
        missing = set(self.config.channels) - set(spec_by_name)
        if missing:
            raise ValueError(f"channels absent from the batch spec: {sorted(missing)}")

        # No BatchNorm anywhere: its int64 `num_batches_tracked` buffer would make
        # SharedStateSpec.all_shared_floating refuse to build the federated state spec.
        self.history_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    spec_by_name[name].vocab_size,
                    self.config.widths[name],
                    padding_idx=spec_by_name[name].pad_id,
                )
                for name in self.config.channels
            }
        )

        width = sum(self.config.widths[name] for name in self.config.channels)
        if self.config.use_gap:
            width += self.batch_spec.history_continuous_dim

        hidden = self.config.hidden
        self.input_projection = nn.Linear(width, hidden)
        self.encoder = nn.GRU(hidden, hidden, num_layers=self.config.layers, batch_first=True)
        self.dropout = nn.Dropout(self.config.dropout)

        # --- heads ---------------------------------------------------------------
        # T1 is trained in S2-DS-01. The other two exist because RawModelOutput requires
        # all three tensors; they are trained in S2-DS-06 and S2-DS-07 on this same
        # encoder.
        #
        # Exactly CATEGORY_COUNT outputs, not CATEGORY_COUNT + 1. The S2-SMOKE model
        # carried an extra OOV output class, but every frozen T1 label is a real code in
        # 0..587 - a decision whose target category was unseen in TRAIN was dropped
        # upstream - so that class could never be correct. It was dead weight, and
        # `validate_raw_model_output` requires the logits to be exactly
        # [B, category_count] anyway.
        self.t1_head = nn.Linear(hidden, CATEGORY_COUNT)

        query_spec = {channel.name: channel for channel in self.batch_spec.query_categorical}
        self.query_embeddings = nn.ModuleDict(
            {name: nn.Embedding(channel.vocab_size, 16) for name, channel in query_spec.items()}
        )
        query_width = 16 * len(query_spec) + self.batch_spec.query_continuous_dim
        self.query_projection = nn.Linear(query_width, hidden)
        self.t2_head = nn.Linear(hidden * 2, 1)

        candidate_spec = {c.name: c for c in self.batch_spec.candidate_categorical}
        self.candidate_id_embedding = nn.Embedding(
            self.batch_spec.candidate_id_vocab_size,
            24,
            padding_idx=self.batch_spec.candidate_id_pad_id,
        )
        self.candidate_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(channel.vocab_size, 8, padding_idx=channel.pad_id)
                for name, channel in candidate_spec.items()
            }
        )
        candidate_width = 24 + 8 * len(candidate_spec) + self.batch_spec.candidate_continuous_dim
        self.candidate_projection = nn.Linear(candidate_width, hidden)

        # T3 as a residual reranker, not a from-scratch scorer.
        #
        # The first T3 attempt asked the model to out-rank a popularity-and-co-occurrence
        # ordering while giving it no way to see that ordering, and it scored 0.0996
        # against a 0.2046 baseline. Handing it the retrieval rank as a feature fixed the
        # collapse but left the model still obliged to *rediscover* the ordering before it
        # could improve on it - so most of its capacity went on reproducing a number we
        # already had, and any gain was entangled with how well it did so.
        #
        # These two parameters remove that obligation. `t3_rank_weight` starts at -1 so the
        # score is exactly the negated retrieval rank, and `t3_residual_scale` starts at 0
        # so the learned term contributes nothing. **At initialisation the model therefore
        # scores precisely the frozen retrieval order**, which the ladder asserts before
        # training: an epoch-0 NDCG that is not the baseline to the fourth decimal means
        # the prior is wired wrong.
        #
        # Every later point is then a measured departure from that ordering, and the gain
        # is the thing being reported rather than a difference of two independent fits.
        # The zero scale gets gradient immediately - d(loss)/d(scale) is the raw score, not
        # zero - so this delays learning rather than preventing it. (This is ReZero.)
        # Shape [1], not a scalar: `LocalTrainerCore` hashes every state_dict tensor by
        # viewing it as uint8, and a 0-dimensional tensor cannot be viewed. A scalar here
        # made the federated trainer step raise, which the contract test caught.
        self.t3_rank_weight = nn.Parameter(torch.tensor([-1.0]))

        # The cross-feature reranker. The previous residual was `session . candidate`, and
        # the query item never entered it - measured by permuting every query tensor across
        # a batch and watching the T3 scores move by **exactly 0.0** while T2 moved by 0.81.
        #
        # That made the comparison unwinnable rather than merely hard. The frozen retrieval
        # order is co-occurrence between the *query item* and each candidate, so a model
        # blind to the query cannot represent what the baseline does, let alone improve on
        # it. The 0.2505-against-0.2707 result it produced is evidence about a
        # session-candidate dot product, and about nothing else.
        #
        # Inputs per candidate, with h the session vector, q the query vector and c the
        # candidate vector: the three vectors, two multiplicative interactions that let the
        # network express "does this candidate match the query", an absolute difference, and
        # the retrieval rank.
        #
        # The final layer is zero-initialised, so `delta` is exactly zero before training and
        # the score is exactly the frozen ordering. Its *input* is not zero, so gradient
        # reaches it immediately - unlike gating the whole branch by a zero scalar, which
        # would leave both the gate and the branch with zero gradient and train neither.
        cross_width = 6 * hidden + 1
        self.t3_cross = nn.Sequential(
            nn.Linear(cross_width, self.config.t3_hidden),
            nn.ReLU(),
            nn.Linear(self.config.t3_hidden, 1),
        )
        nn.init.zeros_(self.t3_cross[-1].weight)
        nn.init.zeros_(self.t3_cross[-1].bias)

    # -- Phase1Model -------------------------------------------------------------
    def shared_state_spec(self) -> SharedStateSpec:
        return SharedStateSpec.all_shared_floating(self)

    # -- encoder -----------------------------------------------------------------
    def encode_history(self, batch: Phase1Batch) -> Tensor:
        """One vector per row summarising the session up to and including the decision."""
        parts = [
            self.history_embeddings[name](batch.history_categorical_ids[name])
            for name in self.config.channels
        ]
        if self.config.use_gap and self.batch_spec.history_continuous_dim > 0:
            parts.append(batch.history_continuous_features)
        projected = torch.tanh(self.input_projection(torch.cat(parts, dim=-1)))

        # The whole padded sequence, then the state at the decision position. Padding
        # follows that position and a unidirectional GRU cannot look forward, so this
        # equals packing - and unlike packing it exports.
        sequence, _ = self.encoder(projected)
        last = torch.clamp(batch.lengths - 1, min=0)
        rows = torch.arange(batch.batch_size, device=batch.lengths.device)
        gathered = sequence[rows, last]
        # A row with no history must contribute exactly zero, not the state of a
        # padding step.
        return torch.where(batch.lengths.unsqueeze(1) > 0, gathered, torch.zeros_like(gathered))

    def encode_query(self, batch: Phase1Batch) -> Tensor:
        parts = [
            self.query_embeddings[name](batch.query_categorical_ids[name])
            for name in self.query_embeddings
        ]
        if self.batch_spec.query_continuous_dim > 0:
            parts.append(batch.query_continuous_features)
        return torch.tanh(self.query_projection(torch.cat(parts, dim=-1)))

    def forward(self, batch: Phase1Batch) -> RawModelOutput:
        session = self.encode_history(batch)
        dropped = self.dropout(session)

        t1_logits = self.t1_head(dropped)

        query = self.encode_query(batch)
        t2_logit = self.t2_head(torch.cat([dropped, query], dim=-1))

        candidate_parts = [self.candidate_id_embedding(batch.candidate_ids)]
        candidate_parts.extend(
            self.candidate_embeddings[name](batch.candidate_categorical_ids[name])
            for name in self.candidate_embeddings
        )
        if self.batch_spec.candidate_continuous_dim > 0:
            candidate_parts.append(batch.candidate_continuous_features)
        candidates = torch.tanh(self.candidate_projection(torch.cat(candidate_parts, dim=-1)))

        # The retrieval rank is channel 0 of the candidate continuous features, normalised
        # to [0, 1) with 0 the best. Negated, it *is* the frozen ordering.
        if self.batch_spec.candidate_continuous_dim > 0:
            rank = batch.candidate_continuous_features[..., 0]
            width = candidates.shape[1]
            history = dropped.unsqueeze(1).expand(-1, width, -1)
            asked = query.unsqueeze(1).expand(-1, width, -1)
            cross = torch.cat([history, asked, candidates,
                               history * candidates, asked * candidates,
                               (asked - candidates).abs(),
                               rank.unsqueeze(-1)], dim=-1)
            t3_scores = (self.t3_rank_weight * rank
                         + self.t3_cross(cross).squeeze(-1))
        else:
            t3_scores = torch.einsum("bh,bkh->bk", dropped, candidates)

        # No sigmoid, no softmax, no sorting. The contract requires raw activations so
        # that loss and metric code owns those choices.
        return RawModelOutput(t1_logits=t1_logits, t2_logit=t2_logit, t3_scores=t3_scores)


def build_model(
    seed: int, *, batch_spec: Phase1BatchSpec | None = None, config: SessionGRUConfig | None = None
) -> SessionGRU:
    """Construct the model deterministically from a seed, without disturbing global RNG.

    `fork_rng` matters: building a model must not shift the stream that shuffles batches,
    or two runs that differ only in architecture would also differ in batch order and the
    comparison between them would measure both.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SessionGRU(batch_spec=batch_spec, config=config)


def common_initialization(
    seed: int, *, config: SessionGRUConfig, batch_spec: Phase1BatchSpec | None = None
) -> tuple[dict, str]:
    """The starting weights every regime must share, plus their digest.

    `S2-PR-09` runs the final centralized R1 across seeds 13, 42 and 2026, and `S2-PR-06`
    trains R2a from *the same* start. Without that, part of the measured gap between
    centralized and federated is just a different random initialisation - and the
    project's headline number is the size of that gap.

    **`config` is required, and that is the whole point of this signature.** It used to
    default to `None`, which built the default five-channel encoder rather than the
    selected two-channel one - a different architecture with a different parameter count
    and a different digest. `common_initialization(13)` therefore looked like a shared
    starting point while silently being the wrong model, and a federated lane following the
    handoff literally would have started somewhere the centralized lane never was. Every
    caller now has to say which architecture it means.

    Returns the state dict and a sha256 over it, so a later run can prove it started where
    it claimed to.
    """
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS}, got {seed}")
    model = build_model(seed, batch_spec=batch_spec, config=config)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        digest.update(buffer.getvalue())
    return state, digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
