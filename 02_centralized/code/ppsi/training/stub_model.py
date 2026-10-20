"""Small deterministic three-head model used only to validate the trainer contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ppsi.training.batch import Phase1Batch, Phase1BatchSpec
from ppsi.training.outputs import RawModelOutput
from ppsi.training.state import SharedStateSpec


@dataclass(frozen=True, slots=True)
class StubModelConfig:
    """Non-scientific model dimensions for S1-PR-05 contract tests."""

    category_count: int
    embedding_dim: int = 4
    hidden_dim: int = 8

    def __post_init__(self) -> None:
        if self.category_count <= 1:
            raise ValueError("category_count must be > 1")
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")


class Phase1StubModel(nn.Module):
    """Contract-coverage model; it is not a final GRU or research baseline."""

    def __init__(self, *, batch_spec: Phase1BatchSpec, config: StubModelConfig) -> None:
        super().__init__()
        self.batch_spec = batch_spec
        self.config = config
        self.category_count = config.category_count

        self.history_embeddings = nn.ModuleDict(
            {
                channel.name: nn.Embedding(
                    _require_vocab(channel.name, channel.vocab_size),
                    config.embedding_dim,
                    padding_idx=channel.pad_id,
                )
                for channel in batch_spec.history_categorical
            }
        )
        self.query_embeddings = nn.ModuleDict(
            {
                channel.name: nn.Embedding(
                    _require_vocab(channel.name, channel.vocab_size),
                    config.embedding_dim,
                )
                for channel in batch_spec.query_categorical
            }
        )
        self.candidate_id_embedding = nn.Embedding(
            _require_vocab("candidate_ids", batch_spec.candidate_id_vocab_size),
            config.embedding_dim,
            padding_idx=batch_spec.candidate_id_pad_id,
        )
        self.candidate_embeddings = nn.ModuleDict(
            {
                channel.name: nn.Embedding(
                    _require_vocab(channel.name, channel.vocab_size),
                    config.embedding_dim,
                    padding_idx=channel.pad_id,
                )
                for channel in batch_spec.candidate_categorical
            }
        )

        history_input_dim = (
            len(batch_spec.history_categorical) * config.embedding_dim
            + batch_spec.history_continuous_dim
        )
        if history_input_dim <= 0:
            raise ValueError("Stub requires at least one history feature")
        query_input_dim = (
            len(batch_spec.query_categorical) * config.embedding_dim
            + batch_spec.query_continuous_dim
        )
        candidate_input_dim = (
            config.embedding_dim
            + len(batch_spec.candidate_categorical) * config.embedding_dim
            + batch_spec.candidate_continuous_dim
        )

        self.history_input_projection = nn.Linear(history_input_dim, config.hidden_dim)
        self.history_encoder = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.history_norm = nn.LayerNorm(config.hidden_dim)

        self.query_projection = (
            nn.Linear(query_input_dim, config.hidden_dim) if query_input_dim > 0 else None
        )
        self.context_projection = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.candidate_projection = nn.Linear(candidate_input_dim, config.hidden_dim)

        self.t1_head = nn.Linear(config.hidden_dim, config.category_count)
        self.t2_head = nn.Linear(config.hidden_dim, 1)

    def shared_state_spec(self) -> SharedStateSpec:
        return SharedStateSpec.all_shared_floating(self)

    def _history_inputs(self, batch: Phase1Batch) -> Tensor:
        features: list[Tensor] = [
            self.history_embeddings[channel.name](batch.history_categorical_ids[channel.name])
            for channel in self.batch_spec.history_categorical
        ]
        if self.batch_spec.history_continuous_dim > 0:
            features.append(batch.history_continuous_features)
        return torch.cat(features, dim=-1)

    def encode_history(self, batch: Phase1Batch) -> Tensor:
        """Return exact zero history representation for semantic length zero."""

        projected = torch.tanh(self.history_input_projection(self._history_inputs(batch)))
        sequence_output, _ = self.history_encoder(projected)
        last_index = torch.clamp(batch.lengths - 1, min=0)
        row_index = torch.arange(batch.batch_size, device=batch.lengths.device)
        gathered = sequence_output[row_index, last_index]
        gathered = self.history_norm(gathered)
        # Final override occurs after history-only transformation/normalization.
        return torch.where(
            batch.lengths.unsqueeze(1) > 0,
            gathered,
            torch.zeros_like(gathered),
        )

    def _encode_query(self, batch: Phase1Batch) -> Tensor:
        features: list[Tensor] = [
            self.query_embeddings[channel.name](batch.query_categorical_ids[channel.name])
            for channel in self.batch_spec.query_categorical
        ]
        if self.batch_spec.query_continuous_dim > 0:
            features.append(batch.query_continuous_features)
        if not features:
            return torch.zeros(
                batch.batch_size,
                self.config.hidden_dim,
                dtype=batch.history_continuous_features.dtype,
                device=batch.lengths.device,
            )
        assert self.query_projection is not None
        return torch.tanh(self.query_projection(torch.cat(features, dim=-1)))

    def _encode_candidates(self, batch: Phase1Batch) -> Tensor:
        features: list[Tensor] = [self.candidate_id_embedding(batch.candidate_ids)]
        features.extend(
            self.candidate_embeddings[channel.name](batch.candidate_categorical_ids[channel.name])
            for channel in self.batch_spec.candidate_categorical
        )
        if self.batch_spec.candidate_continuous_dim > 0:
            features.append(batch.candidate_continuous_features)
        return torch.tanh(self.candidate_projection(torch.cat(features, dim=-1)))

    def forward(self, batch: Phase1Batch) -> RawModelOutput:
        history = self.encode_history(batch)
        query = self._encode_query(batch)
        context = torch.tanh(self.context_projection(torch.cat([history, query], dim=-1)))
        candidates = self._encode_candidates(batch)
        scores = torch.einsum("bd,bkd->bk", context, candidates) / math.sqrt(self.config.hidden_dim)
        return RawModelOutput(
            t1_logits=self.t1_head(context),
            t2_logit=self.t2_head(context),
            t3_scores=scores,
        )


def _require_vocab(name: str, vocab_size: int | None) -> int:
    if vocab_size is None or vocab_size <= 0:
        raise ValueError(f"Stub model requires vocab_size for {name}")
    return vocab_size
