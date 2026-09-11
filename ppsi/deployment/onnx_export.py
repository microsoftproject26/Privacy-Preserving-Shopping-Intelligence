"""Export a Phase 1 model to ONNX and prove the export computes the same thing.

`Phase1Batch` is a dataclass of named tensors; ONNX takes a flat list of tensors in a
fixed order. :class:`SessionGRUExportWrapper` is the one place that translation lives, so
the exported graph has a stable signature that does not move when the model's internals
do. Nothing here changes the model to make it easier to export.

Parity is measured by running the exported graph in ONNX Runtime, the runtime that will
actually serve it, rather than by re-reading the PyTorch model. A parity number that was
never executed in the deployment runtime is not evidence of anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ppsi.models.batch_spec import (
    CANDIDATE_CHANNELS,
    HISTORY_CHANNELS,
    QUERY_CHANNELS,
    phase1_batch_spec_v1,
)
from ppsi.training.batch import Phase1Batch, Phase1BatchSpec

OPSET_VERSION = 17
"""Pinned. The opset decides which operators exist, so it is part of the artifact."""

HISTORY_LENGTH_AXIS = "history_length"
CANDIDATE_WIDTH_AXIS = "candidate_width"
BATCH_AXIS = "batch"

HISTORY_GAP_INPUT = "history_gap"
LENGTHS_INPUT = "lengths"
CANDIDATE_IDS_INPUT = "candidate_ids"
CANDIDATE_RANK_INPUT = "candidate_rank"

INPUT_NAMES: tuple[str, ...] = (
    *HISTORY_CHANNELS,
    HISTORY_GAP_INPUT,
    LENGTHS_INPUT,
    *QUERY_CHANNELS,
    CANDIDATE_IDS_INPUT,
    *CANDIDATE_CHANNELS,
    CANDIDATE_RANK_INPUT,
)

OUTPUT_NAMES: tuple[str, ...] = ("t1_logits", "t2_logit", "t3_scores")

_HISTORY_COUNT = len(HISTORY_CHANNELS)
_QUERY_COUNT = len(QUERY_CHANNELS)
_CANDIDATE_COUNT = len(CANDIDATE_CHANNELS)


def _dynamic_axes() -> dict[str, dict[int, str]]:
    """Only the axes that genuinely vary: batch size, history length, candidate width.

    Leaving an axis dynamic that never varies costs shape inference; pinning one that does
    vary produces a model that silently refuses real input.
    """

    axes: dict[str, dict[int, str]] = {}
    for name in HISTORY_CHANNELS:
        axes[name] = {0: BATCH_AXIS, 1: HISTORY_LENGTH_AXIS}
    axes[HISTORY_GAP_INPUT] = {0: BATCH_AXIS, 1: HISTORY_LENGTH_AXIS}
    axes[LENGTHS_INPUT] = {0: BATCH_AXIS}
    for name in QUERY_CHANNELS:
        axes[name] = {0: BATCH_AXIS}
    axes[CANDIDATE_IDS_INPUT] = {0: BATCH_AXIS, 1: CANDIDATE_WIDTH_AXIS}
    for name in CANDIDATE_CHANNELS:
        axes[name] = {0: BATCH_AXIS, 1: CANDIDATE_WIDTH_AXIS}
    axes[CANDIDATE_RANK_INPUT] = {0: BATCH_AXIS, 1: CANDIDATE_WIDTH_AXIS}
    axes["t1_logits"] = {0: BATCH_AXIS}
    axes["t2_logit"] = {0: BATCH_AXIS}
    axes["t3_scores"] = {0: BATCH_AXIS, 1: CANDIDATE_WIDTH_AXIS}
    return axes


class SessionGRUExportWrapper(nn.Module):
    """Flat tensors in, three raw head outputs out.

    The wrapper rebuilds a `Phase1Batch` and calls the model unchanged. Fields the
    forward pass never reads - the masks and the four target tensors - are filled with
    zeros of the right shape rather than being accepted as graph inputs, so the exported
    signature carries only what actually affects the outputs.
    """

    def __init__(self, model: nn.Module, *, batch_spec: Phase1BatchSpec | None = None) -> None:
        super().__init__()
        self.model = model
        self.batch_spec = batch_spec or getattr(model, "batch_spec", None) or phase1_batch_spec_v1()

    def forward(self, *tensors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if len(tensors) != len(INPUT_NAMES):
            raise ValueError(f"expected {len(INPUT_NAMES)} inputs, got {len(tensors)}")

        cursor = 0
        history_ids = dict(zip(HISTORY_CHANNELS, tensors[cursor : cursor + _HISTORY_COUNT]))
        cursor += _HISTORY_COUNT
        history_gap = tensors[cursor]
        cursor += 1
        lengths = tensors[cursor]
        cursor += 1
        query_ids = dict(zip(QUERY_CHANNELS, tensors[cursor : cursor + _QUERY_COUNT]))
        cursor += _QUERY_COUNT
        candidate_ids = tensors[cursor]
        cursor += 1
        candidate_cat = dict(zip(CANDIDATE_CHANNELS, tensors[cursor : cursor + _CANDIDATE_COUNT]))
        cursor += _CANDIDATE_COUNT
        candidate_rank = tensors[cursor]

        rows = lengths.shape[0]
        width = history_gap.shape[1]
        candidates = candidate_ids.shape[1]
        positions = torch.arange(width, device=lengths.device).unsqueeze(0)

        batch = Phase1Batch(
            history_categorical_ids=history_ids,
            history_continuous_features=history_gap,
            lengths=lengths,
            history_mask=positions < lengths.unsqueeze(1),
            query_categorical_ids=query_ids,
            query_continuous_features=torch.zeros(
                (rows, self.batch_spec.query_continuous_dim), dtype=torch.float32
            ),
            candidate_ids=candidate_ids,
            candidate_categorical_ids=candidate_cat,
            candidate_continuous_features=candidate_rank,
            candidate_mask=torch.ones((rows, candidates), dtype=torch.bool),
            t1_target=torch.zeros(rows, dtype=torch.long),
            t2_target=torch.zeros(rows, dtype=torch.float32),
            t3_gains=torch.zeros((rows, candidates), dtype=torch.float32),
            t1_present=torch.zeros(rows, dtype=torch.bool),
            t2_present=torch.zeros(rows, dtype=torch.bool),
            t3_present=torch.zeros(rows, dtype=torch.bool),
        )
        output = self.model(batch)
        return output.t1_logits, output.t2_logit, output.t3_scores


def deterministic_example_batch(
    spec: Phase1BatchSpec | None = None,
    *,
    rows: int = 4,
    history: int = 6,
    candidates: int = 5,
    seed: int = 13,
    empty_history_row: bool = True,
) -> Phase1Batch:
    """A reproducible batch shaped to the frozen contract, for export and parity.

    Real task examples carry `user_id` and are deliberately not in this repository, so
    parity is measured on synthetic rows. What matters for parity is shape, dtype and the
    edge positions, not realism: the batch deliberately includes a padded tail, an
    out-of-vocabulary-adjacent id at the top of each range, and by default one row with
    no history at all, because that row takes the `ZERO_HIDDEN` path that an export is
    most likely to get wrong.
    """

    spec = spec or phase1_batch_spec_v1()
    generator = torch.Generator().manual_seed(seed)

    lengths = torch.randint(1, history + 1, (rows,), generator=generator, dtype=torch.long)
    if empty_history_row and rows > 0:
        lengths[0] = 0
    positions = torch.arange(history).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)

    history_ids: dict[str, Tensor] = {}
    for channel in spec.history_categorical:
        drawn = torch.randint(
            0, channel.vocab_size, (rows, history), generator=generator, dtype=torch.long
        )
        # Padding positions must carry the channel's declared pad id, not a leftover value.
        history_ids[channel.name] = torch.where(mask, drawn, torch.full_like(drawn, channel.pad_id))

    query_ids = {
        channel.name: torch.randint(
            0, channel.vocab_size, (rows,), generator=generator, dtype=torch.long
        )
        for channel in spec.query_categorical
    }

    candidate_cat = {
        channel.name: torch.randint(
            0, channel.vocab_size, (rows, candidates), generator=generator, dtype=torch.long
        )
        for channel in spec.candidate_categorical
    }

    gap = torch.rand((rows, history, spec.history_continuous_dim), generator=generator)
    gap = torch.where(mask.unsqueeze(-1), gap, torch.zeros_like(gap))

    rank = (
        torch.arange(candidates, dtype=torch.float32).div(candidates).repeat(rows, 1).unsqueeze(-1)
    )

    return Phase1Batch(
        history_categorical_ids=history_ids,
        history_continuous_features=gap.to(torch.float32),
        lengths=lengths,
        history_mask=mask,
        query_categorical_ids=query_ids,
        query_continuous_features=torch.zeros((rows, spec.query_continuous_dim)),
        candidate_ids=torch.randint(
            1,
            spec.candidate_id_vocab_size,
            (rows, candidates),
            generator=generator,
            dtype=torch.long,
        ),
        candidate_categorical_ids=candidate_cat,
        candidate_continuous_features=rank,
        candidate_mask=torch.ones((rows, candidates), dtype=torch.bool),
        t1_target=torch.zeros(rows, dtype=torch.long),
        t2_target=torch.zeros(rows, dtype=torch.float32),
        t3_gains=torch.zeros((rows, candidates), dtype=torch.float32),
        t1_present=torch.ones(rows, dtype=torch.bool),
        t2_present=torch.ones(rows, dtype=torch.bool),
        t3_present=torch.ones(rows, dtype=torch.bool),
    )


def batch_to_onnx_inputs(batch: Phase1Batch) -> dict[str, Any]:
    """The exact named arrays ONNX Runtime expects, taken from a real batch."""

    ordered = _ordered_tensors(batch)
    return {name: tensor.detach().cpu().numpy() for name, tensor in zip(INPUT_NAMES, ordered)}


def _ordered_tensors(batch: Phase1Batch) -> tuple[Tensor, ...]:
    return (
        *(batch.history_categorical_ids[name] for name in HISTORY_CHANNELS),
        batch.history_continuous_features,
        batch.lengths,
        *(batch.query_categorical_ids[name] for name in QUERY_CHANNELS),
        batch.candidate_ids,
        *(batch.candidate_categorical_ids[name] for name in CANDIDATE_CHANNELS),
        batch.candidate_continuous_features,
    )


def export_session_gru(
    model: nn.Module,
    destination: Path | str,
    *,
    example_batch: Phase1Batch,
    batch_spec: Phase1BatchSpec | None = None,
) -> Path:
    """Write the model to ONNX in eval mode and return the path.

    Eval mode is not a detail: dropout is active in this model, so exporting a training-
    mode module would produce a graph whose outputs are random.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    wrapper = SessionGRUExportWrapper(model, batch_spec=batch_spec)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            _ordered_tensors(example_batch),
            str(destination),
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=_dynamic_axes(),
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            dynamo=False,
        )
    return destination


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Worst disagreement per head between PyTorch and ONNX Runtime."""

    max_absolute_difference: dict[str, float]
    max_relative_difference: dict[str, float]
    tolerance: float

    @property
    def worst_absolute(self) -> float:
        return max(self.max_absolute_difference.values(), default=0.0)

    @property
    def within_tolerance(self) -> bool:
        return self.worst_absolute <= self.tolerance

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "OnnxParityReport",
            "version": 1,
            "task_id": "S2-SE-01",
            "opset_version": OPSET_VERSION,
            "tolerance": self.tolerance,
            "max_absolute_difference": dict(self.max_absolute_difference),
            "max_relative_difference": dict(self.max_relative_difference),
            "within_tolerance": self.within_tolerance,
        }


def compare_against_onnx(
    model: nn.Module,
    onnx_path: Path | str,
    batches: Sequence[Phase1Batch] | Phase1Batch,
    *,
    tolerance: float = 1e-4,
) -> ParityReport:
    """Run identical inputs through PyTorch and ONNX Runtime and report the worst gap.

    Every batch is fed to both runtimes; the report keeps the worst disagreement seen on
    any of them, so one well-behaved batch cannot hide a badly-behaved one.
    """

    import onnxruntime  # imported here so the module stays importable without a runtime

    if isinstance(batches, Phase1Batch):
        batches = [batches]
    if not batches:
        raise ValueError("at least one batch is required to measure parity")

    model = model.eval()
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    _require_signature(session)

    absolute = dict.fromkeys(OUTPUT_NAMES, 0.0)
    relative = dict.fromkeys(OUTPUT_NAMES, 0.0)

    for batch in batches:
        with torch.no_grad():
            reference = model(batch)
        expected = {
            "t1_logits": reference.t1_logits,
            "t2_logit": reference.t2_logit,
            "t3_scores": reference.t3_scores,
        }
        produced = session.run(list(OUTPUT_NAMES), batch_to_onnx_inputs(batch))

        for name, array in zip(OUTPUT_NAMES, produced):
            want = expected[name].detach().cpu()
            got = torch.from_numpy(array)
            if want.shape != got.shape:
                raise ValueError(
                    f"{name} shape disagrees: PyTorch {tuple(want.shape)} "
                    f"vs ONNX {tuple(got.shape)}"
                )
            gap = (want - got).abs()
            absolute[name] = max(absolute[name], float(gap.max()))
            scale = want.abs().clamp(min=1e-12)
            relative[name] = max(relative[name], float((gap / scale).max()))

    return ParityReport(
        max_absolute_difference=absolute,
        max_relative_difference=relative,
        tolerance=tolerance,
    )


def _require_signature(session: Any) -> None:
    """Fail loudly if the graph's names drifted, rather than on a confusing shape error."""

    produced_inputs = tuple(item.name for item in session.get_inputs())
    produced_outputs = tuple(item.name for item in session.get_outputs())
    if produced_inputs != INPUT_NAMES:
        raise ValueError(f"exported input names drifted: {produced_inputs}")
    if produced_outputs != OUTPUT_NAMES:
        raise ValueError(f"exported output names drifted: {produced_outputs}")


def serialized_size_bytes(onnx_path: Path | str) -> int:
    """Serialized bytes on disk, the figure a deployment decision actually uses."""

    return Path(onnx_path).stat().st_size


def check_graph(onnx_path: Path | str) -> Mapping[str, Any]:
    """Validate the graph with the ONNX checker and report its identity."""

    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    return {
        "ir_version": model.ir_version,
        "opset_imports": {entry.domain or "ai.onnx": entry.version for entry in model.opset_import},
        "producer_name": model.producer_name,
        "graph_inputs": [item.name for item in model.graph.input],
        "graph_outputs": [item.name for item in model.graph.output],
    }
