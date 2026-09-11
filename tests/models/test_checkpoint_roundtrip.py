"""A checkpoint is only useful if someone else can rebuild the model from it.

`S2-SE-01` receives a `.pt` file and has to reconstruct the encoder to export it; `S2-PR-09`
has to reload it to reproduce the centralized reference. Both will discover a checkpoint
that cannot be rebuilt at the worst moment - halfway through a task that depends on it,
with no clue why.

So these tests take the saved payload and rebuild **from nothing but its own contents**. No
imported constant, no remembered default, no config from the training script. If the file
does not carry enough to reconstruct the model, that fails here rather than in another
lane's task a week later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from ppsi.data.batching import windows_to_batch
from ppsi.data.sequences import build_windows
from ppsi.models.batch_spec import phase1_batch_spec_v1
from ppsi.models.session_gru import SessionGRU, SessionGRUConfig, build_model
from ppsi.training.outputs import validate_raw_model_output


def save_payload(model: SessionGRU, config: SessionGRUConfig, seed: int, path) -> None:
    """The exact shape `finalize.py` writes."""
    torch.save(
        {
            "seed": seed,
            "config": {
                "channels": list(config.channels),
                "use_gap": config.use_gap,
                "widths": config.widths,
                "hidden": config.hidden,
                "layers": config.layers,
                "dropout": config.dropout,
            },
            "state_dict": model.state_dict(),
        },
        path,
    )


def rebuild_from_payload(payload: dict) -> SessionGRU:
    """Reconstruct using only what the file contains, the way another lane would."""
    stored = payload["config"]
    config = SessionGRUConfig(
        channels=tuple(stored["channels"]),
        use_gap=stored["use_gap"],
        widths=dict(stored["widths"]),
        hidden=stored["hidden"],
        layers=stored["layers"],
        dropout=stored["dropout"],
    )
    model = SessionGRU(batch_spec=phase1_batch_spec_v1(), config=config)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


@pytest.fixture
def batch():
    frame = pd.DataFrame(
        {
            "session": [1, 1, 1, 2, 2],
            "order": [0, 1, 2, 3, 4],
            "user": [1, 1, 1, 2, 2],
            "category": np.array([3, 11, 42, 7, 7], dtype="int32"),
            "product_bucket": np.array([31, 77, 88, 12, 19], dtype="int32"),
            "event_code": np.array([1, 2, 3, 1, 1], dtype="int8"),
            "brand_bucket": np.array([4, 6, 6, 2, 2], dtype="int32"),
            "price_band": np.array([1, 3, 2, 4, 0], dtype="int8"),
            "event_time": pd.to_datetime(
                [
                    "2019-10-01 00:00:00",
                    "2019-10-01 00:00:10",
                    "2019-10-01 00:02:00",
                    "2019-10-02 08:00:00",
                    "2019-10-02 08:00:30",
                ],
                utc=True,
            ),
        }
    )
    windows = build_windows(frame, np.array([1, 2, 4]), np.array([42, 7, 7]), history_length=4)
    return windows_to_batch(windows, np.arange(3), phase1_batch_spec_v1())


# The selected S2-DS-01 configuration: two channels, one layer, dropout 0.3.
SELECTED = SessionGRUConfig(
    channels=("category_id", "event_type_id"), use_gap=True, dropout=0.3
)


def test_a_rebuilt_model_produces_identical_outputs(tmp_path, batch):
    """The point of the whole exercise: same file, same numbers."""
    original = build_model(13, config=SELECTED).eval()
    path = tmp_path / "checkpoint.pt"
    save_payload(original, SELECTED, 13, path)

    rebuilt = rebuild_from_payload(torch.load(path, weights_only=False))
    with torch.no_grad():
        before, after = original(batch), rebuilt(batch)

    for name in ("t1_logits", "t2_logit", "t3_scores"):
        difference = (getattr(before, name) - getattr(after, name)).abs().max().item()
        assert difference == 0.0, f"{name} differs by {difference:.3e} after a round trip"


def test_loading_is_strict_so_a_silently_partial_load_cannot_happen(tmp_path):
    """`strict=True` is the check. Without it a renamed layer loads as random weights."""
    model = build_model(13, config=SELECTED)
    path = tmp_path / "checkpoint.pt"
    save_payload(model, SELECTED, 13, path)

    payload = torch.load(path, weights_only=False)
    payload["state_dict"].pop("t1_head.bias")
    with pytest.raises(RuntimeError, match="Missing key"):
        rebuild_from_payload(payload)


def test_the_stored_config_alone_rebuilds_the_right_architecture(tmp_path):
    """A file that needs an outside constant to be read is not a handoff."""
    model = build_model(42, config=SELECTED)
    path = tmp_path / "checkpoint.pt"
    save_payload(model, SELECTED, 42, path)

    payload = torch.load(path, weights_only=False)
    rebuilt = rebuild_from_payload(payload)

    assert rebuilt.config.channels == SELECTED.channels
    assert rebuilt.config.hidden == SELECTED.hidden
    assert rebuilt.config.layers == SELECTED.layers
    assert rebuilt.config.dropout == SELECTED.dropout
    assert set(rebuilt.state_dict()) == set(payload["state_dict"])


def test_a_rebuilt_model_still_satisfies_the_output_contract(tmp_path, batch):
    model = build_model(2026, config=SELECTED)
    path = tmp_path / "checkpoint.pt"
    save_payload(model, SELECTED, 2026, path)

    rebuilt = rebuild_from_payload(torch.load(path, weights_only=False))
    with torch.no_grad():
        output = rebuilt(batch)
    validate_raw_model_output(output, batch, category_count=rebuilt.category_count)


def test_a_checkpoint_from_a_different_configuration_refuses_to_load(tmp_path):
    """Mixing a two-channel checkpoint into a five-channel model must not half-succeed."""
    narrow = build_model(13, config=SELECTED)
    path = tmp_path / "checkpoint.pt"
    save_payload(narrow, SELECTED, 13, path)

    payload = torch.load(path, weights_only=False)
    payload["config"]["channels"] = list(SessionGRUConfig().channels)
    with pytest.raises(RuntimeError):
        rebuild_from_payload(payload)
