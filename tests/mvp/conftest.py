"""A tiny synthetic prepared bundle, shaped exactly like the real one.

These tests must exercise the MVP integration without the private dataset, so the fixture
writes the same file names, dtypes and manifest keys that ``scripts/mvp/prepare_mvp.py``
writes. If the real preparation changes shape, these tests fail rather than passing
against a format nothing produces.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ppsi.federated.mvp_data import MANIFEST_NAME, WINDOW_FIELDS

HISTORY = 20
CATEGORIES = 588


def _window_arrays(rows: int, *, rng: np.random.Generator) -> dict[str, np.ndarray]:
    lengths = rng.integers(1, HISTORY + 1, size=rows).astype("int64")
    category = np.full((rows, HISTORY), CATEGORIES, dtype="int32")  # pad id
    for row, length in enumerate(lengths):
        category[row, :length] = rng.integers(0, CATEGORIES, size=length)
    keep = np.arange(HISTORY)[None, :] < lengths[:, None]
    return {
        "category": category,
        "product": np.where(keep, rng.integers(1, 500, size=(rows, HISTORY)), 0).astype("int32"),
        "event": np.where(keep, rng.integers(1, 4, size=(rows, HISTORY)), 0).astype("int8"),
        "brand": np.where(keep, rng.integers(1, 200, size=(rows, HISTORY)), 0).astype("int32"),
        "price_band": np.where(keep, rng.integers(1, 5, size=(rows, HISTORY)), 0).astype("int8"),
        "gap": np.where(keep, rng.random((rows, HISTORY)).astype("float32"), 0.0).astype("float32"),
        "lengths": lengths,
        "target": rng.integers(0, CATEGORIES, size=rows).astype("int64"),
        "query_category": rng.integers(0, CATEGORIES, size=rows).astype("int32"),
        "query_product": rng.integers(1, 500, size=rows).astype("int32"),
        "query_brand": rng.integers(1, 200, size=rows).astype("int32"),
        "query_price_band": rng.integers(1, 5, size=rows).astype("int8"),
        "client": rng.integers(1, 1000, size=rows).astype("int64"),
    }


@pytest.fixture
def prepared_dir(tmp_path: Path) -> Path:
    """A four-client, two-round prepared bundle with the real file layout."""
    rng = np.random.default_rng(13)
    directory = tmp_path / "prepared"
    directory.mkdir()

    clients = [f"client-v1-{hashlib.sha256(str(i).encode()).hexdigest()}" for i in range(4)]
    train_rows, validation_rows = 12, 10
    client_rows = {client: list(range(i * 3, i * 3 + 3)) for i, client in enumerate(clients)}

    manifest_splits = {}
    for split, rows in (("TRAIN", train_rows), ("VALIDATION", validation_rows)):
        arrays = _window_arrays(rows, rng=rng)
        digests = {}
        for name in WINDOW_FIELDS:
            path = directory / f"{split.lower()}_{name}.npy"
            np.save(path, arrays[name], allow_pickle=False)
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_splits[split] = {"rows": rows, "array_sha256": digests}
        if split == "VALIDATION":
            np.save(
                directory / "validation_current_category.npy",
                rng.integers(0, CATEGORIES, size=rows).astype("int32"),
                allow_pickle=False,
            )
            np.save(
                directory / "validation_category_changed.npy",
                np.ones(rows, dtype=bool),
                allow_pickle=False,
            )
            row_clients = [clients[i % len(clients)] for i in range(rows)]
            (directory / "validation_clients.json").write_text(
                json.dumps(
                    {
                        "client_ids": row_clients,
                        "train_history_counts": {c: 40 + i for i, c in enumerate(clients)},
                        "history_count_basis": "TRUE_RAW_TRAIN_EVENT_ROWS_PER_CLIENT",
                    }
                ),
                encoding="utf-8",
            )

    (directory / "pilot_client_rows.json").write_text(
        json.dumps({"population": clients, "client_rows": client_rows}), encoding="utf-8"
    )
    (directory / "round_schedule.json").write_text(
        json.dumps(
            {
                "sampler_version": "client_sampler_v1",
                "experiment_seed": 13,
                "round_index_basis": "ZERO_BASED_round_index = server_round - 1",
                "rounds": [
                    {
                        "server_round": 1,
                        "round_index": 0,
                        "selected_client_ids": clients[:2],
                        "selected_digest": "a" * 64,
                    },
                    {
                        "server_round": 2,
                        "round_index": 1,
                        "selected_client_ids": clients[2:],
                        "selected_digest": "b" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (directory / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema": "mvp_prepare_manifest_v1",
                "version": "1",
                "run": "mvp-t1-001",
                "history_length": HISTORY,
                "splits": manifest_splits,
                "data_manifest_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def policy() -> dict:
    """The real execution policy, with a small model so tests stay fast."""
    root = Path(__file__).resolve().parents[2]
    loaded = json.loads((root / "config/mvp/execution.v1.json").read_text(encoding="utf-8"))
    loaded["model"] = dict(loaded["model"], hidden=8)
    loaded["pilot"] = dict(loaded["pilot"], rounds=2, clients_per_round=2, batch_size=2)
    return loaded
