"""Loading the prepared scoped-T1 MVP windows and the per-client row index.

This module owns no policy. It reads what ``scripts/mvp/prepare_mvp.py`` wrote, checks
that the arrays on disk are the ones the manifest describes, and hands back read-only
memory maps plus the existing :class:`ppsi.data.sequences.Windows` view over them.

Two rules it exists to enforce:

* A prepared directory is immutable. Every array is opened ``mmap_mode='r'`` with
  ``allow_pickle=False``, so a training run can never edit the inputs it is measured on.
* A client's local rows are the complete set of that client's frozen decisions. There is
  no cap, no sampling with replacement, and no silent truncation, so the row index is
  validated to be a partition of the split's rows rather than an arbitrary subset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ppsi.data.sequences import Windows

# The field names of ``Windows``, which are also the ``.npy`` basenames on disk.
WINDOW_FIELDS: tuple[str, ...] = (
    "category",
    "product",
    "event",
    "brand",
    "price_band",
    "gap",
    "lengths",
    "target",
    "query_category",
    "query_product",
    "query_brand",
    "query_price_band",
    "client",
)

MANIFEST_NAME = "prepare_manifest.v1.json"


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    """One split's memory-mapped window arrays and its declared row count."""

    split: str
    arrays: dict[str, np.ndarray]
    rows: int

    def windows(self) -> Windows:
        """The existing project view over these arrays; no data is copied."""
        return Windows(**{name: self.arrays[name] for name in WINDOW_FIELDS})

    def subset_rows(self, rows: np.ndarray) -> np.ndarray:
        """Validate caller-supplied row positions against this split."""
        values = np.asarray(rows)
        if values.ndim != 1 or values.dtype.kind not in "iu" or values.size == 0:
            raise ValueError("row selection must be a nonempty integer vector")
        if values.min() < 0 or values.max() >= self.rows:
            raise ValueError("row selection is outside the prepared split")
        if len(np.unique(values)) != len(values):
            raise ValueError("row selection contains duplicates")
        return values.astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class PreparedMVP:
    """Everything a training stage needs from preparation, already verified."""

    root: Path
    manifest: dict
    train: PreparedSplit
    validation: PreparedSplit
    client_rows: dict[str, np.ndarray]
    population: list[str]
    schedule: list[dict]

    @property
    def run_id(self) -> str:
        return str(self.manifest["run"])

    @property
    def data_manifest_sha256(self) -> str:
        return str(self.manifest["data_manifest_sha256"])

    def rows_for(self, client_id: str) -> np.ndarray:
        """The complete TRAIN row positions of one pilot client."""
        rows = self.client_rows.get(client_id)
        if rows is None:
            raise KeyError("client is not part of the prepared pilot population")
        return rows

    def validation_metadata(self) -> dict:
        """What the frozen evaluator needs beside the ranks, in frozen decision order.

        ``category_changed`` is read here and used only after predictions exist. The
        history counts are TRUE raw TRAIN event counts per client, not task-example
        counts, and a client with no TRAIN events keeps its measured zero rather than
        being dropped from the stratification.
        """
        changed = np.load(
            self.root / "validation_category_changed.npy", mmap_mode="r", allow_pickle=False
        )
        if changed.shape[0] != self.validation.rows:
            raise ValueError("validation evaluation flags do not cover the prepared rows")
        index = json.loads((self.root / "validation_clients.json").read_text(encoding="utf-8"))
        client_ids = list(index["client_ids"])
        if len(client_ids) != self.validation.rows:
            raise ValueError("validation client column does not cover the prepared rows")
        counts = {str(k): int(v) for k, v in index["train_history_counts"].items()}
        if not set(client_ids).issubset(counts):
            raise ValueError("a validation client has no measured TRAIN history count")
        return {
            "category_changed": np.asarray(changed, dtype=bool),
            "client_ids": client_ids,
            "train_history_counts": counts,
            "history_count_basis": index["history_count_basis"],
        }


def _load_arrays(directory: Path, split: str, manifest_split: dict) -> PreparedSplit:
    rows = int(manifest_split["rows"])
    arrays: dict[str, np.ndarray] = {}
    for name in WINDOW_FIELDS:
        path = directory / f"{split.lower()}_{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"prepared array missing: {path.name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape[0] != rows:
            raise ValueError(f"{path.name} holds {array.shape[0]} rows, manifest says {rows}")
        arrays[name] = array
    return PreparedSplit(split=split, arrays=arrays, rows=rows)


def load_prepared(prepared_dir: Path | str) -> PreparedMVP:
    """Open a prepared directory read-only and validate its internal consistency."""
    directory = Path(prepared_dir)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepared manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "mvp_prepare_manifest_v1":
        raise ValueError("unexpected prepared manifest schema")

    train = _load_arrays(directory, "TRAIN", manifest["splits"]["TRAIN"])
    validation = _load_arrays(directory, "VALIDATION", manifest["splits"]["VALIDATION"])

    index = json.loads((directory / "pilot_client_rows.json").read_text(encoding="utf-8"))
    client_rows = {
        client: np.asarray(rows, dtype=np.int64) for client, rows in index["client_rows"].items()
    }
    population = list(index["population"])
    if sorted(client_rows) != sorted(population):
        raise ValueError("row index and pilot population disagree")

    covered = np.concatenate([client_rows[c] for c in population]) if population else np.array([])
    if len(covered) != train.rows or len(np.unique(covered)) != train.rows:
        raise ValueError(
            "the per-client row index must partition every prepared pilot TRAIN row exactly once"
        )

    schedule = json.loads((directory / "round_schedule.json").read_text(encoding="utf-8"))["rounds"]
    for entry in schedule:
        unknown = [c for c in entry["selected_client_ids"] if c not in client_rows]
        if unknown:
            raise ValueError("scheduled client is absent from the prepared population")

    return PreparedMVP(
        root=directory,
        manifest=manifest,
        train=train,
        validation=validation,
        client_rows=client_rows,
        population=population,
        schedule=schedule,
    )
