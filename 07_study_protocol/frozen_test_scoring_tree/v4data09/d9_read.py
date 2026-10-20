"""v4data09.d9_read (owner: Data engineer, DATA-09) - the ONLY function in PPSI V4 that scans the frozen ledger / feature
partitions with split IN (TRAIN, VALIDATION_SELECT, VALIDATION_CONFIRM, TEST).

read_with_test(path, columns, gate) raises TestGateRefused unless `gate` is a GateRecord that still verifies
(d9_gate.revalidate: the lock file and the binding MANIFEST_BUILD register entry are unchanged). There is no default
gate, no module-level gate and no cache of one: every call re-checks. Everything else in V4 keeps reading through the
locked v4data.manifests.read_non_test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads

from v4data.common import NON_TEST_SPLITS, TestGateError

from .d9_gate import revalidate

ALL_SPLITS = tuple(NON_TEST_SPLITS) + ("TEST",)


def read_with_test(path: Path, columns: Sequence[str], gate) -> pa.Table:
    revalidate(gate)                                     # raises without a valid, unchanged GateRecord
    cols = list(columns) if "split" in columns else list(columns) + ["split"]
    t = pads.dataset(str(path), format="parquet", partitioning=None).to_table(
        columns=cols, filter=pc.field("split").isin(list(ALL_SPLITS)))
    seen = set(pc.unique(t["split"]).to_pylist())
    if not seen <= set(ALL_SPLITS):
        raise TestGateError(f"scan of {path} returned non-permitted split values {sorted(seen - set(ALL_SPLITS))}")
    return t


__all__ = ["read_with_test", "ALL_SPLITS"]
