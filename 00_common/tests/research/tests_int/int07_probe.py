"""Subprocess probe for test_int07_test_path.py (not a test module): in a FRESH interpreter, builds the VALIDATION
manifest-scoped store (v4int.cache.build_subset_store, the unchanged default path, USER_HISTORY_L50) of the synthetic
four-partition dataset and prints its bitwise identity (every event array, decision-set field, session id, and the
digest of every served batch) as JSON - either plainly (v4int.retro_store never imported), or after importing
v4int.retro_store and running a GATED RETRO_TEST subset-store build in the same process.

  python -B int07_probe.py <dataset> plain|after_test [<lock> <register>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for _p in (str(CODE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True
import ppsi_env  # noqa: E402,F401

import v4train  # noqa: E402,F401

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def eval_ids(ds: Path, splits) -> np.ndarray:
    out = []
    for s in splits:
        t = pq.read_table(ds / "decisions_index" / f"split={s}" / "user_bucket=0" / "data_0.parquet",
                          columns=["decision_id", "eval_mask"])
        d = np.asarray(t["decision_id"].to_numpy(), dtype=np.int64)
        out.append(d[np.asarray(t["eval_mask"].to_numpy(zero_copy_only=False), dtype=bool)])
    return np.sort(np.concatenate(out))


def main() -> int:
    from int07kit import batch_digest, store_digest
    from v4data.common import VALIDATION_SPLITS
    from v4int.cache import build_subset_store
    from v4int.providers import add_session_offsets
    ds, mode = Path(sys.argv[1]), sys.argv[2]
    q = lambda *_: None  # noqa: E731
    extra = {}
    if mode == "after_test":
        from v4data09.d9_gate import check_test_gate
        from v4int import retro_store as RS
        g = check_test_gate(Path(sys.argv[3]), Path(sys.argv[4]))
        tids = eval_ids(ds, ("TEST",))
        st, tds, info = build_subset_store(tids, splits=RS.TEST_DECISION_SPLITS, kind="eval", state_sha256=None,
                                           history_policy="USER_HISTORY_L50", ds_root=ds, buckets=[0], log=q,
                                           test_gate=g)
        extra = {"test_decisions": int(tds.decision_id.size), "test_store_splits": list(info.store_splits)}
    vids = eval_ids(ds, VALIDATION_SPLITS)
    store, vds, info = build_subset_store(vids, splits=VALIDATION_SPLITS, kind="eval", state_sha256=None,
                                          history_policy="USER_HISTORY_L50", ds_root=ds, buckets=[0], log=q)
    pos = np.arange(vds.decision_id.size, dtype=np.int64)
    b = add_session_offsets(store.batch(vds, pos), store, vds, pos)
    print(json.dumps({"mode": mode, "store": store_digest(store, vds), "batch": batch_digest(b),
                      "store_splits": list(info.store_splits), "n": int(vds.decision_id.size),
                      "retro_store_imported": "v4int.retro_store" in sys.modules, **extra}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
