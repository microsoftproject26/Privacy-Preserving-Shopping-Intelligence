"""BASE-02: v4base.identity — real parquet reads against tiny, hand-built "clients"/"decisions"/"users" tables
(the documented MANIFEST_SPEC schema), each verified against a provider's own recorded file_sha256 before any value
is trusted, plus the dense-rank and out-of-range defensive checks and the client_ord vs client_key_hex distinction.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from basekit import nc
from v4base import identity


def _write_parquet(path: Path, columns: dict) -> str:
    pq.write_table(pa.table(columns), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Verified:
    def __init__(self, tables: dict, canonical_rows):
        self.tables = tables
        self.canonical_rows = np.asarray(canonical_rows, dtype=np.int64)


class _Provider:
    def __init__(self, verified: _Verified):
        self.verified = verified


def _train_provider(tmp_path, *, client_ords=(0, 1, 2), hexes=("A" * 64, "B" * 64, "C" * 64),
                    decision_client_ords=(0, 1, 1, 2), canonical_rows=(0, 1, 2, 3)) -> _Provider:
    clients_path = tmp_path / "M.clients.parquet"
    decisions_path = tmp_path / "M.decisions.parquet"
    clients_sha = _write_parquet(clients_path, {"client_ord": list(client_ords), "client_key_hex": list(hexes)})
    decisions_sha = _write_parquet(decisions_path, {"client_ord": list(decision_client_ords)})
    verified = _Verified({"clients": {"file": clients_path.name, "file_sha256": clients_sha},
                          "decisions": {"file": decisions_path.name, "file_sha256": decisions_sha}},
                         canonical_rows)
    return _Provider(verified)


def test_train_client_ord_by_canonical_position(tmp_path):
    p = _train_provider(tmp_path)
    got = identity.train_client_ord_by_canonical_position(p, manifest_dir=tmp_path)
    assert got.tolist() == [0, 1, 1, 2]


def test_train_client_key_hex_by_ord(tmp_path):
    p = _train_provider(tmp_path)
    got = identity.train_client_key_hex_by_ord(p, manifest_dir=tmp_path)
    assert got.tolist() == ["A" * 64, "B" * 64, "C" * 64]


def test_train_client_key_hex_by_canonical_position_composes_both(tmp_path):
    p = _train_provider(tmp_path)
    got = identity.train_client_key_hex_by_canonical_position(p, manifest_dir=tmp_path)
    assert got.tolist() == ["A" * 64, "B" * 64, "B" * 64, "C" * 64]


def test_eval_client_key_hex_by_eval_user_key(tmp_path):
    users_path = tmp_path / "M.users.parquet"
    sha = _write_parquet(users_path, {"eval_user_key": [0, 1], "client_key_hex": ["X" * 64, "Y" * 64]})
    verified = _Verified({"users": {"file": users_path.name, "file_sha256": sha}}, canonical_rows=[])
    got = identity.eval_client_key_hex_by_eval_user_key(_Provider(verified), manifest_dir=tmp_path)
    assert got.tolist() == ["X" * 64, "Y" * 64]


def test_wrong_file_sha256_is_refused(tmp_path):
    p = _train_provider(tmp_path)
    p.verified.tables["clients"]["file_sha256"] = "0" * 64        # deliberately wrong
    with pytest.raises(ValueError):
        identity.train_client_key_hex_by_ord(p, manifest_dir=tmp_path)


def test_non_dense_client_ord_is_refused(tmp_path):
    p = _train_provider(tmp_path, client_ords=(0, 2), hexes=("A" * 64, "C" * 64))   # missing 1: not dense 0..n-1
    with pytest.raises(ValueError):
        identity.train_client_key_hex_by_ord(p, manifest_dir=tmp_path)


def test_out_of_range_client_ord_is_refused(tmp_path):
    p = _train_provider(tmp_path, decision_client_ords=(0, 1, 5, 2))   # 5 is outside the 3-row clients table
    with pytest.raises(ValueError):
        identity.train_client_key_hex_by_canonical_position(p, manifest_dir=tmp_path)


@nc("without the file_sha256 verification, a corrupted/substituted clients table would be silently trusted")
def test_nc_wrong_file_would_be_trusted_without_the_hash_check(tmp_path):
    p = _train_provider(tmp_path)
    p.verified.tables["clients"]["file_sha256"] = "0" * 64
    refused = False
    try:
        identity.train_client_key_hex_by_ord(p, manifest_dir=tmp_path)
    except ValueError:
        refused = True
    assert not refused
