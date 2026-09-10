"""Tests for ppsi.federated.task_examples (S1-PR-07).

All tests use toy temporary parquets. No private REES46 data is accessed.
Never skip/xfail to hide defects.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
import pytest
import torch

from ppsi.federated.clients import client_id_from_user
from ppsi.federated.task_examples import (
    _REQUIRED_COLUMNS,
    _build_client_id_map,
    _row_to_query_features,
    default_batch_spec,
    load_t1_category_spec,
    load_t1_client_counts,
    make_t1_smoke_batches,
    prepare_t1_smoke_slices,
    validate_and_convert_t1_labels,
    verify_t1_task_example_file,
)
from ppsi.training.batch import validate_canonical_phase1_batch

# ---------------------------------------------------------------------------
# Helpers — build toy parquets
# ---------------------------------------------------------------------------


def _make_toy_train_df(
    clients: list[int] | None = None,
    n_per_client: int = 5,
    *,
    split: str = "TRAIN",
    cohort: str = "C1",
    task_mask: bool = True,
    status: str = "OBSERVED",
    label_value: str | None = None,
) -> pl.DataFrame:
    if clients is None:
        clients = [1001, 1002]
    rows = []
    for c in clients:
        for i in range(n_per_client):
            rows.append(
                {
                    "client": c,
                    "session": f"sess_{c}_{i // 3}",
                    "decision_order": i,
                    "label_value": label_value if label_value is not None else str(c % 10 + 1),
                    "task_mask": task_mask,
                    "status": status,
                    "cohort": cohort,
                    "split": split,
                }
            )
    return pl.DataFrame(rows)


def _write_toy_train(tmp_path: Path, **kwargs: object) -> Path:
    df = _make_toy_train_df(**kwargs)
    p = tmp_path / "train.parquet"
    df.write_parquet(p)
    return p


def _make_vocab(tmp_path: Path, category_ids: list[str] | None = None) -> Path:
    if category_ids is None:
        category_ids = [str(i) for i in range(1, 11)]
    code_of = {cat: idx for idx, cat in enumerate(category_ids)}
    vocab = {"categories": {"code_of_category_id": code_of}}
    p = tmp_path / "vocabulary.json"
    p.write_text(json.dumps(vocab), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# required columns
# ---------------------------------------------------------------------------


def test_required_columns_present() -> None:
    assert "client" in _REQUIRED_COLUMNS
    assert "label_value" in _REQUIRED_COLUMNS
    assert "split" in _REQUIRED_COLUMNS
    assert "task_mask" in _REQUIRED_COLUMNS
    assert "status" in _REQUIRED_COLUMNS
    assert "cohort" in _REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# verify_t1_task_example_file
# ---------------------------------------------------------------------------


def test_verify_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_t1_task_example_file(
            tmp_path / "nonexistent.parquet",
            expected_sha_prefix="0" * 16,
            expected_row_count=0,
            expected_split="TRAIN",
            file_label="test",
        )


def test_verify_wrong_sha_prefix(tmp_path: Path) -> None:
    p = _write_toy_train(tmp_path)
    with pytest.raises(ValueError, match="prefix mismatch"):
        verify_t1_task_example_file(
            p,
            expected_sha_prefix="0000000000000000",
            expected_row_count=10,
            expected_split="TRAIN",
            file_label="test",
        )


def test_verify_wrong_row_count(tmp_path: Path) -> None:
    import hashlib

    p = _write_toy_train(tmp_path)
    real_digest = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="row count mismatch"):
        verify_t1_task_example_file(
            p,
            expected_sha_prefix=real_digest[:16],
            expected_row_count=999,
            expected_split="TRAIN",
            file_label="test",
        )


def test_verify_wrong_split(tmp_path: Path) -> None:
    import hashlib

    df = _make_toy_train_df(split="VALIDATION")
    p = tmp_path / "val.parquet"
    df.write_parquet(p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="split != 'TRAIN'"):
        verify_t1_task_example_file(
            p,
            expected_sha_prefix=digest[:16],
            expected_row_count=len(df),
            expected_split="TRAIN",
            file_label="test",
        )


def test_verify_masked_row_rejected(tmp_path: Path) -> None:
    import hashlib

    df = _make_toy_train_df(task_mask=False)
    p = tmp_path / "bad.parquet"
    df.write_parquet(p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="task_mask"):
        verify_t1_task_example_file(
            p,
            expected_sha_prefix=digest[:16],
            expected_row_count=len(df),
            expected_split="TRAIN",
            file_label="test",
        )


@pytest.mark.parametrize(
    "col_name",
    [
        "client",
        "session",
        "decision_order",
        "label_value",
        "task_mask",
        "status",
        "cohort",
        "split",
    ],
)
def test_verify_null_column_rejected(tmp_path: Path, col_name: str) -> None:
    import hashlib

    rows = _make_toy_train_df().to_dicts()
    rows[0][col_name] = None
    df = pl.DataFrame(rows)
    p = tmp_path / f"null_{col_name}.parquet"
    df.write_parquet(p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=f"null {col_name}"):
        verify_t1_task_example_file(
            p,
            expected_sha_prefix=digest[:16],
            expected_row_count=len(df),
            expected_split="TRAIN",
            file_label="test",
        )


def test_validate_and_convert_t1_labels_valid_codes() -> None:
    df = pl.DataFrame({"label_value": [0.0, 60.0, 587.0]})
    converted = validate_and_convert_t1_labels(df, category_count=588)
    assert "label_code" in converted.columns
    assert converted["label_code"].to_list() == [0, 60, 587]


@pytest.mark.parametrize(
    ("bad_val", "expected_match"),
    [
        (-1.0, "out of range"),
        (588.0, "out of range"),
        (60.5, "non-integral"),
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
        (None, "null label_value"),
    ],
)
def test_validate_and_convert_t1_labels_rejections(bad_val: object, expected_match: str) -> None:
    df = pl.DataFrame({"label_value": [bad_val]})
    with pytest.raises(ValueError, match=expected_match):
        validate_and_convert_t1_labels(df, category_count=588)


def test_load_t1_category_spec_valid(tmp_path: Path) -> None:
    vocab = {
        "categories": {
            "count": 588,
            "code_of_category_id": {f"cat_{i}": i for i in range(588)},
        }
    }
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(vocab), encoding="utf-8")
    count, codes = load_t1_category_spec(p)
    assert count == 588
    assert codes == set(range(588))


def test_load_t1_category_spec_wrong_count(tmp_path: Path) -> None:
    vocab = {
        "categories": {
            "count": 10,
            "code_of_category_id": {f"cat_{i}": i for i in range(10)},
        }
    }
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(vocab), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected category_count == 588"):
        load_t1_category_spec(p)


# ---------------------------------------------------------------------------
# #19 identity mapping
# ---------------------------------------------------------------------------


def test_client_id_from_user_consistency() -> None:
    """Same raw client always maps to the same opaque ID."""
    raw = 12345
    cid_a = client_id_from_user(str(raw))
    cid_b = client_id_from_user(str(raw))
    assert cid_a == cid_b
    assert cid_a.startswith("client-v1-")


def test_build_client_id_map_unique() -> None:
    df = _make_toy_train_df(clients=[10, 20, 30])
    cmap = _build_client_id_map(df)
    assert len(cmap) == 3
    assert all(v.startswith("client-v1-") for v in cmap.values())


# ---------------------------------------------------------------------------
# load_t1_client_counts
# ---------------------------------------------------------------------------


def test_load_t1_client_counts_basic(tmp_path: Path) -> None:
    clients = [1001, 1002, 1003]
    train_df = _make_toy_train_df(clients=clients[:2], n_per_client=4)
    val_df = _make_toy_train_df(clients=clients[1:], n_per_client=2, split="VALIDATION")
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    train_df.write_parquet(train_p)
    val_df.write_parquet(val_p)

    all_client_ids = {client_id_from_user(str(c)) for c in clients}
    result = load_t1_client_counts(train_p, val_p, all_client_ids)

    assert "client_id" in result.columns
    assert "t1_train_example_count" in result.columns
    assert "t1_validation_example_count" in result.columns
    assert "eligible_for_t1_smoke" in result.columns
    # client 1001: train=4, val=0 → eligible
    cid1001 = client_id_from_user("1001")
    row = result.filter(pl.col("client_id") == cid1001)
    assert len(row) == 1
    assert int(row["t1_train_example_count"][0]) == 4
    assert int(row["t1_validation_example_count"][0]) == 0
    assert bool(row["eligible_for_t1_smoke"][0]) is True


def test_client_outside_base_manifest_fails(tmp_path: Path) -> None:
    train_df = _make_toy_train_df(clients=[9999])
    # val_df has the same client (non-empty) to avoid schema inference issues
    val_df = _make_toy_train_df(clients=[9999], split="VALIDATION", n_per_client=1)
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    train_df.write_parquet(train_p)
    val_df.write_parquet(val_p)

    # Base manifest contains a different client ID → should fail
    different_set = {client_id_from_user("0")}
    with pytest.raises(ValueError, match="outside the base C1 manifest"):
        load_t1_client_counts(train_p, val_p, different_set)


# ---------------------------------------------------------------------------
# prepare_t1_smoke_slices — 32/client cap and stable order
# ---------------------------------------------------------------------------


def test_smoke_slices_cap_32(tmp_path: Path) -> None:
    clients = [100, 101]
    train_df = _make_toy_train_df(clients=clients, n_per_client=50)
    val_df = _make_toy_train_df(clients=clients, n_per_client=5, split="VALIDATION")
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    train_df.write_parquet(train_p)
    val_df.write_parquet(val_p)

    selected = [client_id_from_user(str(c)) for c in clients]

    train_out = tmp_path / "train_slice.parquet"
    val_out = tmp_path / "val_slice.parquet"
    prepare_t1_smoke_slices(
        train_p,
        val_p,
        selected,
        category_count=20,
        valid_codes=set(range(20)),
        max_train_examples_per_client=32,
        max_validation_examples=256,
        train_output_path=train_out,
        validation_output_path=val_out,
    )

    out_df = pl.read_parquet(train_out)
    # Each client should have at most 32 rows
    for cid in selected:
        client_rows = out_df.filter(pl.col("client_id") == cid)
        assert len(client_rows) <= 32


def test_smoke_slices_omits_raw_client(tmp_path: Path) -> None:
    train_df = _make_toy_train_df(clients=[200])
    val_df = _make_toy_train_df(clients=[200], split="VALIDATION")
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    train_df.write_parquet(train_p)
    val_df.write_parquet(val_p)

    selected = [client_id_from_user("200")]

    train_out = tmp_path / "train_slice.parquet"
    val_out = tmp_path / "val_slice.parquet"
    prepare_t1_smoke_slices(
        train_p,
        val_p,
        selected,
        category_count=20,
        valid_codes=set(range(20)),
        train_output_path=train_out,
        validation_output_path=val_out,
    )

    out_df = pl.read_parquet(train_out)
    assert "client" not in out_df.columns
    assert "label_value" not in out_df.columns
    assert "client_id" in out_df.columns
    assert "label_code" in out_df.columns


def test_smoke_slices_stable_order(tmp_path: Path) -> None:
    """Two runs produce identical slices."""
    train_df = _make_toy_train_df(clients=[300, 301], n_per_client=10)
    val_df = _make_toy_train_df(clients=[300, 301], split="VALIDATION", n_per_client=3)
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    train_df.write_parquet(train_p)
    val_df.write_parquet(val_p)

    selected = [client_id_from_user(str(c)) for c in [300, 301]]

    train_out_a = tmp_path / "train_a.parquet"
    val_out_a = tmp_path / "val_a.parquet"
    train_out_b = tmp_path / "train_b.parquet"
    val_out_b = tmp_path / "val_b.parquet"

    sha_a = prepare_t1_smoke_slices(
        train_p,
        val_p,
        selected,
        category_count=20,
        valid_codes=set(range(20)),
        train_output_path=train_out_a,
        validation_output_path=val_out_a,
    )
    sha_b = prepare_t1_smoke_slices(
        train_p,
        val_p,
        selected,
        category_count=20,
        valid_codes=set(range(20)),
        train_output_path=train_out_b,
        validation_output_path=val_out_b,
    )
    assert sha_a == sha_b


def test_out_of_bounds_label_fails(tmp_path: Path) -> None:
    # label_value "99" exceeds category_count 20
    train_df = _make_toy_train_df(clients=[400], label_value="99")
    val_df = _make_toy_train_df(clients=[400], split="VALIDATION", label_value="99")
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    train_df.write_parquet(train_p)
    val_df.write_parquet(val_p)

    selected = [client_id_from_user("400")]

    train_out = tmp_path / "train_slice.parquet"
    val_out = tmp_path / "val_slice.parquet"
    with pytest.raises(ValueError, match="out of range"):
        prepare_t1_smoke_slices(
            train_p,
            val_p,
            selected,
            category_count=20,
            valid_codes=set(range(20)),
            train_output_path=train_out,
            validation_output_path=val_out,
        )


# ---------------------------------------------------------------------------
# Phase1Batch construction
# ---------------------------------------------------------------------------


def test_make_batches_canonical_validation() -> None:
    """make_t1_smoke_batches produces batches that pass canonical validation."""
    df = pl.DataFrame(
        {
            "client_id": ["client-v1-abc"] * 12,
            "session": ["s1"] * 12,
            "decision_order": list(range(12)),
            "label_code": [i % 5 for i in range(12)],
        }
    )
    spec = default_batch_spec()
    batches = make_t1_smoke_batches(df, spec=spec, batch_size=8)
    assert len(batches) == 2  # 12 rows → 2 batches (8 + 4)
    for batch in batches:
        validate_canonical_phase1_batch(batch, spec)


def test_make_batches_zero_semantic_history() -> None:
    """All batches have semantic length 0 (zero-history)."""
    df = pl.DataFrame(
        {
            "client_id": ["c1"] * 4,
            "session": ["s"] * 4,
            "decision_order": [0, 1, 2, 3],
            "label_code": [0, 1, 2, 0],
        }
    )
    batches = make_t1_smoke_batches(df, batch_size=4)
    for batch in batches:
        assert torch.all(batch.lengths == 0), "semantic history must be 0"
        assert not batch.history_mask.any(), "no history positions should be active"


def test_make_batches_t1_only_masks() -> None:
    """T1 present=True, T2/T3 present=False for all rows."""
    df = pl.DataFrame(
        {
            "client_id": ["c1"] * 3,
            "session": ["s"] * 3,
            "decision_order": [0, 1, 2],
            "label_code": [0, 1, 2],
        }
    )
    batches = make_t1_smoke_batches(df, batch_size=3)
    assert len(batches) == 1
    b = batches[0]
    assert b.t1_present.all()
    assert not b.t2_present.any()
    assert not b.t3_present.any()


def test_row_to_query_features_formula() -> None:
    """Verify exact formula for query features."""
    for do in [0, 1, 5, 7, 100, 1000, 2000]:
        qcid, f0, f1 = _row_to_query_features(do)
        assert qcid == 1 + (max(do, 0) % 7)
        assert abs(f0 - math.log1p(max(do, 0))) < 1e-6
        expected_f1 = min(max(do, 0), 1000) / 1000.0
        assert abs(f1 - expected_f1) < 1e-6


def test_make_batches_empty_df() -> None:
    """Empty slice produces no batches."""
    df = pl.DataFrame(
        {"client_id": [], "session": [], "decision_order": [], "label_code": []},
        schema={
            "client_id": pl.String,
            "session": pl.String,
            "decision_order": pl.Int64,
            "label_code": pl.Int64,
        },
    )
    batches = make_t1_smoke_batches(df)
    assert batches == []


def test_deterministic_batch_counts() -> None:
    """Batch count is deterministic for a given slice size and batch_size."""
    df = pl.DataFrame(
        {
            "client_id": ["c"] * 20,
            "session": ["s"] * 20,
            "decision_order": list(range(20)),
            "label_code": [i % 3 for i in range(20)],
        }
    )
    batches = make_t1_smoke_batches(df, batch_size=8)
    # 20 rows → ceil(20/8) = 3 batches: 8 + 8 + 4
    assert len(batches) == 3
    assert batches[0].batch_size == 8
    assert batches[1].batch_size == 8
    assert batches[2].batch_size == 4
