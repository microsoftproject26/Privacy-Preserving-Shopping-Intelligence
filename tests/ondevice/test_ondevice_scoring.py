"""Input-key derivation and the ScoringModule wrapper (pure torch: no onnx / onnxruntime)."""
from __future__ import annotations

import pytest
import torch
from ondevice_testkit import K_SMALL, assert_raises, nc, tiny_scoring

from ppsi.ondevice.scoring import ScoringError, batch_to_inputs, example_batch
from ppsi.seqrec.features import SASREC_KEYS


@pytest.mark.parametrize("family", ["GRU", "SASREC"])
def test_input_keys_start_with_item_tokens(family):
    made = tiny_scoring(family)
    keys = made["scoring"].input_keys
    assert keys[0] == "item_tokens"
    assert len(set(keys)) == len(keys)         # no duplicates


def test_gru_input_keys_exclude_position_ids_and_include_rich_channels():
    made = tiny_scoring("GRU")
    keys = made["scoring"].input_keys
    assert "position_ids" not in keys           # ContextGRU.query never takes position_ids
    assert keys[:3] == ("item_tokens", "lengths", "attention_mask")
    for ch in ("category_tokens", "brand_tokens", "main_category_tokens", "daypart_tokens", "weekday_tokens"):
        assert ch in keys                       # the default build keeps every rich side channel
    assert "event_numeric_features" in keys and "event_quality_flags" in keys
    assert "user_context" in keys and "user_context_masks" in keys


def test_sasrec_input_keys_equal_the_adapter_keys():
    made = tiny_scoring("SASREC")
    assert made["scoring"].input_keys == tuple(SASREC_KEYS)


def test_scoring_module_matches_adapter_logits_of_query_sasrec():
    """SASRec's forward is exactly `adapter.logits(adapter.query(.))` (no reimplementation): bit-identical."""
    made = tiny_scoring("SASREC", seed=2026, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    batch = example_batch(built, n=5, seed=123)
    inputs = batch_to_inputs(scoring, batch)
    with torch.no_grad():
        got = scoring(*inputs)
        want = built.adapter.logits(built.adapter.query(batch))
    assert got.shape == (5, built.K)
    assert torch.equal(got, want)


def test_scoring_module_matches_adapter_logits_of_query_gru_to_tolerance():
    """GRU's forward uses `gru_dense_query` (a pack-free reimplementation ONNX can trace; see its docstring), so it
    is checked to float32 tolerance against the real, packed `ContextGRU.query`, not bit equality."""
    made = tiny_scoring("GRU", seed=2026, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    batch = example_batch(built, n=5, seed=123, min_len=1)          # varying real lengths, not all full-length
    inputs = batch_to_inputs(scoring, batch)
    with torch.no_grad():
        got = scoring(*inputs)
        want = built.adapter.logits(built.adapter.query(batch))
    assert got.shape == (5, built.K)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)     # different kernels (dense vs packed): close, not exact


@pytest.mark.parametrize("family", ["GRU", "SASREC"])
def test_scoring_module_eval_mode_is_deterministic(family):
    made = tiny_scoring(family, seed=7, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    inputs = batch_to_inputs(scoring, example_batch(built, n=4, seed=9))
    scoring.eval()
    with torch.no_grad():
        a = scoring(*inputs)
        b = scoring(*inputs)
    assert torch.equal(a, b)                    # eval mode: dropout off, no randomness left in the forward


def test_batch_to_inputs_refuses_missing_key():
    made = tiny_scoring("GRU", k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    batch = example_batch(built, n=1, seed=1)
    del batch[scoring.input_keys[-1]]
    assert_raises(ScoringError, batch_to_inputs, scoring, batch)


def test_scoring_module_refuses_wrong_input_count():
    made = tiny_scoring("GRU", k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    inputs = batch_to_inputs(scoring, example_batch(built, n=1, seed=1))
    assert_raises(ScoringError, scoring, *inputs[:-1])


# --------------------------------------------------------------------------------------------- negative controls
@nc("GRU never consumes position_ids (ContextGRU.query has no such parameter)")
def test_nc_gru_input_keys_would_include_position_ids():
    made = tiny_scoring("GRU")
    assert "position_ids" in made["scoring"].input_keys


@nc("two differently-seeded builds must NOT score identically (the wrapper must use its own weights)")
def test_nc_scoring_module_ignores_its_own_weights():
    a = tiny_scoring("GRU", seed=1, k=K_SMALL)
    b = tiny_scoring("GRU", seed=2, k=K_SMALL)
    inputs = batch_to_inputs(a["scoring"], example_batch(a["built"], n=3, seed=0))
    with torch.no_grad():
        sa = a["scoring"](*inputs)
        sb = b["scoring"](*inputs)
    assert torch.equal(sa, sb)
