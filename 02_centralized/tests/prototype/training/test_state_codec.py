from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from ppsi.training.fixtures import make_stub_model
from ppsi.training.state import (
    SharedStateError,
    load_shared_state,
    pack_shared_state,
    shared_state_digest,
)


def test_state_pack_load_round_trip():
    source = make_stub_model()
    target = make_stub_model(seed=42)
    spec = source.shared_state_spec()
    packed = pack_shared_state(source, spec)
    load_shared_state(target, packed, spec)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)


def test_digest_is_deterministic():
    model = make_stub_model()
    packed = pack_shared_state(model, model.shared_state_spec())
    assert shared_state_digest(packed) == shared_state_digest(deepcopy(packed))


def test_missing_extra_shape_dtype_and_nan_fail():
    model = make_stub_model()
    spec = model.shared_state_spec()
    packed = pack_shared_state(model, spec)
    key = next(iter(packed))

    missing = dict(packed)
    missing.pop(key)
    with pytest.raises(SharedStateError, match="keys mismatch"):
        load_shared_state(model, missing, spec)

    extra = dict(packed)
    extra["unexpected"] = torch.ones(1)
    with pytest.raises(SharedStateError, match="keys mismatch"):
        load_shared_state(model, extra, spec)

    bad_shape = dict(packed)
    bad_shape[key] = bad_shape[key].reshape(-1)
    if tuple(bad_shape[key].shape) == tuple(packed[key].shape):
        bad_shape[key] = bad_shape[key][:-1]
    with pytest.raises(SharedStateError, match="shape mismatch"):
        load_shared_state(model, bad_shape, spec)

    bad_dtype = dict(packed)
    bad_dtype[key] = bad_dtype[key].double()
    with pytest.raises(SharedStateError, match="dtype mismatch"):
        load_shared_state(model, bad_dtype, spec)

    bad_nan = dict(packed)
    bad_nan[key] = bad_nan[key].clone()
    bad_nan[key].view(-1)[0] = float("nan")
    with pytest.raises(SharedStateError, match="finite floating"):
        load_shared_state(model, bad_nan, spec)


def test_float_buffer_is_explicit_and_nonfloating_buffer_is_rejected():
    from torch import nn

    from ppsi.training.state import SharedStateSpec, StateKind

    class FloatBufferModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.register_buffer("running_value", torch.zeros(2, dtype=torch.float32))

    float_model = FloatBufferModel()
    spec = SharedStateSpec.all_shared_floating(float_model)
    by_key = {entry.key: entry for entry in spec.entries}
    assert by_key["weight"].kind == StateKind.PARAMETER
    assert by_key["running_value"].kind == StateKind.BUFFER

    class IntegerBufferModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    with pytest.raises(SharedStateError, match="non-floating"):
        SharedStateSpec.all_shared_floating(IntegerBufferModel())
