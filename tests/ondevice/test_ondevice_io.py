"""Reading a checkpoint (a {"model", "meta"} record or a bare state dict) and building its scoring module."""
from __future__ import annotations

import torch
from ondevice_testkit import K_SMALL, assert_raises, nc, save_state_dict, tiny_scoring

from ppsi.ondevice.io import CheckpointIOError, build_scoring_module, load_state_dict


def test_fresh_init_has_no_checkpoint_meta():
    made = tiny_scoring("GRU", k=K_SMALL)
    assert made["checkpoint_meta"] == {}
    assert made["state_dict_keys"] > 0


def test_build_scoring_module_loads_bare_state_dict(tmp_path):
    src = tiny_scoring("GRU", seed=11, k=K_SMALL)
    ckpt = tmp_path / "weights.pt"
    save_state_dict(src["built"].module, ckpt)

    other_init = tiny_scoring("GRU", seed=22, k=K_SMALL)
    src_sd = src["built"].module.state_dict()
    other_sd = other_init["built"].module.state_dict()
    assert any(not torch.equal(other_sd[n], src_sd[n]) for n in src_sd), \
        "seeds 11 and 22 produced a bitwise-identical init (this setup would not exercise the load path)"

    loaded = build_scoring_module("GRU", ckpt_path=ckpt, seed=22, k=K_SMALL)
    loaded_sd = loaded["built"].module.state_dict()
    for name, p in src_sd.items():
        assert torch.equal(loaded_sd[name], p), name
    assert loaded["checkpoint_meta"] == {}


def test_build_scoring_module_loads_a_checkpoint_record(tmp_path):
    src = tiny_scoring("SASREC", seed=5, k=K_SMALL)
    ckpt = tmp_path / "record.pt"
    meta = {"config_sha256": "abc", "step": 12}
    torch.save({"kind": "endpoint_6.0", "meta": meta, "model": dict(src["built"].module.state_dict())}, str(ckpt))
    state, got = load_state_dict(ckpt)
    assert got == meta
    assert set(state) == set(src["built"].module.state_dict())

    loaded = build_scoring_module("SASREC", ckpt_path=ckpt, seed=999, k=K_SMALL)
    assert loaded["checkpoint_meta"] == meta
    for name, p in src["built"].module.state_dict().items():
        assert torch.equal(loaded["built"].module.state_dict()[name], p), name


def test_load_state_dict_refuses_non_mapping(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save([1, 2, 3], str(p))
    assert_raises(CheckpointIOError, load_state_dict, p)


def test_build_scoring_module_refuses_mismatched_state_dict(tmp_path):
    src = tiny_scoring("GRU", seed=1, k=K_SMALL)
    bad = dict(src["built"].module.state_dict())
    bad.pop(next(iter(bad)))                     # drop one required key
    p = tmp_path / "incomplete.pt"
    torch.save(bad, str(p))
    assert_raises(CheckpointIOError, build_scoring_module, "GRU", ckpt_path=p, seed=1, k=K_SMALL)


# --------------------------------------------------------------------------------------------- negative controls
@nc("a list is not a checkpoint record or a state dict; load_state_dict must refuse it, not accept it")
def test_nc_load_state_dict_would_accept_a_list(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save([1, 2, 3], str(p))
    try:
        load_state_dict(p)
        refused = False
    except CheckpointIOError:
        refused = True
    assert not refused, "load_state_dict correctly refused the non-mapping checkpoint (this NC claims the opposite)"
