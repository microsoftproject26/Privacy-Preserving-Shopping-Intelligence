"""Virtual / serialized byte counts, with the tied alias counted once.

Derived from comm.py's documented formulas (report virtual communication bytes even though tensors never leave the
process), hand-computed from TinyRec's own declared shapes in synthetic.py, independently of the main tests.

Hand-computed shapes for make_tiny_adapter(K=5, d=3, tied=True):
  V = K + 3 = 8
  item_embed.weight   (V, d) = (8, 3)  -> 24 elem  (SHARED; canonical key for the tied alias)
  enc.weight          (d, d) = (3, 3)  -> 9 elem   (SHARED)
  enc.bias            (d,)   = (3,)    -> 3 elem   (SHARED)
  output_bias         (K,)   = (5,)    -> 5 elem   (SHARED)
  -> shared_numel = 24+9+3+5 = 41 ; shared_bytes (fp32) = 41*4 = 164
  out_proj.weight     (V, d) = (8, 3)  -> 24 elem, ALIAS of item_embed.weight -> 24*4 = 96 bytes (not sent twice)
  class_map (int64)   (K,)   = (5,)    -> 5*8    = 40 bytes   (BUFFER, FIXED)
  input_scale (fp32)  ()     -> 1 elem -> 1*4    = 4 bytes    (BUFFER, SERVER_COPY)
  -> bootstrap_bytes_once = 40 + 4 = 44
"""
from __future__ import annotations

import pytest
import torch
from fedsim_crosscheck_kit import fresh_adapter

from ppsi.fedsim.adapter import BufferRule, Entry, ParamManifest, Role
from ppsi.fedsim.comm import Ledger, VirtualChannel, payload_bytes, visit_bytes


def test_tied_alias_byte_formulas_hand_computed():
    adapter = fresh_adapter(K=5, d=3, tied=True, seed=0)
    m = adapter.manifest
    assert m.shared_numel == 41, "hand count: item_embed(24)+enc.weight(9)+enc.bias(3)+output_bias(5)"
    assert m.shared_bytes == 164, "41 * 4 bytes/fp32"
    alias_bytes = sum(m.entries[k].nbytes for k in m.alias_keys)
    assert alias_bytes == 96, "out_proj.weight aliases item_embed.weight: (8,3) fp32 = 96 B, its OWN entry size"
    assert m.buffer_bytes(BufferRule.FIXED) == 40, "class_map: 5 int64 * 8 B = 40 B (FIXED)"
    assert m.buffer_bytes(BufferRule.SERVER_COPY) == 4, "input_scale: 1 fp32 * 4 B = 4 B (SERVER_COPY)"

    vb = visit_bytes(m)
    assert vb["download_bytes"] == 164, "comm.py: download = every SHARED tensor once"
    assert vb["upload_bytes"] == 164 + 8, "upload = shared + 8-byte n_consumed header"
    assert vb["send_plus_receive_bytes"] == 2 * 164 + 8
    assert vb["alias_bytes_not_sent"] == 96, "the tied alias's OWN bytes, reported but never re-sent"
    assert vb["bootstrap_bytes_once"] == 44, "FIXED(40) + SERVER_COPY(4), once per device, not per visit"
    assert vb["private_bytes_sent"] == 0
    assert vb["private_resident_bytes"] == 0, "no private_extra declared on this manifest"


def test_payload_bytes_counts_a_tied_storage_tensor_once():
    # The claim itself: two DIFFERENT payload keys that are the exact same Parameter (a tied alias) must
    # be counted once, not twice, in the theoretical payload byte count.
    t = torch.randn(8, 3, dtype=torch.float32)   # 24 elem * 4 B = 96 B
    payload_two_names_same_tensor = {"item_embed.weight": t, "out_proj.weight": t}
    assert payload_bytes(payload_two_names_same_tensor) == 96, (
        "comm.py payload_bytes: 'tensors sharing one storage (tied aliases) are counted once'")
    payload_two_distinct_tensors = {"a": t, "b": t.clone()}
    assert payload_bytes(payload_two_distinct_tensors) == 192, "two DISTINCT tensors must be counted separately"


def test_empty_tensor_contributes_zero_bytes_and_is_not_deduplicated_against_itself():
    empty = torch.zeros(0, dtype=torch.float32)
    assert payload_bytes({"e1": empty, "e2": empty}) == 0


def test_retry_bytes_are_counted_separately_from_ordinary_download_bytes():
    entries = {"w": Entry("w", Role.SHARED, "torch.float32", (25,), 25, 4)}   # 25*4 = 100 bytes
    manifest = ParamManifest(entries)
    ledger = Ledger()
    ch = VirtualChannel(manifest, ledger=ledger)

    ch.download(round_idx=0, key="c1")                      # ordinary: +100 to bytes_down
    ch.download(round_idx=0, key="c1", retry=True)          # retry: +100 to retry_bytes_down, NOT bytes_down
    ch.upload(round_idx=0, key="c1", upload={"w": torch.zeros(25)})   # +100+8 to bytes_up

    assert ledger.bytes_down == 100, "comm.py: 'A retried visit re-downloads theta_r; ... counted separately'"
    assert ledger.retry_bytes_down == 100
    assert ledger.bytes_up == 108
    assert ledger.messages == 3
    r0 = ledger.by_round[0]
    assert r0 == {"down": 100, "up": 108, "retry_down": 100, "messages": 3}


def test_channel_upload_refuses_a_key_set_that_is_not_exactly_shared():
    entries = {"w": Entry("w", Role.SHARED, "torch.float32", (4,), 4, 4)}
    manifest = ParamManifest(entries)
    ch = VirtualChannel(manifest, ledger=Ledger())
    with pytest.raises(ValueError):
        ch.upload(0, "c1", {"w": torch.zeros(4), "p": torch.zeros(3)})   # an extra (private-looking) key
    with pytest.raises(ValueError):
        ch.upload(0, "c1", {})                                          # missing the shared key entirely
