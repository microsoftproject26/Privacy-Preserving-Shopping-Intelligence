"""T33 — virtual byte counts: send AND receive per visit, tied aliases once, private state 0, retries separate.

Proves: the byte formulas against hand-computed values for a model with a genuine tied alias, the round ledger totals,
that PF adds no wire bytes, and (small K) that each executed family's per-visit payload is 4 bytes x its de-duplicated
parameter count (the full-K numbers are checked in test_families_smoke). Does not measure network time.
"""
from __future__ import annotations

import torch

from fl.adapter import BufferRule
from fl.client import FaultPlan, PFConfig
from fl.comm import UPLOAD_HEADER_BYTES, Ledger, VirtualChannel, payload_bytes, serialized_bytes, visit_bytes
from fl.personal import PersonalStore
from fl.server import run_round
from fl_testkit import K, D, clients, nc, server_and_worker, solver, tiny

V = K + 3
SHARED_TINY_TIED = 4 * (K + V * D + D * D + D)          # output_bias, item_embed (alias out_proj once), enc.w, enc.b


def naive_bytes(adapter) -> int:
    """WRONG: every state_dict entry, aliases and buffers included."""
    return sum(t.numel() * t.element_size() for t in adapter.module.state_dict().values())


def check_visit_bytes(count_fn):
    a = tiny(1, tied=True)
    assert count_fn(a) == SHARED_TINY_TIED, f"{count_fn(a)} != {SHARED_TINY_TIED}"


def _manifest_bytes(a):
    return visit_bytes(a.manifest)["download_bytes"]


def test_t33_hand_computed_visit_bytes():
    check_visit_bytes(_manifest_bytes)
    a = tiny(1, tied=True)
    vb = visit_bytes(a.manifest.with_private("pf.p_u", (D,)).with_private("pf.adam.exp_avg", (D,))
                     .with_private("pf.adam.exp_avg_sq", (D,)).with_private("pf.adam.step", ()))
    assert vb["upload_bytes"] == SHARED_TINY_TIED + UPLOAD_HEADER_BYTES
    assert vb["send_plus_receive_bytes"] == 2 * SHARED_TINY_TIED + UPLOAD_HEADER_BYTES
    assert vb["private_bytes_sent"] == 0 and vb["private_resident_bytes"] == 4 * (3 * D + 1)
    assert vb["bootstrap_bytes_once"] == 8 * K + 4                  # int64 class map + float scale, once per device
    assert vb["alias_bytes_not_sent"] == 4 * V * D
    up = a.extract_shared()
    up_with_alias = dict(up); up_with_alias["out_proj.weight"] = a.module.out_proj.weight.detach()
    assert payload_bytes(up) == payload_bytes(up_with_alias) == SHARED_TINY_TIED
    assert serialized_bytes(up) > payload_bytes(up)                  # serializer framing is reported separately


def test_t33_round_ledger_send_receive_and_retry():
    srv, w = server_and_worker(seed=2)
    cs = clients(5, seed=81, sizes=[10, 4, 22, 7, 16])
    led = Ledger()
    ch = VirtualChannel(srv.manifest, ledger=led)
    rep = run_round(srv, w, cs, solver(), round_idx=0, seed=1, channel=ch,
                    fault=FaultPlan(frozenset({(cs[2].key, 0, 1)})))
    assert rep.bytes_down == led.bytes_down == 5 * SHARED_TINY_TIED
    assert rep.bytes_up == led.bytes_up == 5 * (SHARED_TINY_TIED + UPLOAD_HEADER_BYTES)
    assert rep.retry_bytes_down == led.retry_bytes_down == SHARED_TINY_TIED
    assert led.messages == 5 + 5 + 1


def test_t33_pf_adds_no_wire_bytes():
    cs = clients(4, seed=82, sizes=[10, 4, 22, 7])
    a, wa = server_and_worker(seed=2)
    b, wb = server_and_worker(seed=2)
    fa = run_round(a, wa, cs, solver(), round_idx=0, seed=1)
    pf = run_round(b, wb, cs, solver(), round_idx=0, seed=1, pf=PFConfig(lr=0.05),
                   store=PersonalStore(b.adapter.query_dim))
    assert (fa.bytes_down, fa.bytes_up) == (pf.bytes_down, pf.bytes_up)


def test_t33_untied_and_families_small_k():
    a = tiny(1, tied=False)
    assert visit_bytes(a.manifest)["download_bytes"] == 4 * (K * D + K + V * D + D * D + D)
    from fl.families import build_v2
    for fam in ("GRU", "SASREC"):
        f = build_v2(fam, seed=2026, K=64).adapter
        n = sum(p.numel() for p in f.module.parameters())           # de-duplicated
        assert visit_bytes(f.manifest)["download_bytes"] == 4 * n
        if fam == "GRU":
            assert f.manifest.buffer_bytes(BufferRule.FIXED) == 8 * 64  # product_idx_of_class, bootstrap only


@nc("naive state_dict byte count (tied alias twice, buffers per visit)")
def test_nc_t33_naive_count():
    check_visit_bytes(naive_bytes)
