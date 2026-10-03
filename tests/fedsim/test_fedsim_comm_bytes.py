"""Virtual byte counts: send AND receive per visit, tied aliases once, private state 0, retries separate.

Proves: the byte formulas against hand-computed values for a model with a genuine tied alias and for an untied
output table, the round ledger totals, and that PF adds no wire bytes. Does not measure network time.
"""
from __future__ import annotations

from fedsim_testkit import D, K, clients, nc, server_and_worker, solver, tiny

from ppsi.fedsim.adapter import BufferRule
from ppsi.fedsim.client import FaultPlan, PFConfig
from ppsi.fedsim.comm import (
    UPLOAD_HEADER_BYTES,
    Ledger,
    VirtualChannel,
    payload_bytes,
    serialized_bytes,
    visit_bytes,
)
from ppsi.fedsim.personal import PersonalStore
from ppsi.fedsim.server import run_round

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


def test_hand_computed_visit_bytes():
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


def test_round_ledger_send_receive_and_retry():
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


def test_pf_adds_no_wire_bytes():
    cs = clients(4, seed=82, sizes=[10, 4, 22, 7])
    a, wa = server_and_worker(seed=2)
    b, wb = server_and_worker(seed=2)
    fa = run_round(a, wa, cs, solver(), round_idx=0, seed=1)
    pf = run_round(b, wb, cs, solver(), round_idx=0, seed=1, pf=PFConfig(lr=0.05),
                   store=PersonalStore(b.adapter.query_dim))
    assert (fa.bytes_down, fa.bytes_up) == (pf.bytes_down, pf.bytes_up)


def test_untied_visit_bytes():
    a = tiny(1, tied=False)
    assert visit_bytes(a.manifest)["download_bytes"] == 4 * (K * D + K + V * D + D * D + D)
    n = sum(p.numel() for p in a.module.parameters())               # de-duplicated
    assert visit_bytes(a.manifest)["download_bytes"] == 4 * n
    assert a.manifest.buffer_bytes(BufferRule.FIXED) == 8 * K       # class_map, bootstrap only


@nc("naive state_dict byte count (tied alias twice, buffers per visit)")
def test_nc_naive_count():
    check_visit_bytes(naive_bytes)
