"""TRAIN-04 — SASREC_D64_B2 variant (PREG1_AMENDMENT_03 §1/§3: downstream SMALL SASRec, code only).

d_model 64, 4 heads (head dim 16), ff x4, 2 blocks, untied head, C7 recentering; everything else identical to
SASREC_D256_B2 (heads count, ff multiplier, tying, dropout sites, recentering). Parameter-matched to GRU_D128 (the
downstream SMALL GRU size). NOT a P2 contest size: not in `v4models.variants.SIZE_ORDER`; the downstream SMALL size
per family lives in the new `v4models.variants.DOWNSTREAM_SMALL` mapping.

No dataset row is read anywhere in this file (`v4models.build.build(..., K=...)` only, exactly the pattern used by
v4train.theta0_variants and v4models.inventory for a K-override / small synthetic build).

  * the variant is registered with the exact spec fields, is a SASREC family entry, and is current_build False;
  * it is NOT in SIZE_ORDER (a P2 contest size), and DOWNSTREAM_SMALL == {"GRU": "GRU_D128", "SASREC": "SASREC_D64_B2"};
  * adding it left every PRE-EXISTING variant's topology/spec bitwise identical (a pinned sha256 of the five 02R1/02R2
    entries, DEFAULT_VARIANT and SIZE_ORDER, computed before this task's edit and asserted again here);
  * the inventory numbers (v4models.build + fl.comm.visit_bytes) at the full catalogue: shared_numel approx 20.4M
    (parameter-matched to GRU_D128, both measured the same way) and the per-visit dense payload (download + upload,
    FP32) <= 170 MB (PREG1_AMENDMENT_03 §1's SMALL cap);
  * theta0 determinism (two builds, same seed) and seed dependence (2026 vs 2027), same contract as the other variants;
  * the model builds on CPU with a tiny synthetic vocab and runs a forward pass (module.forward and adapter.query),
    output shape [batch, K_SMALL].
Negative control: a size comparison that assumes SASREC_D64_B2 has as many parameters as the current SASREC_D256_B2
build (it must be strictly smaller) is deliberately wrong and must fail.
"""
from __future__ import annotations

import hashlib
import json

import torch

from models_testkit import K_SMALL, V2, batch, forward, nc
from fl.comm import visit_bytes
from v4models import variants as V
from v4models.build import build, module_forward

FULL_K = 158_486
SMALL_CAP_BYTES = 170_000_000                      # PREG1_AMENDMENT_03 §1: <= 170 MB per-visit dense payload
NAME = "SASREC_D64_B2"

# The five 02R1 / 02R2 variants + DEFAULT_VARIANT + SIZE_ORDER, exactly as they stood before TRAIN-04 (pinned so this
# task's addition is proven to leave them bitwise unchanged; canonicalisation matches v4models.variants.self_sha256).
PRE_EXISTING_VARIANTS = ("GRU_D128", "GRU_D256", "SASREC_D256_B2", "SASREC_D256_B3", "SASREC_D384_B2")
PRE_EXISTING_VARIANTS_SHA256 = "97ba5cf06ca032c5cba13179607b8a58097c75a8505218b1a61e5079860df664"
DEFAULT_VARIANT_SHA256 = "893837f8898e9d03dbd0b8400c435bae821c1c24bab33fecff25fdad2292c6ba"
SIZE_ORDER_SHA256 = "d23cacf3f4c79ff50cc57765855c346ffa02f4a46522b4d2a6d38daac04a5605"


def _canon_sha256(obj) -> str:
    enc = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(enc.encode("utf-8")).hexdigest()


def test_variant_registered_with_the_exact_spec():
    e = V.VARIANTS[NAME]
    assert e["family"] == "SASREC" and e["current_build"] is False
    assert e["head"] == "untied" and e["recenter"] == "C7"
    assert e["dropout_sites"] == ["all"] and e["dropout_default"] == [0.2]
    entry = e["sasrec_entry"]
    assert entry["d_model"] == 64 and entry["blocks"] == 2 and entry["heads"] == 4
    assert entry["d_model"] // entry["heads"] == 16                  # head dim 16
    assert entry["ff_multiplier"] == 4 and entry["dropout"] == 0.2
    assert entry["group"] == "rich_frozen_channels" and entry["id"] == NAME
    # everything else identical to SASREC_D256_B2 except the fields the spec names as different
    d256 = V.VARIANTS["SASREC_D256_B2"]
    assert e["head"] == d256["head"] and e["recenter"] == d256["recenter"]
    assert e["dropout_sites"] == d256["dropout_sites"] and e["dropout_default"] == d256["dropout_default"]
    d256_entry = d256["sasrec_entry"]
    assert entry["heads"] == d256_entry["heads"] and entry["ff_multiplier"] == d256_entry["ff_multiplier"]
    assert entry["group"] == d256_entry["group"] and entry["dropout"] == d256_entry["dropout"]
    name, resolved = V.resolve("SASREC", NAME)
    assert name == NAME and resolved == e and not V.is_current_build(NAME)


def test_downstream_small_not_a_p2_contest_size():
    """SASREC_D64_B2 is a downstream SMALL size (PREG1_AMENDMENT_03 §1), not a P2 SIZE_ORDER entry."""
    assert NAME not in V.SIZE_ORDER["SASREC"]
    assert NAME not in V.SIZE_ORDER["GRU"]
    assert V.DOWNSTREAM_SMALL == {"GRU": "GRU_D128", "SASREC": NAME}
    # SIZE_ORDER itself (the P2 contest ladder) is unchanged by this addition
    assert _canon_sha256({k: list(v) for k, v in V.SIZE_ORDER.items()}) == SIZE_ORDER_SHA256


@nc("SASREC_D64_B2 wrongly believed to be a P2 contest size (it must NOT be in SIZE_ORDER)")
def test_nc_size_order_would_wrongly_include_it():
    assert NAME in V.SIZE_ORDER["SASREC"]


def test_every_pre_existing_variant_topology_unchanged():
    """Adding SASREC_D64_B2 must leave every existing variant's topology/spec hash identical (HARD RULE)."""
    assert set(V.VARIANTS) == set(PRE_EXISTING_VARIANTS) | {NAME}
    pinned = {k: V.VARIANTS[k] for k in PRE_EXISTING_VARIANTS}
    assert _canon_sha256(pinned) == PRE_EXISTING_VARIANTS_SHA256
    assert _canon_sha256(V.DEFAULT_VARIANT) == DEFAULT_VARIANT_SHA256
    assert NAME not in V.DEFAULT_VARIANT.values()


@nc("a tampered SASREC_D256_B2 entry must be caught by the pinned pre-existing-variants hash")
def test_nc_pre_existing_variant_hash_check_has_teeth():
    tampered = {k: V.VARIANTS[k] for k in PRE_EXISTING_VARIANTS}
    tampered["SASREC_D256_B2"] = dict(tampered["SASREC_D256_B2"], head="tied")     # deliberately wrong
    assert _canon_sha256(tampered) == PRE_EXISTING_VARIANTS_SHA256


def test_inventory_numbers_full_catalogue():
    """Parameter count and per-visit payload at the full catalogue (no dataset row read: K-override, matching the
    v4train.theta0_variants / v4models.inventory production numbers, since only item_tokens depends on K)."""
    from v4models.schema import load_schema
    schema = load_schema()
    bm = build("SASREC", schema, 2026, "cpu", K=FULL_K, variant=NAME)
    man = bm.manifest
    vb = visit_bytes(man)
    assert man.shared_numel == 20_710_048
    assert 20_000_000 <= man.shared_numel <= 21_000_000               # spec: approx 20.4M
    assert bm.inventory["total"] == man.shared_numel
    assert vb["send_plus_receive_bytes"] == 165_680_392
    assert vb["send_plus_receive_bytes"] <= SMALL_CAP_BYTES           # PREG1_AMENDMENT_03 §1: <= 170 MB
    # parameter-matched to GRU_D128, measured the same way
    gru = build("GRU", schema, 2026, "cpu", K=FULL_K, variant="GRU_D128")
    assert gru.manifest.shared_numel == 20_963_806
    ratio = man.shared_numel / gru.manifest.shared_numel
    assert 0.95 <= ratio <= 1.02


@nc("SASREC_D64_B2 must be strictly smaller than the current SASREC_D256_B2 build, not equal to it")
def test_nc_would_wrongly_match_sasrec_d256_b2_size():
    from v4models.schema import load_schema
    schema = load_schema()
    small = build("SASREC", schema, 2026, "cpu", K=FULL_K, variant=NAME).manifest.shared_numel
    current = build("SASREC", schema, 2026, "cpu", K=FULL_K, variant="SASREC_D256_B2").manifest.shared_numel
    assert small == current


def test_theta0_determinism_and_seed_dependence():
    from v4models.schema import load_schema
    schema = load_schema()
    a = build("SASREC", schema, 2026, "cpu", K=FULL_K, variant=NAME)
    b = build("SASREC", schema, 2026, "cpu", K=FULL_K, variant=NAME)
    assert (a.init_sha256, a.raw_init_sha256, a.manifest_digest) == (b.init_sha256, b.raw_init_sha256, b.manifest_digest)
    assert a.recentered is True                                       # C7: SASRec theta0 is always recentered
    c = build("SASREC", schema, 2027, "cpu", K=FULL_K, variant=NAME)
    assert a.init_sha256 != c.init_sha256


def test_forward_pass_tiny_synthetic_vocab():
    """CPU build with a tiny synthetic vocab and vocab; both module.forward() and the FL adapter (scores/query) run."""
    bm = build("SASREC", V2, 2026, "cpu", K=K_SMALL, variant=NAME)
    m, a = bm.module, bm.adapter
    assert m.cfg.d_model == 64 and m.cfg.blocks == 2 and m.cfg.heads == 4
    b = batch(6, K_SMALL, widths=V2, seed=3)
    m.eval()
    with torch.no_grad():
        out = forward(m, b)                                          # the module's own forward: logits over K
        assert out.shape == (6, K_SMALL) and torch.isfinite(out).all()
        assert torch.equal(a.scores(b), module_forward(a, b)), "adapter scores must equal the module's own forward"
        assert torch.equal(a.scores(b), out)
        q = a.query(b)
    assert q.shape == (6, a.query_dim) and a.query_dim == 64          # query_dim == d_model for SASRec
    assert a.head_weight().shape == (K_SMALL, a.query_dim)
