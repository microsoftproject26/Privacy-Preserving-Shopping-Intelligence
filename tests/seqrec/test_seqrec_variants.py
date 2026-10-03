"""Model size variants.

  * the variant table: families, defaults, the cloud size order and the small on-device size per family;
  * the small SASRec (SASREC_D64_B2: d_model 64, 4 heads of 16, ff x4, 2 blocks, untied head, recentering) differs
    from the default SASRec only in d_model, is parameter-matched to the default GRU at the full catalogue, and its
    per-visit dense payload (download + upload, FP32) stays under 170 MB;
  * GRU_D256 changes only the item dimension, and the GRU built at item_dim 128 is the default GRU;
  * dropout values: the variant default, explicit values and refusals; dropout has no parameters, so theta0 does
    not depend on it;
  * theta0 determinism and seed dependence; a forward pass on a tiny catalogue.
Negative control: assuming the small SASRec has as many parameters as the default SASRec must fail.
"""
from __future__ import annotations

import pytest
import torch
from seqrec_testkit import K_SMALL, W, assert_raises, batch, forward, nc

from ppsi.fedsim.comm import visit_bytes
from ppsi.seqrec import variants as V
from ppsi.seqrec.adapters import module_forward
from ppsi.seqrec.build import build

FULL_K = 158_486                # the REES46 catalogue size
SMALL_CAP_BYTES = 170_000_000
NAME = "SASREC_D64_B2"


def test_variant_table():
    assert set(V.VARIANTS) == {"GRU_D128", "GRU_D256", "SASREC_D256_B2", "SASREC_D256_B3", "SASREC_D384_B2", NAME}
    assert V.DEFAULT_VARIANT == {"GRU": "GRU_D128", "SASREC": "SASREC_D256_B2"}
    assert V.SMALL_VARIANT == {"GRU": "GRU_D128", "SASREC": NAME}
    assert NAME not in V.SIZE_ORDER["SASREC"] and NAME not in V.DEFAULT_VARIANT.values()
    for fam, names in V.SIZE_ORDER.items():
        assert names[0] == V.DEFAULT_VARIANT[fam] and all(V.VARIANTS[n]["family"] == fam for n in names)
    assert_raises(V.VariantError, V.resolve, "GRU", "SASREC_D64_B2")
    assert_raises(V.VariantError, V.resolve, "GRU", "GRU_D512")
    assert_raises(V.VariantError, V.resolve, "LSTM")
    assert V.topology(NAME)["model_variant"] == NAME


def test_small_sasrec_spec():
    e = V.VARIANTS[NAME]
    entry = e["sasrec_entry"]
    assert e["family"] == "SASREC" and e["current_build"] is False and e["head"] == "untied"
    assert entry["d_model"] == 64 and entry["blocks"] == 2 and entry["heads"] == 4 and entry["ff_multiplier"] == 4
    base = V.VARIANTS["SASREC_D256_B2"]
    assert {k: v for k, v in e.items() if k not in ("sasrec_entry", "current_build")} == \
        {k: v for k, v in base.items() if k not in ("sasrec_entry", "current_build")}
    diff = {k for k in entry if entry[k] != base["sasrec_entry"][k]}
    assert diff == {"id", "d_model"}
    name, resolved = V.resolve("SASREC", NAME)
    assert name == NAME and resolved == e and not V.is_current_build(NAME)


def test_inventory_numbers_full_catalogue():
    """Parameter count and per-visit payload at the full catalogue (only item_tokens depends on K)."""
    bm = build("SASREC", W, 2026, "cpu", K=FULL_K, variant=NAME)
    man = bm.manifest
    vb = visit_bytes(man)
    assert man.shared_numel == 20_710_048 and bm.inventory["total"] == man.shared_numel
    assert vb["send_plus_receive_bytes"] == 165_680_392 <= SMALL_CAP_BYTES
    g = build("GRU", W, 2026, "cpu", K=FULL_K, variant="GRU_D128")
    assert g.manifest.shared_numel == 20_963_806
    assert 0.95 <= man.shared_numel / g.manifest.shared_numel <= 1.02


@nc("the small SASRec must be strictly smaller than the default SASRec, not equal to it")
def test_nc_small_sasrec_as_large_as_the_default():
    small = build("SASREC", W, 2026, "cpu", K=4096, variant=NAME).manifest.shared_numel
    default = build("SASREC", W, 2026, "cpu", K=4096).manifest.shared_numel
    assert small == default


def test_gru_item_dim_variant():
    d128 = build("GRU", W, 2026, K=K_SMALL)
    explicit = build("GRU", W, 2026, K=K_SMALL, variant="GRU_D128")
    assert d128.init_sha256 == explicit.init_sha256
    d256 = build("GRU", W, 2026, K=K_SMALL, variant="GRU_D256")
    m = d256.module
    assert m.item_embed.weight.shape[1] == 256 and m.input_proj.out_features == 256
    assert m.gru.input_size == 256 and m.gru.hidden_size == 256 and m.readout_proj.out_features == 256
    assert d256.adapter.query_dim == 256 and d128.adapter.query_dim == 128
    assert [k for k, _ in m.named_parameters()] == [k for k, _ in d128.module.named_parameters()]


def test_dropout_values():
    assert V.dropout_of("GRU", None) == (0.2, 0.1) and V.dropout_of("SASREC", NAME) == (0.2,)
    assert V.dropout_of("GRU", None, (0.3, 0.0)) == (0.3, 0.0) and V.dropout_of("SASREC", None, 0.1) == (0.1,)
    assert_raises(V.VariantError, V.dropout_of, "GRU", None, 0.1)            # GRU needs two values
    assert_raises(V.VariantError, V.dropout_of, "SASREC", None, 1.0)
    a = build("SASREC", W, 2026, K=K_SMALL, dropout=0.1)
    b = build("SASREC", W, 2026, K=K_SMALL)
    assert a.module.cfg.dropout == 0.1 and b.module.cfg.dropout == 0.2
    assert a.init_sha256 == b.init_sha256, "dropout has no parameters: theta0 must not depend on it"


def test_theta0_determinism_and_seed_dependence():
    a = build("SASREC", W, 2026, "cpu", K=4096, variant=NAME)
    b = build("SASREC", W, 2026, "cpu", K=4096, variant=NAME)
    assert (a.init_sha256, a.raw_init_sha256, a.manifest_digest) == (b.init_sha256, b.raw_init_sha256,
                                                                     b.manifest_digest)
    assert a.recentered is True
    c = build("SASREC", W, 2027, "cpu", K=4096, variant=NAME)
    assert a.init_sha256 != c.init_sha256


@pytest.mark.parametrize("variant", ["SASREC_D256_B3", "SASREC_D384_B2", NAME])
def test_forward_pass_tiny_catalogue(variant):
    bm = build("SASREC", W, 2026, "cpu", K=K_SMALL, variant=variant)
    m, a = bm.module, bm.adapter
    b = batch(6, K_SMALL, seed=3)
    m.eval()
    with torch.no_grad():
        out = forward(m, b)
        assert out.shape == (6, K_SMALL) and torch.isfinite(out).all()
        assert torch.equal(a.scores(b), module_forward(a, b))
        q = a.query(b)
    assert q.shape == (6, a.query_dim) and a.query_dim == m.cfg.d_model
    assert a.head_weight().shape == (K_SMALL, a.query_dim)
