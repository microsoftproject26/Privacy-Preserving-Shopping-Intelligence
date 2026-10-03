"""Declared input widths: the models refuse inputs whose width differs from the declared layout.

  * both models refuse every wrong float-block width, including the OFFSETTING cases (numeric + flags or user ctx
    + masks summing to the right concatenated width), which a Linear layer alone would accept silently;
  * the adapters also refuse wrong name arrays and a wrong feature_view;
  * a checkpoint of another width cannot be loaded (hard shape refusal);
  * a GRU spec without user widths and a rich SASRec without widths are refused.
Negative control: a model without the explicit width check accepts the offsetting widths silently, which is why the
check exists.
"""
from __future__ import annotations

import pytest
import torch
from seqrec_testkit import (
    FAMS,
    K_SMALL,
    W,
    adapter,
    assert_raises,
    batch,
    forward,
    kwargs_for,
    model,
    nc,
    poc,
    vocab,
)

from ppsi.seqrec import gru, sasrec
from ppsi.seqrec.features import InputWidths, SchemaError, WidthError, check_batch_layout
from ppsi.seqrec.variants import VARIANTS

WRONG = {  # key -> (numeric, flags, user ctx, masks) widths of a bad batch
    "numeric_19": (19, 15, 12, 4),
    "flags_14": (15, 14, 12, 4),
    "user_ctx_11": (15, 15, 11, 4),
    "masks_6": (15, 15, 12, 6),
    "offset_numeric19_flags11": (19, 11, 12, 4),
    "offset_uctx14_masks2": (15, 15, 14, 2),
    "all_wider": (19, 15, 12, 6),
}


def _bad_batch(widths4):
    return batch(4, widths=InputWidths(*widths4, view_id="bad"), seed=1)


def test_input_widths_validation():
    assert_raises(SchemaError, InputWidths, -1, 15, 12, 4)
    assert_raises(SchemaError, InputWidths, 2, 0, 0, 0, numeric_names=("a",))
    w = InputWidths(2, 1, 0, 0, numeric_names=("a", "b"), flag_names=("f",))
    assert w.gru_spec()["user_context"] is False and w.as_json()["numeric_names"] == ["a", "b"]


@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("case", sorted(WRONG))
def test_models_refuse_wrong_widths(family, case):
    m = model(family, 2026).eval()
    b = _bad_batch(WRONG[case])
    with torch.no_grad():
        assert_raises(ValueError, forward, m, b)
        assert_raises(ValueError, lambda: m.query(**kwargs_for(m, b)))
    a = adapter(family)
    assert_raises(WidthError, a.query, b)


@pytest.mark.parametrize("family", FAMS)
def test_models_accept_the_declared_widths(family):
    m = model(family, 2026).eval()
    with torch.no_grad():
        y = forward(m, batch(4, seed=1))
    assert y.shape == (4, K_SMALL) and torch.isfinite(y).all()


@pytest.mark.parametrize("family", FAMS)
def test_adapter_refuses_wrong_name_arrays_and_view(family):
    a = adapter(family)
    a.module.eval()
    good = batch(3, seed=2, with_names=True)
    with torch.no_grad():
        a.query(good)                                                          # accepted
    bad = dict(good)
    n = list(good["event_numeric_feature_names"])
    n[1], n[2] = n[2], n[1]
    bad["event_numeric_feature_names"] = n
    assert_raises(WidthError, a.query, bad)
    bad = dict(good)
    bad["user_context_mask_names"] = [*list(good["user_context_mask_names"])[:3], "another_mask"]
    assert_raises(WidthError, a.query, bad)
    bad = dict(good)
    bad["feature_view"] = "other_view"
    assert_raises(WidthError, a.query, bad)
    assert_raises(WidthError, check_batch_layout, {"event_quality_flag_names": "price_missing"}, W)


@pytest.mark.parametrize("family", FAMS)
def test_checkpoint_of_another_width_cannot_load(family):
    other = model(family, 2026, widths=InputWidths(19, 15, 12, 6)).state_dict()
    m = model(family, 2026)
    assert_raises(RuntimeError, m.load_state_dict, other, strict=True)
    assert_raises(RuntimeError, m.load_state_dict, other, strict=False)       # size mismatch refuses even non-strict


def test_gru_spec_without_user_widths_is_refused():
    spec = dict(W.gru_spec())
    del spec["user_mask_dim"]
    assert_raises(KeyError, gru.build_model, spec, vocab(64), product_idx_of_class=list(range(3, 67)))
    assert_raises(KeyError, gru.build_model, "F4_rich_side_information", vocab(64),
                  product_idx_of_class=list(range(3, 67)))                      # no silent width by group name


def test_sasrec_rich_without_widths_is_refused():
    cfg = sasrec.SASRecConfig.from_entry(VARIANTS["SASREC_D256_B2"]["sasrec_entry"])
    assert_raises(ValueError, sasrec.SASRecCE, cfg, vocab_sizes=vocab(8), product_idx_of_class=poc(8), seed=1)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a GRU input layer without the width check silently accepts numeric 19 + flags 11 = 30")
def test_nc_unchecked_linear_accepts_offsetting_widths():
    m = model("GRU", 2026).eval()
    b = _bad_batch((19, 11, 12, 4))
    nf = torch.cat([b["event_numeric_features"], b["event_quality_flags"].to(torch.float32)], dim=-1)
    with torch.no_grad():
        assert_raises(RuntimeError, m.numeric_flags_proj, nf)                  # the Linear alone does not refuse
