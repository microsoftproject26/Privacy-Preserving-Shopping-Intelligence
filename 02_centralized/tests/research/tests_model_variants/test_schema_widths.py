"""Schema-driven widths (OI-05): the schema object, and the models refuse inputs whose width differs from the schema.

  * FEATURE_SCHEMA_CLEAN_V2 2.1.0: self-hash verified; widths 15 / 15 / 12 / 4 from the name arrays, equal to every
    other place the schema states them; 53 channels; a tampered schema is refused;
  * both V2 models refuse every wrong float-block width, including the OFFSETTING cases (numeric + flags or user ctx
    + masks summing to the right concatenated width), which a Linear layer alone would accept silently;
  * the adapters also refuse wrong Clean V2 name arrays and a wrong feature_view;
  * a v1-width checkpoint cannot be loaded into a V2 model (and vice versa): hard shape refusal.
Negative control: the SNAPSHOT models (no width check) accept the offsetting widths silently — the reason the
explicit check exists.
"""
from __future__ import annotations

import copy

import pytest
import torch

from models_testkit import FAMS, K_SMALL, SCHEMA, V1, V2, assert_raises, batch, forward, kwargs_for, nc, snap_model, \
    v2_adapter, v2_model
from v4models.schema import (SchemaError, WidthError, canonical_self_hash, check_batch_layout, load_schema,
                             widths_from_schema_obj)

WRONG = {  # key -> (numeric, flags, user ctx, masks) widths of a bad batch
    "numeric_v1_19": (19, 15, 12, 4),
    "flags_14": (15, 14, 12, 4),
    "user_ctx_11": (15, 15, 11, 4),
    "masks_v1_6": (15, 15, 12, 6),
    "offset_numeric19_flags11": (19, 11, 12, 4),
    "offset_uctx14_masks2": (15, 15, 14, 2),
    "all_v1_widths": (19, 15, 12, 6),
}


def _bad_batch(widths4):
    from v4models.schema import InputWidths
    w = InputWidths(*widths4, view_id="bad")
    return batch(4, widths=w, seed=1)


# ------------------------------------------------------------------------------------------------ schema object
def test_schema_widths_and_self_hash():
    assert SCHEMA.schema_version == "2.1.0" and SCHEMA.schema_id == "FEATURE_SCHEMA_CLEAN_V2"
    assert SCHEMA.self_sha256 == canonical_self_hash(SCHEMA.raw) == \
        "55a244e1608bcdcdf6f1728595f534871b52154faebf2977e11ac1fb487052f9"
    assert (V2.numeric_dim, V2.flags_dim, V2.user_context_dim, V2.user_mask_dim) == (15, 15, 12, 4)
    assert SCHEMA.channels_total == 53 and V2.view_id == "clean_v2_rich"
    assert V2.numeric_names[0] == "log1p_price" and "local_hour_sin" not in V2.numeric_names
    assert "prefix_item_count" not in V2.numeric_names and "log1p_prefix_elapsed_s" not in V2.numeric_names
    assert V2.user_mask_names == ("recent_category_match_supported", "recent_brand_match_supported",
                                  "recent_price_supported", "prev_session_available")
    assert len(SCHEMA.dead_in_train_flags) == 6 and set(SCHEMA.dead_in_train_flags) <= set(V2.flag_names)
    assert (V1.numeric_dim, V1.flags_dim, V1.user_context_dim, V1.user_mask_dim) == (19, 15, 12, 6)
    assert V2.gru_spec() == SCHEMA.raw["family_sizing"]["GRU"]["group_spec"] | {"categorical": list(V2.categorical)}


def test_schema_tampering_is_refused():
    obj = copy.deepcopy(SCHEMA.raw)
    obj["tensor_layout_v2"]["user_context_masks"]["names"].append("has_history_mask")        # 5 masks
    assert_raises(SchemaError, widths_from_schema_obj, obj)
    obj = copy.deepcopy(SCHEMA.raw)
    obj["family_sizing"]["SASRec"]["constants_v2"]["USER_MASK_DIM"] = 6
    assert_raises(SchemaError, widths_from_schema_obj, obj)
    obj = copy.deepcopy(SCHEMA.raw)
    obj["summary_counts"]["final_widths"]["channels_total"] = 59
    assert_raises(SchemaError, widths_from_schema_obj, obj)
    obj = copy.deepcopy(SCHEMA.raw)
    obj["schema_version"] = "2.1.1"                                   # any edit breaks the self-hash
    assert canonical_self_hash(obj) != SCHEMA.self_sha256
    assert load_schema().widths == V2                                 # the file on disk still verifies


# ------------------------------------------------------------------------------------------------ model refusals
@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("case", sorted(WRONG))
def test_models_refuse_wrong_widths(family, case):
    m = v2_model(family, 2026).eval()
    b = _bad_batch(WRONG[case])
    with torch.no_grad():
        assert_raises(ValueError, forward, m, b)
        assert_raises(ValueError, lambda: m.query(**kwargs_for(m, b)))
    a = v2_adapter(family)
    assert_raises(WidthError, a.query, b)


@pytest.mark.parametrize("family", FAMS)
def test_models_accept_the_schema_widths(family):
    m = v2_model(family, 2026).eval()
    with torch.no_grad():
        y = forward(m, batch(4, seed=1))
    assert y.shape == (4, K_SMALL) and torch.isfinite(y).all()


@pytest.mark.parametrize("family", FAMS)
def test_adapter_refuses_wrong_name_arrays_and_view(family):
    a = v2_adapter(family)
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
    bad["user_context_mask_names"] = list(good["user_context_mask_names"])[:3] + ["has_history_mask"]
    assert_raises(WidthError, a.query, bad)
    bad = dict(good)
    bad["feature_view"] = "v1_rich"
    assert_raises(WidthError, a.query, bad)
    assert_raises(WidthError, check_batch_layout, {"event_quality_flag_names": "price_missing"}, V2)


@pytest.mark.parametrize("family", FAMS)
def test_v1_checkpoint_cannot_load_into_v2(family):
    v1_state = snap_model(family, 2026).state_dict()
    v2 = v2_model(family, 2026)
    assert_raises(RuntimeError, v2.load_state_dict, v1_state, strict=True)
    assert_raises(RuntimeError, v2.load_state_dict, v1_state, strict=False)       # size mismatch refuses even non-strict
    assert_raises(RuntimeError, snap_model(family, 2026).load_state_dict, v2.state_dict(), strict=True)


def test_gru_spec_without_user_widths_is_refused():
    from v4models import gru_v2
    spec = dict(V2.gru_spec())
    del spec["user_mask_dim"]
    assert_raises(KeyError, gru_v2.build_model, spec, {"item_tokens": 67, "category_tokens": 663,
                                                       "brand_tokens": 3956, "main_category_tokens": 16,
                                                       "daypart_tokens": 8, "weekday_tokens": 10},
                  product_idx_of_class=list(range(3, 67)))
    assert_raises(KeyError, gru_v2.build_model, "F4_rich_side_information", {"item_tokens": 67, "category_tokens": 663,
                  "brand_tokens": 3956, "main_category_tokens": 16, "daypart_tokens": 8, "weekday_tokens": 10},
                  product_idx_of_class=list(range(3, 67)))                      # no silent v1 width by group name


def test_sasrec_rich_without_widths_is_refused():
    from fl.families import SASREC_ENTRY
    from v4models import sasrec_v2
    from models_testkit import poc, vocab
    cfg = sasrec_v2.SASRecConfig.from_registry(SASREC_ENTRY)
    assert_raises(ValueError, sasrec_v2.SASRecCE, cfg, vocab_sizes=vocab(8), product_idx_of_class=poc(8), seed=1)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("the snapshot GRU (hard-coded widths, no check) silently accepts numeric 23 + flags 11 = 34")
def test_nc_snapshot_gru_accepts_offsetting_widths():
    m = snap_model("GRU", 2026).eval()
    with torch.no_grad():
        assert_raises(ValueError, forward, m, _bad_batch((23, 11, 12, 6)))


@nc("the snapshot SASRec (hard-coded widths, no check) silently accepts user ctx 14 + masks 4 = 18")
def test_nc_snapshot_sasrec_accepts_offsetting_widths():
    m = snap_model("SASREC", 2026).eval()
    with torch.no_grad():
        assert_raises(ValueError, forward, m, _bad_batch((19, 15, 14, 4)))
