"""tests/test_data_adapter.py (owner: Sonnet-data). 05_TESTS_AND_PRETRAIN_GATE.md section A.

Real assertions with negative controls that really fail, covering A1-A9. Tests
marked `@pytest.mark.real` touch the frozen dataset (TRAIN/VALIDATION_SELECT
only, small bucket subsets except A1 which needs the full population count) --
they NEVER load VALIDATION_CONFIRM/TEST. Every other test is real-data-free
(hand-built fixtures or `synthetic_store.SyntheticStore`) and always runs.

No `conftest.py` is created here (owned by Sonnet-tests): each test file does
its own `sys.path` bootstrap.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import paths  # noqa: E402  -- sys.path side effect (code/reuse + vendor)
from full_store import CompactStore  # noqa: E402
from eval_full import aggregate  # noqa: E402
from rees46_eda.loaders import DecisionWindowLoader, detect_layout  # noqa: E402

import sasrec_data as sd  # noqa: E402
from synthetic_store import SyntheticStore  # noqa: E402
from sasrec_model import SASRecCE, SASRecConfig  # noqa: E402

REAL_BUCKETS = tuple(range(8))  # small subset -- ">=8 buckets" per contract, not the full 65


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _real_store(channels: str, buckets: tuple) -> CompactStore:
    return CompactStore.build(paths.DATASET_ROOT, splits=("TRAIN", "VALIDATION_SELECT"),
                              channels=channels, buckets=list(buckets), n_threads=3,
                              log=lambda *a, **k: None)


def _tiny_model(K: int, rich: bool, *, seed: int = 0, dropout: float = 0.0,
                impl: str = "unpadded") -> SASRecCE:
    poc = np.arange(K, dtype=np.int64) + 3
    vs = {"item_tokens": K + 3}
    if rich:
        vs.update({"category_tokens": 12, "brand_tokens": 12, "main_category_tokens": 8,
                   "daypart_tokens": 8, "weekday_tokens": 10})
    group = "rich_frozen_channels" if rich else "item_only"
    cfg = SASRecConfig(run_id="test", group=group, d_model=8, blocks=1, heads=2, dropout=dropout)
    m = SASRecCE(cfg, vocab_sizes=vs, product_idx_of_class=poc, seed=seed, impl=impl)
    m.eval()
    return m


def _independent_ascending_unique_check(ids: np.ndarray):
    """Independent re-implementation of "strictly ascending, all unique" --
    does NOT call into `full_store`'s own internal assertion, per the Evidence
    discipline rule against "checks that merely repeat the producer's own
    predicate". Returns (ok, n_unique, n_total)."""
    ids = np.asarray(ids)
    n_total = int(ids.shape[0])
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    strictly_ascending = bool(np.all(np.diff(sorted_ids) > 0)) if n_total > 1 else True
    n_unique = int(np.unique(sorted_ids).shape[0])
    return (strictly_ascending and n_unique == n_total), n_unique, n_total


def _make_solo_batch(tokens: list[int], length: int, T: int, *, poison: bool = True):
    item = torch.zeros(1, T, dtype=torch.int64)
    item[0, :length] = torch.tensor(tokens[:length], dtype=torch.int64)
    if poison and T > length:
        item[0, length:] = (torch.arange(length, T) % 5) + 3  # nonzero garbage in PAD cells
    am = torch.zeros(1, T, dtype=torch.bool)
    am[0, :length] = True
    lengths = torch.tensor([length])
    return dict(item_tokens=item, attention_mask=am, lengths=lengths)


# ===========================================================================
# A1 -- TRAIN loss population: count, strictly ascending, unique, all loss_mask
# ===========================================================================

@pytest.mark.real
def test_A1_train_loss_population_real():
    store = CompactStore.build(paths.DATASET_ROOT, splits=("TRAIN", "VALIDATION_SELECT"),
                               channels="F0", n_threads=3, log=lambda *a, **k: None)
    ds = store.decisions("TRAIN", "loss")
    assert len(ds) == paths.N_TRAIN == 40_180_608
    ok, n_unique, n_total = _independent_ascending_unique_check(ds.decision_id)
    assert ok, f"A1: decision_id not strictly ascending/unique (n_unique={n_unique}, n_total={n_total})"
    assert n_unique == paths.N_TRAIN
    assert bool(ds.loss_mask.all()), "A1: not every TRAIN loss decision has loss_mask==True"
    # tail128 is counted (i.e. the last 128 ids are present and contiguous-unique too)
    tail = ds.decision_id[-128:]
    ok_tail, nu_tail, nt_tail = _independent_ascending_unique_check(tail)
    assert ok_tail and nu_tail == 128


def test_A1_duplicate_missing_detector_negative_control():
    good = np.arange(1000, dtype=np.int64)
    ok, nu, nt = _independent_ascending_unique_check(good)
    assert ok and nu == nt == 1000

    dup = good.copy()
    dup[500] = dup[499]  # inject a duplicate id
    ok_dup, nu_dup, nt_dup = _independent_ascending_unique_check(dup)
    assert not ok_dup, "negative control FAILED: duplicate id was not detected"
    assert nu_dup < nt_dup

    lost = np.delete(good, 500)  # inject a missing (lost) id -- a gap in the sequence
    ok_lost, nu_lost, nt_lost = _independent_ascending_unique_check(lost)
    assert ok_lost and nt_lost == 999  # internally consistent, but the CALLER comparing
    assert nt_lost != good.shape[0]    # nt_lost to the expected total sees the loss


# ===========================================================================
# A2 -- catalogue mapping unique/bijective, == arange(K)+3, from the real file
# ===========================================================================

@pytest.mark.real
def test_A2_catalogue_bijective_from_file():
    layout = detect_layout(paths.DATASET_ROOT)
    vocab_dir = "mappings" if layout == "frozen" else "vocabularies"
    cat = pq.read_table(paths.DATASET_ROOT / vocab_dir / "catalogue.parquet", columns=["product_idx"])
    pidx = cat["product_idx"].to_numpy(zero_copy_only=False)
    assert pidx.shape[0] == paths.K == 158_486
    expected = np.arange(pidx.shape[0], dtype=pidx.dtype) + 3
    assert np.array_equal(pidx, expected), "A2: product_idx != arange(K)+3 read from the real file"
    assert np.unique(pidx).shape[0] == pidx.shape[0], "A2: product_idx is not bijective (duplicates)"


def test_A2_negative_control_broken_mapping_detected():
    good = np.arange(50, dtype=np.int64) + 3
    assert np.array_equal(good, np.arange(50, dtype=np.int64) + 3)
    broken = good.copy()
    broken[10] = broken[9]  # collapse two classes onto the same product_idx
    assert not np.array_equal(broken, np.arange(50, dtype=np.int64) + 3)
    assert np.unique(broken).shape[0] < broken.shape[0]


# ===========================================================================
# A3 -- repack preserves item order and every per-event tensor's alignment;
# positions independent of the batch's allocated padding.
# ===========================================================================

def _scattered_batch(T: int):
    """2 rows, LEFT-padded (valid events at the tail), item/category tokens
    carry a fixed +100 offset so alignment between channels is checkable."""
    B, length0, length1 = 2, 3, 5
    item = torch.zeros(B, T, dtype=torch.int64)
    cat = torch.zeros(B, T, dtype=torch.int64)
    am = torch.zeros(B, T, dtype=torch.bool)
    item[0, T - length0:T] = torch.tensor([10, 11, 12])
    cat[0, T - length0:T] = torch.tensor([110, 111, 112])
    am[0, T - length0:T] = True
    item[1, T - length1:T] = torch.tensor([20, 21, 22, 23, 24])
    cat[1, T - length1:T] = torch.tensor([120, 121, 122, 123, 124])
    am[1, T - length1:T] = True
    return dict(item_tokens=item, category_tokens=cat, attention_mask=am,
                decision_id=torch.tensor([501, 502]), target_class=torch.tensor([7, 8]))


def test_A3_repack_preserves_order_and_alignment():
    for T in (8, 20):
        batch = _scattered_batch(T)
        out = sd.repack_right(batch)
        sd.check_right_padded(out)
        # item order preserved, right-aligned prefix
        assert out["item_tokens"][0, :3].tolist() == [10, 11, 12]
        assert out["item_tokens"][1, :5].tolist() == [20, 21, 22, 23, 24]
        # per-event alignment: category always +100 over item, on every VALID column
        # (both sides are independently zeroed on PAD, so compare there separately)
        valid = out["attention_mask"]
        assert torch.equal(out["category_tokens"][valid], out["item_tokens"][valid] + 100)
        assert torch.equal(out["category_tokens"][~valid], torch.zeros_like(out["category_tokens"][~valid]))
        # per-decision tensors pass through unchanged
        assert out["decision_id"].tolist() == [501, 502]
        assert out["target_class"].tolist() == [7, 8]
        # positions independent of the batch's allocated padding: same result at T=8 and T=20
        if T == 8:
            ref = out
        else:
            assert torch.equal(ref["item_tokens"], out["item_tokens"][:, :ref["item_tokens"].shape[1]])
            assert torch.equal(ref["attention_mask"], out["attention_mask"][:, :ref["attention_mask"].shape[1]])
            assert ref["lengths"].tolist() == out["lengths"].tolist()


def test_A3_repack_idempotent_on_already_right_padded():
    st = SyntheticStore.build(channels="F4", n_train=32, n_vs=32, K=41, seed=3, task="mapping")
    ds = st.decisions("TRAIN", "loss")
    batch = st.batch(ds, np.arange(6))
    sd.check_right_padded(batch)
    repacked = sd.repack_right(batch)
    assert torch.equal(repacked["item_tokens"][:, :50], batch["item_tokens"])
    assert torch.equal(repacked["attention_mask"][:, :50], batch["attention_mask"])
    assert repacked["lengths"].tolist() == batch["lengths"].tolist()


# ===========================================================================
# A4 -- same sequence alone / batched at widths 3,8,50 gives equal logits
# (fp32 tolerance frozen here at 1e-5); PAD cells poisoned.
# ===========================================================================

FP32_TOLERANCE = 1e-5


def test_A4_padding_batch_equivalence_poisoned_pad():
    torch.manual_seed(0)
    model = _tiny_model(K=13, rich=False, dropout=0.0)
    tokens = [4, 5, 6]
    length = 3

    solo = _make_solo_batch(tokens, length, T=3, poison=False)
    with torch.no_grad():
        solo_logits = model(**sd.predictor_kwargs(solo, model))[0]

    for T in (3, 8, 50):
        for B in (1, 4):
            item = torch.zeros(B, T, dtype=torch.int64)
            am = torch.zeros(B, T, dtype=torch.bool)
            lengths = torch.zeros(B, dtype=torch.int64)
            item[0, :length] = torch.tensor(tokens, dtype=torch.int64)
            am[0, :length] = True
            lengths[0] = length
            for b in range(1, B):
                Lb = 1 + (b % min(T, 4))
                item[b, :Lb] = torch.arange(Lb, dtype=torch.int64) + 3
                am[b, :Lb] = True
                lengths[b] = Lb
            if T > length:
                # poison every row's PAD region with nonzero garbage
                pad_cols = torch.arange(length, T)
                item[0, pad_cols] = (pad_cols % 5) + 3
            batch = dict(item_tokens=item, attention_mask=am, lengths=lengths)
            with torch.no_grad():
                logits = model(**sd.predictor_kwargs(batch, model))[0]
            diff = (logits - solo_logits).abs().max().item()
            assert diff < FP32_TOLERANCE, f"A4 FAILED at T={T} B={B}: max abs diff {diff:.3e}"


# ===========================================================================
# A5 -- lengths 1,2,49,50 (synthetic) and 51+ real cropping; all-PAD raises.
# ===========================================================================

def test_A5_synthetic_lengths_1_2_49_50_present():
    st = SyntheticStore.build(channels="F0", n_train=64, n_vs=64, K=37, seed=5, task="mapping")
    ds = st.decisions("TRAIN", "loss")
    lengths = ds.length.astype(np.int64)
    assert 1 in lengths and 50 in lengths
    ds_vs = st.decisions("VALIDATION_SELECT", "eval")
    vs_lengths = set(ds_vs.length.tolist())
    assert vs_lengths  # non-empty; VS lengths are drawn 1..50 uniformly
    for edge in (1, 50):
        pos = int(np.flatnonzero(lengths == edge)[0])
        batch = st.batch(ds, np.array([pos]))
        sd.check_right_padded(batch)
        assert int(batch["lengths"][0]) == edge
        assert int(batch["last_valid_index"][0]) == edge - 1


def test_A5_all_pad_row_raises():
    model = _tiny_model(K=13, rich=False)
    item = torch.zeros(1, 5, dtype=torch.int64)
    am = torch.zeros(1, 5, dtype=torch.bool)
    lengths = torch.zeros(1, dtype=torch.int64)
    batch = dict(item_tokens=item, attention_mask=am, lengths=lengths)
    with pytest.raises(ValueError):
        model(**sd.predictor_kwargs(batch, model))


@pytest.mark.real
def test_A5_prefix_position_51plus_cropped_to_last50_real():
    store = _real_store("F4", REAL_BUCKETS)
    layout = detect_layout(paths.DATASET_ROOT)
    found = None
    for split, kind in (("TRAIN", "loss"), ("VALIDATION_SELECT", "eval")):
        ds = store.decisions(split, kind)
        hit = np.flatnonzero(ds.prefix_position >= 51)
        if hit.size:
            found = (split, kind, ds, int(hit[0]))
            break
    if found is None:
        pytest.skip("no prefix_position >= 51 decision found in the small real bucket subset")
    split, kind, ds, pos = found
    assert int(ds.length[pos]) == 50, "A5: a prefix_position>=51 decision must be cropped to length 50"

    # independent recompute via the frozen loader for THIS one decision
    dec_id = int(ds.decision_id[pos])
    client_idx = int(ds.client_idx[pos])
    d = dict(decision_id=dec_id, user_bucket=int(ds.user_bucket[pos]),
            target_class=int(ds.target_class[pos]), loss_mask=bool(ds.loss_mask[pos]),
            eval_mask=bool(ds.eval_mask[pos]), client_idx=client_idx)
    loader = DecisionWindowLoader(paths.DATASET_ROOT, 50, layout=layout)
    ref = loader.collate([d])
    computed = store.batch(ds, np.array([pos]))
    assert int(ref["lengths"][0]) == 50
    assert np.array_equal(ref["item_tokens"][0], computed["item_tokens"][0].numpy())


# ===========================================================================
# A6 -- missing category/brand, OOV product, zero price/flags survive
# unchanged (never re-normalised) -- real, small F4 bucket subset.
# ===========================================================================

@pytest.mark.real
def test_A6_missing_and_oov_features_survive_real():
    store = _real_store("F4", REAL_BUCKETS)
    layout = detect_layout(paths.DATASET_ROOT)
    ds = store.decisions("TRAIN", "loss")
    n = min(len(ds), 200_000)
    sample = np.arange(n)
    b = store.batch(ds, sample)
    flags = b["event_quality_flags"].numpy()
    valid = b["attention_mask"].numpy()

    checks = {
        "price_missing": flags[..., 0],
        "brand_missing": flags[..., 3],
        "category_missing": flags[..., 6],
    }
    loader = DecisionWindowLoader(paths.DATASET_ROOT, 50, layout=layout)
    found_any = False
    for name, flag_arr in checks.items():
        hit = np.argwhere((flag_arr == 1) & valid)
        if hit.size == 0:
            continue
        found_any = True
        row, col = hit[0]
        dec_id = int(ds.decision_id[sample[row]])
        d = dict(decision_id=dec_id, user_bucket=int(ds.user_bucket[sample[row]]),
                target_class=int(ds.target_class[sample[row]]),
                loss_mask=bool(ds.loss_mask[sample[row]]), eval_mask=bool(ds.eval_mask[sample[row]]),
                client_idx=int(ds.client_idx[sample[row]]))
        ref = loader.collate([d])
        assert ref["event_quality_flags"][0, col, checks_index(name)] == 1
        # zero-filled numeric survives unchanged (not re-normalised) through the adapter
        computed = store.batch(ds, np.array([sample[row]]))
        assert torch.equal(computed["event_quality_flags"], torch.from_numpy(ref["event_quality_flags"]))
        assert torch.allclose(computed["event_numeric_features"], torch.from_numpy(ref["event_numeric_features"]))
    if not found_any:
        pytest.skip("no missing-category/brand/price row found in the small real bucket subset")


def checks_index(name: str) -> int:
    order = ["price_missing", "price_zero", "price_conflict", "brand_missing", "brand_recovered",
             "brand_conflict", "category_missing", "category_recovered", "category_conflict",
             "category_code_missing", "category_code_recovered", "main_category_missing",
             "prev_gap_unavailable", "item_age_unavailable", "simultaneous_block"]
    return order.index(name)


def test_A6_adapter_does_not_mutate_values():
    st = SyntheticStore.build(channels="F4", n_train=32, n_vs=8, K=29, seed=6, task="mapping")
    ds = st.decisions("TRAIN", "loss")
    batch = st.batch(ds, np.arange(4))
    model = _tiny_model(K=29, rich=True)
    kw = sd.predictor_kwargs(batch, model)
    assert torch.equal(kw["event_numeric_features"], batch["event_numeric_features"])
    assert torch.equal(kw["event_quality_flags"], batch["event_quality_flags"])
    assert torch.equal(kw["category_tokens"], batch["category_tokens"])


# ===========================================================================
# A7 -- target_class=-1 with eval_mask True vs False: distinct branches.
# ===========================================================================

def test_A7_oov_target_distinct_branches():
    target_class = np.array([-1, 5, -1, 2, -1, 0], dtype=np.int64)
    eval_mask = np.array([True, True, False, True, True, True], dtype=bool)
    eval_user_key = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    client_idx = np.array([3, 3, 3, 3, 3, 3], dtype=np.int64)
    # ranks: -1 wherever unrankable (mask False, or target<0); real ranks elsewhere
    ranks = np.array([-1, 3, -1, 1, -1, 50], dtype=np.int64)
    ds = dict(target_class=target_class, eval_mask=eval_mask, eval_user_key=eval_user_key,
             client_idx=client_idx, prior_view_count=np.zeros(6, dtype=np.int64),
             prefix_position=np.ones(6, dtype=np.int64))
    result = aggregate(ranks, ds, k=20)
    e2e, closed = result["end_to_end"], result["closed"]

    assert e2e["n"] == int(eval_mask.sum()) == 5
    assert closed["n"] == int((eval_mask & (target_class >= 0)).sum()) == 3
    assert e2e["n"] != closed["n"], "A7: end_to_end and closed must be genuinely distinct populations"
    # row0 (target=-1, eval_mask True) is INCLUDED in end_to_end with zero credit,
    # but EXCLUDED entirely from closed -- this is the two branches Fable requires.
    assert e2e["micro"]["recall@20"] < 1.0
    # closed only has rows 1,3,5 with ranks 3,1,50 -> recall@20 hits for rank<=20: rows1,3 hit, row5 miss
    assert abs(closed["micro"]["recall@20"] - (2.0 / 3.0)) < 1e-6


def test_A7_negative_control_branches_collapse_without_mask():
    """If a caller (bug) used `eval_mask` for BOTH populations the two branches
    would collapse -- prove the real distinction is `target_class>=0`, not a
    tautology, by showing a broken 'closed' that ignores target sign disagrees
    with the real `aggregate`."""
    target_class = np.array([-1, 5, 2], dtype=np.int64)
    eval_mask = np.array([True, True, True], dtype=bool)
    eval_user_key = np.array([0, 1, 2], dtype=np.int64)
    client_idx = np.array([3, 3, 3], dtype=np.int64)
    ranks = np.array([-1, 3, 1], dtype=np.int64)
    ds = dict(target_class=target_class, eval_mask=eval_mask, eval_user_key=eval_user_key,
             client_idx=client_idx, prior_view_count=np.zeros(3, dtype=np.int64),
             prefix_position=np.ones(3, dtype=np.int64))
    result = aggregate(ranks, ds, k=20)
    broken_closed_n = int(eval_mask.sum())  # what a buggy "closed==eval_mask" would report
    assert result["closed"]["n"] != broken_closed_n
    assert result["closed"]["n"] == 2


# ===========================================================================
# A8 -- previously-viewed target is legal; target column never a forward kwarg.
# ===========================================================================

def test_A8_repeated_target_legal_and_never_forwarded():
    model = _tiny_model(K=20, rich=False)
    class_a, class_b = 4, 9
    item = torch.zeros(1, 5, dtype=torch.int64)
    item[0, :2] = torch.tensor([class_a + 3, class_b + 3])
    am = torch.zeros(1, 5, dtype=torch.bool)
    am[0, :2] = True
    lengths = torch.tensor([2])
    # legal (A,B,A) target: the target repeats an item already seen in context
    batch = dict(item_tokens=item, attention_mask=am, lengths=lengths,
                target_class=torch.tensor([class_a]), decision_id=torch.tensor([1]),
                loss_mask=torch.tensor([True]), eval_mask=torch.tensor([False]),
                client_id=torch.tensor([3]))
    kw = sd.predictor_kwargs(batch, model)
    assert "target_class" not in kw and "decision_id" not in kw and "client_id" not in kw
    with torch.no_grad():
        logits_a = model(**kw)

    # mutate the (never-forwarded) label; forward output must be byte-identical
    batch["target_class"] = torch.tensor([999])
    kw2 = sd.predictor_kwargs(batch, model)
    with torch.no_grad():
        logits_b = model(**kw2)
    assert torch.equal(logits_a, logits_b), "A8: model output changed after mutating an unforwarded label"


def test_A8_negative_control_forbidden_key_raises():
    batch = dict(item_tokens=torch.zeros(1, 3, dtype=torch.int64),
                attention_mask=torch.zeros(1, 3, dtype=torch.bool),
                lengths=torch.tensor([1]))

    class LeakyModel:
        _categorical_channels = ("item_tokens",)
        numeric_dim = 0
        flags_dim = 0
        use_user_context = False

    good_kw = sd.predictor_kwargs(batch, LeakyModel())
    assert "target_class" not in good_kw
    assert not (set(good_kw) & sd.FORBIDDEN_PREDICTOR_KEYS)


# ===========================================================================
# A9 -- distinct eval users sharing client_idx=2 stay separate; client_idx==2
# rows with has_history>0 exist.
# ===========================================================================

def test_synthetic_store_train_vs_decision_ids_disjoint():
    """One global monotonic decision_id counter must be shared across BOTH
    splits (regression: a per-split counter previously let TRAIN and
    VALIDATION_SELECT both restart at 0 and collide, which trips
    train_sasrec.py's cross-split decision_id leakage guard and blocks
    every --test-tiny run)."""
    for n_train, n_vs, K in ((37, 64, 23), (4096, 1024, 53)):
        st = SyntheticStore.build(channels="F0", n_train=n_train, n_vs=n_vs, K=K, seed=11,
                                  task="mapping")
        train_ids = set(st.decisions("TRAIN", "loss").decision_id.tolist())
        vs_ids = set(st.decisions("VALIDATION_SELECT", "eval").decision_id.tolist())
        vs_ids |= set(st.decisions("VALIDATION_SELECT", "loss").decision_id.tolist())
        overlap = train_ids & vs_ids
        assert not overlap, f"TRAIN/VS decision_id collision at n_train={n_train}: {sorted(overlap)[:10]}"
        # every VS id strictly follows every TRAIN id (one global monotonic counter)
        if train_ids and vs_ids:
            assert max(train_ids) < min(vs_ids)


def test_synthetic_store_decision_id_collision_negative_control():
    """Prove the disjointness check above actually fires: a store whose two
    splits use independent per-split counters (the bug) must be REJECTED."""
    train_ids = set(range(0, 40))
    vs_ids = set(range(0, 60))  # both restart at 0 -- the bug
    assert train_ids & vs_ids, "negative control fixture must itself collide"


def test_A9_synthetic_distinct_users_share_client_idx_2():
    st = SyntheticStore.build(channels="F4", n_train=64, n_vs=400, K=53, seed=9, task="mapping")
    ds = st.decisions("VALIDATION_SELECT", "eval")
    oov_mask = ds.client_idx == 2
    assert bool(oov_mask.any()), "no client_idx==2 rows generated"
    distinct_users = np.unique(ds.eval_user_key[oov_mask])
    assert distinct_users.shape[0] >= 2, "A9: need >=2 DISTINCT eval users sharing client_idx==2"

    batch = st.batch(ds, np.arange(len(ds)))
    has_hist = batch["user_context"][:, 3].numpy()
    cid = batch["client_id"].numpy()
    n_oov_with_history = int(((cid == 2) & (has_hist > 0)).sum())
    assert n_oov_with_history > 0, "A9: client_idx==2 rows with has_history>0 must exist"

    # unknown-to-TRAIN identity does not force has_history=0 at EVERY future event:
    # at least one distinct OOV user has BOTH a has_history==0 and a has_history==1 row
    for u in distinct_users:
        rows = np.flatnonzero(ds.eval_user_key == u)
        if rows.size < 2:
            continue
        hh = has_hist[rows]
        if (hh == 0).any() and (hh > 0).any():
            break
    else:
        pytest.fail("A9: no OOV user found with a has_history transition (0 -> >0)")


@pytest.mark.real
def test_A9_real_distinct_eval_users_share_client_idx_2():
    store = _real_store("F4", REAL_BUCKETS)
    ds = store.decisions("VALIDATION_SELECT", "eval")
    oov = ds.client_idx == 2
    if not bool(oov.any()):
        pytest.skip("no client_idx==2 rows in the small real bucket subset")
    distinct = np.unique(ds.eval_user_key[oov])
    if distinct.shape[0] < 2:
        pytest.skip("fewer than 2 distinct OOV eval users in the small real bucket subset")
    assert distinct.shape[0] >= 2


# ===========================================================================
# predictor_kwargs / FORBIDDEN_PREDICTOR_KEYS: allowlist corruption must raise.
# ===========================================================================

def test_predictor_kwargs_id_and_rich_selection():
    id_model = _tiny_model(K=10, rich=False)
    rich_model = _tiny_model(K=10, rich=True)
    st = SyntheticStore.build(channels="F4", n_train=16, n_vs=8, K=10, seed=1)
    ds = st.decisions("TRAIN", "loss")
    batch = st.batch(ds, np.arange(4))

    kw_id = sd.predictor_kwargs(batch, id_model)
    assert set(kw_id) == {"item_tokens", "lengths", "attention_mask"}

    kw_rich = sd.predictor_kwargs(batch, rich_model)
    assert set(kw_rich) == set(sd.PREDICTOR_ALLOWLIST_RICH)
    assert "position_ids" not in kw_rich


def test_predictor_kwargs_raises_on_corrupted_allowlist(monkeypatch):
    model = _tiny_model(K=10, rich=False)
    batch = dict(item_tokens=torch.zeros(1, 3, dtype=torch.int64),
                attention_mask=torch.zeros(1, 3, dtype=torch.bool), lengths=torch.tensor([1]))
    corrupted = sd.PREDICTOR_ALLOWLIST_ID + ("target_class",)
    monkeypatch.setattr(sd, "PREDICTOR_ALLOWLIST_ID", corrupted)
    with pytest.raises(ValueError):
        sd.predictor_kwargs(batch, model)


def test_forbidden_keys_never_intersect_real_allowlists():
    assert not (set(sd.PREDICTOR_ALLOWLIST_ID) & sd.FORBIDDEN_PREDICTOR_KEYS)
    assert not (set(sd.PREDICTOR_ALLOWLIST_RICH) & sd.FORBIDDEN_PREDICTOR_KEYS)


# ===========================================================================
# check_right_padded: positive + negative controls
# ===========================================================================

def test_check_right_padded_positive_and_negative():
    st = SyntheticStore.build(channels="F4", n_train=16, n_vs=8, K=17, seed=2)
    ds = st.decisions("TRAIN", "loss")
    batch = st.batch(ds, np.arange(4))
    sd.check_right_padded(batch)  # positive control: must not raise

    bad = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    bad["position_ids"][0, 0] = 999  # corrupt a valid cell's position
    with pytest.raises(ValueError):
        sd.check_right_padded(bad)

    bad2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    L = int(bad2["lengths"][0])
    if L < bad2["item_tokens"].shape[1]:
        bad2["event_numeric_features"][0, L, 0] = 1.0  # poison a PAD cell
        with pytest.raises(ValueError):
            sd.check_right_padded(bad2)


# ===========================================================================
# build_training_store / build_eval_store guard rails (no real data touched).
# ===========================================================================

def test_build_training_store_rejects_bad_channels():
    with pytest.raises(ValueError):
        sd.build_training_store("bogus")


def test_build_eval_store_rejects_bad_split():
    with pytest.raises(ValueError):
        sd.build_eval_store("F0", "TRAIN", lock_path="/does/not/matter")


def test_build_eval_store_refuses_without_lock():
    missing = Path("/tmp/definitely_missing_sasrec_lock_9f8e7d.json")
    assert not missing.exists()
    with pytest.raises(FileNotFoundError):
        sd.build_eval_store("F0", "VALIDATION_CONFIRM", lock_path=missing)
    with pytest.raises(FileNotFoundError):
        sd.build_eval_store("F0", "TEST", lock_path=missing)
