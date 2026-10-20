"""BM-W2 tests: converters on small fixtures + sampled-metric math vs brute force. CPU only, no real data."""
import gzip
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4bench_w2 import amazon2023, fmt, mbht, sampled_eval as se, verify  # noqa: E402

WAVE1_COLUMNS = ["user_id", "item_id", "rating", "datetime", "weight"]            # ml_1m / ml_20m release
WAVE1_STAT_COLUMNS = ["n_users", "n_items", "n_interactions", "avg_length", "sparsity", "min_date", "max_date",
                      "leave_one_out_n_train_interactions", "leave_one_out_n_holdout_interactions",
                      "leave_one_out_holdout_ratio", "loo_correction_leave_one_out"]


# ------------------------------------------------------------------------------------------------ Amazon fixture
def _write_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp", "history"])
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)


@pytest.fixture()
def amazon_raw(tmp_path):
    raw = tmp_path / "Amazon2023-Toy"
    seqs = {"UB": ["i3", "i1", "i2", "i4", "i9"], "UA": ["i1", "i2", "i3", "i5", "i6", "i7"], "UC": ["i2", "i8", "i1", "i4", "i5"]}
    tr, va, te = [], [], []
    for n, (u, items) in enumerate(seqs.items()):
        ts = [1_600_000_000_000 + 1000 * (100 * n + k) + 7 * k for k in range(len(items))]
        for k, it in enumerate(items[:-2]):
            tr.append((u, it, 4.0, ts[k], ""))
        va.append((u, items[-2], 5.0, ts[-2], ""))
        te.append((u, items[-1], 3.0, ts[-1], ""))
    for part, rows in (("train", tr), ("valid", va), ("test", te)):
        _write_gz(raw / "last_out_w_his" / f"Toy.{part}.csv.gz", rows)
    # timestamp variant: a global-time split, user UA has two test rows
    allrows = sorted([r for r in tr + va + te], key=lambda r: r[3])
    n = len(allrows)
    t_tr, t_va, t_te = allrows[: n - 6], allrows[n - 6: n - 3], allrows[n - 3:]
    for part, rows in (("train", t_tr), ("valid", t_va), ("test", t_te)):
        _write_gz(raw / "timestamp_w_his" / f"Toy.{part}.csv.gz", rows)
    return raw, seqs


def test_amazon_counts_one_holdout_no_leak_format(amazon_raw, tmp_path):
    raw, seqs = amazon_raw
    out = tmp_path / "out"
    amazon2023.CATEGORIES["Toy"] = "amazon2023_toy"
    s = amazon2023.convert_last_out(raw, "Toy", out)
    n_rows = sum(len(v) for v in seqs.values())
    assert s["n_users"] == 3 and s["n_interactions"] == n_rows
    assert s["train_rows"] == n_rows - 3 and s["holdout_rows"] == 3          # train + valid vs one test row per user
    assert s["source_rows"] == {"train": n_rows - 6, "valid": 3, "test": 3}
    v = verify.verify_layout(out)
    assert v["columns_ok"] and v["stat_columns_ok"] and v["dtypes_ok"] and v["stats_match"] and v["train_sorted_by_user"]
    assert v["one_holdout_per_user"] and v["holdout_users_in_train"]
    assert v["holdout_in_user_train_pairs"] == 0                              # holdout item never in the user's train
    # semantics: holdout = official test row (last item), train keeps valid as its last row
    um = pd.read_csv(out / "user_map.csv"); im = pd.read_csv(out / "item_map.csv")
    tr = pd.read_csv(out / "leave_one_out" / "train.csv"); ho = pd.read_csv(out / "leave_one_out" / "holdout.csv")
    for _, h in ho.iterrows():
        u = um.set_index("user_id").orig_user_id[h.user_id]
        assert im.set_index("item_id").orig_parent_asin[h.item_id] == seqs[u][-1]
        mine = tr[tr.user_id == h.user_id]
        assert [im.set_index("item_id").orig_parent_asin[i] for i in mine.item_id] == seqs[u][:-1]
        assert mine.datetime.is_monotonic_increasing and mine.datetime.iloc[-1] < h.datetime
    # sha sums exist and match
    sums = dict(l.split("  ", 1)[::-1] for l in (out / "SHA256SUMS").read_text().splitlines())
    assert fmt.sha256_file(out / "leave_one_out" / "holdout.csv") == sums["leave_one_out/holdout.csv"].strip()


def test_amazon_datetime_is_ms_utc(amazon_raw, tmp_path):
    raw, _ = amazon_raw
    amazon2023.CATEGORIES["Toy"] = "amazon2023_toy"
    amazon2023.convert_last_out(raw, "Toy", tmp_path / "o")
    ho = pd.read_csv(tmp_path / "o" / "leave_one_out" / "holdout.csv")
    assert ho.datetime.str.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}$").all()
    assert fmt.dt_from_ms([1_600_000_000_123])[0] == "2020-09-13 12:26:40.123"


def test_amazon_timestamp_variant_is_not_one_per_user(amazon_raw):
    raw, _ = amazon_raw
    d = amazon2023.timestamp_diagnostics(raw, "Toy")
    assert d["users_with_test_rows"] < d["users_total"] or d["users_with_gt1_test_rows"] > 0
    assert d["test_ts_min"] > d["valid_ts_max"] > d["train_ts_max"]


def test_amazon_rejects_two_test_rows(amazon_raw, tmp_path):
    raw, _ = amazon_raw
    p = raw / "last_out_w_his" / "Toy.test.csv.gz"
    df = pd.read_csv(p)
    with gzip.open(p, "wt") as f:
        pd.concat([df, df.iloc[:1]]).to_csv(f, index=False)
    with pytest.raises(ValueError):
        amazon2023.convert_last_out(raw, "Toy", tmp_path / "o")


def test_format_identical_to_wave1_declaration():
    assert fmt.COLUMNS == WAVE1_COLUMNS and fmt.STAT_COLUMNS == WAVE1_STAT_COLUMNS
    # statistics definitions reproduce the Wave-1 ml_1m row (6040 users / 3706 items / 1,000,209 inter / 994,169 + 6,038)
    n_inter, U, I, t, h = 1000209, 6040, 3706, 994169, 6038
    assert abs(n_inter / U - 165.5975165562914) < 1e-9
    assert abs((1 - n_inter / (U * I)) - 0.9553163743776872) < 1e-12
    assert abs(h / (t + h) - 0.0060367503926687) < 1e-12 and abs(h / U - 0.9996688741721854) < 1e-12


# ------------------------------------------------------------------------------------------------ MBHT fixture
def _inter(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("session_id:token\titem_id_list:token_seq\titem_type_list:token_seq\titem_id:token\n")
        for sid, il, tl, t in rows:
            f.write(f"{sid}\t{' '.join(map(str, il))}\t{' '.join(map(str, tl))}\t{t}\n")


@pytest.fixture()
def mbht_raw(tmp_path):
    d = tmp_path / "MBHT_dataset" / "tmall_beh"
    # session 1: three augmented rows (history before each purchase); session 2: one row
    s1_full = [11, 12, 13, 14, 15, 16, 17]
    rows = [(1, [11, 12], [2, 2], 13), (1, [11, 12, 13, 14], [2, 2, 4, 1], 15), (1, [11, 12, 13, 14, 15, 16], [2, 2, 4, 1, 4, 0], 17),
            (2, [21, 22, 23], [0, 1, 2], 24)]
    _inter(d / "tmall_beh.train.inter", rows)
    _inter(d / "tmall_beh.test.inter", [(1, [11, 31, 32, 33], [0, 1, 2, 3], 34), (2, [41, 42, 43], [2, 2, 2], 11)])
    return d, s1_full


def test_mbht_longest_plus_target_and_holdout(mbht_raw, tmp_path):
    d, s1 = mbht_raw
    out = tmp_path / "o"
    s = mbht.convert(d, "tmall_beh", out)
    assert s["dataset"] == "mbht_taobao" and s["train_sessions"] == 2 and s["test_sessions"] == 2 and s["n_users"] == 4
    tr = pd.read_csv(out / "leave_one_out" / "train.csv"); ho = pd.read_csv(out / "leave_one_out" / "holdout.csv")
    # counts: train sessions = longest + target = 7 and 4 events; test histories = 4 and 3
    assert len(tr) == 7 + 4 + 4 + 3 and len(ho) == 2
    assert tr[tr.user_id == 1].item_id.tolist() == s1                                  # longest list + its target
    assert tr[tr.user_id == 2].item_id.tolist() == [21, 22, 23, 24]
    assert ho.user_id.tolist() == [3, 4] and ho.item_id.tolist() == [34, 11]          # MBHT test targets
    assert tr[tr.user_id == 3].item_id.tolist() == [11, 31, 32, 33]
    v = verify.verify_layout(out)
    assert v["columns_ok"] and v["dtypes_ok"] and v["stats_match"] and v["one_holdout_per_user"] and v["holdout_users_in_train"]
    assert v["holdout_in_user_train_pairs"] == 0 and s["holdout_item_in_user_train"] == 0
    # holdout comes after the user's history; side file keeps the behaviour type per interaction
    side = pd.read_csv(out / "item_type.csv")
    assert len(side) == len(tr) + len(ho) and set(side.part) == {"train", "holdout"}
    assert side[side.user_id == 1].item_type.tolist() == [2, 2, 4, 1, 4, 0, 4]
    assert side[(side.user_id == 3) & (side.part == "holdout")].item_type.tolist() == [4]
    assert pd.read_csv(out / "user_map.csv").orig_session_id.tolist() == [1, 2, 1, 2]


def test_mbht_leak_is_counted_not_hidden(mbht_raw, tmp_path):
    d, _ = mbht_raw
    _inter(d / "tmall_beh.test.inter", [(1, [11, 31, 32, 33], [0, 1, 2, 3], 31)])
    s = mbht.convert(d, "tmall_beh", tmp_path / "o")
    assert s["holdout_item_in_user_train"] == 1
    assert verify.verify_layout(tmp_path / "o")["holdout_in_user_train_pairs"] == 1


# ------------------------------------------------------------------------------------------------ sampled metrics
def _brute_expected(rank, n_cand, N, ks):
    """Enumerate every size-N subset of the other n_cand-1 items (r-1 better, rest worse)."""
    M = n_cand - 1
    items = [1] * (rank - 1) + [0] * (M - rank + 1)                     # 1 = better than the target
    hr = {k: 0.0 for k in ks}; nd = {k: 0.0 for k in ks}; mrr = 0.0; cnt = 0
    for sub in itertools.combinations(range(M), N):
        sr = 1 + sum(items[i] for i in sub)
        cnt += 1
        mrr += 1.0 / sr
        for k in ks:
            if sr <= k:
                hr[k] += 1; nd[k] += 1 / math.log2(sr + 1)
    return {**{f"HR@{k}": hr[k] / cnt for k in ks}, **{f"NDCG@{k}": nd[k] / cnt for k in ks}, "MRR": mrr / cnt}


def test_expected_matches_brute_force_enumeration():
    ks = (1, 2, 3, 5)
    for rank, ncand, N in [(1, 9, 3), (4, 9, 3), (9, 9, 3), (3, 7, 6), (2, 12, 4), (5, 8, 7)]:
        got = se.expected_sampled_metrics([rank], [ncand], N, ks)
        want = _brute_expected(rank, ncand, N, ks)
        for k_, v in want.items():
            assert got[k_] == pytest.approx(v, abs=1e-12), (rank, ncand, N, k_)


def test_expected_averages_over_users_and_clips_n():
    r, nc = np.array([1, 4, 9]), np.array([9, 9, 9])
    got = se.expected_sampled_metrics(r, nc, 100, (5,))                  # N > M -> all items used, rank unchanged
    assert got["HR@5"] == pytest.approx(np.mean([1, 1, 0]))
    want = np.mean([1 / math.log2(rr + 1) if rr <= 5 else 0 for rr in r])
    assert got["NDCG@5"] == pytest.approx(want)


def _fixture(U=6, K=30, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(U, K))
    target = rng.integers(0, K, U)
    seen = [rng.choice([c for c in range(K) if c != target[u]], 5, replace=False) for u in range(U)]
    return scores, target, seen


def test_full_ranks_vs_sort_with_seen_filter_and_tie_rule():
    scores, target, seen = _fixture()
    scores[0, :] = 1.0                                                   # all tied: rank = 1 + #unseen cols below target
    r, nc = se.full_ranks(scores, target, seen)
    for u in range(len(target)):
        cand = [c for c in range(scores.shape[1]) if c == target[u] or c not in set(seen[u])]
        order = sorted(cand, key=lambda c: (-scores[u, c], c))
        assert r[u] == order.index(int(target[u])) + 1 and nc[u] == len(cand)


def test_sampled_uniform_matches_brute_force_and_is_seeded():
    scores, target, seen = _fixture()
    a = se.evaluate(scores, target, seen, scheme="uniform", n_neg=7, ks=(3, 5), seed=11)
    b = se.evaluate(scores, target, seen, scheme="uniform", n_neg=7, ks=(3, 5), seed=11)
    c = se.evaluate(scores, target, seen, scheme="uniform", n_neg=7, ks=(3, 5), seed=12)
    assert a == b and a["label"] == se.LABEL and a["metrics"] != c["metrics"]
    neg = se.draw_uniform(scores.shape[1], target, seen, 7, 11)
    ranks = []
    for u in range(len(target)):
        assert len(set(neg[u])) == 7 and target[u] not in neg[u] and not (set(neg[u]) & set(seen[u]))
        ranks.append(1 + sum(1 for n in neg[u] if (scores[u, n], -n) > (scores[u, target[u]], -target[u])))
    want = se.metrics_from_ranks(np.array(ranks), (3, 5))
    assert a["metrics"] == want


def test_uniform_mean_converges_to_expected_value():
    rng = np.random.default_rng(3)
    U, K, N = 4000, 60, 9
    scores = rng.normal(size=(U, K)); target = rng.integers(0, K, U)
    full, nc = se.full_ranks(scores, target, None)
    exp = se.expected_sampled_metrics(full, nc, N, (5, 10))
    emp = se.evaluate(scores, target, None, scheme="uniform", n_neg=N, ks=(5, 10), seed=1)["metrics"]
    for k in exp:
        assert emp[k] == pytest.approx(exp[k], abs=0.02)


def test_popularity_scheme_excludes_seen_target_and_follows_counts():
    scores, target, seen = _fixture(U=300, K=40)
    counts = np.arange(1, 41, dtype=float) ** 2
    neg = se.draw_popularity(counts, target, seen, 5, seed=5)
    assert all(len(set(n)) == 5 and t not in n and not (set(n) & set(s)) for n, t, s in zip(neg, target, seen))
    assert neg.mean() > 25                                               # skewed to high-count (high-index) items
    out = se.evaluate(scores, target, seen, scheme="popularity", n_neg=5, ks=(3,), seed=5, counts=counts)
    assert out["label"] == se.LABEL


def test_fixed_scheme_replaces_seen_and_keeps_rest():
    catalog = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    fixed = se.fixed_list_columns([20, 40, 999, 60], catalog)            # 999 not in catalogue -> -1
    assert fixed.tolist() == [1, 3, -1, 5]
    scores = np.random.default_rng(0).normal(size=(3, 8)); target = np.array([0, 2, 7])
    seen = [np.array([1]), np.array([], dtype=int), np.array([3, 5])]
    neg = se.draw_fixed(fixed, 8, target, seen, seed=9)
    assert neg.shape == (3, 4)
    assert neg[1].tolist()[:2] == [1, 3] and neg[1][3] == 5              # unseen fixed ids stay
    assert neg[0][0] != 1 and neg[0][0] not in {1}                       # seen id replaced by an unseen one
    assert all(int(x) not in set(seen[2].tolist()) for x in neg[2])
    assert (neg >= 0).all() and (neg < 8).all()
    out = se.evaluate(scores, target, seen, scheme="fixed", ks=(1, 2), seed=9, fixed_cols=fixed)
    assert out["n_neg"] == 4 and out["label"] == se.LABEL


def test_mbht_fixed_lists_have_100_ids():
    assert len(se.MBHT_FIXED_100["tmall_beh"]) == 100 and len(se.MBHT_FIXED_100["ijcai_beh"]) == 100
    assert len(set(se.MBHT_FIXED_100["tmall_beh"])) == 100


def test_label_on_every_output():
    out = se.evaluate_expected([2, 3], [10, 10], n_neg=4, ks=(2,))
    assert out["label"] == "SECONDARY / for comparison with sampled-metric papers only"
