"""The Amazon Reviews 2023 converter and the shared layout on a small synthetic fixture (CPU, no real data)."""
import gzip

import pandas as pd
import pytest

from ppsi.benchmarks import amazon2023, check_layout, layout

RELEASE_COLUMNS = ["user_id", "item_id", "rating", "datetime", "weight"]            # ml_1m / ml_20m release
RELEASE_STAT_COLUMNS = ["n_users", "n_items", "n_interactions", "avg_length", "sparsity", "min_date", "max_date",
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
    v = check_layout.verify_layout(out)
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
    assert layout.sha256_file(out / "leave_one_out" / "holdout.csv") == sums["leave_one_out/holdout.csv"].strip()


def test_amazon_datetime_is_ms_utc(amazon_raw, tmp_path):
    raw, _ = amazon_raw
    amazon2023.CATEGORIES["Toy"] = "amazon2023_toy"
    amazon2023.convert_last_out(raw, "Toy", tmp_path / "o")
    ho = pd.read_csv(tmp_path / "o" / "leave_one_out" / "holdout.csv")
    assert ho.datetime.str.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}$").all()
    assert layout.dt_from_ms([1_600_000_000_123])[0] == "2020-09-13 12:26:40.123"


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


def test_format_identical_to_the_release_declaration():
    assert layout.COLUMNS == RELEASE_COLUMNS and layout.STAT_COLUMNS == RELEASE_STAT_COLUMNS
    # statistics definitions reproduce the released ml_1m row (6040 users / 3706 items / 1,000,209 inter / 994,169 + 6,038)
    n_inter, U, I, t, h = 1000209, 6040, 3706, 994169, 6038
    assert abs(n_inter / U - 165.5975165562914) < 1e-9
    assert abs((1 - n_inter / (U * I)) - 0.9553163743776872) < 1e-12
    assert abs(h / (t + h) - 0.0060367503926687) < 1e-12 and abs(h / U - 0.9996688741721854) < 1e-12


