"""The MBHT session converter on a small synthetic fixture (CPU, no real data)."""
import pandas as pd
import pytest

from ppsi.benchmarks import check_layout, mbht


# ------------------------------------------------------------------------------------------------ MBHT fixture
def _inter(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("session_id:token\titem_id_list:token_seq\titem_type_list:token_seq\titem_id:token\n")
        f.writelines(f"{sid}\t{' '.join(map(str, il))}\t{' '.join(map(str, tl))}\t{t}\n" for sid, il, tl, t in rows)


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
    v = check_layout.verify_layout(out)
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
    assert check_layout.verify_layout(tmp_path / "o")["holdout_in_user_train_pairs"] == 1


