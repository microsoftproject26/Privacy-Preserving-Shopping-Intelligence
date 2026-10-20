"""BASE-01: `v4base.run.execute()` end to end (fit -> score -> evaluate -> BASELINE rows -> receipt) against
synthetic providers — the CLI-independent core `main()` wires to the real store on the server (never exercised
here; no `data_v3` manifests on this host, see company/handoffs/BASE-01.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from basekit import K, MINIMAL_KEY_BASE, FixtureEvalProvider, FixtureTrainProvider
from v4base import knn_emb as base_knn_emb
from v4base.guard import RssGuard
from v4base.run import execute


def _providers():
    return FixtureTrainProvider(n=80, K=K, seed=0), FixtureEvalProvider(n=40, K=K, seed=1)


def _check_receipt_and_rows(out_dir: Path, baseline: str, *, method: str = None) -> None:
    method = method or baseline
    rows = json.loads((out_dir / "rows.json").read_text(encoding="utf-8"))
    receipt = json.loads((out_dir / "RECEIPT.json").read_text(encoding="utf-8"))
    assert rows, "no rows written"
    assert all(r["run_label"] == "BASELINE" and r["method"] == method for r in rows)
    assert receipt["baseline"] == baseline
    assert receipt["n_rows"] == len(rows)
    assert receipt["v4base_code_sha256"]
    assert receipt["wall_s"] >= 0.0


def test_execute_pop(tmp_path):
    tp, ep = _providers()
    execute("POP", train_provider=tp, eval_provider=ep, key_base=MINIMAL_KEY_BASE, out_dir=tmp_path, K=K,
           guard=RssGuard())
    _check_receipt_and_rows(tmp_path, "POP")


def test_execute_last(tmp_path):
    tp, ep = _providers()
    execute("LAST", train_provider=tp, eval_provider=ep, key_base=MINIMAL_KEY_BASE, out_dir=tmp_path, K=K,
           guard=RssGuard())
    _check_receipt_and_rows(tmp_path, "LAST")


def test_execute_sknn_direct(tmp_path):
    tp, ep = _providers()
    execute("SKNN", train_provider=tp, eval_provider=ep, key_base=MINIMAL_KEY_BASE, out_dir=tmp_path, K=K,
           sknn_k=5, sknn_sample_size=20, guard=RssGuard())
    _check_receipt_and_rows(tmp_path, "SKNN")
    receipt = json.loads((tmp_path / "RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["config"]["sknn_k"] == 5 and receipt["config"]["sknn_sample_size"] == 20
    assert receipt["grid"] is None


def test_execute_sknn_grid(tmp_path):
    tp, ep = _providers()
    out = execute("SKNN", train_provider=tp, eval_provider=ep, key_base=MINIMAL_KEY_BASE, out_dir=tmp_path, K=K,
                 grid=True, guard=RssGuard())
    _check_receipt_and_rows(tmp_path, "SKNN")
    receipt = json.loads((tmp_path / "RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["grid"] is not None
    assert len(receipt["grid"]["points"]) == 12                          # 4 k's x 3 sample sizes
    sel = receipt["grid"]["selected"]
    assert receipt["config"]["sknn_k"] == sel["k"] and receipt["config"]["sknn_sample_size"] == sel["sample_size"]
    assert out["rows"]


def test_execute_rule(tmp_path):
    tp, ep = _providers()
    user_keys = list(range(tp.n_decisions))          # a distinct user per decision: the dedup is a no-op here
    execute("RULE", train_provider=tp, eval_provider=ep, key_base=MINIMAL_KEY_BASE, out_dir=tmp_path, K=K,
           user_keys=user_keys, guard=RssGuard())
    _check_receipt_and_rows(tmp_path, "RULE")


def test_execute_rule_needs_user_keys(tmp_path):
    tp, ep = _providers()
    try:
        execute("RULE", train_provider=tp, eval_provider=ep, key_base=MINIMAL_KEY_BASE, out_dir=tmp_path, K=K,
               guard=RssGuard())
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_execute_knn_emb(tmp_path):
    tp, ep = _providers()
    ckpt = tmp_path / "small.pt"
    torch.save({base_knn_emb.SASREC_TABLE_KEY: torch.randn(K, 8)}, ckpt)
    execute("KNN_EMB", train_provider=tp, eval_provider=ep, key_base=MINIMAL_KEY_BASE, out_dir=tmp_path, K=K,
           item_table_path=str(ckpt), family="SASREC", table_role="SMALL", guard=RssGuard())
    _check_receipt_and_rows(tmp_path, "KNN_EMB", method="KNN_EMB_SMALL")
    receipt = json.loads((tmp_path / "RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["config"]["item_table_path"] == str(ckpt) and receipt["config"]["family"] == "SASREC"
    assert receipt["config"]["table_role"] == "SMALL"
    assert receipt["config"]["item_table_sha256"]
    rows = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    assert all(r["family"] == "BASELINE_KNN_EMB_SMALL" for r in rows)


def test_execute_knn_emb_big_gets_a_different_label(tmp_path):
    tp, ep = _providers()
    ckpt = tmp_path / "big.pt"
    torch.save({base_knn_emb.SASREC_TABLE_KEY: torch.randn(K, 8)}, ckpt)
    execute("KNN_EMB", train_provider=tp, eval_provider=ep, key_base=MINIMAL_KEY_BASE, out_dir=tmp_path, K=K,
           item_table_path=str(ckpt), family="SASREC", table_role="BIG", guard=RssGuard())
    rows = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    assert all(r["method"] == "KNN_EMB_BIG" and r["family"] == "BASELINE_KNN_EMB_BIG" for r in rows)


def test_execute_knn_emb_needs_item_table_and_family(tmp_path):
    tp, ep = _providers()
    try:
        execute("KNN_EMB", train_provider=tp, eval_provider=ep, key_base=MINIMAL_KEY_BASE, out_dir=tmp_path, K=K,
               guard=RssGuard())
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_grid_refused_for_non_sknn_baselines(tmp_path):
    tp, ep = _providers()
    try:
        execute("POP", train_provider=tp, eval_provider=ep, key_base=MINIMAL_KEY_BASE, out_dir=tmp_path, K=K,
               grid=True)
        raised = False
    except ValueError:
        raised = True
    assert raised
