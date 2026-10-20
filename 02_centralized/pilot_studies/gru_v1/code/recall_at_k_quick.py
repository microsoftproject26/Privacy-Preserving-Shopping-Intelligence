"""Descriptive Recall/MRR at k in {1,2,5,10,20} on the pilot 10k-user VS cohort (owner: Opus).

Owner question on 2026-09-23 (mid-training): Recall@2/5/10/20. The training-time evals record
only @20, so this script re-ranks the 112,365 cohort decisions (a subset of VALIDATION_SELECT)
for each arm's CURRENT best.pt and for POP/TRANS refitted on all 40,180,608 TRAIN loss
decisions. Reporting only; it feeds no selection, stopping or LR decision (those are fixed in
FULL_RUN_PROTOCOL.md and read only the trainer's own full-VS macro MRR@20).
Data path: the pilot cohort cache (byte-exact to CompactStore, gate i). No VC/TEST access.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
CODE_DIR = Path(__file__).resolve().parent
for _p in (str(CODE_DIR), str(CODE_DIR / "vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
import train as pilot_train  # noqa: E402
import model as model_mod  # noqa: E402
import eval_full  # noqa: E402
import baselines_full as bf  # noqa: E402
from window_cache import CacheReader  # noqa: E402
from rees46_eda.loaders import detect_layout  # noqa: E402

W = CODE_DIR.parent
ROOT = Path("/home/jovyan/shared_data/rees46_nextitem_v1")
OUT = W / "reports_full" / "recall_at_k_cohort.json"
KS = (1, 2, 5, 10, 20)
ARMS = {"F0_item_only_h256": "F0_item_only", "F4_rich_side_information_h256": "F4_rich_side_information"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class DS:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __len__(self):
        return int(self.target_class.shape[0])


class CohortStore:
    def __init__(self, reader):
        self.reader = reader

    def batch(self, ds, positions):
        return self.reader.read(np.asarray(positions))


def at_k(ranks, target, eval_mask, user_key):
    out = {}
    for k in KS:
        hit = (ranks >= 1) & (ranks <= k)
        rr = np.where(hit, 1.0 / np.maximum(ranks, 1), 0.0)
        row = {}
        for name, credit in (("recall", hit.astype(np.float64)), ("mrr", rr)):
            e2e = credit[eval_mask]
            closed = credit[eval_mask & (target >= 0)]
            keys = user_key[eval_mask]
            uniq, inv = np.unique(keys, return_inverse=True)
            per_user = np.bincount(inv, weights=e2e) / np.bincount(inv)
            row[f"{name}_micro_e2e"] = float(e2e.mean())
            row[f"{name}_micro_closed"] = float(closed.mean())
            row[f"{name}_macro_e2e"] = float(per_user.mean())
        out[f"@{k}"] = row
    return out


def main():
    pilot_train.set_determinism(pilot_train.SEED)
    device = torch.device("cuda:0")
    reader = CacheReader(W / "cache" / "vs")
    n = reader.header["row_count"]
    raw = reader.read(np.arange(n))
    coh = pq.read_table(W / "artifacts" / "validation_select_cohort.parquet", columns=["eval_user_key"])
    tgt = raw["target_class"].numpy()
    em = raw["eval_mask"].numpy().astype(bool)
    uk = coh["eval_user_key"].to_numpy().astype(np.int64)
    ds = DS(target_class=tgt, eval_mask=em, eval_user_key=uk, client_idx=raw["client_id"].numpy(),
            prior_view_count=np.zeros(n, np.int64), prefix_position=np.ones(n, np.int64))
    store = CohortStore(reader)
    out = {"population": {"decisions": int(n), "users": int(np.unique(uk).shape[0]),
                          "oov_targets": int((em & (tgt < 0)).sum()),
                          "note": "pilot 10k-user cohort = subset of VALIDATION_SELECT"}, "models": {}}

    poc = pilot_train.load_product_idx_of_class()
    for arm, group in ARMS.items():
        run = W / "experiments" / "full" / arm
        ck = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
        m = model_mod.build_model(group, pilot_train.VOCAB_SIZES, product_idx_of_class=poc).to(device)
        m.load_state_dict(ck["model_state_dict"])
        res = eval_full.evaluate_model(m, store, ds, device, eval_batch=1024, return_ranks=True)
        sel = ck.get("selection_state", {})
        out["models"][arm] = {"best_epoch_0based": sel.get("best_epoch"), "best_full_vs_macro_mrr20": sel.get("best_metric"),
                              "checkpoint_update": int(ck["update"]),
                              "metrics": at_k(res["ranks"], tgt, em, uk),
                              "check_macro_mrr20_via_eval_full": res["end_to_end"]["macro"]["mrr@20"]}
        log(f"{arm}: {out['models'][arm]['metrics']['@20']}")
        del m
        torch.cuda.empty_cache()

    layout = detect_layout(ROOT)
    mr = bf.load_model_ready_columns(ROOT)
    fit = bf.fit_from_parquet(ROOT, layout, product_idx_table=mr, expect_n_rows=40_180_608, log=log)
    del mr
    last_item = raw["item_tokens"].numpy()[np.arange(n), raw["last_valid_index"].numpy()]
    need = em & (tgt >= 0)
    for name, ranks in (("POP_full", bf.score_and_rank_pop(fit, tgt, need, device, eval_batch=4096)),
                        ("TRANS_full", bf.score_and_rank_trans(fit, last_item, tgt, need, device, eval_batch=1024))):
        out["models"][name] = {"metrics": at_k(ranks, tgt, em, uk)}
        log(f"{name}: {out['models'][name]['metrics']['@20']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("RECALL_AT_K_DONE")


if __name__ == "__main__":
    main()
