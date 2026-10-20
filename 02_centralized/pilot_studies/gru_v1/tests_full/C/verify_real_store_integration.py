"""Ad hoc integration smoke test: eval_full.evaluate_model + baselines_full's
store-backed readers against a REAL (now available) full_store.CompactStore,
restricted to a tiny 2-bucket subset for speed. NOT one of Sonnet C's four
owned files -- throwaway, run once from tests_full/C/, safe to delete after.

Never touches VALIDATION_CONFIRM/TEST. Only TRAIN (for the fit) and
VALIDATION_SELECT (for eval), both restricted to buckets [0, 1].
"""
import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True

import full_store  # Sonnet A's real module
import model as model_mod
import train as train_mod
from eval_full import evaluate_model
import baselines_full as bf

WS = "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1"
DATASET_ROOT = bf.DATASET_ROOT
device = torch.device("cuda:0")

print("=== building a small CompactStore (buckets=[0,1], TRAIN+VALIDATION_SELECT, channels=F4) ===")
t0 = time.time()
store = full_store.CompactStore.build(
    DATASET_ROOT, splits=("TRAIN", "VALIDATION_SELECT"), channels="F4",
    buckets=[0, 1], context_length=50, n_threads=3,
)
print(f"build took {time.time()-t0:.1f}s, n_rows={store.n_rows}, version_hash={store.version_hash[:16]}...")

print("\n=== store.decisions + evaluate_model with the pilot F0 checkpoint ===")
ds_eval = store.decisions("VALIDATION_SELECT", "eval")
print("VALIDATION_SELECT eval ds: len =", len(ds_eval), "eval_user_key dtype", ds_eval.eval_user_key.dtype)

run_dir = WS + "/experiments/gru/T-M_F0_item_only"
ckpt = torch.load(run_dir + "/checkpoints/best.pt", map_location="cpu", weights_only=False)
poc = train_mod.load_product_idx_of_class()
gru_model = model_mod.build_model("F0_item_only", train_mod.VOCAB_SIZES, product_idx_of_class=poc)
gru_model.load_state_dict(ckpt["model_state_dict"])
gru_model.to(device)
gru_model.eval()

result = evaluate_model(gru_model, store, ds_eval, device, eval_batch=256, k=20)
print("evaluate_model OK: n =", result["end_to_end"]["n"], "user_count =",
      result["end_to_end"]["macro"]["user_count"])
print("macro mrr@20 =", result["end_to_end"]["macro"]["mrr@20"],
      "elapsed_s =", result["_elapsed_s"], "rows_per_s =", result["_rows_per_s"])

print("\n=== baselines_full.fit_from_store + run_pop_baseline/run_trans_baseline via the real store ===")
ds_loss = store.decisions("TRAIN", "loss")
print("TRAIN loss ds: len =", len(ds_loss))
fit = bf.fit_from_store(store, ds_loss, K=bf.EXPECTED_K)
print("fit OK: n_fit_rows =", fit.n_fit_rows, "max_pop =", fit.max_pop, "max_trans =", fit.max_trans)

pop_result = bf.run_pop_baseline(fit, ds_eval, device, k=20, eval_batch=4096)
print("POP (small-store) macro mrr@20 =", pop_result["end_to_end"]["macro"]["mrr@20"])

eval_last_item = store.decision_item_tokens(ds_eval)
trans_result = bf.run_trans_baseline(fit, ds_eval, device, k=20, eval_batch=1024,
                                      last_item=eval_last_item)
print("TRANS (small-store) macro mrr@20 =", trans_result["end_to_end"]["macro"]["mrr@20"])

print("\n=== ALL REAL-STORE INTEGRATION CHECKS PASSED ===")
