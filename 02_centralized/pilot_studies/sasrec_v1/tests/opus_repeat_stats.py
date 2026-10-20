import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/code")
import paths, numpy as np, torch
torch.set_num_threads(4)
S = paths.S
t0 = time.time()
from full_store import CompactStore
st = CompactStore.build(paths.DATASET_ROOT, splits=("TRAIN", "VALIDATION_SELECT"), channels="F0", n_threads=4)
ds = st.decisions("VALIDATION_SELECT", "eval")
N = len(ds)
tgt = np.asarray(ds.target_class); em = np.asarray(ds.eval_mask, bool)
in_ctx = np.zeros(N, bool); is_last = np.zeros(N, bool); ctx_len = np.zeros(N, np.int16)
for s0 in range(0, N, 65536):
    pos = np.arange(s0, min(N, s0 + 65536))
    b = st.batch(ds, pos)
    items = b["item_tokens"].numpy(); L = b["lengths"].numpy()
    t = tgt[pos] + 3
    in_ctx[pos] = (items == t[:, None]).any(1) & (tgt[pos] >= 0)
    is_last[pos] = (items[np.arange(len(pos)), L - 1] == t) & (tgt[pos] >= 0)
    ctx_len[pos] = L
rank = {}
for r in ("A0_ID256_s2026", "A1_RICH256_s2026"):
    z = np.load(S / f"runs/{r}/ranks/epoch3.npz"); rank[r] = z["ranks"].astype(np.int64)
ev = em
out = {"n_eval": int(ev.sum()), "frac_target_in_context_window": float(in_ctx[ev].mean()),
       "frac_target_equals_last_item": float(is_last[ev].mean()),
       "frac_oov_target": float((tgt[ev] < 0).mean())}
for r, rk in rank.items():
    rr = rk[ev]; ic = in_ctx[ev]; ok = rr >= 1
    def rec(mask, k): return float(((rr >= 1) & (rr <= k) & mask).sum() / max(1, mask.sum()))
    def mrr(mask): 
        v = np.where((rr >= 1) & (rr <= 20), 1.0 / np.maximum(rr, 1), 0.0); return float(v[mask].mean())
    out[r] = {"recall20_target_in_context": rec(ic, 20), "recall20_target_not_in_context": rec(~ic, 20),
              "recall1_in_ctx": rec(ic, 1), "recall1_not_in_ctx": rec(~ic, 1),
              "mrr20_in_ctx": mrr(ic), "mrr20_not_in_ctx": mrr(~ic), "micro_recall20_all": rec(np.ones_like(ic), 20)}
out["seconds"] = time.time() - t0
open(S / "review/deep/repeat_stats.json", "w").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
