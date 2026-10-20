import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/code")
import paths, numpy as np, torch
torch.set_num_threads(2)
S = paths.S
t0 = time.time()
# TRAIN target frequency per class (read-only store, F0)
from full_store import CompactStore
st = CompactStore.build(paths.DATASET_ROOT, splits=("TRAIN",), channels="F0", n_threads=2)
tc = np.asarray(st.decisions("TRAIN", "loss").target_class)
freq = np.bincount(tc, minlength=paths.K)
del st
buckets = [("0", freq == 0), ("1-4", (freq >= 1) & (freq <= 4)), ("5-49", (freq >= 5) & (freq <= 49)),
           ("50-499", (freq >= 50) & (freq <= 499)), ("500+", freq >= 500)]
out = {"freq_bucket_sizes": {n: int(m.sum()) for n, m in buckets}}
def describe(tag, E, bias=None, v=None):
    En = np.linalg.norm(E[3:3 + paths.K], axis=1)
    d = {}
    for n, m in buckets:
        if m.sum() == 0: continue
        row = {"norm_median": float(np.median(En[m])), "norm_p99": float(np.percentile(En[m], 99))}
        if bias is not None: row["bias_median"] = float(np.median(bias[m]))
        if v is not None: row["adam_v_rowmean_median"] = float(np.median(v[3:3 + paths.K][m].mean(1)))
        d[n] = row
    out[tag] = d
ck = torch.load(S / "runs/A0_ID256_s2026/checkpoints/previous.pt", map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]; osd = ck["optimizer_state_dict"]
E = sd["item_embed.weight"].numpy(); b = sd["output_bias"].numpy()
# item_embed is the first param in the decay group
v = None
for pid, stt in osd["state"].items():
    if stt["exp_avg_sq"].shape == sd["item_embed.weight"].shape:
        v = stt["exp_avg_sq"].numpy(); break
describe("A0_update_%d" % ck["update"], E, b, v)
out["A0_final_norm_gain_mean"] = float(sd["final_norm.weight"].abs().mean())
g = torch.load(paths.GRU_WORKSPACE / "experiments/full/F0_item_only_h256/checkpoints/best.pt", map_location="cpu", weights_only=False)
gsd = g.get("model_state_dict", g)
describe("GRU_F0_best", gsd["item_embed.weight"].numpy(), gsd["output_bias"].numpy())
out["seconds"] = time.time() - t0
open(S / "bench/diag_item_embeddings.json", "w").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
