import sys, time, json, os
sys.path.insert(0, "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/code")
import paths, torch, numpy as np, precision
from sasrec_model import SASRecConfig, SASRecCE
from sasrec_objective import ce_sum
dev = torch.device("cuda:0")
rng = np.random.default_rng(0)
pool = np.r_[np.ones(18), np.full(2, 50), rng.integers(2, 50, 80)].astype(int)
def batch(B):
    L = torch.as_tensor(rng.choice(pool, B), device=dev); M = torch.arange(50, device=dev)[None] < L[:, None]
    return torch.randint(3, 158489, (B, 50), device=dev) * M, L, M, torch.randint(0, 158486, (B,), device=dev)
res = {}
bs = [batch(256) for _ in range(12)]
lm = torch.ones(256, dtype=torch.bool, device=dev)
def train_bench(net, opt, n=60, label=""):
    ts = []
    t_first = None
    for i in range(n):
        I, L, M, tg = bs[i % 12]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        l, _ = ce_sum(net(item_tokens=I, lengths=L, attention_mask=M), tg, lm); (l / 256).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0; ts.append(dt)
        if i == 0: t_first = dt
    return {"ms_step_median": 1000 * float(np.median(ts[15:])), "first_step_s": t_first, "sum_first15_s": float(sum(ts[:15]))}
for mode in ("eager", "compile_blocks"):
    precision.configure_strict_fp32(0); precision.training_precision(True)
    cfg = SASRecConfig(run_id="t", group="item_only", d_model=256, blocks=2, heads=4, dropout=0.2)
    net = SASRecCE(cfg, vocab_sizes={"item_tokens": 158489}, product_idx_of_class=np.arange(158486) + 3, seed=0).to(dev).train()
    if mode == "compile_blocks":
        for blk in net.blocks:
            blk._ffn = torch.compile(blk._ffn, dynamic=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.98), fused=True)
    try:
        t0 = time.time(); r = train_bench(net, opt); r["wall_s"] = time.time() - t0
    except Exception as e:
        r = {"error": repr(e)[:300]}
    res[mode] = r; print(mode, json.dumps(r), flush=True)
    del net, opt; torch.cuda.empty_cache()
# exact eval batch size (strict FP32)
precision.configure_strict_fp32(0)
cfg = SASRecConfig(run_id="t", group="item_only", d_model=256, blocks=2, heads=4, dropout=0.2)
net = SASRecCE(cfg, vocab_sizes={"item_tokens": 158489}, product_idx_of_class=np.arange(158486) + 3, seed=0).to(dev).eval()
sys.path.insert(0, "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/code/reuse")
from eval_full import rank_targets
for EB in (1024, 2048, 4096):
    et = []
    with torch.no_grad(), precision.strict_eval():
        for j in range(8):
            I, L, M, tg = batch(EB)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            s = net(item_tokens=I, lengths=L, attention_mask=M); r = rank_targets(s, tg)
            torch.cuda.synchronize(); et.append(time.perf_counter() - t0)
    res[f"eval_B{EB}"] = {"rows_per_s": EB / float(np.median(et[2:])), "peak_gib": torch.cuda.max_memory_allocated(dev) / 2**30}
    print(EB, json.dumps(res[f"eval_B{EB}"]), flush=True)
open("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/bench/compile_evalbatch_synthetic.json", "w").write(json.dumps(res, indent=1))
