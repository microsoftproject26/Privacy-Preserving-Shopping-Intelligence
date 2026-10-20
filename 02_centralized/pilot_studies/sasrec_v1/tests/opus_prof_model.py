import sys, time, json
sys.path.insert(0, "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/code")
import paths, torch, numpy as np, precision
from sasrec_model import SASRecConfig, SASRecCE
from sasrec_objective import ce_sum
dev = torch.device("cuda:0")
rng = np.random.default_rng(0)
lens = rng.choice(np.r_[np.ones(18), np.full(2, 50), rng.integers(2, 50, 80)].astype(int), 256)
def build(det):
    precision.configure_strict_fp32(0)
    if not det: torch.use_deterministic_algorithms(False)
    cfg = SASRecConfig(run_id="t", group="item_only", d_model=256, blocks=2, heads=4, dropout=0.2)
    net = SASRecCE(cfg, vocab_sizes={"item_tokens": 158489}, product_idx_of_class=np.arange(158486) + 3, seed=0).to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.98), fused=True)
    return net, opt
L_ = torch.as_tensor(lens, device=dev); M_ = torch.arange(50, device=dev)[None] < L_[:, None]
I_ = torch.randint(3, 158489, (256, 50), device=dev) * M_
tg = torch.randint(0, 158486, (256,), device=dev); lm = torch.ones(256, dtype=torch.bool, device=dev)
def step(net, opt):
    opt.zero_grad(set_to_none=True)
    l, n = ce_sum(net(item_tokens=I_, lengths=L_, attention_mask=M_), tg, lm); (l / 256).backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
res = {}
for det in (True, False):
    net, opt = build(det)
    ts = []
    for i in range(60):
        torch.cuda.synchronize(); t0 = time.perf_counter(); step(net, opt); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    res[f"det={det}"] = 1000 * float(np.median(ts[15:]))
net, opt = build(True)
for _ in range(10): step(net, opt)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for _ in range(10): step(net, opt)
    torch.cuda.synchronize()
print(json.dumps(res))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=28, max_name_column_width=60))
