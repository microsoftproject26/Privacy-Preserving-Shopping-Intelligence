import sys, time, json
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
for d, h in ((128, 4), (256, 4), (384, 6)):
  for tf32 in (False, True):
    for B in ((256, 512, 1024) if d == 256 else (256,)):
        precision.configure_strict_fp32(0)
        torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        cfg = SASRecConfig(run_id="t", group="item_only", d_model=d, blocks=2, heads=h, dropout=0.2)
        net = SASRecCE(cfg, vocab_sizes={"item_tokens": 158489}, product_idx_of_class=np.arange(158486) + 3, seed=0).to(dev).train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.98), fused=True)
        bs = [batch(B) for _ in range(4)]
        lm = torch.ones(B, dtype=torch.bool, device=dev)
        ts = []
        for i in range(45):
            I, L, M, tg = bs[i % 4]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            l, _ = ce_sum(net(item_tokens=I, lengths=L, attention_mask=M), tg, lm); (l / B).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        ms = 1000 * float(np.median(ts[10:]))
        # eval: forward + exact rank count at batch 1024
        net.eval(); et = []
        with torch.no_grad():
            for j in range(8):
                I, L, M, tg = batch(1024)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                s = net(item_tokens=I, lengths=L, attention_mask=M); st = s.gather(1, tg.view(-1, 1))
                r = 1 + (s > st).sum(1) + ((s == st) & (torch.arange(158486, device=dev) < tg.view(-1, 1))).sum(1)
                torch.cuda.synchronize(); et.append(time.perf_counter() - t0)
        key = f"d{d}_B{B}_tf32={tf32}"
        res[key] = {"ms_step": round(ms, 1), "train_dec_per_s": round(B / ms * 1000), "eval_rows_per_s": round(1024 / float(np.median(et[2:]))),
                    "train_h_per_epoch": round(40180608 / (B / ms * 1000) / 3600, 2),
                    "eval_h_full_vs": round(8826488 / (1024 / float(np.median(et[2:]))) / 3600, 2)}
        print(key, json.dumps(res[key]), flush=True)
        del net, opt; torch.cuda.empty_cache()
open("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/bench/speed_levers_synthetic.json", "w").write(json.dumps(res, indent=1))
