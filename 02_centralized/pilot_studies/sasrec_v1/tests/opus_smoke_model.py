import sys, time, math, json
sys.path.insert(0, "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/code")
import paths, torch, numpy as np, precision
from sasrec_model import SASRecConfig, SASRecCE
from sasrec_objective import ce_sum, ce_sum_reference
precision.configure_strict_fp32(0)
dev = torch.device("cuda:0")
K = 997; V = K + 3
vs = {"item_tokens": V, "category_tokens": 20, "brand_tokens": 30, "main_category_tokens": 6, "daypart_tokens": 8, "weekday_tokens": 10}
poc = np.arange(K) + 3
rng = np.random.default_rng(0)
B, T = 64, 50
lengths = torch.as_tensor(rng.choice([1, 2, 5, 17, 49, 50], B), device=dev)
m = torch.arange(T, device=dev)[None] < lengths[:, None]
items = torch.randint(3, V, (B, T), device=dev) * m
pos_ids = torch.where(m, torch.arange(T, device=dev)[None] + 1, torch.zeros_like(items))
side = {"category_tokens": torch.randint(0, 20, (B, T), device=dev) * m, "brand_tokens": torch.randint(0, 30, (B, T), device=dev) * m,
        "main_category_tokens": torch.randint(0, 6, (B, T), device=dev) * m, "daypart_tokens": torch.randint(0, 8, (B, T), device=dev) * m,
        "weekday_tokens": torch.randint(0, 10, (B, T), device=dev) * m,
        "event_numeric_features": torch.randn(B, T, 19, device=dev) * m[..., None], "event_quality_flags": (torch.rand(B, T, 15, device=dev) < .2).to(torch.int8) * m[..., None],
        "user_context": torch.randn(B, 12, device=dev), "user_context_masks": (torch.rand(B, 6, device=dev) < .5).float()}
out = {}
for group in ("item_only", "rich_frozen_channels"):
    cfg = SASRecConfig(run_id="t", group=group, d_model=64, blocks=2, heads=4, dropout=0.2)
    a = SASRecCE(cfg, vocab_sizes=vs, product_idx_of_class=poc, seed=1, impl="unpadded").to(dev).eval()
    b = SASRecCE(cfg, vocab_sizes=vs, product_idx_of_class=poc, seed=1, impl="padded_reference").to(dev).eval()
    kw = dict(item_tokens=items, lengths=lengths, attention_mask=m, position_ids=pos_ids)
    if group != "item_only": kw.update(side)
    la, lb = a(**kw), b(**kw)
    out[group] = {"max_abs_logit_diff_unpadded_vs_padded": float((la - lb).abs().max()),
                  "init_loss": float(ce_sum(la, torch.randint(0, K, (B,), device=dev), torch.ones(B, dtype=torch.bool, device=dev))[0] / B),
                  "lnK": math.log(K)}
    # training-mode gradient equivalence with dropout disabled
    for mdl in (a, b):
        mdl.train()
        for mod in mdl.modules():
            if hasattr(mod, "p"): mod.p = 0.0
    tgt = torch.randint(0, K, (B,), device=dev); lm = torch.ones(B, dtype=torch.bool, device=dev)
    import sasrec_model
    a.cfg.__class__  # frozen dataclass: dropout via F.dropout(p=cfg.dropout) at input
    object.__setattr__(a.cfg, "dropout", 0.0); object.__setattr__(b.cfg, "dropout", 0.0)
    ga = torch.autograd.grad(ce_sum(a(**kw), tgt, lm)[0], list(a.parameters()), allow_unused=True)
    gb = torch.autograd.grad(ce_sum(b(**kw), tgt, lm)[0], list(b.parameters()), allow_unused=True)
    out[group]["max_rel_grad_diff"] = max(float((x - y).norm() / (y.norm() + 1e-12)) for x, y in zip(ga, gb) if x is not None and y is not None)
# speed, production model, synthetic tokens, real-like lengths
torch.manual_seed(0)
lens = rng.choice(np.r_[np.ones(18), np.full(2, 50), rng.integers(2, 50, 80)].astype(int), 256)
cfg = SASRecConfig(run_id="t", group="item_only", d_model=256, blocks=2, heads=4, dropout=0.2)
net = SASRecCE(cfg, vocab_sizes={"item_tokens": 158489}, product_idx_of_class=np.arange(158486) + 3, seed=0).to(dev).train()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.98), fused=True)
L_ = torch.as_tensor(lens, device=dev); M_ = torch.arange(50, device=dev)[None] < L_[:, None]
I_ = torch.randint(3, 158489, (256, 50), device=dev) * M_
tg = torch.randint(0, 158486, (256,), device=dev); lm = torch.ones(256, dtype=torch.bool, device=dev)
ts = []
for i in range(80):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    l, n = ce_sum(net(item_tokens=I_, lengths=L_, attention_mask=M_), tg, lm); (l / 256).backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
out["d256_prod_model_ms_step_det"] = 1000 * float(np.median(ts[20:]))
print(json.dumps(out, indent=1))
