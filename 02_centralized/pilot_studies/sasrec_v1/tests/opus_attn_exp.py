import sys, time, json, math, warnings
sys.path.insert(0, "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/code")
import paths, torch, numpy as np, precision
import torch.nn.functional as F
import sasrec_model as sm
from sasrec_model import SASRecConfig, SASRecCE
from sasrec_objective import ce_sum
from torch.nn.attention import sdpa_kernel, SDPBackend
dev = torch.device("cuda:0")
rng = np.random.default_rng(0)
lens = rng.choice(np.r_[np.ones(18), np.full(2, 50), rng.integers(2, 50, 80)].astype(int), 256)
L_ = torch.as_tensor(lens, device=dev); M_ = torch.arange(50, device=dev)[None] < L_[:, None]
I_ = torch.randint(3, 158489, (256, 50), device=dev) * M_
tg = torch.randint(0, 158486, (256,), device=dev); lm = torch.ones(256, dtype=torch.bool, device=dev)
orig_ft = sm.CausalBlock.forward_tokens
def ft_single(self, xf, groups, causal, _state={}):
    # single padded buffer (pre-bucketing version)
    B, T, flat_idx = _state["B"], _state["T"], _state["flat"]
    qkv = self.qkv(self.ln1(xf))
    buf = qkv.new_zeros(B * T, 3 * self.d).index_copy(0, flat_idx, qkv)
    o = self._attend(buf.view(B, T, 3 * self.d), causal).reshape(B * T, self.d)
    u = xf + F.dropout(self.out(o.index_select(0, flat_idx)), self.p, self.training)
    return self._ffn(u)
def attend_sdpa(self, qkv_bt, causal):
    B, T, _ = qkv_bt.shape
    q, k, v = qkv_bt.view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0, is_causal=True)
    return o.transpose(1, 2).reshape(B, T, self.d)
orig_attend = sm.CausalBlock._attend
def run(mode, det=True, n=50):
    precision.configure_strict_fp32(0)
    if not det: torch.use_deterministic_algorithms(False)
    sm.CausalBlock.forward_tokens = orig_ft if mode.startswith("bucket") else ft_single
    sm.CausalBlock._attend = attend_sdpa if "sdpa" in mode else orig_attend
    cfg = SASRecConfig(run_id="t", group="item_only", d_model=256, blocks=2, heads=4, dropout=0.2)
    net = SASRecCE(cfg, vocab_sizes={"item_tokens": 158489}, product_idx_of_class=np.arange(158486) + 3, seed=0).to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.98), fused=True)
    T = int(L_.max()); ft_single.__defaults__[0].update(B=256, T=T, flat=M_[:, :T].reshape(-1).nonzero(as_tuple=True)[0])
    ts, losses = [], []
    torch.manual_seed(5); torch.cuda.manual_seed_all(5)
    for i in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        l, _ = ce_sum(net(item_tokens=I_, lengths=L_, attention_mask=M_), tg, lm); (l / 256).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0); losses.append(float(l))
    return 1000 * float(np.median(ts[10:])), losses, net.item_embed.weight.detach().clone()
out = {}
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for mode in ("single_math", "bucket_math", "single_sdpa", "bucket_sdpa"):
        for backend in ([None] if "math" in mode else [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            ctx = sdpa_kernel([backend]) if backend is not None else __import__("contextlib").nullcontext()
            try:
                with ctx:
                    ms, l1, w1 = run(mode, n=40)
                    _, l2, w2 = run(mode, n=40)
                out[f"{mode}/{backend}"] = {"ms": ms, "repeat_bitwise_equal": bool(torch.equal(w1, w2)) and l1 == l2}
            except Exception as e:
                out[f"{mode}/{backend}"] = {"error": repr(e)[:200]}
    out["warnings"] = sorted({str(x.message)[:160] for x in w})
print(json.dumps(out, indent=1))
