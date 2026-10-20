"""Pre-Gate-P sizing micro-benchmark (owner: Opus). SYNTHETIC tokens only.

No real-data parameter updates: the frozen data is read only to obtain the TRAIN
window-length histogram (a read-only statistic). The throwaway module here is NOT
the campaign model; it only sizes compute for d128/d256/d384 and compares the
padded, cropped and unpadded encoder layouts. Output: $S/bench/sizing_pre_gate_p.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
S = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1")
sys.path[:0] = [str(S / "code/reuse"), str(S / "code/reuse/vendor")]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(3)
K, V, L, B = 158486, 158489, 50, 256


class Block(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.f1, self.f2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.p = p

    def attn(self, qkv_bt):  # [B,T,3d] -> [B,T,d], causal
        Bb, T, _ = qkv_bt.shape
        q, k, v = qkv_bt.view(Bb, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=s.device).tril(), float("-inf"))
        a = F.dropout(s.softmax(-1), self.p, self.training)
        return (a @ v).transpose(1, 2).reshape(Bb, T, -1)

    def forward_padded(self, x, m):
        u = x + F.dropout(self.o(self.attn(self.qkv(self.ln1(x)))), self.p, self.training)
        y = u + F.dropout(self.f2(F.dropout(F.gelu(self.f1(self.ln2(u))), self.p, self.training)), self.p, self.training)
        return y * m.unsqueeze(-1)

    def forward_unpad(self, xf, idx, Bb, T):  # xf [N,d] valid tokens only
        qkv_f = self.qkv(self.ln1(xf))
        buf = qkv_f.new_zeros(Bb * T, qkv_f.shape[1]).index_copy(0, idx, qkv_f)
        a = self.attn(buf.view(Bb, T, -1)).reshape(Bb * T, -1).index_select(0, idx)
        u = xf + F.dropout(self.o(a), self.p, self.training)
        return u + F.dropout(self.f2(F.dropout(F.gelu(self.f1(self.ln2(u))), self.p, self.training)), self.p, self.training)


class Net(nn.Module):
    def __init__(self, d, h, p=0.2, blocks=2):
        super().__init__()
        self.d = d
        self.E = nn.Embedding(V, d, padding_idx=0)
        self.P = nn.Embedding(L, d)
        self.blocks = nn.ModuleList(Block(d, h, p) for _ in range(blocks))
        self.lnf = nn.LayerNorm(d)
        self.bias = nn.Parameter(torch.zeros(K))
        self.p = p

    def forward(self, items, lengths, mode):
        Bb = items.shape[0]
        if mode == "head_only":
            q = self.E(items.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1))
        else:
            T = L if mode == "padded50" else int(lengths.max())
            it = items[:, :T]
            m = torch.arange(T, device=it.device).unsqueeze(0) < lengths.unsqueeze(1)
            pos = torch.arange(T, device=it.device).unsqueeze(0).expand(Bb, T)
            x = (self.E(it) * math.sqrt(self.d) + self.P(pos)) * m.unsqueeze(-1)
            x = F.dropout(x, self.p, self.training)
            if mode == "unpad":
                idx = m.reshape(-1).nonzero().squeeze(1)
                xf = x.reshape(Bb * T, -1).index_select(0, idx)
                for b in self.blocks:
                    xf = b.forward_unpad(xf, idx, Bb, T)
                x = xf.new_zeros(Bb * T, self.d).index_copy(0, idx, xf).view(Bb, T, -1)
            else:
                for b in self.blocks:
                    x = b.forward_padded(x, m)
            q = self.lnf(x.gather(1, (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.d)).squeeze(1))
        return q @ self.E.weight[3:3 + K].T + self.bias


def lengths_hist():
    from full_store import CompactStore
    st = CompactStore.build(Path("/home/jovyan/shared_data/rees46_nextitem_v1"), splits=("TRAIN",), channels="F0")
    ln = np.asarray(st.decisions("TRAIN", "loss").length).astype(np.int64)
    del st
    return ln


def main():
    dev = torch.device("cuda:0")
    t0 = time.time()
    ln = lengths_hist()
    hist = np.bincount(ln, minlength=L + 1)
    rng = np.random.default_rng(0)
    batch_max = [int(rng.choice(ln, B).max()) for _ in range(2000)]
    out = {"n_train": int(ln.size), "mean_len": float(ln.mean()), "median_len": float(np.median(ln)),
           "frac_len_1": float((ln == 1).mean()), "frac_len_50": float((ln == 50).mean()),
           "hist": hist.tolist(), "batch256_max_len_mean": float(np.mean(batch_max)),
           "batch256_max_len_frac_50": float(np.mean(np.array(batch_max) == 50)),
           "store_seconds": time.time() - t0, "results": []}
    print(json.dumps({k: v for k, v in out.items() if k != "hist"}), flush=True)
    pool = []
    for _ in range(16):
        lb = torch.as_tensor(rng.choice(ln, B), device=dev)
        items = torch.randint(3, V, (B, L), device=dev) * (torch.arange(L, device=dev) < lb.unsqueeze(1))
        pool.append((items, lb, torch.randint(0, K, (B,), device=dev)))
    for d, h in ((128, 4), (256, 4), (384, 6)):
        for mode in ("head_only", "padded50", "crop", "unpad"):
            torch.manual_seed(0)
            net = Net(d, h).to(dev).train()
            opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.98), fused=True)
            torch.cuda.reset_peak_memory_stats(dev)
            times = []
            for i in range(130):
                items, lb, tgt = pool[i % 16]
                torch.cuda.synchronize(dev); ts = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(net(items, lb, mode), tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                torch.cuda.synchronize(dev); times.append(time.perf_counter() - ts)
            ms = 1000 * float(np.median(times[30:]))
            r = {"d": d, "heads": h, "mode": mode, "median_ms_step": ms, "decisions_per_s": B / ms * 1000,
                 "params": sum(p.numel() for p in net.parameters()),
                 "peak_alloc_gib": torch.cuda.max_memory_allocated(dev) / 2**30}
            if mode == "unpad":  # eval-style forward + exact rank count, batch 1024
                net.eval()
                with torch.no_grad():
                    et = []
                    for j in range(12):
                        lb = torch.as_tensor(rng.choice(ln, 1024), device=dev)
                        items = torch.randint(3, V, (1024, L), device=dev) * (torch.arange(L, device=dev) < lb.unsqueeze(1))
                        tgt = torch.randint(0, K, (1024,), device=dev)
                        torch.cuda.synchronize(dev); ts = time.perf_counter()
                        s = net(items, lb, "unpad")
                        st_ = s.gather(1, tgt.view(-1, 1))
                        rk = 1 + (s > st_).sum(1) + ((s == st_) & (torch.arange(K, device=dev) < tgt.view(-1, 1))).sum(1)
                        torch.cuda.synchronize(dev); et.append(time.perf_counter() - ts)
                r["eval_rows_per_s"] = 1024 / float(np.median(et[2:]))
            out["results"].append(r)
            print(json.dumps(r), flush=True)
            del net, opt
            torch.cuda.empty_cache()
    (S / "bench").mkdir(exist_ok=True)
    (S / "bench/sizing_pre_gate_p.json").write_text(json.dumps(out, indent=2))
    print("WROTE", S / "bench/sizing_pre_gate_p.json")


if __name__ == "__main__":
    main()
