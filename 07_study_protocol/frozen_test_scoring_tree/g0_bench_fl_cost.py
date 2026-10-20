"""g0_bench_fl_cost.py (owner: Opus, lead) — PPSI V4 G0 client-update cost benchmark. SYNTHETIC inputs only (shapes and
vocabularies of the frozen v1 data); it reads no ledger rows, writes only its JSON, strict FP32 (TF32 off).

  python code/g0_bench_fl_cost.py --family GRU|SASREC --out g0/BENCH_<family>.json

For the executed architectures (GRU: ContextGRU F4_rich_side_information, item128/hidden256, tied head; SASRec: v3 A1
d256/2 blocks/4 heads, untied head + recentering) it times, on one GPU:
  - copy of the shared state into a worker model (load_state_dict, device to device);
  - creation of a fresh AdamW (the FA/FP local solver resets each round);
  - one local optimisation step (forward, exact full-K CE, backward, clip 1.0, AdamW step, SASRec recentering) at
    local batch sizes b in {1, 4, 16, 64, 256};
  - accumulation of the client's weighted state into an FP32 aggregator;
  - serialisation of the shared state (virtual bytes);
  - a central step at B = 256 for reference.
Each timing is the median over repeats after warm-up, with torch.cuda.synchronize() around the timed region.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppsi_env  # noqa: E402  (must come before any snapshot module)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train as reuse_train  # noqa: E402  (snapshot copy)
import model as gru_model  # noqa: E402
from sasrec_model import SASRecConfig, SASRecCE  # noqa: E402

L, NUM, FLAGS, UC, UM = 50, 19, 15, 12, 6
SASREC_ENTRY = {"id": "A1_RICH256", "group": "rich_frozen_channels", "d_model": 256, "blocks": 2, "heads": 4,
                "ff_multiplier": 4, "dropout": 0.2}


def build(family: str, device):
    poc = reuse_train.load_product_idx_of_class()
    if family == "GRU":
        m = gru_model.build_model("F4_rich_side_information", reuse_train.VOCAB_SIZES, product_idx_of_class=poc)
    else:
        m = SASRecCE(SASRecConfig.from_registry(SASREC_ENTRY), vocab_sizes=dict(reuse_train.VOCAB_SIZES),
                     product_idx_of_class=poc, seed=2026)
    return m.to(device)


def batch(b: int, device, gen: torch.Generator, K: int, gru: bool, model) -> dict:
    V = reuse_train.VOCAB_SIZES
    lengths = torch.randint(1, L + 1, (b,), generator=gen)
    am = torch.arange(L)[None, :] < lengths[:, None]
    tok = lambda n: torch.randint(3, n, (b, L), generator=gen) * am
    out = {"item_tokens": tok(V["item_tokens"]), "category_tokens": tok(V["category_tokens"]), "brand_tokens": tok(V["brand_tokens"]),
           "main_category_tokens": tok(V["main_category_tokens"]), "daypart_tokens": tok(V["daypart_tokens"]),
           "weekday_tokens": tok(V["weekday_tokens"]), "lengths": lengths, "attention_mask": am,
           "event_numeric_features": torch.randn(b, L, NUM, generator=gen) * am[..., None],
           "event_quality_flags": (torch.rand(b, L, FLAGS, generator=gen) < 0.05).to(torch.int8),
           "user_context": torch.randn(b, UC, generator=gen), "user_context_masks": torch.ones(b, UM),
           "position_ids": (torch.arange(1, L + 1)[None, :] * am), "target_class": torch.randint(0, K, (b,), generator=gen)}
    out = {k: v.to(device) for k, v in out.items()}
    if gru:
        kw = reuse_train.build_named_kwargs(out, model)
    else:
        kw = {k: out[k] for k in ("item_tokens", "lengths", "attention_mask", "position_ids", "category_tokens", "brand_tokens",
                                   "main_category_tokens", "daypart_tokens", "weekday_tokens", "event_numeric_features",
                                   "event_quality_flags", "user_context", "user_context_masks")}
    return kw, out["target_class"]


def timed(fn, reps: int, warm: int = 3) -> dict:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(xs), "p90_ms": sorted(xs)[int(0.9 * (len(xs) - 1))], "n": reps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("GRU", "SASREC"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    seen = ppsi_env.assert_isolated()
    ppsi_env.verify_snapshot()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(2026)
    g = build(a.family, dev)
    w = build(a.family, dev)
    K = int(reuse_train.load_product_idx_of_class().shape[0])
    gru = a.family == "GRU"
    params = {n: p.numel() for n, p in g.named_parameters()}
    shared_state = {k: v for k, v in g.state_dict().items()}
    buf = io.BytesIO(); torch.save(shared_state, buf)
    res = {"family": a.family, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
           "precision": "strict FP32 (TF32 off)", "K": K, "n_params": sum(params.values()),
           "param_bytes_fp32": 4 * sum(params.values()), "state_dict_serialized_bytes": buf.getbuffer().nbytes,
           "largest_params": sorted(params.items(), key=lambda kv: -kv[1])[:6], "isolation_modules_checked": len(seen)}
    res["t_copy_state"] = timed(lambda: w.load_state_dict(shared_state), a.reps)
    res["t_fresh_adamw"] = timed(lambda: torch.optim.AdamW(w.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5), a.reps)
    acc = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in shared_state.items() if v.is_floating_point()}

    def accumulate():
        sd = w.state_dict()
        for k, v in acc.items():
            v.add_(sd[k], alpha=7.0)
    res["t_accumulate"] = timed(accumulate, a.reps)
    res["t_serialize_state"] = timed(lambda: torch.save({k: v for k, v in w.state_dict().items()}, io.BytesIO()), max(5, a.reps // 4))
    opt = torch.optim.AdamW(w.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    w.train()
    steps = {}
    for b in (1, 4, 16, 64, 256):
        kw, tgt = batch(b, dev, gen, K, gru, w)

        def step():
            opt.zero_grad(set_to_none=True)
            logits = w(**kw)
            loss = F.cross_entropy(logits, tgt, reduction="sum") / tgt.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(w.parameters(), 1.0)
            opt.step()
            if not gru:
                w.recenter_output_()
        steps[str(b)] = timed(step, a.reps)
        steps[str(b)]["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    res["t_local_step_by_batch"] = steps
    res["central_decisions_per_s_B256"] = 256 / (steps["256"]["median_ms"] / 1e3)
    kw, tgt = batch(256, dev, gen, K, gru, w)
    w.eval()
    with torch.no_grad():
        res["t_eval_forward_B256_fullK"] = timed(lambda: w(**kw), a.reps)
    res["peak_alloc_gib_total"] = torch.cuda.max_memory_allocated() / 2**30
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("family", "n_params", "param_bytes_fp32")} | {"copy_ms": res["t_copy_state"]["median_ms"],
          "step_ms": {b: round(s["median_ms"], 2) for b, s in steps.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
