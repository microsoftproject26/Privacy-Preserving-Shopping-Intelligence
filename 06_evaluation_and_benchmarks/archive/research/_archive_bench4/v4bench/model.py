"""v4bench.model — the ID-only REES46 SASRec (v4models.sasrec_v2.SASRecCE) at a benchmark catalogue size.

Exactly v4models.build.construct_module + build for family SASREC, with two registered differences only:
  * group "item_only" (no side channels: the benchmarks' main, paper-comparable variant; BENCHMARK_PLAN_V1 §4) in
    place of the variant entry's "rich_frozen_channels" — every other SASRecConfig field (d_model, blocks, heads,
    ff_multiplier) comes from v4models.variants.VARIANTS[<variant>]["sasrec_entry"]; the ID parameters and their
    init order are those of the rich model (sasrec_v2 initialises the rich branch from a separate RNG scope);
  * the catalogue is the benchmark's: product_idx_of_class = arange(3, 3 + K), item vocab K + 3.
Dropout = the recipe's (REES46 02B_SASREC / SMALL_SASREC_D64_B2: [0.1]) through variants.dropout_of. The registered
theta0 is the C7-recentered init (recentered once, on CPU), hashed with fl.numerics.state_digest. The FL / harness
adapter is fl.families.SASRecAdapter (query / head / logits / post_step = recenter_output_).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .common import BenchRefused

ID_ONLY_GROUP = "item_only"


def build_model(variant: str, K: int, seed: int, dropout=(0.1,), max_len=None) -> dict:
    import ppsi_env  # noqa: F401  (isolation first)
    import torch
    from fl.families import SASRecAdapter
    from fl.numerics import seeded_global_rng, state_digest
    from v4models import sasrec_v2, variants
    vname, ventry = variants.resolve("SASREC", variant)
    if ventry.get("head") != "untied" or ventry.get("recenter") != "C7":
        raise BenchRefused(f"{vname}: not an untied / C7 SASRec variant")
    drop = variants.dropout_of("SASREC", vname, dropout)
    entry = dict(ventry["sasrec_entry"], group=ID_ONLY_GROUP)
    cfg = sasrec_v2.SASRecConfig.from_registry(entry)
    if drop[0] != cfg.dropout:
        cfg = replace(cfg, dropout=drop[0])
    if max_len is not None and int(max_len) != cfg.max_len:      # Track B (max_len 200 on ML-1M); Track A keeps 50
        cfg = replace(cfg, max_len=int(max_len))
    K = int(K)
    poc = np.arange(3, 3 + K, dtype=np.int64)
    with seeded_global_rng(int(seed), torch.device("cpu")):
        m = sasrec_v2.SASRecCE(cfg, vocab_sizes={"item_tokens": K + 3}, product_idx_of_class=poc, seed=int(seed),
                               impl="unpadded")
    raw = state_digest(SASRecAdapter(m).broadcast_state())
    m.recenter_output_()                                   # C7: the registered theta0 is the recentered init
    adapter = SASRecAdapter(m)
    init = state_digest(adapter.broadcast_state())
    rec = {"family": "SASREC", "model_variant": vname, "group": ID_ONLY_GROUP, "K": K, "seed": int(seed),
           "dropout": list(drop), "d_model": cfg.d_model, "blocks": cfg.blocks, "heads": cfg.heads,
           "ff_multiplier": cfg.ff_multiplier, "max_len": cfg.max_len, "raw_init_sha256": raw,
           "init_sha256": init, "shared_numel": int(adapter.manifest.shared_numel),
           "manifest_digest": adapter.manifest.digest(), "inventory": sasrec_v2.parameter_inventory(m)}
    return {"module": m, "adapter": adapter, "init_sha256": init, "record": rec}
