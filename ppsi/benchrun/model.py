"""The ID-only SASRec of the benchmark runner (ppsi.seqrec.sasrec.SASRecCE) at a benchmark catalogue size.

Exactly the ppsi.seqrec builder for family SASREC, with two differences only:
  * group "item_only" (no side channels: the benchmarks' paper-comparable variant) in place of the variant entry's
    rich group; every other SASRecConfig field (d_model, blocks, heads, ff_multiplier) comes from
    ppsi.seqrec.variants.VARIANTS[<variant>]["sasrec_entry"]; the ID parameters and their init order are those of the
    rich model (the rich branch is initialised from a separate RNG scope);
  * the catalogue is the benchmark's: product_idx_of_class = arange(3, 3 + K), item vocab K + 3.
Dropout through variants.dropout_of. The initial state is the recentered init (recentered once, on CPU), hashed with
ppsi.fedsim.numerics.state_digest. The federated / harness adapter is ppsi.seqrec.adapters.SASRecAdapter (query /
head / logits / post_step = recenter_output_). `device` (default CPU) only moves the finished module, AFTER the
initial state is built, recentered and hashed on CPU, so init_sha256 is the same on every device.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .common import BenchRefused

ID_ONLY_GROUP = "item_only"


def build_model(variant: str, K: int, seed: int, dropout=(0.1,), max_len=None, device=None) -> dict:
    import torch

    from ppsi.fedsim.numerics import seeded_global_rng, state_digest
    from ppsi.seqrec import sasrec, variants
    from ppsi.seqrec.adapters import SASRecAdapter
    from ppsi.seqrec.features import InputWidths
    vname, ventry = variants.resolve("SASREC", variant)
    if ventry.get("head") != "untied" or ventry.get("recenter") != "per_step":
        raise BenchRefused(f"{vname}: not an untied, recentered SASRec variant")
    drop = variants.dropout_of("SASREC", vname, dropout)
    entry = dict(ventry["sasrec_entry"], group=ID_ONLY_GROUP)
    cfg = sasrec.SASRecConfig.from_entry(entry)
    if drop[0] != cfg.dropout:
        cfg = replace(cfg, dropout=drop[0])
    if max_len is not None and int(max_len) != cfg.max_len:      # a longer context (e.g. 200 on ML-1M)
        cfg = replace(cfg, max_len=int(max_len))
    K = int(K)
    poc = np.arange(3, 3 + K, dtype=np.int64)
    widths = InputWidths(0, 0, 0, 0, view_id=ID_ONLY_GROUP)
    with seeded_global_rng(int(seed), torch.device("cpu")):
        m = sasrec.SASRecCE(cfg, vocab_sizes={"item_tokens": K + 3}, product_idx_of_class=poc, seed=int(seed),
                            impl="unpadded")
    raw = state_digest(SASRecAdapter(m, widths).broadcast_state())
    m.recenter_output_()                                   # the initial state is the recentered init
    adapter = SASRecAdapter(m, widths)
    init = state_digest(adapter.broadcast_state())
    if device is not None and torch.device(device).type != "cpu":   # move only after the CPU init + hash
        m = m.to(torch.device(device))
        adapter = SASRecAdapter(m, widths)
    rec = {"family": "SASREC", "model_variant": vname, "group": ID_ONLY_GROUP, "K": K, "seed": int(seed),
           "dropout": list(drop), "d_model": cfg.d_model, "blocks": cfg.blocks, "heads": cfg.heads,
           "ff_multiplier": cfg.ff_multiplier, "max_len": cfg.max_len, "raw_init_sha256": raw,
           "init_sha256": init, "shared_numel": int(adapter.manifest.shared_numel),
           "manifest_digest": adapter.manifest.digest(), "inventory": sasrec.parameter_inventory(m)}
    return {"module": m, "adapter": adapter, "init_sha256": init, "record": rec}
