"""Local CPU mechanics test for vs_systems.py on SYNTHETIC data (no server, no real data).

Checks:
(1) Member ranks captured by forward hooks through TargetStashStore (prefetch=False) equal each
    member's own evaluate_model ranks exactly. Negative control: a shifted stash start gives a mismatch.
(2) The ensemble ranks from evaluate_model equal a manual mean-log-softmax ranking.
(3) average_state_dicts + strict load works, and state_dict_sha is deterministic.
(4) complementarity() and boot() run and are self-consistent: identical inputs give 0 exclusive hits
    and a zero bootstrap difference.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vs_systems as V
import eval_full, ensembles
from synthetic_store import SyntheticStore
from sasrec_model import SASRecConfig, SASRecCE

torch.manual_seed(0)
st = SyntheticStore.build(channels="F4", n_train=64, n_vs=700, K=97, seed=0, task="mapping")
ds = st.decisions("VALIDATION_SELECT", "eval")


def model(group, seed):
    reg = {"id": "A1_RICH256" if group == "rich_frozen_channels" else "A0_ID256", "group": group, "d_model": 16, "blocks": 1, "heads": 2,
           "ff_multiplier": 4, "dropout": 0.0, "peak_lr": 1e-3, "label_smoothing": 0.0}
    m = SASRecCE(SASRecConfig.from_registry(reg), vocab_sizes=st.vocab_sizes, product_idx_of_class=st.product_idx_of_class, seed=seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
        if m.cfg.rich:
            m.side_gate.fill_(0.5); m.user_gate.fill_(0.5)
    return m.eval()


a0, a1 = model("item_only", 1), model("rich_frozen_channels", 2)
dev = torch.device("cpu")
ens = ensembles.LogProbEnsemble([a0, a1]).eval()
proxy = V.TargetStashStore(st)
arrays, handles = V.member_rank_hooks(ens, proxy, len(ds), dev)
r_ens = eval_full.evaluate_model(ens, proxy, ds, dev, eval_batch=128, k=20, return_ranks=True, prefetch=False)["ranks"]
for h in handles:
    h.remove()
r0 = eval_full.evaluate_model(a0, st, ds, dev, eval_batch=128, k=20, return_ranks=True)["ranks"]
r1 = eval_full.evaluate_model(a1, st, ds, dev, eval_batch=128, k=20, return_ranks=True)["ranks"]
assert np.array_equal(arrays[0].numpy(), r0) and np.array_equal(arrays[1].numpy(), r1), "hook ranks differ from standalone"
print("(1) hook member ranks == standalone evaluate_model ranks (A0, A1):", int((r0 >= 0).sum()), "scored rows")

# negative control: corrupt the stash offset -> must NOT match
class BadProxy(V.TargetStashStore):
    def batch(self, d, positions):
        b = super().batch(d, positions); self.stash["start"] += 1; return b
bp = BadProxy(st); arr_b, hb = V.member_rank_hooks(ens, bp, len(ds) + 1, dev)
eval_full.evaluate_model(ens, bp, ds, dev, eval_batch=128, k=20, return_ranks=True, prefetch=False)
for h in hb:
    h.remove()
assert not np.array_equal(arr_b[0].numpy()[: len(ds)], r0), "negative control failed to detect misalignment"
print("(1-neg) shifted stash offset detected: mismatch as expected")

# (2) manual ensemble ranking equals the evaluator's ensemble ranks
import sasrec_data
b = st.batch(ds, np.arange(len(ds)))
kw0 = {k: v for k, v in sasrec_data.predictor_kwargs(b, a0).items()}
kw1 = {k: v for k, v in sasrec_data.predictor_kwargs(b, a1).items()}
with torch.no_grad():
    lp = (torch.log_softmax(a0(**kw0), 1) + torch.log_softmax(a1(**kw1), 1)) / 2
tgt, em = b["target_class"], b["eval_mask"]
need = em & (tgt >= 0); idx = need.nonzero(as_tuple=True)[0]
manual = np.full(len(ds), -1); manual[idx.numpy()] = eval_full.rank_targets(lp[idx], tgt[idx]).numpy()
assert np.array_equal(manual, r_ens), "ensemble ranks differ from manual mean log-softmax"
print("(2) ensemble ranks == manual mean-log-softmax ranks")

# (3) averaging + strict load + deterministic sha
import tempfile
tmp = tempfile.mkdtemp(); ps = []
for s in (3, 4, 5):
    m = model("rich_frozen_channels", s); p = os.path.join(tmp, f"e{s}.pt"); torch.save({"state_dict": m.state_dict()}, p); ps.append(p)
avg = ensembles.average_state_dicts(ps)
m = SASRecCE(SASRecConfig.from_registry({"id": "A1_RICH256", "group": "rich_frozen_channels", "d_model": 16, "blocks": 1, "heads": 2,
                                         "ff_multiplier": 4, "dropout": 0.0, "peak_lr": 1e-3, "label_smoothing": 0.0}),
             vocab_sizes=st.vocab_sizes, product_idx_of_class=st.product_idx_of_class, seed=0)
m.load_state_dict(avg, strict=True)
assert V.state_dict_sha(avg) == V.state_dict_sha(ensembles.average_state_dicts(ps))
print("(3) S4a-style average loads strictly; sha deterministic", V.state_dict_sha(avg)[:12])

# (4) complementarity / bootstrap self-consistency (ds from SyntheticStore)
c = V.complementarity(r1, r1, ds, ("x", "y"))
assert c["x_only"] == 0 and c["y_only"] == 0
c2 = V.complementarity(r0, r1, ds)
assert c2["both_hit@20"] + c2["A0_e5_only"] + c2["A1_e5_only"] + c2["neither"] == c2["n"]
bt = V.boot(r1, r1, ds)
assert abs(bt["diff"]) == 0.0
print("(4) complementarity consistent; bootstrap(identical)=0; example:", {k: c2[k] for k in ("both_hit@20", "A0_e5_only", "A1_e5_only", "neither")})
print("ALL LOCAL MECHANICS CHECKS PASSED")
