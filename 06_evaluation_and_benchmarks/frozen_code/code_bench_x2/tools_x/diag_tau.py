"""POST_HOC_X diagnostic (monitoring only): run `v4bench.run run ... --smoke` with a read-only wrapper around
fl.server.FedAdam.step that records, per round and per tensor, how the aggregated pseudo-gradient |delta| compares with
FedAdam's tau and how large the server step is. Usage: python -B tools_x/diag_tau.py <out.json> <v4bench.run args...>"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import fl.server as S  # noqa: E402

STATS = []
_orig = S.FedAdam.step


def _step(self, theta_r, new_state):
    tau, rec = float(self.cfg.tau), {}
    for k in self.keys:
        d = (new_state[k].to(theta_r[k].device) - theta_r[k]).abs().flatten().double()
        nz = d[d > 0]
        rec[k] = {"numel": d.numel(), "frac_nonzero": float((d > 0).double().mean()),
                  "frac_below_tau": float((d < tau).double().mean()),
                  "frac_nonzero_below_tau": float((nz < tau).double().mean()) if nz.numel() else None,
                  "med_abs_delta_nonzero": float(nz.median()) if nz.numel() else 0.0, "max_abs_delta": float(d.max())}
    out = _orig(self, theta_r, new_state)
    for k in self.keys:
        st = (out[k] - theta_r[k]).abs().flatten().double()
        nzs = st[st > 0]
        rec[k]["med_step_nonzero"] = float(nzs.median()) if nzs.numel() else 0.0
        rec[k]["max_step"] = float(st.max())
    STATS.append(rec)
    return out


S.FedAdam.step = _step
from v4bench.run import main  # noqa: E402

rc = main(sys.argv[2:])
with open(sys.argv[1], "w") as f:
    json.dump(STATS, f)
skip = min(25, max(0, len(STATS) - 5))          # skip the FL_CONST warm-up rounds (first 1 % of T)
tail = STATS[skip:]
print(f"[diag_tau] rounds recorded {len(STATS)}, summarised rounds {skip}..{len(STATS) - 1}")
for k in (tail[0] if tail else {}):
    def avg(f):
        v = [r[k][f] for r in tail if r[k][f] is not None]
        return sum(v) / len(v) if v else None
    print(f"  {k:28s} n={tail[0][k]['numel']:>8d} nonzero={avg('frac_nonzero'):.3f} below_tau={avg('frac_below_tau'):.3f} "
          f"nz_below_tau={avg('frac_nonzero_below_tau') or 0:.3f} med|d|nz={avg('med_abs_delta_nonzero'):.2e} "
          f"max|d|={avg('max_abs_delta'):.2e} med_step_nz={avg('med_step_nonzero'):.2e} max_step={avg('max_step'):.2e}")
sys.exit(rc)
