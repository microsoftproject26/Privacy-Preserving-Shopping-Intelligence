"""X-REES46 (POST_HOC_EXPLORATORY, owner approval DL 539): the UNCHANGED v4int.launch_fl P2_W path (code_x1 = a copy of
code_p2r4) with three in-process changes, nothing else:
  (1) FedAdam tau = --x-tau (default 1e-6; the registered P2 recipe path fixes tau 1e-3, which W* used). The ARM_SPEC cfg
      server_opt records the tau actually used, so the spec digest differs from every confirmatory run.
  (2) a non-smoke MAIN run from a --p2-recipe file (the confirmatory path admits MAIN only from the registered
      P2_WARM_SELECTION_FINAL); admitted ONLY with --run-dir under runs_x_rees46.
  (3) a FedAdam diagnostic per round (first --x-diag-first rounds, then every --x-diag-every): fraction of
      |aggregate delta| below tau_ref (1e-3) and below the tau used, |delta| quantiles, server step RMS / L2 / max ->
      <run-dir>/X_TAU_DIAG.jsonl. Reads tensors only; the step itself is the original fl.server.FedAdam.step.
Refused outside runs_x_rees46. Same args as launch_fl.py plus --x-tau / --x-diag-every / --x-diag-first."""
import copy
import json
import os
import sys
import time


def _pop(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


X_TAU = float(_pop("--x-tau", "1e-6"))
DIAG_EVERY = int(_pop("--x-diag-every", "50"))
DIAG_FIRST = int(_pop("--x-diag-first", "20"))
TAU_REF = 1e-3
if not (X_TAU > 0 and X_TAU < 1):
    sys.exit(f"launch_fl_x: --x-tau {X_TAU} out of range")
if "--run-dir" not in sys.argv or "runs_x_rees46" not in sys.argv[sys.argv.index("--run-dir") + 1]:
    sys.exit("launch_fl_x: refused - --run-dir must be under runs_x_rees46 (POST_HOC_EXPLORATORY)")
RUN_DIR = sys.argv[sys.argv.index("--run-dir") + 1]
if "--gpu" in sys.argv:                       # as exp_pb100: set before any snapshot import can default it to 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(sys.argv[sys.argv.index("--gpu") + 1]))

sys.path.insert(0, os.environ.get("X_CODE") or os.getcwd())
from v4int import launch_fl as F  # noqa: E402
import fl.server as S  # noqa: E402

_orig_so = F.p2_server_opt


def p2_server_opt_x(rec):
    so = _orig_so(rec)
    if so is not None and so.get("name") == "fedadam":
        so = dict(so, tau=X_TAU)
    return so


F.p2_server_opt = p2_server_opt_x

_orig_tag = F.p2_run_tag


def p2_run_tag_x(rec):
    t = _orig_tag(rec)
    return t + f"_XTAU{X_TAU:g}" if rec.get("server_opt") == "fedadam" else t


F.p2_run_tag = p2_run_tag_x

_orig_pf = F.p2_preflight


def p2_preflight_x(args):
    if (getattr(args, "role", "MAIN") == "MAIN" and getattr(args, "p2_selection", None) is None
            and getattr(args, "p2_recipe", None) is not None):
        a = copy.copy(args)
        a.dry_run = True                       # the only MAIN-specific refusal is "MAIN needs the registered selection"
        return _orig_pf(a)
    return _orig_pf(args)


F.p2_preflight = p2_preflight_x

_orig_step = S.FedAdam.step
_N = {"r": 0}


def _diag_pre(self, theta_r, new_state):
    import torch
    ds = []
    for k in self.keys:
        th = theta_r[k]
        ds.append(torch.sub(new_state[k].to(th.device), th).abs().flatten())
    d = torch.cat(ds)
    n = d.numel()
    nz = d[d > 0]
    q = {}
    sample = d if n <= 4_000_000 else d[torch.randperm(n, device=d.device)[:4_000_000]]
    for p in (0.5, 0.9, 0.99):
        q[f"absdelta_q{int(p * 100)}"] = float(torch.quantile(sample.float(), p))
    rec = {"n": int(n), "n_nonzero": int(nz.numel()),
           "frac_lt_tau_ref_1e-3": float((d < TAU_REF).float().mean()),
           "frac_lt_tau_used": float((d < X_TAU).float().mean()),
           "frac_nonzero_lt_tau_ref_1e-3": float((nz < TAU_REF).float().mean()) if nz.numel() else None,
           "frac_nonzero_lt_tau_used": float((nz < X_TAU).float().mean()) if nz.numel() else None,
           "absdelta_mean": float(d.mean()), "delta_l2": float(torch.linalg.vector_norm(d)), **q}
    return rec


def step_x(self, theta_r, new_state):
    r = _N["r"]
    _N["r"] += 1
    rec = None
    if r < DIAG_FIRST or r % DIAG_EVERY == 0:
        try:
            rec = _diag_pre(self, theta_r, new_state)
        except Exception as e:                 # the diagnostic never blocks a step
            rec = {"diag_error": f"{type(e).__name__}: {e}"}
    out = _orig_step(self, theta_r, new_state)
    if rec is not None:
        try:
            import torch
            st = torch.cat([torch.sub(out[k], theta_r[k]).flatten() for k in self.keys])
            rec.update({"step_l2": float(torch.linalg.vector_norm(st)), "step_rms": float(st.pow(2).mean().sqrt()),
                        "step_absmax": float(st.abs().max()), "eta": float(self.cfg.lr), "tau_used": float(self.cfg.tau)})
        except Exception as e:
            rec["diag_error_post"] = f"{type(e).__name__}: {e}"
        rec.update({"round_index": r, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "label": "POST_HOC_EXPLORATORY"})
        try:
            with open(os.path.join(RUN_DIR, "X_TAU_DIAG.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        except OSError:
            pass
    return out


S.FedAdam.step = step_x

if __name__ == "__main__":
    sys.exit(F.main())
