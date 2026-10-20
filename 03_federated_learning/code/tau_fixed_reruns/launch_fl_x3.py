"""X-REES46 C1 / C2 (POST_HOC_EXPLORATORY, owner "fix every arm trained wrongly"): the UNCHANGED v4int.launch_fl
PHASE-1 path (cold FA / DP arms, code_x1 = code_p2r4 bytes) with in-process changes only:
  (1) FedAdam tau 1e-6 admitted (--server-tau; registered {1e-3, 1e-5}) and FedAdam eta {1e-4, 3e-4, 3e-3} admitted
      beside the registered {0.01, 0.1, 0.003, 0.001}
  (2) a non-smoke MAIN run takes the tuning point from the CLI options (the registered FL_TUNING_SELECTION binding is
      skipped: bind_main_selection -> None); admitted ONLY with --run-dir under runs_x_rees46/COLD or runs_x_rees46/DP
  (3) --x-efe E (6 / 12): budget E EFE, rounds = fa_rounds x E / 6 (ceil), marks every 0.5 EFE
  (4) the FedAdam per-round diagnostic of launch_fl_x.py -> <run-dir>/X_TAU_DIAG.jsonl
Same args as launch_fl.py plus --x-efe / --x-diag-every / --x-diag-first."""
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


DIAG_EVERY = int(_pop("--x-diag-every", "50"))
DIAG_FIRST = int(_pop("--x-diag-first", "20"))
X_EFE = float(_pop("--x-efe", "6.0"))
TAU_REF = 1e-3
X_TAU = float(sys.argv[sys.argv.index("--server-tau") + 1]) if "--server-tau" in sys.argv else 1e-3
if X_EFE not in (6.0, 12.0):
    sys.exit("launch_fl_x3: --x-efe in {6, 12}")
RD = sys.argv[sys.argv.index("--run-dir") + 1] if "--run-dir" in sys.argv else ""
if not any(f"runs_x_rees46/{d}/" in RD for d in ("COLD", "DP")):
    sys.exit("launch_fl_x3: refused - --run-dir must be under runs_x_rees46/COLD or runs_x_rees46/DP")
RUN_DIR = RD
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(sys.argv[sys.argv.index("--gpu") + 1]))

sys.path.insert(0, os.environ.get("X_CODE") or os.getcwd())
from v4int import launch_fl as F  # noqa: E402
import fl.server as S  # noqa: E402

F.FEDADAM_TAUS = tuple(F.FEDADAM_TAUS) + (1e-6,)
F.FEDADAM_EDGE_LRS = tuple(F.FEDADAM_EDGE_LRS) + (1e-4, 3e-4, 3e-3)
_orig_bind = F.bind_main_selection


def bind_main_selection_x(args, *a, **k):
    return None                                # exploratory: the CLI tuning point is recorded in the spec instead


F.bind_main_selection = bind_main_selection_x
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import x_execute_fix  # noqa: E402
X_FIX = x_execute_fix.apply(F)       # the run-own endpoint unit (budget != 6 EFE); see x_execute_fix.py
F.KNOBS["endpoint_efe"] = X_EFE
_orig_rounds = F.fa_rounds


def fa_rounds_x(n_clients, passes=2):
    r = _orig_rounds(n_clients, passes)
    return r if X_EFE == 6.0 else -(-(int(round(10 * X_EFE)) * int(n_clients)) // (F.KNOBS["group"] * 9 * int(passes)))


F.fa_rounds = fa_rounds_x

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
    try:
        _rc = F.main()
    finally:
        try:                                   # the patch record beside the run's own files (never in ARM_SPEC: resume
            if os.path.isdir(RUN_DIR):         # compares the spec; the fix does not change any spec value)
                with open(os.path.join(RUN_DIR, "X_PATCHES.json"), "w", encoding="utf-8") as _f:
                    json.dump(dict(X_FIX, wrapper=os.path.basename(__file__), x_efe=X_EFE), _f, indent=1)
        except OSError:
            pass
    sys.exit(_rc)
