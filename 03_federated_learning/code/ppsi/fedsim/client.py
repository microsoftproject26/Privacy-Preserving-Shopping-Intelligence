"""The client update (FA / FP / PF) and the local-pass engine shared with local-only training (LO).

Methods: FA = FedAvg, FP = FedProx, PF = FedAvg with a personal on-device query vector.

Client visit (FA / FP / PF), `client_update`:
  1. receive theta_r: the broadcast shared state is copied into the worker's module (aliases follow; FIXED buffers
     are verified identical);
  2. a FRESH AdamW (the given lr, betas, eps, weight decay with the central ndim >= 2 rule) and gradient clipping of
     the shared parameters to `clip` (total L2 norm, aliases once);
  3. `passes` local passes over the client's OWN valid examples (target_class >= 0 and loss_mask where present); each
     pass is an independent permutation from a seed derived from (seed, round, client key, pass); local batch 16 and
     the last partial batch at its own size;
  4. per-step loss (the stated normalization):
         L_t = (1/|B_t|) sum_{i in B_t} CE_full_K(i)                              decision-weighted mean over the batch
             + (mu/2) * sum_{j in shared scalars, aliases once} (theta_j - stopgrad(theta_r,j))^2   once per step, FP
             + (lambda/2) * ||p_u||^2                                                 once per step, PF only
     The proximal term is applied through its exact gradient mu * (theta - theta_r), added to the CE gradient BEFORE
     clipping (identical to putting the penalty in the scalar loss; `prox_penalty` gives the scalar for checks).
     The anchor is the broadcast theta_r itself (for SASRec, the server-recentered state, i.e. the same gauge as the
     per-step recentered local model). Parameters that receive no gradient are not moved by AdamW and stay at theta_r,
     so their proximal gradient is exactly 0 and is skipped;
  5. optimizer step, then the family's post_step (SASRec recentering of W_out / b only: never optimizer moments,
     never p_u);
  6. return the upload (shared canonical tensors) and n_consumed = passes * n_valid (every loss-contributing exposure,
     tails included); PF also returns the updated private state, which stays with the client.

PF (`pf` given): q_u = q_shared + p_u with p_u in R^{query_dim}, zero-initialized, a separate PERSISTENT AdamW
(same betas / eps, weight decay 0; the L2 penalty is in the loss), personal LR `pf.lr`, and a separate clip of p's
gradient, so the shared update rule is exactly the FA rule when lambda = 0 and the personal LR is 0.

Client SGD with momentum: `LocalSolver(optimizer="sgd_m")` replaces the fresh AdamW by a FRESH torch SGD per visit (momentum 0.9,
dampening 0, no Nesterov, no weight decay; the momentum buffer is created at the visit's first step and discarded with
the visit, so clients stay stateless); clipping to `clip` is unchanged; betas / eps / weight_decay / fused are unused.

RNG: dropout draws from the global RNG, which is forked and seeded per visit from derive_seed(seed, "visit", round,
key) (no method name, so FA and FP(mu=0) draw identical streams; no attempt number, so a retry is identical).

Item-table learning-rate multiplier (OFF unless an ItemLRSolver is given): `ItemLRSolver(item_lr_mult=m)` (a LocalSolver subclass, so
LocalSolver's fields and every digest that serialises a plain LocalSolver are unchanged) makes `make_optimizer` put the
trainable item tables (ITEM_LR_KEYS) into their own param group(s) at LR x m (tag "lr_mult"); `_set_lr` keeps the ratio.
A plain LocalSolver builds exactly the optimizer it built before, and `_set_lr` writes exactly `lr` into every group
without an "lr_mult" tag.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F
from torch import Tensor

from .numerics import assert_strict_fp32, derive_seed, generator, seeded_global_rng


# ------------------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class LocalSolver:
    lr: float
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-5
    clip: float | None = 1.0
    batch_size: int = 16
    passes: int = 2
    optimizer: str = "adamw"      # "sgd" exists ONLY for the pooled-step sanity test; "sgd_m" = client SGD with momentum
    fused: bool | None = None  # AdamW fused kernel (the GPU recipe uses fused=True); None = torch default

    def __post_init__(self):
        if self.optimizer not in ("adamw", "sgd", SGD_M):
            raise ValueError(f"unknown local optimizer {self.optimizer!r}")
        if self.batch_size < 1 or self.passes < 1:
            raise ValueError("batch_size and passes must be >= 1")


SGD_M = "sgd_m"                   # client SGD with heavy-ball momentum
SGD_M_MOMENTUM = 0.9              # fixed constant (not a LocalSolver field, so existing config digests stay unchanged)
ITEM_LR_KEYS = ("item_embed.weight", "output_embed", "output_bias")   # = runtime.FROZEN_ITEM_KEYS


@dataclass(frozen=True)
class ItemLRSolver(LocalSolver):
    """A LocalSolver whose optimizer gives the trainable item tables (ITEM_LR_KEYS present among
    the shared parameters) their own param group(s) at LR x `item_lr_mult` (group key "lr_mult"). A SUBCLASS, not a
    LocalSolver field: LocalSolver's field set (and so every RunConfig / LORecipe digest that serialises it) is unchanged;
    only this solver carries `item_lr_mult` (runtime.RunConfig.from_dict builds it via solver_from_dict)."""
    item_lr_mult: float | None = None

    def __post_init__(self):
        super().__post_init__()
        import math
        m = self.item_lr_mult
        if isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(m) or m <= 0:
            raise ValueError(f"item_lr_mult must be a finite float > 0, not {m!r}")


def solver_from_dict(d: Mapping) -> LocalSolver:
    """The solver of a config dict — an ItemLRSolver iff the dict carries a non-None item_lr_mult, else the plain
    LocalSolver exactly as before (a None item_lr_mult key is dropped)."""
    d = dict(d)
    if d.get("item_lr_mult") is not None:
        return ItemLRSolver(**d)
    d.pop("item_lr_mult", None)
    return LocalSolver(**d)


@dataclass(frozen=True)
class PFConfig:
    lr: float                     # personal LR (default choice: equal to the shared peak LR)
    lam: float = 1e-4             # coefficient of ||p||^2 / 2
    clip: float | None = 1.0   # separate clip of p's gradient


@dataclass
class PersonalState:
    """A client's private state: p_u and its persistent AdamW moments. Never uploaded."""
    p: Tensor
    opt_state: dict | None = None
    visits: int = 0
    n_consumed: int = 0

    def clone(self) -> PersonalState:
        st = None if self.opt_state is None else {k: v.detach().clone() for k, v in self.opt_state.items()}
        return PersonalState(self.p.detach().clone(), st, self.visits, self.n_consumed)


@dataclass
class ClientData:
    key: str                      # logical client key (collision-free; also the private-state key)
    examples: dict                # the client's OWN examples only: tensors with leading dim n_u


@dataclass
class ClientResult:
    key: str
    upload: OrderedDict[str, Tensor]   # shared canonical tensors only
    n_consumed: int
    n_valid: int
    steps: int
    steps_per_pass: int
    loss_sum: float                       # sum over steps of the batch-mean CE (monitoring only)
    personal: PersonalState | None = None
    attempts: int = 1
    wasted_steps: int = 0                 # steps done by failed attempts (never aggregated)


class InjectedFault(RuntimeError):
    def __init__(self, msg: str, steps_done: int = 0):
        super().__init__(msg)
        self.steps_done = steps_done

    def __reduce__(self):  # picklable across the process pool
        return (InjectedFault, (str(self), self.steps_done))


@dataclass(frozen=True)
class FaultPlan:
    """Test-only failure injection: raise after `step` optimizer steps of (client key, attempt)."""
    faults: frozenset = field(default_factory=frozenset)

    def check(self, key: str, attempt: int, step: int) -> None:
        if (key, attempt, step) in self.faults:
            raise InjectedFault(f"injected failure: client {key} attempt {attempt} after step {step}", step)


# ------------------------------------------------------------------------------------------------ helpers
def valid_rows(examples: Mapping[str, Tensor], target_key: str = "target_class",
               loss_mask_key: str = "loss_mask") -> Tensor:
    ok = examples[target_key] >= 0
    if loss_mask_key in examples:
        ok = ok & examples[loss_mask_key].to(torch.bool)
    return ok.nonzero(as_tuple=True)[0]


def n_examples(examples: Mapping[str, Tensor]) -> int:
    sizes = {int(v.shape[0]) for v in examples.values() if isinstance(v, Tensor)}
    if len(sizes) != 1:
        raise ValueError(f"client examples must share one leading dimension, got {sorted(sizes)}")
    return sizes.pop()


def select_rows(examples: Mapping[str, Tensor], idx: Tensor, device: torch.device | None = None) -> dict:
    """Row subset of every per-row tensor; non-tensor metadata (feature-name arrays, feature_view) passes through."""
    out = {}
    for k, v in examples.items():
        if not isinstance(v, Tensor):
            out[k] = v
            continue
        t = v.index_select(0, idx.to(v.device))
        out[k] = t.to(device) if device is not None and k != "lengths" else t
    return out


def item_lr_groups(adapter, groups: list, lr: float, mult: float) -> list:
    """`groups` (adapter.param_groups) with the item-table parameters moved into their own groups (one per source
    group, same options) at LR lr x mult, tagged "lr_mult" (so _set_lr keeps the ratio). Refused when no item table is a
    trainable shared parameter (frozen tables / a family without them): a multiplier must never be a silent no-op."""
    item_ids = {id(p) for k, p in zip(adapter.manifest.shared_keys, adapter.shared_parameters()) if k in ITEM_LR_KEYS}
    if not item_ids:
        raise ValueError(f"item_lr_mult needs trainable item tables {ITEM_LR_KEYS} among the shared parameters "
                         "(frozen item tables take no item-LR multiplier)")
    rest, items = [], []
    for g in groups:
        keep = [p for p in g["params"] if id(p) not in item_ids]
        mine = [p for p in g["params"] if id(p) in item_ids]
        rest.append(dict(g, params=keep))
        if mine:
            items.append(dict(g, params=mine, lr=lr * float(mult), lr_mult=float(mult)))
    return rest + items


def make_optimizer(adapter, solver: LocalSolver, lr: float | None = None) -> torch.optim.Optimizer:
    lr = solver.lr if lr is None else lr
    mult = getattr(solver, "item_lr_mult", None)          # ItemLRSolver only; None = the plain path
    if mult is not None:
        return _make_optimizer_item_lr(adapter, solver, lr, float(mult))
    if solver.optimizer == "sgd":
        return torch.optim.SGD(adapter.param_groups(0.0), lr=lr, momentum=0.0)
    if solver.optimizer == SGD_M:                         # fresh per visit, heavy-ball 0.9, no decay
        return torch.optim.SGD(adapter.param_groups(0.0), lr=lr, momentum=SGD_M_MOMENTUM, dampening=0.0,
                               nesterov=False, weight_decay=0.0)
    return torch.optim.AdamW(adapter.param_groups(solver.weight_decay), lr=lr, betas=tuple(solver.betas),
                             eps=solver.eps, fused=solver.fused)


def _make_optimizer_item_lr(adapter, solver: LocalSolver, lr: float, mult: float) -> torch.optim.Optimizer:
    """make_optimizer's three optimizers with the item tables in their own lr x mult group(s)."""
    if solver.optimizer == "sgd":
        return torch.optim.SGD(item_lr_groups(adapter, adapter.param_groups(0.0), lr, mult), lr=lr, momentum=0.0)
    if solver.optimizer == SGD_M:
        return torch.optim.SGD(item_lr_groups(adapter, adapter.param_groups(0.0), lr, mult), lr=lr,
                               momentum=SGD_M_MOMENTUM, dampening=0.0, nesterov=False, weight_decay=0.0)
    return torch.optim.AdamW(item_lr_groups(adapter, adapter.param_groups(solver.weight_decay), lr, mult), lr=lr,
                             betas=tuple(solver.betas), eps=solver.eps, fused=solver.fused)


def _add_prox_grad_(params: list, anchors: list, mu: float) -> None:
    """grad += mu * (theta - theta_r): the exact gradient of (mu/2)||theta - stopgrad(theta_r)||^2."""
    ps = [p for p in params if p.grad is not None]
    if not ps:
        return
    an = [a for p, a in zip(params, anchors) if p.grad is not None]
    diffs = torch._foreach_sub([p.detach() for p in ps], an)
    torch._foreach_add_([p.grad for p in ps], diffs, alpha=float(mu))


def prox_penalty(params: list, anchors: list, mu: float) -> Tensor:
    """(mu/2) * sum ||theta - stopgrad(theta_r)||^2 over shared canonical params (aliases once) — reference scalar."""
    return 0.5 * mu * sum(((p - a.detach()) ** 2).sum() for p, a in zip(params, anchors))


def batch_ce(logits: Tensor, target: Tensor) -> Tensor:
    """Decision-weighted mean CE over the full K for one batch (every row here is a valid decision)."""
    return F.cross_entropy(logits, target, reduction="sum") / target.shape[0]


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr if "lr_mult" not in g else lr * g["lr_mult"]   # an item-LR group keeps its ratio


def local_passes(adapter, ex: Mapping[str, Tensor], n: int, *, passes: int, batch_size: int, visit_seed: int,
                 opt: torch.optim.Optimizer, clip: float | None, lr_fn: Callable[[int], float] | None = None,
                 mu: float = 0.0, anchors: list | None = None, p: Tensor | None = None,
                 popt: torch.optim.Optimizer | None = None, pf: PFConfig | None = None,
                 fault: FaultPlan | None = None, key: str = "", attempt: int = 0,
                 after_pass: Callable[[int], None] | None = None, pass0: int = 0, step0: int = 0) -> tuple:
    """Run passes [pass0, pass0 + passes) over `ex` (valid rows only). Returns (steps_done, loss_sum)."""
    shared = adapter.shared_parameters()
    tkey = adapter.target_key
    steps, loss_sum = step0, 0.0
    adapter.module.train()
    for ps in range(pass0, pass0 + passes):
        perm = torch.randperm(n, generator=generator(derive_seed(visit_seed, "order", ps)))
        for s in range(0, n, batch_size):
            b = select_rows(ex, perm[s:s + batch_size])
            if lr_fn is not None:
                _set_lr(opt, lr_fn(steps))
            logits = adapter.scores(b, p)
            ce = batch_ce(logits, b[tkey])
            loss = ce + (0.5 * pf.lam) * (p * p).sum() if (p is not None and pf is not None and pf.lam > 0) else ce
            opt.zero_grad(set_to_none=True)
            if popt is not None:
                popt.zero_grad(set_to_none=True)
            loss.backward()
            if mu > 0.0:
                _add_prox_grad_(shared, anchors, mu)
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(shared, clip)
            if p is not None and pf is not None and pf.clip is not None:
                torch.nn.utils.clip_grad_norm_([p], pf.clip)
            opt.step()
            if popt is not None:
                popt.step()
            adapter.post_step()
            steps += 1
            loss_sum += float(ce.detach())
            if fault is not None:
                fault.check(key, attempt, steps)
        if after_pass is not None:
            after_pass(ps + 1)
    return steps - step0, loss_sum


def _restore_personal(adapter, personal: PersonalState | None, pf: PFConfig, solver: LocalSolver):
    dev = adapter.device
    if personal is None:
        p = torch.zeros(adapter.query_dim, dtype=torch.float32, device=dev)
    else:
        if tuple(personal.p.shape) != (adapter.query_dim,):
            raise ValueError(f"p_u shape {tuple(personal.p.shape)} != query dim ({adapter.query_dim},)")
        p = personal.p.detach().to(dev).clone()
    p.requires_grad_(True)
    popt = torch.optim.AdamW([p], lr=pf.lr, betas=tuple(solver.betas), eps=solver.eps, weight_decay=0.0,
                             fused=solver.fused)
    if personal is not None and personal.opt_state is not None:
        # AdamW keeps `step` on the CPU unless fused/capturable, where it lives on the parameter's device
        step_dev = dev if (popt.defaults.get("fused") or popt.defaults.get("capturable")) else torch.device("cpu")
        popt.state[p] = {k: v.detach().to(step_dev if k == "step" else dev).clone()
                         for k, v in personal.opt_state.items()}
    return p, popt


def _assert_anchor_not_aliased(adapter, anchors: list) -> None:
    own = {p.data_ptr() for p in adapter.shared_parameters()}
    if any(a.data_ptr() in own for a in anchors if a.numel()):
        raise RuntimeError("the proximal anchor aliases the worker's own parameters (worker == server module?)")


# ------------------------------------------------------------------------------------------------ client update
def client_update(adapter, theta_r: Mapping[str, Tensor], client: ClientData, solver: LocalSolver, *,
                  round_idx: int, seed: int, mu: float = 0.0, pf: PFConfig | None = None,
                  personal: PersonalState | None = None, fault: FaultPlan | None = None, attempt: int = 0,
                  clone_upload: bool = False) -> ClientResult:
    """One FA / FP / PF client visit (see the module docstring). `theta_r` is the broadcast state, used as-is as the
    proximal anchor (stop-gradient: plain tensors, never parameters of this worker)."""
    assert_strict_fp32()
    if mu < 0:
        raise ValueError("mu must be >= 0")
    if pf is None and personal is not None:
        raise ValueError("personal state given without a PF config")
    adapter.load_state_(theta_r)
    keys = adapter.manifest.shared_keys
    anchors = [theta_r[k] for k in keys]
    if mu > 0:
        _assert_anchor_not_aliased(adapter, anchors)
    n_examples(client.examples)
    idx = valid_rows(client.examples, adapter.target_key, adapter.loss_mask_key)
    n = int(idx.numel())
    spp = -(-n // solver.batch_size) if n else 0
    if n == 0:   # no-op client: no update; PF state (possibly None -> p = 0) is unchanged
        return ClientResult(client.key, adapter.extract_shared(clone_upload), 0, 0, 0, 0, 0.0,
                            personal.clone() if personal is not None else None)
    ex = select_rows(client.examples, idx, adapter.device)
    visit_seed = derive_seed(int(seed), "visit", int(round_idx), client.key)
    opt = make_optimizer(adapter, solver)
    p = popt = None
    if pf is not None:
        p, popt = _restore_personal(adapter, personal, pf, solver)
    with seeded_global_rng(derive_seed(visit_seed, "dropout"), adapter.device):
        steps, loss_sum = local_passes(adapter, ex, n, passes=solver.passes, batch_size=solver.batch_size,
                                       visit_seed=visit_seed, opt=opt, clip=solver.clip, mu=float(mu),
                                       anchors=anchors, p=p, popt=popt, pf=pf, fault=fault, key=client.key,
                                       attempt=attempt)
    new_personal = None
    if pf is not None:
        prev_visits = personal.visits if personal is not None else 0
        prev_n = personal.n_consumed if personal is not None else 0
        new_personal = PersonalState(p.detach().to("cpu").clone(),
                                     {k: v.detach().to("cpu").clone() for k, v in popt.state[p].items()},
                                     prev_visits + 1, prev_n + solver.passes * n)
    return ClientResult(client.key, adapter.extract_shared(clone_upload), solver.passes * n, n, steps, spp,
                        loss_sum, new_personal)


def with_attempts(res: ClientResult, attempts: int, wasted_steps: int) -> ClientResult:
    return replace(res, attempts=attempts, wasted_steps=wasted_steps)
