"""Full-model on-device fine-tuning: FT_C (fine-tune the cloud model), FT_FA (fine-tune the federated model) and
FT_KD (fine-tune a distilled small model).

For each evaluation client u:
  1. copy the given base SMALL model (FT_C: central C_SMALL; FT_FA: the federated FA model; FT_KD: KD_SMALL) — its
     broadcast-state digest must equal the recipe's `base_digest` (provenance, checked once per runner);
  2. fine-tune ALL trainable parameters (the adapter's parameter groups, exactly the client-update set) on the
     client's OWN permitted support rows: the same valid-row definition as FA / PF / CPF (client.valid_rows:
     target_class >= 0 and loss_mask), with the same conventions as the client update — a FRESH AdamW (betas, eps,
     the ndim >= 2 weight-decay rule), gradient clip of the shared parameters (total L2), local batch 16 with the tail
     batch at its own size, one independent permutation per pass, the family's post_step after every step — and a
     CONSTANT LR = lr_frac x the SMALL peak LR, no early stopping;
  3. after each budget (passes 1, 2, 4 of ONE trajectory; with a constant LR and per-pass permutations
     seeded by (seed, "FT", client key, pass), the state after k passes equals a separate k-pass run bit for bit,
     which the tests check), call the caller's eval hook (the shared evaluator on the client's validation rows)
     in eval mode under a forked RNG;
  4. discard the copy (the worker is overwritten by the next client's base load; nothing is aggregated or uploaded).
Deterministic per (seed, client key); the stream deliberately excludes the arm and the LR, so FT_C / FT_FA / FT_KD and
both LR fractions see the same batch order (paired comparison).

Causality (strictly causal support): the CALLER supplies the support rows (warm clients: TRAIN examples with
target_ts < TRAIN_END; cold / new users: the support labels before decision_ts(q)). `support_cutoff` is REQUIRED
(None is refused): every valid support row must carry `ts_key` < cutoff, otherwise the run is REFUSED (never silently
filtered). Warm clients: cutoff = TRAIN_END. A client evaluated on SEVERAL queries is fine-tuned ONCE with
cutoff = the EARLIEST decision_ts among its evaluated queries (`support_cutoff_for`), so the one fine-tuned copy is
causal for every query it scores (support labels between two queries are then not used; a per-query variant would
need one fine-tune per query and is not provided).
Worker isolation: FTRunner overwrites its adapter, so an adapter that is a Server model (fl_role "server") or a live
FLRun worker (fl_role "worker") is refused; use a dedicated FT worker model.

Per client record: n_support_events (valid support rows), updates (optimizer steps) per budget, consumed exposures
per budget, bytes_up = 0 (nothing leaves the device), bytes_down_once = the base model's shared bytes (the one model
download the arm needs anyway), the final state digest, and the eval hook's outputs per budget.
Budget selection (passes x lr_frac) is done once, on a fixed calibration panel; this module only provides the grid.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .client import ClientData, LocalSolver, local_passes, make_optimizer, select_rows, valid_rows
from .numerics import assert_strict_fp32, derive_seed, seeded_global_rng, state_digest

FT_ARMS = ("FT_C", "FT_FA", "FT_KD")
FT_PASSES = (1, 2, 4)                  # budget grid (passes)
FT_LR_FRACS = (0.1, 0.3)               # LR grid (x the SMALL peak LR)


class FTProvenanceError(RuntimeError):
    pass


class CausalityError(ValueError):
    pass


@dataclass(frozen=True)
class FTRecipe:
    arm: str                           # FT_C | FT_FA | FT_KD
    base_digest: str                   # state_digest of the base model's broadcast state
    seed: int
    peak_lr: float                     # the SMALL peak LR (absolute)
    lr_frac: float                     # in FT_LR_FRACS
    passes: tuple = FT_PASSES          # evaluation budgets on one trajectory (subset of FT_PASSES)
    batch_size: int = 16
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-5
    clip: float | None = 1.0
    fused: bool | None = None
    eval_base: bool = False            # also evaluate the untouched base copy (budget 0)

    def __post_init__(self):
        if self.arm not in FT_ARMS:
            raise ValueError(f"arm must be one of {FT_ARMS}")
        if self.lr_frac not in FT_LR_FRACS:
            raise ValueError(f"lr_frac must be one of {FT_LR_FRACS}")
        ps = tuple(self.passes)
        if not ps or list(ps) != sorted(set(ps)) or not set(ps) <= set(FT_PASSES):
            raise ValueError(f"passes must be a strictly increasing subset of {FT_PASSES}")
        if not self.peak_lr > 0:
            raise ValueError("peak_lr must be > 0")
        if self.batch_size != 16:
            raise ValueError("the fine-tuning batch size is 16")

    @property
    def lr(self) -> float:
        return float(self.lr_frac) * float(self.peak_lr)

    def solver(self) -> LocalSolver:
        return LocalSolver(lr=self.lr, betas=tuple(self.betas), eps=self.eps, weight_decay=self.weight_decay,
                           clip=self.clip, batch_size=self.batch_size, passes=max(self.passes), fused=self.fused)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class FTResult:
    key: str
    arm: str
    recipe_digest: str
    lr_frac: float
    lr: float
    n_support_events: int
    updates: dict                      # budget (passes) -> optimizer steps
    n_consumed: dict                   # budget -> passes x n_support_events
    evals: dict                        # budget -> eval hook output (0 = base, if eval_base)
    bytes_up: int
    bytes_down_once: int
    final_digest: str
    no_support: bool

    def record(self) -> dict:
        return asdict(self)


def support_cutoff_for(decision_ts) -> object:
    """The FT cutoff for a client with several evaluated queries: the earliest decision_ts (see the docstring)."""
    ts = list(decision_ts)
    if not ts:
        raise CausalityError("no decision_ts given: the cutoff cannot be derived")
    return min(ts)


def check_support_causal(examples: Mapping[str, Tensor], idx: Tensor, *, cutoff, ts_key: str = "target_ts") -> None:
    if ts_key not in examples:
        raise CausalityError(f"support_cutoff given but the support has no {ts_key!r} column")
    ts = examples[ts_key].index_select(0, idx.to(examples[ts_key].device))
    bad = int((ts >= cutoff).sum())
    if bad:
        raise CausalityError(f"{bad} support row(s) at or after the cutoff: FT support must be strictly causal")


class FTRunner:
    """Per-client full-model fine-tuning on ONE worker; the base is verified once and kept as a private copy."""

    def __init__(self, adapter, base_state: Mapping[str, Tensor], recipe: FTRecipe):
        assert_strict_fp32()
        role = getattr(adapter, "fl_role", None)
        if role in ("server", "worker"):
            raise ValueError(f"FTRunner would overwrite a {role} model: give it a dedicated FT worker adapter")
        if state_digest(base_state) != recipe.base_digest:
            raise FTProvenanceError(f"{recipe.arm}: base model digest does not match the recipe's base_digest")
        self.adapter, self.recipe = adapter, recipe
        self.base = {k: v.detach().to(adapter.device).clone() for k, v in base_state.items()}
        self.bytes_down_once = int(adapter.manifest.shared_bytes)
        adapter.fl_role = "ft"

    def run(self, client: ClientData, eval_hook: Callable | None = None, *, support_cutoff,
            ts_key: str = "target_ts") -> FTResult:
        """`support_cutoff` is required: TRAIN_END (warm) or `support_cutoff_for(decision_ts of the client's
        queries)` (cold / U_NEW)."""
        assert_strict_fp32()
        if support_cutoff is None:
            raise CausalityError("support_cutoff is required (TRAIN_END, or the earliest decision_ts of the client)")
        adapter, recipe = self.adapter, self.recipe
        solver = recipe.solver()
        adapter.load_state_(self.base)                                  # a fresh copy of the base for this client
        idx = valid_rows(client.examples, adapter.target_key, adapter.loss_mask_key)
        n = int(idx.numel())
        if n:
            check_support_causal(client.examples, idx, cutoff=support_cutoff, ts_key=ts_key)
        dev = adapter.device
        fork_devs = [dev] if dev.type == "cuda" else []
        evals: dict = {}
        updates: dict = {}
        spp = -(-n // solver.batch_size) if n else 0

        def _eval(budget: int) -> None:
            if eval_hook is None:
                return
            with torch.random.fork_rng(devices=fork_devs), torch.no_grad():
                adapter.module.eval()
                evals[budget] = eval_hook(adapter, budget)
            adapter.module.train()

        if recipe.eval_base:
            _eval(0)
        if n == 0:                                                      # no support: the base itself, recorded
            for b in recipe.passes:
                updates[b] = 0
                _eval(b)
            return FTResult(client.key, recipe.arm, recipe.digest(), recipe.lr_frac, recipe.lr, 0, updates,
                            {b: 0 for b in recipe.passes}, evals, 0, self.bytes_down_once,
                            state_digest(adapter.extract_shared()), True)
        ex = select_rows(client.examples, idx, dev)
        opt = make_optimizer(adapter, solver)                           # fresh, persistent over the trajectory
        visit_seed = derive_seed(int(recipe.seed), "FT", client.key)
        budgets = set(recipe.passes)

        def _after(k: int) -> None:
            if k in budgets:
                updates[k] = k * spp
                _eval(k)

        with seeded_global_rng(derive_seed(visit_seed, "dropout"), dev):
            steps, _ = local_passes(adapter, ex, n, passes=solver.passes, batch_size=solver.batch_size,
                                    visit_seed=visit_seed, opt=opt, clip=solver.clip, after_pass=_after)
        if steps != solver.passes * spp:
            raise RuntimeError("FT step count mismatch")
        final = state_digest(adapter.extract_shared())
        del opt                                                         # discard: never aggregated, never uploaded
        return FTResult(client.key, recipe.arm, recipe.digest(), recipe.lr_frac, recipe.lr, n, updates,
                        {b: b * n for b in recipe.passes}, evals, 0, self.bytes_down_once, final, False)


def run_ft_grid(adapter, base_state: Mapping[str, Tensor], client: ClientData, *, arm: str, base_digest: str,
                seed: int, peak_lr: float, eval_hook: Callable | None = None,
                lr_fracs: Sequence[float] = FT_LR_FRACS, passes: Sequence[int] = FT_PASSES,
                support_cutoff, ts_key: str = "target_ts", **recipe_kw) -> dict:
    """The full grid for one client: one trajectory per LR fraction, evaluated at every pass budget.
    Returns {"cells": {(lr_frac, passes): eval}, "records": [FTResult.record() per lr_frac]}."""
    cells, records = {}, []
    for f in lr_fracs:
        rec = FTRecipe(arm=arm, base_digest=base_digest, seed=seed, peak_lr=peak_lr, lr_frac=f,
                       passes=tuple(passes), **recipe_kw)
        res = FTRunner(adapter, base_state, rec).run(client, eval_hook, support_cutoff=support_cutoff,
                                                     ts_key=ts_key)
        for b in rec.passes:
            cells[(f, b)] = res.evals.get(b)
        records.append(res.record())
    return {"cells": cells, "records": records}
