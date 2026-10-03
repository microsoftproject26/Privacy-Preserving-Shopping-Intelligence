"""Local-only training (LO): one client trained alone from theta0 on its own device, procedurally.

Recipe (`LORecipe`, frozen and digested BEFORE any held-out query is evaluated): theta0 digest, seed, total passes,
evaluation budgets (e.g. 2 / 4 / 6 passes), local solver. `run_local_only`:
  * checks the shared-bootstrap provenance: the given theta0 must hash to the recipe's theta0 digest (the same
    per-seed initialization as C / FA / FP), otherwise it refuses;
  * loads theta0 once into the worker model; there is NO channel, NO download after the start, NO upload and NO
    aggregation (the process-wide comm ledger is untouched);
  * ONE persistent AdamW for the whole trajectory; `passes` passes over the client's own valid examples at the local
    batch size (tail batch at its own size); an optional LR schedule over the client's own steps; the family's
    post_step (SASRec recentering) after every step, exactly as in the central recipe;
  * after each evaluation budget, the eval hook runs (eval mode, under a forked RNG so it cannot perturb training);
  * returns metrics + checksums only; the dense state is discarded (the worker model is overwritten by the next
    client's theta0 load). A client with no valid example gets no update and is evaluated with theta0 (recorded).
Replay = the same call with the same recipe; results are checksum-repeatable.

LR shape: `lr_shape="s_efe"` is the study's LO recipe — step t of T uses
solver.lr (the peak) * s(6 * (t + 1) / T) with `exposure_point="end"`, and the recipe MUST carry exposure_point
explicitly (a missing key is refused; `from_dict` refuses a config without it). `lr_shape="constant"` (solver.lr at
every step, no exposure point) exists for unit fixtures and older suites only; an external `lr_schedule` callable is
accepted only with the constant shape (regression tests).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor

from .client import ClientData, LocalSolver, local_passes, make_optimizer, select_rows, valid_rows
from .numerics import assert_strict_fp32, derive_seed, seeded_global_rng, state_digest
from .schedule import check_exposure_point, lo_lr_schedule

LR_SHAPES = ("constant", "s_efe")


class ProvenanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class LORecipe:
    theta0_digest: str
    seed: int
    passes: int = 6
    eval_at: tuple = (2, 4, 6)
    solver: LocalSolver = field(default_factory=lambda: LocalSolver(lr=1e-3, passes=6))
    lr_shape: str = "constant"
    exposure_point: str | None = None

    def __post_init__(self):
        if any(b < 1 or b > self.passes for b in self.eval_at) or list(self.eval_at) != sorted(set(self.eval_at)):
            raise ValueError("eval_at must be strictly increasing budgets within [1, passes]")
        if self.solver.passes != self.passes:
            raise ValueError("recipe passes and solver passes must agree (one LO budget, frozen in the recipe)")
        if self.lr_shape not in LR_SHAPES:
            raise ValueError(f"lr_shape must be one of {LR_SHAPES}")
        if self.lr_shape == "s_efe":
            check_exposure_point(self.exposure_point)          # P1: explicit, never a code default
        elif self.exposure_point is not None:
            raise ValueError("a constant-LR LO recipe has no exposure point")

    @classmethod
    def from_dict(cls, d: Mapping) -> LORecipe:
        """The launcher path: an LO run config without an explicit exposure_point key is refused."""
        if d.get("exposure_point") is None:
            raise ValueError("LO run config has no explicit exposure_point")
        d = dict(d)
        if isinstance(d.get("solver"), Mapping):
            d["solver"] = LocalSolver(**d["solver"])
        d["eval_at"] = tuple(d.get("eval_at", (2, 4, 6)))
        d.setdefault("lr_shape", "s_efe")
        return cls(**d)

    def schedule(self) -> Callable[[int, int], float] | None:
        if self.lr_shape == "s_efe":
            return lo_lr_schedule(self.solver.lr, self.passes, exposure_point=self.exposure_point)
        return None

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class LOResult:
    key: str
    recipe_digest: str
    n_valid: int
    n_consumed: int
    steps: int
    evals: dict
    final_digest: str
    no_label: bool

    def record(self) -> dict:
        return asdict(self)


class LORunner:
    """Procedural LO over many clients on ONE worker: theta0 provenance is verified once and a private copy of theta0
    is kept on the worker's device (so it cannot be mutated between clients); each `run` reloads it (no carry-over)."""

    def __init__(self, adapter, theta0: Mapping[str, Tensor], recipe: LORecipe, *,
                 digest_fn: Callable = state_digest, lr_schedule: Callable[[int, int], float] | None = None):
        assert_strict_fp32()
        if state_digest(theta0) != recipe.theta0_digest:
            raise ProvenanceError("LO theta0 does not match the recipe's registered initialization digest")
        if lr_schedule is not None and recipe.lr_shape != "constant":
            raise ValueError("the s_efe recipe defines its own schedule; an external lr_schedule is regression-only")
        self.adapter, self.recipe, self.digest_fn = adapter, recipe, digest_fn
        self.lr_schedule = lr_schedule if lr_schedule is not None else recipe.schedule()
        self.theta0 = {k: v.detach().to(adapter.device).clone() for k, v in theta0.items()}

    def run(self, client: ClientData, eval_hook: Callable | None = None) -> LOResult:
        assert_strict_fp32()
        adapter, recipe, solver = self.adapter, self.recipe, self.recipe.solver
        adapter.load_state_(self.theta0)
        idx = valid_rows(client.examples, adapter.target_key, adapter.loss_mask_key)
        n = int(idx.numel())
        evals: dict = {}
        dev = adapter.device
        fork_devs = [dev] if dev.type == "cuda" else []

        def _eval(budget: int) -> None:
            if eval_hook is None:
                return
            with torch.random.fork_rng(devices=fork_devs), torch.no_grad():
                adapter.module.eval()
                evals[budget] = eval_hook(adapter, budget)
            adapter.module.train()

        if n == 0:
            for b in recipe.eval_at:
                _eval(b)
            return LOResult(client.key, recipe.digest(), 0, 0, 0, evals, self.digest_fn(adapter.extract_shared()),
                            True)
        ex = select_rows(client.examples, idx, dev)
        opt = make_optimizer(adapter, solver)                              # persistent for the whole trajectory
        total_steps = recipe.passes * (-(-n // solver.batch_size))
        lr_fn = (lambda step: self.lr_schedule(step, total_steps)) if self.lr_schedule is not None else None
        visit_seed = derive_seed(int(recipe.seed), "LO", client.key)
        budgets = set(recipe.eval_at)
        with seeded_global_rng(derive_seed(visit_seed, "dropout"), dev):
            steps, _ = local_passes(adapter, ex, n, passes=recipe.passes, batch_size=solver.batch_size,
                                    visit_seed=visit_seed, opt=opt, clip=solver.clip, lr_fn=lr_fn,
                                    after_pass=lambda k: _eval(k) if k in budgets else None)
        final = self.digest_fn(adapter.extract_shared())
        del opt                                                             # discard: nothing dense is retained
        return LOResult(client.key, recipe.digest(), n, recipe.passes * n, steps, evals, final, False)


def run_local_only(adapter, theta0: Mapping[str, Tensor], client: ClientData, recipe: LORecipe, *,
                   eval_hook: Callable | None = None,
                   lr_schedule: Callable[[int, int], float] | None = None) -> LOResult:
    """Single-client LO (provenance checked on every call)."""
    return LORunner(adapter, theta0, recipe, lr_schedule=lr_schedule).run(client, eval_hook)
