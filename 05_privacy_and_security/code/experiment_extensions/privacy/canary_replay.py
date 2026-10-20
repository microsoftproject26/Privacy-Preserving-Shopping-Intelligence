"""CPU synthetic controls for fixed-roster canary replay using real FLRun.

No filesystem CLI or real-provider entry point exists. The production study
still needs its own source/data/init/protocol/accountant/S_CAL review. Internal
runtime state is private; model_only_payload is the sole release whitelist.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import lru_cache
import math
import secrets

CALIB_TRAIN = 'CALIB_TRAIN_5_100_32768'
REAL_N = 32768
CANARIES = 1000
REPEATS = 10
POPULATION = REAL_N + CANARIES
REAL_DECISIONS = 646067
SHADOW_DECISIONS = REAL_DECISIONS + CANARIES * REPEATS
DP_M = 1024
DP_T = 384
DELTA = 5e-6
_NOISE_ACTIVE = False


class Refused(ValueError):
    pass


def _integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise Refused(name + ' is outside the declared integer range')
    return value


def exact_role(role):
    if role != CALIB_TRAIN:
        raise Refused('only exact CALIB TRAIN synthetic additions are supported')


@lru_cache(maxsize=1)
def _accountant_record():
    from fl.accountant import registration_record
    return registration_record([8.], m=DP_M, N=POPULATION, T=DP_T, delta=DELTA)['records'][0]


def dp_accountant():
    """Recompute the registered STANDARD record for the new fixed population.

    Only the numeric fixed-cell record is returned: the historical helper's
    outdated fixed-size sampling disclosure is not copied into this study.
    This excludes private clip calibration and prior tuning/composition.
    """
    import copy
    record = copy.deepcopy(_accountant_record())
    if (record['N'], record['m'], record['T'], record['q'], record['delta']) != (
            POPULATION, DP_M, DP_T, DP_M / POPULATION, DELTA) or record['achieved_epsilon'] > 8:
        raise Refused('new population STANDARD accountant failed')
    return record


def public_study_declaration(*, role=CALIB_TRAIN):
    exact_role(role)
    return dict(status='PROTOTYPE_NOT_REAL_RUN_READY', train_manifest_id=CALIB_TRAIN,
        real_clients=REAL_N, potential_canaries=CANARIES, fixed_population=POPULATION,
        potential_decisions_normalizer=SHADOW_DECISIONS, examples_per_included_canary=REPEATS,
        dp_sampling='PRIVATE_POISSON', dp_m=DP_M, dp_q=DP_M / POPULATION,
        dp_rounds=DP_T, dp_passes=2, dp_expected_client_passes=DP_T * DP_M / POPULATION * 2,
        non_dp_recipe='PENDING_EXACT_PB100_SOURCE_BINDING',
        comparison='WHOLE_SCHEDULE_CONTRAST_NOT_MATCHED_NOISE',
        release='FINAL_MODEL_ONLY', private_tuning_accounted=False,
        privacy_certified=False, MAIN_allowed=False, TEST_allowed=False)


@dataclass(frozen=True, repr=False)
class PrivateEntropy:
    participation_seed: int
    noise_seed: int

    def __post_init__(self):
        for s in (self.participation_seed, self.noise_seed):
            _integer(s, 'private entropy', 0, 2**63 - 1)
        if self.participation_seed == self.noise_seed:
            raise Refused('private sampling and Gaussian entropy must be independent streams')

    @classmethod
    def draw(cls):
        sampling, noise = secrets.randbits(63), secrets.randbits(63)
        while sampling == noise:
            noise = secrets.randbits(63)
        return cls(sampling, noise)


@contextlib.contextmanager
def private_gaussian(entropy):
    """Change only the secret stream, keeping the inherited Gaussian law.

    Direct FLRun construction builds its own Poisson plan with the separate
    sampling entropy; no ParticipationPlan classmethod is modified here.
    """
    global _NOISE_ACTIVE
    if not isinstance(entropy, PrivateEntropy):
        raise Refused('independent private entropy required')
    if _NOISE_ACTIVE:
        raise Refused('nested private Gaussian context')
    from fl import dp
    original = dp.noise_like

    def noise(keys, template, *, seed, round_idx, std):
        return original(keys, template, seed=entropy.noise_seed, round_idx=round_idx, std=std)

    _NOISE_ACTIVE = True
    try:
        dp.noise_like = noise
        yield
    finally:
        dp.noise_like = original
        _NOISE_ACTIVE = False


def private_poisson_plan(keys, *, group_size, entropy, public_roster_id):
    if not isinstance(entropy, PrivateEntropy):
        raise Refused('independent private entropy required')
    _integer(group_size, 'fixed denominator')
    if group_size > len(keys):
        raise Refused('fixed denominator may not exceed the potential population')
    if not isinstance(public_roster_id, str) or not public_roster_id.startswith('EID_E2_PUBLIC_ROSTER:'):
        raise Refused('synthetic roster needs a distinct public namespace, not an old manifest hash')
    from fl.participation import ParticipationPlan
    return ParticipationPlan.build(keys, seed=entropy.participation_seed,
        manifest_hash=public_roster_id, group_size=group_size, sampling='poisson')


@dataclass(frozen=True)
class ToySchedule:
    rounds: int = 8
    passes: int = 2
    group_size: int = 3
    optimizer_seed: int = 2026
    dropout_p: float = 0.
    mode: str = 'UNFROZEN'
    clip_norm: float = .2
    noise_multiplier: float = .4

    def __post_init__(self):
        _integer(self.rounds, 'toy rounds', 1, 64)
        _integer(self.passes, 'toy passes', 1, 4)
        _integer(self.group_size, 'toy group size', 1, 32)
        _integer(self.optimizer_seed, 'public optimizer seed', 0, 2**63 - 1)
        if self.mode not in ('FROZEN', 'UNFROZEN', 'DP'):
            raise Refused('unsupported toy replay mode')
        if not math.isfinite(self.dropout_p) or not 0 <= self.dropout_p < 1 or (
                self.mode == 'DP' and self.dropout_p != 0):
            raise Refused('DP forbids dropout; toy dropout probability must be valid')
        if not math.isfinite(self.clip_norm) or self.clip_norm <= 0 or \
                not math.isfinite(self.noise_multiplier) or self.noise_multiplier <= 0:
            raise Refused('finite positive toy clipping and noise required')


class PrivateToyReplay:
    """Private CPU fixture state, never a public checkpoint or DP claim."""
    def __init__(self, run, schedule):
        self._run, self._schedule = run, schedule

    def model_only_payload(self):
        from collections import OrderedDict
        import torch
        theta = self._run.server.adapter.broadcast_state(clone=True)
        if not theta or any(t.is_floating_point() and (t.dtype != torch.float32 or not torch.isfinite(t).all())
                            for t in theta.values()):
            raise Refused('finite FP32 final model required')
        return dict(format='EID_CANARY_SYNTHETIC_MODEL_ONLY_1',
            theta=OrderedDict((k, v.detach().cpu().clone()) for k, v in theta.items()),
            mechanism=dict(scope='CPU_SYNTHETIC_CONTROL_ONLY', mode=self._schedule.mode,
                rounds=self._schedule.rounds, local_passes=self._schedule.passes,
                lr_shape='fl_const', optimizer='FRESH_SGDM_PER_VISIT',
                fixed_denominator=self._schedule.group_size if self._schedule.mode == 'DP' else None),
            fixed_cell_accountant_applicable=False, private_tuning_accounted=False, privacy_certified=False)


def replay_synthetic(real_clients, definitions, private_signs, *, adapter, workers,
                     schedule=ToySchedule(), entropy=None, role=CALIB_TRAIN):
    """Run unchanged actual FLRun/client/server on a tiny CPU-only fixture.

    TinyAdapter and toy_real:: names are required. This function deliberately
    cannot consume real data providers or models, even when a role is correct.
    The actual zero/ten counts stay private; the fixed normalizer uses all
    potential canaries and is independent of the membership signs.
    """
    exact_role(role)
    if not isinstance(schedule, ToySchedule):
        raise Refused('exact CPU toy schedule required')
    from fl.synthetic import TinyAdapter
    from fl.client import ClientData, ItemLRSolver, LocalSolver, valid_rows
    from fl.numerics import state_digest, assert_strict_fp32
    from fl.runtime import FLRun, RunConfig, freeze_item_tables, FROZEN_ITEM_KEYS
    from fl.server import Server, ServerOptConfig
    from fl.participation import ParticipationPlan
    from fl.dp import DPConfig
    from .canary_data import CanaryDefinitions, FixedRosterCanaries
    assert_strict_fp32()
    workers = list(workers)
    if not workers or any(type(a) is not TinyAdapter or not 1 <= a.K <= 64 or
            any(t.device.type != 'cpu' for t in a.module.state_dict().values()) for a in [adapter, *workers]):
        raise Refused('prototype admits only small CPU TinyAdapter fixtures')
    if not isinstance(definitions, CanaryDefinitions) or definitions.catalogue_K != adapter.K or definitions.count > 32:
        raise Refused('toy synthetic definitions must match the small model')
    clients = list(real_clients)
    if not 1 <= len(clients) <= 64 or any(not isinstance(c, ClientData) or not c.key.startswith('toy_real::') for c in clients):
        raise Refused('prototype refuses real-provider client identities')
    by_key = {c.key: c for c in clients}
    if len(by_key) != len(clients):
        raise Refused('duplicate toy clients')
    real_counts = {k: int(valid_rows(c.examples).numel()) for k, c in by_key.items()}
    roster = FixedRosterCanaries(tuple(by_key), by_key.__getitem__, definitions, private_signs,
        train_manifest_id=role, repeats=REPEATS)
    actual_counts = {k: int(valid_rows(roster.get_client(k).examples).numel()) for k in roster.keys}
    shadow_decisions = sum(real_counts.values()) + definitions.count * REPEATS
    if schedule.group_size > len(roster.keys):
        raise Refused('fixed cohort cannot exceed toy potential population')
    if schedule.mode == 'DP':
        if not isinstance(entropy, PrivateEntropy):
            raise Refused('DP toy run requires separate private streams')
        plan = private_poisson_plan(roster.keys, group_size=schedule.group_size, entropy=entropy,
            public_roster_id='EID_E2_PUBLIC_ROSTER:CPU_TOY')
        dp = DPConfig(schedule.clip_norm, schedule.noise_multiplier, schedule.group_size)
    else:
        if entropy is not None:
            raise Refused('non-DP toy schedule does not consume private DP entropy')
        plan = ParticipationPlan.build(roster.keys, seed=schedule.optimizer_seed,
            manifest_hash='EID_E2_PUBLIC_ROSTER:CPU_TOY', group_size=schedule.group_size, sampling='sweep')
        dp = None
    frozen = schedule.mode == 'FROZEN'
    if frozen:
        freeze_item_tables(adapter, FROZEN_ITEM_KEYS)
        theta = adapter.broadcast_state(clone=True)
        for worker in workers:
            freeze_item_tables(worker, FROZEN_ITEM_KEYS, source=theta)
    solver_class = LocalSolver if frozen else ItemLRSolver
    solver = solver_class(lr=.1, optimizer='sgd_m', passes=schedule.passes,
        batch_size=16, clip=1., weight_decay=0., **({} if frozen else {'item_lr_mult': .1}))
    config = RunConfig('EID_CANARY_CPU_SYNTHETIC_CONTROL', method='FA', seed=schedule.optimizer_seed,
        lr_peak_local=.1, solver=solver, endpoint_rounds=schedule.rounds, exposure_point='end',
        lr_shape='fl_const', dropout_p=schedule.dropout_p, dp=dp,
        server_opt=ServerOptConfig(name='fedadam', lr=.001, tau=1e-6))
    server = Server(adapter, init_sha256=state_digest(adapter.broadcast_state()))
    # No evaluator, observer, checkpoint manager, metadata writer or filesystem.
    run = FLRun(config, plan, server, roster.get_client, shadow_decisions,
        workers=workers, n_valid_by_key=actual_counts, eval_fn=None, ckpt=None)
    ctx = private_gaussian(entropy) if schedule.mode == 'DP' else contextlib.nullcontext()
    with ctx:
        for _ in range(schedule.rounds):
            run.step()  # real planned/actual equality, original clients and aggregation
    if not run.state.done or run.state.cursor != schedule.rounds or run.state.evals:
        raise Refused('toy replay did not reach the fixed nonadaptive endpoint')
    return PrivateToyReplay(run, schedule)
