"""CALIB-only fixed-public-roster update-FTRL/tree and local-message candidate.

All raw state, entropy, counters, timing and validation are INTERNAL. A model-only
whitelist is a separate boundary. Accounting is conditional on the fixed public
roster, initializer, vocabulary/preprocessing and configuration; it does not
certify earlier pretraining, model/eta selection or finite-precision randomness.
No inherited source is patched. Real training needs a newly bound protocol,
actual CPU provider proof and independent exact-source/run accountant review.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
from types import SimpleNamespace
import ppsi_paths

BASE = ppsi_paths.SERVER_BASE
EXT = ppsi_paths.EXTENSIONS_ROOT
WORK, CODE = EXT / 'ppsi_v4', EXT / 'ppsi_v4/code'
DATA = BASE / 'shared_data/rees46_nextitem_v1'
DONOR = BASE / 'Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4'
INIT = DONOR / 'runs_exp_pb100/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_PB100K_s2026/PRETRAIN_FOR_FL_INIT.json'
INIT_CHECKPOINT_SHA = '22135cdf132eae1d405dd802acf1a756c9aba122e02c9a01f53876894bd58383'
INIT_DIGEST = '3e1c48efe79a1f1a1377719c144af9fe8e8610060ea7cc18b6bff545b61ce4e2'
MATH_REVIEW = EXT / 'reviews/DPFTRL_MATH_INDEPENDENT_REVIEW_02.json'
TRAIN, EVAL = 'CALIB_TRAIN_5_100_32768', 'CALIB_VALIDATION_5_100_32768'
PRE = 'PRETRAIN_TRAIN_5_100_NONMAIN'
N, DECISIONS, GROUP, SWEEPS, SEED, EPSILON, DELTA = 32768, 646067, 64, 6, 2026, 8., 5e-6
LABEL = 'POST_HOC_EXPLORATORY'


class Refused(ValueError):
    pass


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(Refused('nonfinite JSON ' + s)))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pin(path):
    p = Path(path).resolve()
    if BASE not in p.parents or not p.is_file():
        raise Refused('input absent or outside owner EID copy')
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def fresh(path):
    p = Path(path).absolute()
    if EXT not in p.parents or EXT.resolve() not in p.resolve().parents or p.exists():
        raise Refused('fresh extension output required')
    if any(a.is_symlink() for a in (p, *p.parents) if a == EXT or EXT in a.parents):
        raise Refused('symlink output component refused')
    return p


def write_private(path, value):
    p = fresh(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def declaration():
    return {'role': 'CALIBRATION', 'split': 'VALIDATION', 'TEST_allowed': False, 'label': LABEL,
        'init': 'PB100_seed2026_fixed_endpoint6', 'public_seed': SEED, 'N_public_slots': N,
        'client_optimizer': 'fresh_per_visit_sgd_m', 'momentum': .9, 'client_lr': .1,
        'passes': 1, 'batch_size': 16, 'gradient_clip': 1., 'weight_decay': 0., 'item_lr_mult': .1,
        'model_dropout': 0., 'cohort_dropout': 0., 'sweeps': SWEEPS, 'group_size': GROUP,
        'flat_client_delta_clip': 1., 'clip_selection': 'FIXED_PUBLIC_NO_S_CAL',
        'server': 'quadratic_prefix_FTRL_restart_each_sweep', 'eta_grid': [.1, 1.],
        'tree': 'independent_Gaussian_dyadic_nodes',
        'local_message': 'independent_Gaussian_each_user_each_visit_including_zero_slots',
        'local_boundary': 'CLIENT_LOCAL_RANDOMIZER_BEFORE_NOISY_MESSAGE_SERVER_API',
        'local_client_keys': 'INDEPENDENT_PRIVATE_PER_SLOT_MEMORY_ONLY',
        'transport_isolation_implemented': False,
        'execution_scope': 'SINGLE_PROCESS_ROLE_BOUNDARY_SIMULATOR_NOT_DEPLOYED_TRANSPORT',
        'epsilon': EPSILON, 'delta': DELTA, 'adjacency': 'add_remove_zero_slot',
        'replacement_sensitivity_factor': 2, 'amplification_used': False,
        'FP64_accumulation': True, 'FP32_model': True, 'automatic_retry': False, 'resume': False,
        'public_release': 'SHARED_MODEL_TENSORS_AND_PUBLIC_RECIPE_ONLY',
        'privacy_certified': False, 'pretraining_preprocessing_tuning_accounted': False}


def select_cell(mechanism, eta, *, role='CALIBRATION'):
    if role != 'CALIBRATION' or mechanism not in ('tree', 'local') or isinstance(eta, bool) or eta not in (.1, 1.):
        raise Refused('only fixed CALIB tree/local eta{.1,1} admitted before data access')
    return {'cell_id': mechanism.upper() + ('_ETA01' if eta == .1 else '_ETA1'), 'mechanism': mechanism, 'eta': float(eta)}


def budget(mechanism, *, n=N, group_size=GROUP, sweeps=SWEEPS, epsilon=EPSILON, delta=DELTA):
    from . import dpftrl_statistics as A
    if mechanism not in ('tree', 'local') or any(type(x) is not int or x < 1 for x in (n, group_size, sweeps)):
        raise Refused('fixed positive public geometry required')
    rho = A.rho_for_epsilon(epsilon, delta)
    if not rho > 0:
        raise Refused('positive finite target epsilon required')
    rounds = (n + group_size - 1) // group_size
    z = (A.noise_multiplier_for_epsilon(epsilon, delta, [rounds] * sweeps) if mechanism == 'tree'
         else math.sqrt(sweeps / (2 * rho)))
    achieved = (A.restart_zcdp(z, [rounds] * sweeps) if mechanism == 'tree' else sweeps / (2 * z * z))
    return {'mechanism': mechanism, 'epsilon': A.epsilon_from_zcdp(achieved, delta), 'delta': delta,
        'rho': achieved, 'z': z, 'rounds_per_sweep': rounds, 'max_visits_per_user': sweeps,
        'tree_depth_sum': A.restart_depth_sum([rounds] * sweeps) if mechanism == 'tree' else None,
        'adjacency': 'add_remove_zero_slot', 'amplification_used': False,
        'replacement_rho_if_same_noise': 4 * achieved, 'privacy_certified': False,
        'scope': 'CONDITIONAL_FIXED_INIT_PREPROCESSING_PUBLIC_ROSTER_CONFIG_MODEL_OR_LOCAL_MESSAGES_ONLY'}


def public_orders(roster, *, sweeps=SWEEPS, seed=SEED):
    import numpy as np
    roster = tuple(roster)
    if not roster or len(set(roster)) != len(roster) or any(not isinstance(k, str) or not k for k in roster):
        raise Refused('unique nonempty public potential roster required')
    if type(sweeps) is not int or sweeps < 1 or type(seed) is not int or seed < 0:
        raise Refused('public sweep count and seed required')
    return tuple(tuple(roster[i] for i in np.random.default_rng(seed + s).permutation(len(roster))) for s in range(sweeps))


class SharedVector:
    """Canonical shared trainables once; aliases/buffers/frozen tensors excluded."""
    def __init__(self, adapter):
        import torch
        self.keys = tuple(adapter.manifest.shared_keys)
        self.shapes = tuple(tuple(adapter.manifest.entries[k].shape) for k in self.keys)
        self.sizes = tuple(adapter.manifest.entries[k].numel for k in self.keys)
        if not self.keys or any(adapter.manifest.entries[k].dtype != 'torch.float32' for k in self.keys):
            raise Refused('nonempty FP32 shared trainable manifest required')
        if len(self.keys) != len(set(self.keys)):
            raise Refused('duplicate shared role')
        self.template = OrderedDict((k, v.detach().cpu().clone()) for k, v in adapter.broadcast_state(clone=True).items())
        self.fixed = {k: v.clone() for k, v in self.template.items() if k not in self.keys}
        self.dtype = torch.float32

    def flatten(self, state, *, dtype=None):
        import torch
        if not all(k in state for k in self.keys):
            raise Refused('missing canonical shared trainable')
        chunks = []
        for k, shape in zip(self.keys, self.shapes):
            value = state[k]
            if value.dtype != torch.float32 or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise Refused('changed shared tensor shape/dtype/nonfinite')
            chunks.append(value.detach().cpu().reshape(-1).to(dtype=dtype or torch.float32))
        return torch.cat(chunks)

    def state(self, vector):
        import torch
        if vector.device.type != 'cpu' or vector.dtype != torch.float32 or tuple(vector.shape) != (sum(self.sizes),) or not torch.isfinite(vector).all():
            raise Refused('finite fixed-size CPU FP32 model vector required')
        result, offset = OrderedDict((k, v.clone()) for k, v in self.fixed.items()), 0
        for k, shape, size in zip(self.keys, self.shapes, self.sizes):
            result[k] = vector[offset:offset + size].reshape(shape).clone()
            offset += size
        return result

    def public_geometry(self):
        return {'shared_keys': list(self.keys), 'shared_shapes': [list(s) for s in self.shapes],
                'shared_numel': sum(self.sizes), 'alias_buffer_frozen_excluded': True}


class LocalClientRandomizer:
    """Client-owned clip/noise; the server receives only the returned tensor.

    Private per-slot keys stay on these simulated clients, not on the server.
    This role/API separation is not operating-system or deployed transport
    isolation; finite-precision Torch pseudorandomness is not certified.
    """
    def __init__(self, shape, *, slot, private_key, clip_norm, noise_multiplier):
        import torch
        from .dpftrl_statistics import finite, integer
        self.shape, self.slot = tuple(shape), slot
        if len(self.shape) != 1 or self.shape[0] < 1 or not isinstance(slot, str) or not slot:
            raise Refused('fixed public slot and nonempty flat shape required')
        self.private_key = integer(private_key, 'client private key')
        if not 0 <= self.private_key < 2**64:
            raise Refused('private uint64 client key required')
        self.clip = finite(clip_norm, 'clip_norm', positive=True)
        self.z = finite(noise_multiplier, 'noise_multiplier')
        if self.z < 0 or not math.isfinite(self.clip * self.z):
            raise Refused('finite nonnegative client noise scale required')
        self.released = set()

    def randomize(self, direction, *, sweep):
        import torch
        from .dpftrl_tree import clip_l2, checked_tensor
        if type(sweep) is not int or sweep < 0 or sweep in self.released:
            raise Refused('one fresh local message per client/sweep required')
        value = torch.zeros(self.shape, dtype=torch.float64) if direction is None else clip_l2(
            checked_tensor(direction, self.shape), self.clip)
        key = canonical(['PPSI_CLIENT_LOCAL_MESSAGE_NOISE_2', self.private_key, sweep, self.slot])
        generator = torch.Generator(device='cpu').manual_seed(int(key[:16], 16))
        message = value + torch.randn(self.shape, generator=generator, dtype=torch.float64) * (self.clip * self.z)
        checked_tensor(message, self.shape)
        self.released.add(sweep)
        return message.detach().clone()


def private_client_keys(roster):
    """Independent private entropy for simulated clients; never server state."""
    keys, used = {}, set()
    for slot in roster:
        key = secrets.randbits(64)
        while key in used:
            key = secrets.randbits(64)
        keys[slot] = key; used.add(key)
    return keys


class PublicSweepServer:
    """Streaming group accumulation: O(log(rounds)*D), not 64 client vectors.

Tree uses the independently reviewed FP64 primitive. Local-message mode uses
independent per-slot Gaussian noise, including absent slots; it never reuses a
public training seed for entropy. The two variants share quadratic prefix FTRL.
    """
    def __init__(self, theta, *, roster, orders, mechanism, eta, noise_seed, noise_multiplier,
                 group_size=GROUP, clip_norm=1., postprocess=None):
        import torch
        from .dpftrl_tree import checked_tensor
        from .dpftrl_statistics import finite, integer
        if mechanism not in ('tree', 'local'):
            raise Refused('tree or local-message mechanism required')
        self.roster, self.orders = tuple(roster), tuple(tuple(o) for o in orders)
        if not self.roster or len(set(self.roster)) != len(self.roster) or any(not isinstance(k, str) or not k for k in self.roster):
            raise Refused('unique fixed public roster required')
        if not self.orders or any(len(o) != len(self.roster) or set(o) != set(self.roster) for o in self.orders):
            raise Refused('each public sweep must be an exact roster permutation')
        self.group = integer(group_size, 'group_size')
        self.eta, self.clip = finite(eta, 'eta', positive=True), finite(clip_norm, 'clip_norm', positive=True)
        self.z = finite(noise_multiplier, 'noise_multiplier')
        if mechanism == 'tree':
            self.seed = integer(noise_seed, 'private tree noise seed')
            if not 0 <= self.seed < 2**64:
                raise Refused('private uint64 tree entropy required')
        elif noise_seed is not None:
            raise Refused('local server must not receive any client private noise seed')
        if self.group < 1 or self.z < 0 or not math.isfinite(self.clip * self.z):
            raise Refused('positive group and finite nonnegative std required')
        if theta.dtype != torch.float32 or theta.device.type != 'cpu' or theta.ndim != 1:
            raise Refused('flat CPU FP32 model required')
        self.theta = checked_tensor(theta).to(torch.float32)
        self.mechanism, self.post = mechanism, postprocess
        if postprocess is not None and not callable(postprocess):
            raise Refused('postprocessing must be deterministic callable')
        self.sweep, self.round = -1, 0
        self.begin_sweep()

    @property
    def rounds(self):
        return (len(self.roster) + self.group - 1) // self.group

    def begin_sweep(self):
        import torch
        from .dpftrl_tree import GaussianPrefixTree
        if self.sweep >= 0 and self.round != self.rounds:
            raise Refused('incomplete public sweep cannot restart')
        if self.sweep + 1 >= len(self.orders):
            raise Refused('all declared sweeps already started')
        self.sweep += 1
        self.round = 0
        self.anchor = self.theta.to(torch.float64).clone()
        self.prefix = torch.zeros_like(self.anchor)
        self.tree = GaussianPrefixTree(self.theta, clip_norm=self.clip, noise_multiplier=self.z,
                                      noise_seed=self.seed, epoch=self.sweep) if self.mechanism == 'tree' else None

    def next_group(self):
        if self.round >= self.rounds:
            raise Refused('sweep complete, explicit restart required')
        return self.orders[self.sweep][self.round * self.group:(self.round + 1) * self.group]

    def step(self, visit):
        """Trusted tree aggregator only; local server never receives callbacks."""
        import torch
        from .dpftrl_tree import clip_l2, checked_tensor
        if self.mechanism != 'tree':
            raise Refused('local server accepts already randomized messages only, never raw callbacks')
        total = torch.zeros_like(self.anchor)
        for key in self.next_group():
            value = visit(key)
            if value is not None:
                checked_tensor(value, self.theta.shape)
                total.add_(clip_l2(value, self.clip))
        return self._advance(total)

    def step_messages(self, messages):
        """Server-local API: exact public slots and noisy FP64 tensor payloads only.

        No raw direction, clipping operation, private key, client callback or
        randomizer object crosses this interface. Noisy messages are NOT clipped.
        """
        import torch
        from .dpftrl_tree import checked_tensor
        if self.mechanism != 'local' or not isinstance(messages, dict) or set(messages) != set(self.next_group()):
            raise Refused('exact fixed public group of local noisy messages required')
        total = torch.zeros_like(self.anchor)
        for key in self.next_group():
            value = messages[key]
            if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float64:
                raise Refused('local server receives only fixed-shape CPU FP64 noisy tensor messages')
            checked_tensor(value, self.theta.shape)
            total.add_(value)
        return self._advance(total)

    def _advance(self, total):
        import torch
        from .dpftrl_tree import checked_tensor
        if not torch.isfinite(total).all():
            raise Refused('private group overflow')
        prepared = self.tree._prepare_append(total) if self.tree is not None else None
        prefix = prepared[2] if prepared is not None else self.prefix + total
        candidate = (self.anchor + (self.eta / self.group) * prefix).to(torch.float32)
        checked_tensor(candidate, self.theta.shape)
        if self.post is not None:
            candidate = self.post(candidate.clone())
            checked_tensor(candidate, self.theta.shape)
            if candidate.dtype != torch.float32:
                raise Refused('postprocessor changed model dtype')
        if prepared is not None:
            self.tree._commit(prepared)
        self.prefix = prefix.clone() if self.tree is None else self.prefix
        self.theta = candidate.detach().clone()
        self.round += 1
        return self.theta.clone()

    def weights(self):
        return self.theta.clone()


def math_gate():
    review = read(MATH_REVIEW)
    if sha(MATH_REVIEW) != '8d6ca0f353a42032d971eba04d4c0d5ee65affe99257ccc9919aa27febbee5cc' or review.get('status') != 'PASS':
        raise Refused('exact independently reviewed math primitives required')
    for row in review['source_files']:
        if pin(row['path']) != row:
            raise Refused('frozen math source changed')
    return pin(MATH_REVIEW)


def source_paths():
    package = CODE / 'experiment_extensions'
    return sorted(p for p in CODE.rglob('*.py') if 'experiment_extensions' not in p.parts) + [
        package / '__init__.py', package / 'dpftrl_statistics.py', package / 'test_dpftrl_statistics.py',
        package / 'dpftrl_tree.py', package / 'test_dpftrl_tree.py', package / 'dpftrl_train.py',
        package / 'test_dpftrl_train.py']


def input_paths():
    paths = {p.resolve() for p in (WORK / 'contracts').rglob('*') if p.is_file()}
    registry = read(WORK / 'contracts/manifest_registry/REGISTRY_V2.json')
    if registry['roles']['CALIBRATION'] != {'train': TRAIN, 'eval': EVAL}:
        raise Refused('registry CALIB role changed')
    for mid in (TRAIN, EVAL):
        entry = registry['manifests'][mid]
        paths.update((WORK / entry[k]).resolve() for k in ('path', 'marker'))
        paths.update((WORK / r['file']).resolve() for r in entry['tables'].values())
    pre = read(WORK / 'contracts/manifest_registry/REGISTRY_V2_PRETRAIN_EXT_PB100.json')['manifests'][PRE]
    paths.update((WORK / pre[k]).resolve() for k in ('path', 'marker'))
    paths.add((WORK / pre['tables']['clients']['file']).resolve())
    paths.add((WORK / 'data_v2/TRANSFORM_STATE_CLEAN_V2.json').resolve())
    paths.update(p.resolve() for p in DATA.glob('model_ready/user_bucket=*/features.parquet'))
    for split in ('TRAIN', 'VALIDATION_SELECT', 'VALIDATION_CONFIRM'):
        paths.update(p.resolve() for p in (DATA / 'decisions_index' / ('split=' + split)).glob('user_bucket=*/*.parquet'))
    paths.update(p.resolve() for p in (DATA / 'mappings').glob('*') if p.is_file())
    paths.add(INIT.resolve())
    paths.add((INIT.parent / read(INIT)['fl_init']['path']).resolve())
    return sorted(paths)


def build_protocol(out):
    fresh(out)
    obj = {'format': 'PPSI_DPFTRL_LOCAL_CALIB_PROTOCOL_1', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'declaration': declaration(), 'cells': [select_cell(m, eta) for m in ('tree', 'local') for eta in (.1, 1.)],
        'accountants': {m: budget(m) for m in ('tree', 'local')}, 'math_review': math_gate(),
        'sources': [pin(p) for p in source_paths()], 'inputs': [pin(p) for p in input_paths()],
        'privacy_certified': False, 'real_training_started': False}
    obj['self_sha256'] = canonical(obj)
    write_private(out, obj)
    return {'status': 'BOUND_NOT_TRAINED', 'protocol': str(Path(out)), 'sha256': sha(out)}


def load_protocol(path, expected_sha):
    if sha(path) != expected_sha:
        raise Refused('protocol bytes changed')
    obj = read(path)
    body = {k: v for k, v in obj.items() if k != 'self_sha256'}
    if obj.get('format') != 'PPSI_DPFTRL_LOCAL_CALIB_PROTOCOL_1' or obj.get('self_sha256') != canonical(body):
        raise Refused('protocol schema or self hash changed')
    if obj.get('declaration') != declaration() or obj.get('cells') != [select_cell(m, e) for m in ('tree', 'local') for e in (.1, 1.)]:
        raise Refused('fixed declared recipe/cells changed')
    if obj.get('accountants') != {m: budget(m) for m in ('tree', 'local')} or obj.get('math_review') != math_gate():
        raise Refused('exact separate tree and local accounting required')
    for name, inventory in (('sources', source_paths()), ('inputs', input_paths())):
        if obj.get(name) != [pin(p) for p in inventory]:
            raise Refused('source/data/init inventory or bytes changed')
    return obj


def require_early_cuda(gpu):
    if type(gpu) is not int or gpu not in (0, 1) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise Refused('export exact physical CUDA device before Python import')


def initializer_population_audit():
    """Private declared-population evidence; does not certify earlier training."""
    import pyarrow.parquet as pq
    reg = read(WORK / 'contracts/manifest_registry/REGISTRY_V2_PRETRAIN_EXT_PB100.json')
    tables = []
    for mid, size in ((PRE, 100000), (TRAIN, N)):
        row = reg['manifests'][mid]['tables']['clients']
        path = WORK / row['file']
        if sha(path) != row['file_sha256']:
            raise Refused('registered declared client table drift')
        table = pq.read_table(path, columns=['client_key_hex', 'selection_key']).to_pydict()
        key, selection = set(table['client_key_hex']), set(table['selection_key'])
        if len(key) != size or len(selection) != size or len(table['client_key_hex']) != size:
            raise Refused('registered PRE/CALIB client count or uniqueness changed')
        tables.append((key, selection))
    overlap = len(tables[0][0] & tables[1][0])
    selection_overlap = len(tables[0][1] & tables[1][1])
    return {'declared_PRE_clients': 100000, 'declared_CALIB_clients': N,
        'client_key_overlap': overlap, 'selection_key_overlap': selection_overlap,
        'declared_tables_disjoint': overlap == 0 and selection_overlap == 0,
        'earlier_training_attested': False, 'pretraining_preprocessing_accounted': False}


def require_zero_model_dropout(adapter, *, require_registered_cfg=False):
    """Guard module and functional SASRec dropout before any client update.

    Registered SASRec uses cfg.dropout for input F.dropout and CausalBlock.p
    for functional attention/feed-forward dropout, rather than nn.Dropout.
    Synthetic adapters may lack cfg but every exposed dropout probability is
    still checked. Real CPU/CUDA construction requires the registered cfg.
    """
    import torch
    module = adapter.module
    cfg = getattr(module, 'cfg', None)
    if require_registered_cfg and (cfg is None or not hasattr(cfg, 'dropout')):
        raise Refused('registered model cfg.dropout is required')
    cfg_p = getattr(cfg, 'dropout', None) if cfg is not None else None
    def zero(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value == 0.
    if cfg is not None and not zero(cfg_p):
        raise Refused('fixed zero model cfg.dropout required for functional F.dropout')
    nn_sites, functional_sites = 0, 0
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout):
            nn_sites += 1
            if not zero(child.p):
                raise Refused('fixed zero nn.Dropout required')
        elif hasattr(child, 'p'):
            functional_sites += 1
            if not zero(child.p):
                raise Refused('fixed zero functional dropout probability required')
    return {'cfg_dropout': cfg_p, 'nn_dropout_sites': nn_sites,
        'functional_p_sites': functional_sites, 'all_zero': True,
        'registered_cfg_required': require_registered_cfg}


def actual_inputs(gpu, *, device='cpu', log=lambda *a: None):
    require_early_cuda(gpu)
    os.environ['V4_DATASET_ROOT'] = str(DATA)
    from v4int import launch_central as LC
    LC.export_environment(gpu)
    import torch
    import ppsi_env
    ppsi_env.verify_snapshot(); ppsi_env.assert_isolated()
    from v4int import launch_fl as F
    from v4int.providers import StoreTrainProvider, StoreEvalProvider, comparison_key_base
    from v4int.verify import DataPaths
    from v4models import build
    from v4data.common import dataset_root
    from fl.numerics import enforce_strict_fp32, state_digest
    from fl.client import ClientData
    torch.set_num_threads(4)
    enforce_strict_fp32(warn_only=False)
    paths = DataPaths.default(WORK)
    lock = LC.bind_data_protocol_lock_v2(paths, smoke=False)
    kw = dict(paths=paths, history_policy='USER_HISTORY_L50', manifest_registry=lock['manifest_registry_path'], registry_root=WORK, log=log)
    tp, ep = StoreTrainProvider(TRAIN, **kw), StoreEvalProvider(EVAL, prefetch=0, **kw)
    keys, positions = F.client_index(tp, paths.manifest_dir)
    if len(keys) != N or tp.n_decisions != DECISIONS or sum(len(v) for v in positions.values()) != DECISIONS:
        raise Refused('full actual CALIB population differs')
    recipe = F._small_recipe(SimpleNamespace(small_recipe=None, smoke=False))
    schema = LC.load_production_schema(lock['schema_id'], lock['schema_self_sha256'], lock['schema_registry_path'])
    adapter = build(F.FAMILY, schema, SEED, device, v1_root=dataset_root(), variant=recipe['model_variant'], dropout=(0.,)).adapter
    state, identity, launch = F.load_pretrain_init(INIT, model_variant=recipe['model_variant'])
    if identity.get('checkpoint_sha256') != INIT_CHECKPOINT_SHA or identity.get('state_digest') != INIT_DIGEST or identity.get('seed') != SEED:
        raise Refused('only exact registered PB100 seed2026 initializer allowed')
    F.apply_init_state(adapter, state, INIT_DIGEST)
    dropout_validation = require_zero_model_dropout(adapter, require_registered_cfg=True)
    import numpy as np
    from v4base.identity import eval_client_key_hex_by_eval_user_key
    m = ep.manifest
    external = eval_client_key_hex_by_eval_user_key(ep, manifest_dir=paths.manifest_dir)
    users = np.unique(m.eval_user_key[m.eval_mask])
    rankable = np.unique(m.eval_user_key[m.eval_mask & m.rankable])
    missing = {str(external[u]) for u in users} - set(keys)
    if m.n != 158135 or int(m.eval_mask.sum()) != 127530 or len(users) != 10070 or len(rankable) != 10060 or missing:
        raise Refused('actual CALIB validation E2E/support population differs')
    layout = SharedVector(adapter)
    proof = {'status': 'PASS_PREPARED_NOT_TRAINED', 'role': 'CALIBRATION', 'TRAIN': TRAIN, 'VALIDATION': EVAL,
        'N_clients': N, 'N_decisions': DECISIONS, 'N_positions': sum(len(v) for v in positions.values()),
        'validation_rows': m.n, 'E2E_decisions': int(m.eval_mask.sum()), 'E2E_users': len(users),
        'rankable_users': len(rankable), 'OOV_only_users': len(users) - len(rankable), 'missing_support_users': len(missing),
        'schema_self_sha256': schema.self_sha256, 'model_dropout_validation': dropout_validation, 'initializer': identity, 'initial_state_digest': state_digest(adapter.broadcast_state()),
        'shared_geometry': layout.public_geometry(), 'public_roster_sha256': canonical(keys),
        'public_orders_sha256': canonical(public_orders(keys)), 'comparison_key': comparison_key_base(paths, tp, ep),
        'validation_hashes': dict(m.hashes), 'declared_PRE_population_audit': initializer_population_audit(),
        'TEST_read': False, 'GPU_training_started': False}
    return {'adapter': adapter, 'layout': layout, 'keys': keys, 'tp': tp, 'ep': ep, 'proof': proof,
            'schema': schema, 'variant': recipe['model_variant'],
            'get_client': lambda key: ClientData(key, tp.batch(positions[key]))}


def prepare(protocol_path, protocol_sha, out, gpu):
    protocol = load_protocol(protocol_path, protocol_sha)
    state = actual_inputs(gpu)
    import torch
    if torch.cuda.is_initialized():
        raise Refused('CPU preparation initialized CUDA')
    proof = dict(state['proof'], protocol_sha256=protocol_sha, declaration=protocol['declaration'],
                 module_sha256=sha(__file__), tests_sha256=sha(CODE / 'experiment_extensions/test_dpftrl_train.py'), GPU_initialized=False)
    load_protocol(protocol_path, protocol_sha)
    write_private(out, proof)
    return {'status': proof['status'], 'preparation_sha256': sha(out)}


def review_gate(path, expected, protocol_sha, preparation_sha, mechanism):
    if sha(path) != expected:
        raise Refused('review bytes changed')
    obj = read(path)
    if (obj.get('status') != 'PASS' or obj.get('reviewer') in (None, '', '/root/warm_operator_review')
        or obj.get('protocol_sha256') != protocol_sha or obj.get('preparation_sha256') != preparation_sha
        or obj.get('module_sha256') != sha(__file__)
        or obj.get('tests_sha256') != sha(CODE / 'experiment_extensions/test_dpftrl_train.py')
        or obj.get('run_accountant_concurrence') is not True
        or not isinstance(obj.get('modes'), list) or not obj['modes']
        or any(m not in ('tree', 'local') for m in obj['modes'])
        or len(set(obj['modes'])) != len(obj['modes']) or mechanism not in obj['modes']
        or obj.get('fixed_public_S') != 1. or obj.get('TEST_approved') is not False):
        raise Refused('exact independent source/full-preparation/run-accountant review required')
    return obj


def solver():
    from fl.client import ItemLRSolver
    return ItemLRSolver(lr=.1, optimizer='sgd_m', passes=1, batch_size=16, clip=1., weight_decay=0., item_lr_mult=.1)


def run_clients(adapter, layout, *, roster, get_client, mechanism, eta, private_seed, noise_multiplier,
                orders=None, group_size=GROUP, on_round=None, local_client_seeds=None):
    """Actual unchanged client_update kernel; no private callback output is public."""
    import torch
    from fl.client import client_update
    from fl.numerics import enforce_strict_fp32
    enforce_strict_fp32(warn_only=False)
    require_zero_model_dropout(adapter)
    initial = layout.flatten(adapter.broadcast_state())
    public = tuple(orders) if orders is not None else public_orders(roster)
    def post(vector):
        adapter.load_state_(layout.state(vector))
        adapter.server_post_aggregate()
        return layout.flatten(adapter.broadcast_state())
    if mechanism == 'local':
        if private_seed is not None:
            raise Refused('local client entropy must never be assigned as a server private seed')
        seeds = private_client_keys(roster) if local_client_seeds is None else dict(local_client_seeds)
        if set(seeds) != set(roster) or len(set(seeds.values())) != len(seeds):
            raise Refused('distinct private client keys for exact fixed public roster required')
        clients = {key: LocalClientRandomizer(initial.shape, slot=key, private_key=seeds[key],
            clip_norm=1., noise_multiplier=noise_multiplier) for key in roster}
    elif local_client_seeds is not None:
        raise Refused('tree mechanism must not receive client-local noise keys')
    server = PublicSweepServer(initial, roster=roster, orders=public, mechanism=mechanism, eta=eta,
        noise_seed=private_seed, noise_multiplier=noise_multiplier, group_size=group_size, postprocess=post)
    private = {'valid_exposures': 0, 'optimizer_steps': 0, 'present_visits': 0, 'noop_visits': 0}
    round_idx = 0
    for sweep in range(len(public)):
        if sweep:
            server.begin_sweep()
        for _ in range(server.rounds):
            broadcast = layout.state(server.weights())
            reference = layout.flatten(broadcast, dtype=torch.float64)
            round_idx += 1
            def visit(key):
                client = get_client(key)
                if client is None:
                    private['noop_visits'] += 1
                    return None
                if client.key != key:
                    raise Refused('client callback key differs from fixed public slot')
                result = client_update(adapter, broadcast, client, solver(), round_idx=round_idx, seed=SEED, clone_upload=True)
                private['valid_exposures'] += result.n_consumed
                private['optimizer_steps'] += result.steps
                private['present_visits'] += int(result.n_valid > 0)
                private['noop_visits'] += int(result.n_valid == 0)
                return layout.flatten(result.upload, dtype=torch.float64) - reference if result.n_valid else None
            if mechanism == 'tree':
                server.step(visit)
            else:
                # Simulated clients finish clipping/randomization BEFORE server
                # delivery. The server receives tensors, never this raw callback.
                messages = {key: clients[key].randomize(visit(key), sweep=sweep) for key in server.next_group()}
                server.step_messages(messages)
            if on_round is not None:
                on_round(round_idx, {'sweep_index': sweep, 'round_in_sweep': server.round,
                    'planned_denominator': group_size}, dict(private))
    adapter.load_state_(layout.state(server.weights()))
    return adapter.broadcast_state(clone=True), private


def model_only_payload(theta, layout, cell, accounting):
    """No input hashes, client IDs, buffers, seeds, counts or validation metadata."""
    if cell != select_cell(cell['mechanism'], cell['eta']) or accounting != budget(cell['mechanism']):
        raise Refused('export needs exact public cell/accountant whitelist')
    flat = layout.flatten(theta)
    shared = layout.state(flat)
    return {'format': 'PPSI_DPFTRL_LOCAL_FIXED_CELL_MODEL_ONLY_1',
        'shared_theta': {k: shared[k].detach().cpu().clone() for k in layout.keys},
        'public_cell': dict(cell), 'conditional_fixed_recipe_accountant': dict(accounting),
        'pretraining_preprocessing_tuning_accounted': False, 'privacy_certified': False,
        'execution_scope': 'SINGLE_PROCESS_ROLE_BOUNDARY_SIMULATOR_NOT_DEPLOYED_TRANSPORT',
        'transport_isolation_implemented': False}


def train(protocol_path, protocol_sha, preparation_path, preparation_sha, review_path, review_sha,
          mechanism, eta, out, gpu):
    cell = select_cell(mechanism, eta)
    require_early_cuda(gpu)
    protocol = load_protocol(protocol_path, protocol_sha)
    review_gate(review_path, review_sha, protocol_sha, preparation_sha, mechanism)
    if sha(preparation_path) != preparation_sha:
        raise Refused('CPU preparation bytes changed')
    prep = read(preparation_path)
    if prep.get('status') != 'PASS_PREPARED_NOT_TRAINED' or prep.get('protocol_sha256') != protocol_sha or prep.get('GPU_initialized') is not False:
        raise Refused('actual full CPU provider/init preparation required')
    out = fresh(out)
    import torch
    # Run review and all CPU/data source gates precede the first real CUDA model.
    state = actual_inputs(gpu, device='cpu')
    expected = dict(state['proof'], protocol_sha256=protocol_sha, declaration=protocol['declaration'],
        module_sha256=sha(__file__), tests_sha256=sha(CODE / 'experiment_extensions/test_dpftrl_train.py'), GPU_initialized=False)
    if prep != expected:
        raise Refused('actual provider/init/preparation binding changed')
    from v4int import launch_central as LC
    gpu_check = LC.gpu_checks(SimpleNamespace(gpu=gpu, allow_shared_gpu=False), LC.default_runner, dict(os.environ))
    if not torch.cuda.is_available():
        raise Refused('physical GPU unavailable')
    accounting = protocol['accountants'][mechanism]
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    entropy = secrets.randbits(64) if mechanism == 'tree' else None
    write_private(out / 'PRIVATE_ENTROPY.json',
        {'private_tree_seed': entropy, 'public_training_seed': SEED} if mechanism == 'tree' else
        {'scope': 'CLIENT_LOCAL_PRIVATE_KEYS_MEMORY_ONLY', 'keys_persisted': False,
         'transport_isolation_implemented': False, 'public_training_seed': SEED})
    binding = {'cell': cell, 'protocol_sha256': protocol_sha, 'preparation_sha256': preparation_sha,
        'review_sha256': review_sha, 'module_sha256': sha(__file__), 'role': 'CALIBRATION', 'split': 'VALIDATION',
        'gpu_ownership_check': gpu_check}
    write_private(out / 'STARTED.json', binding)
    # BaseAdapter stores device/parameter/buffer references at construction;
    # reconstruct on CUDA instead of moving an already-built CPU adapter.
    from v4models import build
    from v4data.common import dataset_root
    from v4int import launch_fl as F
    gpu_adapter = build(F.FAMILY, state['schema'], SEED, 'cuda:0', v1_root=dataset_root(),
                        variant=state['variant'], dropout=(0.,)).adapter
    F.apply_init_state(gpu_adapter, state['adapter'].module.state_dict(), INIT_DIGEST)
    require_zero_model_dropout(gpu_adapter, require_registered_cfg=True)
    state['adapter'], state['layout'] = gpu_adapter, SharedVector(gpu_adapter)
    def private_progress(round_idx, public_cursor, counts):
        if round_idx % 8 == 0:
            fd = os.open(out / 'INTERNAL_PROGRESS.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'a') as stream:
                stream.write(json.dumps(dict(round_idx=round_idx, cursor=public_cursor, private_counts=counts), allow_nan=False) + '\n')
    theta, private = run_clients(state['adapter'], state['layout'], roster=state['keys'], get_client=state['get_client'],
        mechanism=mechanism, eta=eta, private_seed=entropy, noise_multiplier=accounting['z'], on_round=private_progress)
    load_protocol(protocol_path, protocol_sha)
    write_private(out / 'INTERNAL_COUNTERS.json', dict(private, actual_EFE=private['valid_exposures'] / DECISIONS,
        counts_norms_steps_timing_outside_public_message_boundary=True))
    target = fresh(out / 'MODEL_ONLY/model.pt')
    target.parent.mkdir(mode=0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        torch.save(model_only_payload(theta, state['layout'], cell, accounting), stream)
    write_private(out / 'COMPLETE.json', {'status': 'TRAIN_ONLY_NOT_UTILITY_COMPLETE', 'binding': binding,
        'model_file_sha256': sha(target), 'TEST_read': False, 'privacy_certified': False,
        'validation_scored': False, 'utility_complete': False,
        'execution_scope': 'SINGLE_PROCESS_ROLE_BOUNDARY_SIMULATOR_NOT_DEPLOYED_TRANSPORT'})
    return {'status': 'TRAIN_ONLY_NOT_UTILITY_COMPLETE', 'output': str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build-protocol'); p.add_argument('--out', required=True, type=Path)
    p = sub.add_parser('prepare')
    for name in ('protocol', 'protocol-sha256', 'out'):
        p.add_argument('--' + name, required=True, type=Path if name in ('protocol', 'out') else str)
    p.add_argument('--gpu', required=True, type=int, choices=(0, 1))
    p = sub.add_parser('train')
    for name in ('protocol', 'protocol-sha256', 'preparation', 'preparation-sha256', 'review', 'review-sha256', 'out'):
        p.add_argument('--' + name, required=True, type=Path if name in ('protocol', 'preparation', 'review', 'out') else str)
    p.add_argument('--gpu', required=True, type=int, choices=(0, 1))
    p.add_argument('--mechanism', required=True, choices=('tree', 'local'))
    p.add_argument('--eta', required=True, type=float, choices=(.1, 1.))
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    if command == 'build-protocol':
        result = build_protocol(**args)
    elif command == 'prepare':
        result = prepare(args['protocol'], args['protocol_sha256'], args['out'], args['gpu'])
    else:
        result = train(args['protocol'], args['protocol_sha256'], args['preparation'], args['preparation_sha256'],
            args['review'], args['review_sha256'], args['mechanism'], args['eta'], args['out'], args['gpu'])
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
