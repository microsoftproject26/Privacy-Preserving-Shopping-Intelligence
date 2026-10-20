"""Separate full-catalogue CALIB VALIDATION utility for frozen DPFTRL03 exports.

This module never trains, samples private entropy, opens INTERNAL*/PRIVATE_* files,
or changes the trainer's TRAIN_ONLY_NOT_UTILITY_COMPLETE receipt. Utility outputs
are INTERNAL exploratory validation, outside the conditional model-only privacy
boundary. Four released models compose; four epsilon-8 cells are not jointly
epsilon 8. No TEST/MAIN route or candidate-pool fit is exposed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from . import dpftrl_train as D

EXT, CODE, WORK = D.EXT, D.CODE, D.WORK
RUN_ROOT = EXT / 'runs/DPFTRL_CALIB_03'
TRAIN_PROTOCOL = EXT / 'protocols/DPFTRL_LOCAL_CALIB_PROTOCOL_03_v2.json'
TRAIN_PROTOCOL_SHA = 'aee1dbc81b96cea144f8179d3eca0d16731810d008a19fb70e77500f523e5c19'
TRAIN_PREPARATION = EXT / 'evidence/DPFTRL_LOCAL_FULL_CALIB_CPU_PREPARATION_03.json'
TRAIN_PREPARATION_SHA = 'e42465eece87f99cc7cf8028589fc209c33cc4130a0fb119e95d4b02e5e0a727'
TRAIN_CPU_REVIEW = EXT / 'reviews/DPFTRL_LOCAL_INDEPENDENT_CPU_REVIEW_03.json'
TRAIN_CPU_REVIEW_SHA = '76121bb8da56b976b15ea04f513a4738f8533fa70b23b80fc3ced3e9401839fb'
TRAIN_MODULE_SHA = 'd3096bb4be8cfa9e97f472a47b8dc50ec0412714d40641a025df51d6fb372eaa'
TRAIN_TESTS_SHA = '4de710248c07d7e3b9c5adb6265c5d1cf56f81180f94399576eca5ac15a0901f'
FORMAT = 'PPSI_DPFTRL03_CALIB_VALIDATION_PROTOCOL_1'
K = 158486
PRODUCTION_SCHEMA_SHA = 'c53d47fe2555dd537e27e11d45a4e8b23739566e6b98179dcf521852aa918f87'
PRODUCTION_SCHEMA = WORK / 'contracts/derived_schemas/L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7.json'
WIDTH_KEYS = ('numeric_dim', 'flags_dim', 'user_context_dim', 'user_mask_dim')
TRUTH_KEYS = frozenset(('target_class', 'target_oov', 'target', 'label', 'labels'))
PUBLIC_KEYS = frozenset(('format', 'shared_theta', 'public_cell', 'conditional_fixed_recipe_accountant',
    'pretraining_preprocessing_tuning_accounted', 'privacy_certified', 'execution_scope',
    'transport_isolation_implemented'))


class Refused(ValueError):
    pass


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def exact_role(role='CALIBRATION', split='VALIDATION'):
    if role != 'CALIBRATION' or split != 'VALIDATION':
        raise Refused('only CALIBRATION VALIDATION is admitted before data access')


def own_path(path, *, root=EXT, exists=True):
    """Require the physical extension path, never a symlink or an alias spelling."""
    p, root = Path(path).absolute(), Path(root).absolute()
    if root not in p.parents or p != p.resolve():
        raise Refused('exact physical extension path required')
    if any(a.is_symlink() for a in (p, *p.parents) if a == root or root in a.parents):
        raise Refused('symlink path component refused')
    if exists and not p.is_file():
        raise Refused('bound own file absent')
    return p


def check_pin(row):
    if not isinstance(row, dict) or D.pin(row['path']) != row:
        raise Refused('bound file changed')
    return Path(row['path'])


def production_contract():
    """Derive exact widths/geometry from the immutable trainer preparation/schema."""
    if D.sha(TRAIN_PREPARATION) != TRAIN_PREPARATION_SHA:
        raise Refused('frozen trainer preparation changed')
    prep, schema = D.read(TRAIN_PREPARATION), D.read(PRODUCTION_SCHEMA)
    if (schema.get('self_sha256') != PRODUCTION_SCHEMA_SHA
        or prep.get('schema_self_sha256') != PRODUCTION_SCHEMA_SHA
        or schema.get('self_sha256') != D.canonical({k: v for k, v in schema.items() if k != 'self_sha256'})):
        raise Refused('exact registered production schema/preparation identity required')
    return {'schema_self_sha256': PRODUCTION_SCHEMA_SHA,
        'widths': {k: schema['widths'][k] for k in WIDTH_KEYS}, 'shared_geometry': prep['shared_geometry']}


def declaration():
    contract = production_contract()
    return {'role': 'CALIBRATION', 'split': 'VALIDATION', 'TEST_allowed': False,
        'label': D.LABEL, 'manifest_id': D.EVAL, 'catalogue_K': K,
        'model': 'SASREC_D64_B2_RICH_ALL_REGISTERED_BRANCHES', 'model_dropout': 0.,
        'production_schema_self_sha256': contract['schema_self_sha256'], 'production_model_widths': contract['widths'],
        'endpoint': 'FIXED_FINAL_SIX_SWEEPS_ONLY_NO_PB_OR_SELECTION',
        'prediction': 'UNCHANGED_v4eval.predict.predict_adapter_FROZEN_T3',
        'metric': 'E2E_CLIENT_MACRO_MRR_20', 'metric_accumulation': 'FP64',
        'eval_batch': 1024, 'eval_score_block': K, 'eval_class_chunk': 32768,
        'eval_cache_bytes': 1073741824, 'eval_stream': 'E2E_ROWS',
        'tie_rule': '1+strictly_greater+equal_score_lower_class_id',
        'candidate_pool': 'FIXED_REGISTERED_TRAIN_CATALOGUE_NO_REFIT_NO_SAMPLING',
        'full_manifest_rows': 158135, 'E2E_decisions': 127530, 'E2E_users': 10070,
        'rankable_users': 10060, 'OOV_only_users': 10,
        'OOV_credit': 0., 'censored_rows_enter_denominator': False,
        'input_whitelist': ['STARTED.json', 'COMPLETE.json', 'MODEL_ONLY/model.pt'],
        'private_training_state_read': False, 'training_receipt_modified': False,
        'utility_scope': 'INTERNAL_EXPLORATORY_NOT_PART_OF_MODEL_ONLY_PRIVACY_RELEASE',
        'privacy_certified': False, 'pretraining_preprocessing_tuning_accounted': False,
        'execution_scope': 'SINGLE_PROCESS_ROLE_BOUNDARY_SIMULATOR_NOT_DEPLOYED_TRANSPORT',
        'transport_isolation_implemented': False, 'four_cell_joint_epsilon8_claim': False,
        'automatic_retry': False, 'resume': False}


def source_paths():
    return D.source_paths() + [CODE / 'experiment_extensions/dpftrl_validation.py',
                               CODE / 'experiment_extensions/test_dpftrl_validation.py']


def training_component_gate():
    if D.sha(D.__file__) != TRAIN_MODULE_SHA or D.sha(CODE / 'experiment_extensions/test_dpftrl_train.py') != TRAIN_TESTS_SHA:
        raise Refused('frozen trainer03 source/tests changed')
    protocol = D.load_protocol(TRAIN_PROTOCOL, TRAIN_PROTOCOL_SHA)
    if D.sha(TRAIN_PREPARATION) != TRAIN_PREPARATION_SHA or D.sha(TRAIN_CPU_REVIEW) != TRAIN_CPU_REVIEW_SHA:
        raise Refused('frozen trainer03 preparation/CPU review changed')
    prep, review = D.read(TRAIN_PREPARATION), D.read(TRAIN_CPU_REVIEW)
    if (prep.get('status') != 'PASS_PREPARED_NOT_TRAINED' or prep.get('GPU_initialized') is not False
        or prep.get('protocol_sha256') != TRAIN_PROTOCOL_SHA or prep.get('module_sha256') != TRAIN_MODULE_SHA
        or review.get('status') != 'CPU_REVIEW_PASS_PENDING_ROOT_GPU'
        or review.get('protocol_sha256') != TRAIN_PROTOCOL_SHA
        or review.get('preparation_sha256') != TRAIN_PREPARATION_SHA
        or review.get('module_sha256') != TRAIN_MODULE_SHA or review.get('real_training_allowed') is not False):
        raise Refused('exact CPU component evidence required; it does not authorize training')
    return protocol


def protocol_body():
    """Called only after exact_role; inventories do not include other extension owners."""
    training_component_gate()
    return {'declaration': declaration(), 'run_root': str(RUN_ROOT),
        'cells': [D.select_cell(m, e) for m in ('tree', 'local') for e in (.1, 1.)],
        'trainer_protocol': D.pin(TRAIN_PROTOCOL), 'trainer_preparation': D.pin(TRAIN_PREPARATION),
        'trainer_cpu_component_review': D.pin(TRAIN_CPU_REVIEW),
        'sources': [D.pin(p) for p in source_paths()],
        'inputs': [D.pin(p) for p in D.input_paths()]}


def build_protocol(out, *, role='CALIBRATION', split='VALIDATION'):
    exact_role(role, split)
    D.fresh(out)
    obj = dict(format=FORMAT, utc=utc(), **protocol_body())
    obj['self_sha256'] = D.canonical(obj)
    D.write_private(out, obj)
    return {'status': 'VALIDATION_PROTOCOL_BOUND_NOT_SCORED', 'protocol': D.pin(out)}


def load_protocol(path, expected_sha, *, role='CALIBRATION', split='VALIDATION'):
    exact_role(role, split)
    own_path(path)
    if D.sha(path) != expected_sha:
        raise Refused('scorer protocol bytes changed')
    obj = D.read(path)
    body = {k: v for k, v in obj.items() if k != 'self_sha256'}
    if obj.get('format') != FORMAT or obj.get('self_sha256') != D.canonical(body):
        raise Refused('scorer protocol schema/self hash changed')
    expected = protocol_body()
    if set(obj) != set(expected) | {'format', 'utc', 'self_sha256'} or any(obj.get(k) != v for k, v in expected.items()):
        raise Refused('scorer declared source/data/init/training identity changed')
    return obj


def require_actual_model(adapter, *, schema=None):
    import torch
    m = adapter.module
    if (getattr(adapter, 'K', None) != K or adapter.query_dim != 64
        or getattr(getattr(m, 'cfg', None), 'rich', None) is not True
        or not m.side_enabled or not m.has_side_branch or not m.has_user_branch
        or (m.cfg.group, m.cfg.d_model, m.cfg.blocks, m.cfg.heads, m.cfg.ff_multiplier,
            m.cfg.max_len, m.cfg.ln_eps) != ('rich_frozen_channels', 64, 2, 4, 4, 50, 1e-5)
        or adapter.head_weight().dtype != torch.float32):
        raise Refused('exact full-K registered rich D64 architecture required')
    contract = production_contract()
    entries, keys = adapter.manifest.entries, adapter.manifest.shared_keys
    geometry = {'shared_keys': list(keys), 'shared_shapes': [list(entries[k].shape) for k in keys],
        'shared_numel': sum(entries[k].numel for k in keys), 'alias_buffer_frozen_excluded': True}
    if (getattr(schema, 'self_sha256', None) != contract['schema_self_sha256']
        or any(getattr(m, k) != v or getattr(schema.widths, k) != v for k, v in contract['widths'].items())
        or geometry != contract['shared_geometry']):
        raise Refused('exact registered production schema/widths/shared geometry required')
    D.require_zero_model_dropout(adapter, require_registered_cfg=True)
    return {'catalogue_K': K, 'query_dim': 64, 'rich': True,
        'side_branch': True, 'user_branch': True, **contract['widths'], 'dropout': 0.,
        'production_schema_self_sha256': contract['schema_self_sha256'], 'shared_geometry': geometry}


def load_model_only(path, adapter, cell):
    """Strict whitelist; fixed buffers come from the registered initializer, not export."""
    import torch
    if cell != D.select_cell(cell['mechanism'], cell['eta']):
        raise Refused('fixed cell required')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != PUBLIC_KEYS:
        raise Refused('model-only payload contains missing/private/unknown fields')
    if (payload['format'] != 'PPSI_DPFTRL_LOCAL_FIXED_CELL_MODEL_ONLY_1'
        or payload['public_cell'] != cell
        or payload['conditional_fixed_recipe_accountant'] != D.budget(cell['mechanism'])
        or payload['privacy_certified'] is not False
        or payload['pretraining_preprocessing_tuning_accounted'] is not False
        or payload['transport_isolation_implemented'] is not False
        or payload['execution_scope'] != 'SINGLE_PROCESS_ROLE_BOUNDARY_SIMULATOR_NOT_DEPLOYED_TRANSPORT'):
        raise Refused('exact model-only public cell/accountant/scope required')
    layout = D.SharedVector(adapter)
    shared = payload['shared_theta']
    if not isinstance(shared, dict) or set(shared) != set(layout.keys):
        raise Refused('exact canonical shared trainables required; buffers/aliases refused')
    for k in layout.keys:
        if not torch.is_tensor(shared[k]) or shared[k].device.type != 'cpu':
            raise Refused('CPU shared tensors required')
    # flatten validates dtype/shape/finiteness; state inserts only frozen registered buffers.
    state = layout.state(layout.flatten(shared))
    adapter.load_state_(state)
    return payload


def query_inputs(batch):
    return {k: v for k, v in batch.items() if k not in TRUTH_KEYS}


def query_batches(ep):
    for batch in ep.batches(ep.frozen.eval_batch):
        yield query_inputs(batch)


def population_proof(manifest):
    import numpy as np
    users = np.unique(manifest.eval_user_key[manifest.eval_mask])
    rankable_users = np.unique(manifest.eval_user_key[manifest.rankable])
    obj = {'manifest_id': manifest.manifest_id, 'manifest_rows': manifest.n,
        'E2E_decisions': int(manifest.eval_mask.sum()), 'E2E_users': len(users),
        'rankable_users': len(rankable_users), 'OOV_only_users': len(users) - len(rankable_users),
        'catalogue_K': manifest.catalogue_K, 'hashes': dict(manifest.hashes)}
    expected = {'manifest_id': D.EVAL, 'manifest_rows': 158135, 'E2E_decisions': 127530,
        'E2E_users': 10070, 'rankable_users': 10060, 'OOV_only_users': 10, 'catalogue_K': K}
    if any(obj[k] != v for k, v in expected.items()):
        raise Refused('full actual CALIB validation population differs')
    return obj


def validation_stats(manifest, ranks):
    """All E2E clients, including OOV-only; censored rows never enter averages."""
    import numpy as np
    from v4eval.metrics import credits_from_ranks
    from v4eval.bootstrap import user_stats
    ranks = np.asarray(ranks)
    if ranks.shape != (manifest.n,) or not np.issubdtype(ranks.dtype, np.integer):
        raise Refused('integer ranks must cover the exact whole manifest')
    if (np.any(ranks[manifest.rankable] < 1) or np.any(ranks[manifest.rankable] > manifest.catalogue_K)
        or np.any(ranks[~manifest.rankable] != -1)):
        raise Refused('invalid full-K ranks/OOV/censored sentinel')
    credit = credits_from_ranks(ranks, (20,))['mrr@20']
    return user_stats(credit, manifest.eval_mask, manifest.rankable, manifest.eval_user_key, 'DPFTRL03_CALIB')


def metric_recount(manifest, art, result, *, actual_population=True):
    from v4eval.bootstrap import point
    stats = validation_stats(manifest, art.ranks)
    if actual_population and (len(stats.user_keys) != 10070 or stats.n_decisions != 127530):
        raise Refused('actual full-client E2E denominator differs')
    value = point(stats, 'macro_e2e')
    try:
        reported = float(result['E2E']['macro']['mrr@20'])
    except (TypeError, ValueError) as exc:
        raise Refused('finite numerical E2E metric required') from exc
    if not math.isfinite(value) or not math.isfinite(reported) or abs(value - reported) > 1e-15:
        raise Refused('independent full-client E2E metric recount differs')
    if (len(stats.user_keys) != result['E2E']['n_users'] or stats.n_decisions != result['E2E']['n_decisions']):
        raise Refused('independent result user/decision denominator differs')
    return value


def evaluate_utility(manifest, art, config):
    from v4eval import evaluator_code_sha256
    from v4eval.evaluate import evaluate
    return evaluate(manifest, art, config=config, evaluator_code_sha256=evaluator_code_sha256())


def utility_rows(result, cell, comparison_key):
    """Own method labels; do not impersonate inherited FA/DP-SGD result rows."""
    rows = []
    for population in ('E2E', 'RANKABLE'):
        block = result[population]
        for aggregation in ('micro', 'macro'):
            for metric, value in block[aggregation].items():
                rows.append({'method': 'DPFTRL_TREE' if cell['mechanism'] == 'tree' else 'LDP_MESSAGE_ROLE_SIMULATOR',
                    'cell': dict(cell), 'role': 'CALIBRATION', 'split': 'VALIDATION', 'label': D.LABEL,
                    'population': population, 'aggregation': aggregation, 'metric': metric, 'value': value,
                    'n_users': block['n_users'], 'n_decisions': block['n_decisions'],
                    'comparison_key': dict(comparison_key), 'model_sha256': result['model_sha256'],
                    'eval_config': result['eval_config'], 'privacy_certified': False,
                    'utility_release_inside_privacy_accountant': False,
                    'transport_isolation_implemented': False})
    return rows


def actual_cpu_state(gpu):
    """Only frozen trainer's documented actual_inputs API; no patching/fallback."""
    state = D.actual_inputs(gpu, device='cpu')
    import torch
    if torch.cuda.is_initialized():
        raise Refused('CPU preparation initialized CUDA')
    original = D.read(TRAIN_PREPARATION)
    observed = dict(state['proof'], protocol_sha256=TRAIN_PROTOCOL_SHA, declaration=D.declaration(),
        module_sha256=TRAIN_MODULE_SHA, tests_sha256=TRAIN_TESTS_SHA, GPU_initialized=False)
    if observed != original:
        raise Refused('actual full provider/init no longer matches frozen trainer03 preparation')
    require_actual_model(state['adapter'], schema=state['schema'])
    population_proof(state['ep'].manifest)
    return state


def prepare(protocol_path, protocol_sha, out, gpu, *, role='CALIBRATION', split='VALIDATION'):
    exact_role(role, split)
    protocol = load_protocol(protocol_path, protocol_sha)
    D.fresh(out)
    state = actual_cpu_state(gpu)
    import torch
    from v4eval.ranking import rank_from_query
    adapter, ep = state['adapter'], state['ep']
    # A separate, explicitly untrained public-initializer export validates the real
    # full architecture's model-only loading without creating a fake trained run.
    cell = D.select_cell('tree', .1)
    payload = D.model_only_payload(adapter.broadcast_state(), state['layout'], cell, D.budget('tree'))
    import io
    stream = io.BytesIO(); torch.save(payload, stream); stream.seek(0)
    load_model_only(stream, adapter, cell)
    first = next(iter(ep.batches(ep.frozen.eval_batch)))
    n = int(first['decision_id'].shape[0])
    batch = {k: v[:2] if torch.is_tensor(v) and v.ndim and v.shape[0] == n else v for k, v in first.items()}
    adapter.module.eval()
    with torch.no_grad():
        query = adapter.query(query_inputs(batch))
        poison = dict(batch, target_class=torch.full_like(batch['target_class'], -999999))
        if not torch.equal(query, adapter.query(query_inputs(poison))):
            raise Refused('target labels affected rich query')
        rows = torch.as_tensor([int(x) for x in batch['target_class']])
        valid = rows >= 0
        if not valid.any():
            raise Refused('first real validation sample lacks a rankable row')
        r = rank_from_query(query[valid], adapter.head_weight(), adapter.head_bias(), rows[valid],
            score_block=K, class_chunk=32768, K_expected=K, cache_bytes=1073741824)
        if not torch.isfinite(query).all() or not ((r.ranks >= 1) & (r.ranks <= K)).all():
            raise Refused('actual rich full-K CPU rank interface failed')
    if torch.cuda.is_initialized():
        raise Refused('CPU readiness initialized CUDA')
    proof = {'status': 'PASS_ACTUAL_CPU_VALIDATION_READY_NOT_SCORED', 'utc': utc(),
        'protocol_sha256': protocol_sha, 'module_sha256': D.sha(__file__),
        'tests_sha256': D.sha(CODE / 'experiment_extensions/test_dpftrl_validation.py'),
        'trainer_preparation': protocol['trainer_preparation'], 'actual_provider_proof': state['proof'],
        'population': population_proof(ep.manifest), 'model': require_actual_model(adapter, schema=state['schema']),
        'provider_fingerprint': ep.fingerprint, 'provider_manifest_binding': ep.manifest_binding,
        'comparison_key': state['proof']['comparison_key'],
        'frozen_T3': {'eval_batch': ep.frozen.eval_batch, 'score_block': ep.frozen.score_block,
            'class_chunk': ep.frozen.class_chunk, 'cache_bytes': ep.frozen.cache_bytes},
        'public_initializer_model_only_roundtrip': True, 'real_validation_forward_rows': int(valid.sum()),
        'actual_CPU_full_K_rank': True, 'target_truth_excluded_from_query': True,
        'trained_model_scored': False, 'GPU_initialized': False, 'TEST_read': False,
        'private_training_state_read': False, 'real_training_started': False, 'privacy_certified': False}
    load_protocol(protocol_path, protocol_sha)
    D.write_private(out, proof)
    return {'status': proof['status'], 'preparation': D.pin(out)}


def preparation_gate(path, expected_sha, protocol_sha):
    own_path(path)
    if D.sha(path) != expected_sha:
        raise Refused('scorer CPU preparation bytes changed')
    p = D.read(path)
    if (p.get('status') != 'PASS_ACTUAL_CPU_VALIDATION_READY_NOT_SCORED'
        or p.get('protocol_sha256') != protocol_sha or p.get('module_sha256') != D.sha(__file__)
        or p.get('tests_sha256') != D.sha(CODE / 'experiment_extensions/test_dpftrl_validation.py')
        or p.get('GPU_initialized') is not False or p.get('trained_model_scored') is not False
        or p.get('TEST_read') is not False or p.get('private_training_state_read') is not False
        or p.get('actual_CPU_full_K_rank') is not True or p.get('target_truth_excluded_from_query') is not True
        or p.get('frozen_T3') != {'eval_batch': 1024, 'score_block': K, 'class_chunk': 32768, 'cache_bytes': 1073741824}):
        raise Refused('exact actual CPU validation readiness required')
    return p


def review_gate(path, expected_sha, protocol_sha, preparation_sha):
    own_path(path)
    if D.sha(path) != expected_sha:
        raise Refused('independent scoring review bytes changed')
    p = D.read(path)
    if (p.get('status') != 'PASS' or p.get('reviewer') in (None, '', '/root/dpftrl_validation_complete')
        or p.get('protocol_sha256') != protocol_sha or p.get('preparation_sha256') != preparation_sha
        or p.get('module_sha256') != D.sha(__file__)
        or p.get('tests_sha256') != D.sha(CODE / 'experiment_extensions/test_dpftrl_validation.py')
        or p.get('CALIB_VALIDATION_scoring_approved') is not True or p.get('TEST_approved') is not False
        or p.get('privacy_certified') is not False):
        raise Refused('exact independent CALIB-only utility review required')
    return p


def own_completed_export(protocol, mechanism, eta, training_review_path, training_review_sha):
    cell = D.select_cell(mechanism, eta)
    directory = RUN_ROOT / cell['cell_id']
    if protocol.get('run_root') != str(RUN_ROOT) or cell not in protocol['cells']:
        raise Refused('exact own declared cell output required')
    files = {name: own_path(directory / name, root=RUN_ROOT) for name in declaration()['input_whitelist']}
    complete, started = D.read(files['COMPLETE.json']), D.read(files['STARTED.json'])
    expected_binding = {'cell': cell, 'protocol_sha256': TRAIN_PROTOCOL_SHA,
        'preparation_sha256': TRAIN_PREPARATION_SHA, 'review_sha256': training_review_sha,
        'module_sha256': TRAIN_MODULE_SHA, 'role': 'CALIBRATION', 'split': 'VALIDATION'}
    if (set(started) != set(expected_binding) | {'gpu_ownership_check'}
        or any(started.get(k) != v for k, v in expected_binding.items())
        or not isinstance(started.get('gpu_ownership_check'), dict)):
        raise Refused('training STARTED identity/source/recipe/role/review differs')
    if (complete.get('status') != 'TRAIN_ONLY_NOT_UTILITY_COMPLETE' or complete.get('binding') != started
        or complete.get('model_file_sha256') != D.sha(files['MODEL_ONLY/model.pt'])
        or complete.get('TEST_read') is not False or complete.get('validation_scored') is not False
        or complete.get('utility_complete') is not False or complete.get('privacy_certified') is not False
        or complete.get('execution_scope') != declaration()['execution_scope']):
        raise Refused('exact completed own model-only export required')
    own_path(training_review_path)
    # This is the FUTURE full root training review, not the frozen CPU-only component.
    training_review = D.review_gate(training_review_path, training_review_sha, TRAIN_PROTOCOL_SHA, TRAIN_PREPARATION_SHA, mechanism)
    expected_outputs = {c['cell_id']: str(RUN_ROOT / c['cell_id']) for c in protocol['cells']}
    if training_review.get('approved_output_directories') != expected_outputs:
        raise Refused('future full training review must bind all exact own output directories')
    return {'cell': cell, 'directory': directory, 'files': {name: D.pin(p) for name, p in files.items()},
        'training_review': D.pin(training_review_path), 'binding': started}


def score(protocol_path, protocol_sha, preparation_path, preparation_sha, review_path, review_sha,
          mechanism, eta, training_review_path, training_review_sha, out, gpu,
          *, role='CALIBRATION', split='VALIDATION'):
    exact_role(role, split)
    D.select_cell(mechanism, eta, role=role)
    D.require_early_cuda(gpu)
    protocol = load_protocol(protocol_path, protocol_sha)
    preparation = preparation_gate(preparation_path, preparation_sha, protocol_sha)
    review_gate(review_path, review_sha, protocol_sha, preparation_sha)
    identity = own_completed_export(protocol, mechanism, eta, training_review_path, training_review_sha)
    out = D.fresh(out)
    state = actual_cpu_state(gpu)
    ep = state['ep']
    if (state['proof'] != preparation['actual_provider_proof'] or ep.fingerprint != preparation['provider_fingerprint']
        or ep.manifest_binding != preparation['provider_manifest_binding']
        or population_proof(ep.manifest) != preparation['population']):
        raise Refused('scorer provider/preparation binding changed')
    import torch
    from v4int import launch_central as LC
    gpu_ownership = LC.gpu_checks(SimpleNamespace(gpu=gpu, allow_shared_gpu=False), LC.default_runner, dict(os.environ))
    if not torch.cuda.is_available():
        raise Refused('CUDA scoring requires actual root-owned physical GPU')
    from v4models import build
    from v4data.common import dataset_root
    from v4int import launch_fl as F
    adapter = build(F.FAMILY, state['schema'], D.SEED, 'cuda:0', v1_root=dataset_root(),
                    variant=state['variant'], dropout=(0.,)).adapter
    F.apply_init_state(adapter, state['adapter'].module.state_dict(), D.INIT_DIGEST)
    require_actual_model(adapter, schema=state['schema'])
    model = identity['files']['MODEL_ONLY/model.pt']
    load_model_only(model['path'], adapter, identity['cell'])
    from v4eval.predict import predict_adapter
    from v4eval.values import to_jsonable
    art = predict_adapter(adapter, query_batches(ep), ep.manifest, config=ep.frozen,
        model_sha256=model['sha256'], device=torch.device('cuda:0'))
    result = evaluate_utility(ep.manifest, art, ep.frozen)
    utility = metric_recount(ep.manifest, art, result)
    # Rebind every input after scoring before producing any completion receipt.
    load_protocol(protocol_path, protocol_sha)
    preparation_gate(preparation_path, preparation_sha, protocol_sha)
    review_gate(review_path, review_sha, protocol_sha, preparation_sha)
    again = own_completed_export(protocol, mechanism, eta, training_review_path, training_review_sha)
    if again != identity:
        raise Refused('training identity changed during scoring')
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    import numpy as np
    artifacts = {}
    for name, value in (('ranks_int64.npy', art.ranks), ('n_tied_int64.npy', art.n_tied),
                        ('eq_before_int64.npy', art.eq_before), ('decision_id_int64.npy', art.decision_id)):
        p = out / name
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            np.save(stream, value, allow_pickle=False)
        artifacts[name] = D.pin(p)
    D.write_private(out / 'RESULT.json', to_jsonable(result))
    artifacts['RESULT.json'] = D.pin(out / 'RESULT.json')
    D.write_private(out / 'ROWS.json', to_jsonable(utility_rows(result, identity['cell'], state['proof']['comparison_key'])))
    artifacts['ROWS.json'] = D.pin(out / 'ROWS.json')
    record = {'status': 'CALIB_VALIDATION_UTILITY_COMPLETE_SEPARATE_FROM_TRAINING', 'utc': utc(),
        'protocol': D.pin(protocol_path), 'preparation': D.pin(preparation_path), 'review': D.pin(review_path),
        'module': D.pin(__file__), 'tests': D.pin(CODE / 'experiment_extensions/test_dpftrl_validation.py'),
        'training': {k: str(v) if isinstance(v, Path) else v for k, v in identity.items()},
        'artifacts': artifacts, 'prediction_header': art.header, 'population': population_proof(ep.manifest),
        'comparison_key': state['proof']['comparison_key'], 'gpu_ownership': gpu_ownership,
        'E2E_CLIENT_MACRO_MRR_20': utility, 'declaration': declaration(),
        'TEST_read': False, 'private_training_state_read': False, 'training_receipt_modified': False,
        'privacy_certified': False, 'joint_epsilon8_claim': False}
    D.write_private(out / 'COMPLETE.json', record)
    return {'status': record['status'], 'completion': D.pin(out / 'COMPLETE.json')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build-protocol'); b.add_argument('--out', type=Path, required=True)
    prep = sub.add_parser('prepare'); scorep = sub.add_parser('score')
    for parser in (b, prep, scorep):
        parser.add_argument('--role', choices=('CALIBRATION',), default='CALIBRATION')
        parser.add_argument('--split', choices=('VALIDATION',), default='VALIDATION')
    for parser in (prep, scorep):
        parser.add_argument('--protocol', type=Path, required=True)
        parser.add_argument('--protocol-sha256', required=True)
        parser.add_argument('--out', type=Path, required=True)
        parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    for name in ('preparation', 'review', 'training-review'):
        scorep.add_argument('--' + name, type=Path, required=True)
        scorep.add_argument('--' + name + '-sha256', required=True)
    scorep.add_argument('--mechanism', choices=('tree', 'local'), required=True)
    scorep.add_argument('--eta', type=float, choices=(.1, 1.), required=True)
    args = vars(p.parse_args(argv)); command = args.pop('command')
    if command == 'build-protocol':
        result = build_protocol(args.pop('out'), **args)
    elif command == 'prepare':
        result = prepare(args.pop('protocol'), args.pop('protocol_sha256'), args.pop('out'), args.pop('gpu'), **args)
    else:
        result = score(args.pop('protocol'), args.pop('protocol_sha256'), args.pop('preparation'),
            args.pop('preparation_sha256'), args.pop('review'), args.pop('review_sha256'),
            args.pop('mechanism'), args.pop('eta'), args.pop('training_review'), args.pop('training_review_sha256'),
            args.pop('out'), args.pop('gpu'), **args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
