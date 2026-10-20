"""Own four-cell CALIB cold-DP grid, with private participation/noise streams.

No inherited source is changed. The fixed-cell accountant is the unchanged
registered Poisson SGM/RDP implementation; private S_CAL and tuning releases are
excluded. Raw outputs, keys, checkpoints, entropy and validation stay private.
Only a separately whitelisted endpoint model can be exported. No TEST route.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
from types import SimpleNamespace
import ppsi_paths

BASE = ppsi_paths.SERVER_BASE
EXT = ppsi_paths.EXTENSIONS_ROOT
WORK = EXT / 'ppsi_v4'
CODE = WORK / 'code'
DATA = BASE / 'shared_data/rees46_nextitem_v1'
V = BASE / 'Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4'
GRID = EXT / 'planning/COLD_DP_CALIB_GRID_01.json'
DONOR = V / 'runs_x_rees46/DP/COLD/DP8_s2026/ARM_SPEC.json'
TRAIN = 'CALIB_TRAIN_5_100_32768'
EVAL = 'CALIB_VALIDATION_5_100_32768'
N, M, DELTA = 32768, 1024, 5e-6
LABEL = 'POST_HOC_EXPLORATORY'
_ACTIVE = False


class Refused(ValueError):
    pass


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def physical(path):
    value = str(path)
    for alias in ppsi_paths.LEGACY_ALIASES:
        if value == alias or value.startswith(alias + '/'):
            value = str(BASE) + value[len(alias):]
            break
    return Path(value).resolve()


def inside(path):
    p = physical(path)
    if p == EXT or EXT not in p.parents:
        raise Refused('output escapes the extension')
    return p


def sha(path):
    h = hashlib.sha256()
    with physical(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(physical(path).read_text(),
        parse_constant=lambda x: (_ for _ in ()).throw(Refused('nonfinite JSON ' + x)))


def canonical(obj):
    return hashlib.sha256(json.dumps({k: v for k, v in obj.items() if k != 'self_sha256'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pin(path):
    p = physical(path)
    if BASE not in p.parents or not p.is_file():
        raise Refused('input outside the EID copy or absent')
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def write_private(path, obj):
    p = inside(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    return p


def load_grid():
    grid = read(GRID)
    if grid.get('format') != 'PPSI_COLD_DP_CALIB_GRID_1' or grid.get('self_sha256') != canonical(grid):
        raise Refused('fixed grid format/hash differs')
    expected = [('A', 96, 2, .1, 1.), ('B', 192, 1, .1, 1.),
                ('C', 96, 2, .03, 1.), ('D', 96, 2, .1, .5)]
    if [(c['cell_id'], c['T'], c['passes'], c['client_lr'], c['S_multiplier']) for c in grid['cells']] != expected:
        raise Refused('four-cell predeclared grid changed')
    for c in grid['cells']:
        if (c['N'], c['planned_m'], c['q'], c['server_eta'], c['server_tau'], c['dropout_p'], c['seed']) != (
                N, M, M / N, .001, 1e-6, 0., 2026):
            raise Refused('fixed grid population/recipe differs')
    if grid.get('role') != 'CALIBRATION' or grid.get('TEST_allowed') is not False:
        raise Refused('only CALIBRATION is admitted')
    return grid


def select_cell(cell_id, *, role='CALIBRATION'):
    if role != 'CALIBRATION' or cell_id not in ('A', 'B', 'C', 'D'):
        raise Refused('fixed CALIBRATION cell required before any data access')
    return next(c for c in load_grid()['cells'] if c['cell_id'] == cell_id)


def numerical_sources():
    return sorted(p for p in CODE.rglob('*.py') if 'experiment_extensions' not in p.parts) + [
        CODE / 'experiment_extensions/cold_dp_tuning.py', CODE / 'experiment_extensions/test_cold_dp_tuning.py']


def declared_inputs():
    paths = set(physical(p) for p in (WORK / 'contracts').rglob('*') if p.is_file())
    reg = read(WORK / 'contracts/manifest_registry/REGISTRY_V2.json')
    if reg['roles']['CALIBRATION'] != {'train': TRAIN, 'eval': EVAL} or reg['manifests'][TRAIN]['n_users'] != N:
        raise Refused('registry is not the actual CALIB population')
    for mid in (TRAIN, EVAL):
        entry = reg['manifests'][mid]
        paths.update([physical(WORK / entry['path']), physical(WORK / entry['marker'])])
        paths.update(physical(WORK / row['file']) for row in entry['tables'].values())
    paths.add(physical(WORK / 'data_v2/TRANSFORM_STATE_CLEAN_V2.json'))
    paths.update(DATA.glob('model_ready/user_bucket=*/features.parquet'))
    for split in ('TRAIN', 'VALIDATION_SELECT', 'VALIDATION_CONFIRM'):
        paths.update((DATA / 'decisions_index' / ('split=' + split)).glob('user_bucket=*/*.parquet'))
    paths.update((DATA / 'mappings').glob('*'))
    paths.update([DONOR, GRID])
    return sorted(p for p in paths if p.is_file())


def build_protocol(out):
    if inside(out).exists():
        raise Refused('protocol must be new')
    from fl import accountant as A
    grid = load_grid()
    for cell in grid['cells']:
        computed = A.registration_record([8.], m=M, N=N, T=cell['T'], delta=DELTA)['records'][0]
        expected = {k: v for k, v in cell['accountant'].items() if k != 'calculation_wall_s'}
        if computed != expected or computed['achieved_epsilon'] > 8:
            raise Refused('registered STANDARD accountant no longer matches fixed cell')
    obj = {'format': 'PPSI_COLD_DP_CALIB_PROTOCOL_1', 'utc': utc(), 'grid_sha256': sha(GRID),
           'grid': grid, 'label': LABEL, 'role': 'CALIBRATION', 'TEST_allowed': False,
           'sources': [pin(p) for p in numerical_sources()], 'inputs': [pin(p) for p in declared_inputs()],
           'release_boundary': 'MODEL_ONLY; internal raw metadata and entropy are private',
           'sampling': 'private Poisson rate1024/32768; fixed denominator1024',
           'noise': 'private Gaussian entropy separate from public optimizerseed2026',
           'prior_private_tuning_accounted': False, 'real_training_started': False}
    if any(row != pin(row['path']) for row in obj['sources']) or obj['grid_sha256'] != sha(GRID):
        raise Refused('source or grid changed during protocol binding')
    obj['self_sha256'] = canonical(obj)
    write_private(out, obj)
    return {'status': 'BOUND_NOT_TRAINED', 'path': str(physical(out)), 'sha256': sha(out)}


def load_protocol(path, expected_sha):
    p = inside(path)
    if sha(p) != expected_sha:
        raise Refused('protocol hash differs')
    obj = read(p)
    if obj.get('format') != 'PPSI_COLD_DP_CALIB_PROTOCOL_1' or obj.get('self_sha256') != canonical(obj):
        raise Refused('protocol self hash/format differs')
    if obj.get('grid_sha256') != sha(GRID) or obj.get('grid') != load_grid() or obj.get('TEST_allowed') is not False:
        raise Refused('protocol does not bind current fixed CALIB grid')
    for name, paths in [('sources', numerical_sources()), ('inputs', declared_inputs())]:
        if [r['path'] for r in obj[name]] != [str(physical(p)) for p in paths]:
            raise Refused(name + ' inventory changed')
        for row in obj[name]:
            if row != pin(row['path']):
                raise Refused('bound input changed: ' + row['path'])
    return obj


def build_spec(cell, *, mode, clip=None):
    if cell != select_cell(cell['cell_id']):
        raise Refused('cell does not equal the exact fixed recipe')
    if mode not in ('S_CAL', 'DP'):
        raise Refused('only own S_CAL or DP mode')
    if mode == 'DP' and not (clip is not None and 0 < float(clip) < float('inf')):
        raise Refused('DP needs the exact finite positive fresh clip binding')
    from v4int import launch_fl as F
    source = read(DONOR)
    if source['arm_spec_sha256'] != F.spec_digest(source['arm_spec']):
        raise Refused('donor spec identity differs')
    spec = copy.deepcopy(source['arm_spec'])
    if spec['arm'] != 'DP' or spec['cfg']['server_opt']['tau'] != 1e-6 or spec.get('init_from'):
        raise Refused('donor is not the tau-fixed cold DP path')
    cfg = spec['cfg']
    cfg.update(seed=2026, lr_peak_local=cell['client_lr'], endpoint_rounds=cell['T'], dropout_p=0., lr_shape='fl_const')
    cfg['solver'].update(lr=cell['client_lr'], passes=cell['passes'], batch_size=16, clip=1., weight_decay=0.)
    cfg['server_opt'].update(lr=.001, tau=1e-6)
    cfg['dp'] = {'clip_norm': None if mode == 'S_CAL' else float(clip),
                 'noise_multiplier': 0. if mode == 'S_CAL' else cell['accountant']['z'], 'denominator': M}
    spec.update(arm='S_CAL' if mode == 'S_CAL' else 'DP', role='CALIBRATION',
                max_rounds=20 if mode == 'S_CAL' else cell['T'], extension_label=LABEL,
                extension_cell=cell['cell_id'], extension_grid_sha256=sha(GRID),
                client_lr=cell['client_lr'], local_passes=cell['passes'])
    spec['knobs'] = dict(spec['knobs'], rounds_1024=cell['T'])
    spec['plan'] = {'group_size': M, 'sampling': 'poisson'}
    spec['poisson_local_rule'] = {'passes': cell['passes'], 'batch_size': 16,
                                'rule': 'own fixed CALIB grid; registered numerical solver unchanged'}
    return spec


@contextlib.contextmanager
def private_randomness(*, participation_seed, noise_seed):
    """Scoped inherited numerical law with independent secret randomness."""
    global _ACTIVE
    if _ACTIVE:
        raise Refused('nested private randomness context')
    if any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**63 for s in (participation_seed, noise_seed)):
        raise Refused('two independent private uint63 seeds required')
    if participation_seed == noise_seed:
        raise Refused('participation and noise entropy must be separate')
    from fl import dp as D
    from fl.participation import ParticipationPlan as P
    original_noise, original_build = D.noise_like, P.__dict__['build']

    def noise(keys, template, *, seed, round_idx, std):
        return original_noise(keys, template, seed=noise_seed, round_idx=round_idx, std=std)

    def build(cls, keys, *, seed, manifest_hash, group_size=64, expected_keys_digest=None, sampling='sweep'):
        if sampling != 'poisson' or group_size != M:
            raise Refused('private CALIB mechanism requires exact Poisson m1024')
        return original_build.__func__(cls, keys, seed=participation_seed, manifest_hash=manifest_hash,
            group_size=group_size, expected_keys_digest=expected_keys_digest, sampling=sampling)

    _ACTIVE = True
    try:
        D.noise_like, P.build = noise, classmethod(build)
        yield
    finally:
        D.noise_like, P.build = original_noise, original_build
        _ACTIVE = False


def _inputs(cell, spec, gpu, *, device='cpu', log=lambda *args: None):
    """Unchanged full client batches; CPU preparation performs no ranking."""
    os.environ['V4_DATASET_ROOT'] = str(DATA)
    from v4int import launch_central as LC
    LC.export_environment(gpu)  # before v4train/torch import
    import ppsi_env
    ppsi_env.verify_snapshot()
    ppsi_env.assert_isolated()
    from v4int import launch_fl as F
    from v4int.verify import DataPaths
    from v4int.providers import StoreTrainProvider, StoreEvalProvider, comparison_key_base
    from v4models import build
    from fl.client import ClientData
    from fl.numerics import enforce_strict_fp32, state_digest
    from v4data.common import dataset_root
    enforce_strict_fp32()
    paths = DataPaths.default(WORK)
    lock = LC.bind_data_protocol_lock_v2(paths, smoke=False)
    kw = dict(history_policy='USER_HISTORY_L50', paths=paths, log=log,
              manifest_registry=lock['manifest_registry_path'], registry_root=WORK)
    tp = StoreTrainProvider(TRAIN, **kw)
    ep = StoreEvalProvider(EVAL, prefetch=0, **kw)
    keys, positions = F.client_index(tp, paths.manifest_dir)
    if len(keys) != N or tp.n_decisions != 646067:
        raise Refused('actual full CALIB TRAIN population differs')
    small = F._small_recipe(SimpleNamespace(small_recipe=None, smoke=False))
    schema = LC.load_production_schema(lock['schema_id'], lock['schema_self_sha256'], lock['schema_registry_path'])
    theta0 = LC._theta0_for(SimpleNamespace(theta0_variants=None, smoke=False), lock['schema_id'],
                            {'model_variant': small['model_variant']}, 2026)
    built = build(F.FAMILY, schema, 2026, device, v1_root=dataset_root(), variant=small['model_variant'],
                  dropout=tuple(small['dropout']))
    if built.init_sha256 != theta0 or state_digest(built.adapter.broadcast_state()) != theta0:
        raise Refused('actual cold theta0 does not match registered initializer')
    return dict(tp=tp, ep=ep, keys=keys, positions=positions, paths=paths, lock=lock, small=small,
                schema=schema, theta0=theta0, built=built,
                key=comparison_key_base(paths, tp, ep),
                get_client=lambda k: ClientData(k, tp.batch(positions[k])),
                n_valid={k: int(v.size) for k, v in positions.items()})


def model_only_payload(theta, cell, protocol_sha):
    """Export whitelist; private protocol/data commitments are not exported.

The protocol_sha argument is an internal identity and intentionally omitted:
it commits to private input-file hashes and is not a DP-safe public metadata.
    """
    import torch
    if not isinstance(theta, dict) or not theta or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in theta.values()):
        raise Refused('finite model state required for explicit model-only export')
    return {'format': 'PPSI_COLD_DP_FIXED_CELL_MODEL_ONLY_1',
            'theta': {k: v.detach().cpu().clone() for k, v in theta.items()},
            'cell_id': cell['cell_id'],
            'accountant_fixed_cell_only': cell['accountant'], 'private_tuning_accounted': False}


def prepare(protocol_path, protocol_sha, cell_id, out, *, gpu=0):
    cell = select_cell(cell_id)
    load_protocol(protocol_path, protocol_sha)
    from v4int.launch_central import export_environment
    export_environment(gpu)
    spec = build_spec(cell, mode='S_CAL')
    state = _inputs(cell, spec, gpu)
    from v4int import launch_fl as F
    proof = {'status': 'PASS_PREPARED_NOT_TRAINED', 'utc': utc(), 'protocol_sha256': protocol_sha,
             'cell_id': cell_id, 'source_sha256': sha(__file__), 'role': 'CALIBRATION',
             'TRAIN': TRAIN, 'VALIDATION': EVAL, 'N_clients': len(state['keys']),
             'N_decisions': state['tp'].n_decisions, 'theta0_digest': state['theta0'],
             'schema_self_sha256': state['schema'].self_sha256,
             'client_positions': sum(len(v) for v in state['positions'].values()),
             'full_client_examples_used': True, 'comparison_key': state['key'],
             'spec_digest': F.spec_digest(spec), 'accountant': cell['accountant'],
             'GPU_training_started': False, 'TEST_read': False}
    write_private(out, proof)
    return proof


def load_clip(path, cell, protocol_sha):
    if path is None:
        raise Refused('exact own fresh clip binding required')
    obj = read(inside(path))
    if obj.get('format') != 'PPSI_COLD_DP_OWN_CLIP_1' or obj.get('self_sha256') != canonical(obj):
        raise Refused('own fresh clip format/hash required')
    if obj['protocol_sha256'] != protocol_sha or obj['S_CAL_group'] != cell['S_CAL_group']:
        raise Refused('clip belongs to another protocol/local recipe')
    if obj['source_record'] != pin(obj['source_record']['path']):
        raise Refused('calibration raw record changed')
    from v4int import launch_fl as F
    raw = F.load_s_record(physical(obj['source_record']['path']), smoke=False)
    group = select_cell(cell['S_CAL_group'])
    run_id = 'COLD_DP_SCAL_' + group['cell_id'] + '_CALIB_s2026'
    expected = F.fl_run_config(build_spec(group, mode='S_CAL'), run_id, 8).digest()
    if raw['run_id'] != run_id or raw['config_digest'] != expected:
        raise Refused('calibration is not this own exact CALIB local recipe')
    source = read(physical(obj['source_record']['path']).parent / 'ARM_SPEC.json')
    if (source['arm_spec'] != build_spec(group, mode='S_CAL')
            or source.get('extension_binding') != obj['source_binding']
            or obj['source_binding'].get('protocol_sha256') != protocol_sha
            or obj['source_binding'].get('mode') != 'S_CAL'
            or obj['source_binding'].get('run_id') != run_id):
        raise Refused('calibration source spec differs from exact fresh cell')
    if float(raw['S']) != obj['base_S'] or raw['init_sha256'] != obj['theta0_digest']:
        raise Refused('calibration S/init identity differs')
    return float(obj['base_S']) * cell['S_multiplier'], pin(path)


@contextlib.contextmanager
def execution_metadata(spec, binding):
    """Own exact-spec admission/labels only; inherited numerical path stays intact."""
    from v4int import launch_fl as F
    saved = {k: getattr(F, k) for k in ('write_json', 'spec_tuning', 'poisson_row_labels')}
    if getattr(saved['write_json'], '_cold_dp_extension', False):
        raise Refused('nested own metadata context')
    cell = select_cell(spec['extension_cell'])
    expected = F.spec_digest(spec)

    def tuning(value):
        if F.spec_digest(value) != expected:
            raise Refused('metadata context received a different source spec')
        return {'optimizer': 'fedadam', 'server_lr': .001, 'server_tau': 1e-6,
                'lr_mult': 1., 'lr_shape': 'fl_const', 'client_opt': 'sgd_m',
                'client_lr': cell['client_lr'], 'local_passes': cell['passes'], 'local_batch': 16}

    def labels(value):
        tuning(value)
        return {'fl_poisson_local_rule': 'OWN_COLD_DP_GRID_01',
                'fl_selected_local_passes': cell['passes'], 'fl_selected_local_batch': 16}

    def write(path, obj):
        if isinstance(obj, dict):
            obj = dict(obj, extension_binding=binding, exploratory_label=LABEL, extension_role='CALIBRATION')
        return saved['write_json'](path, obj)
    write._cold_dp_extension = True
    try:
        F.write_json, F.spec_tuning, F.poisson_row_labels = write, tuning, labels
        yield
    finally:
        for key, value in saved.items():
            setattr(F, key, value)


def run(protocol_path, protocol_sha, cell_id, mode, out, review_path, prep_path,
        *, gpu, clip_path=None, resume=False):
    cell = select_cell(cell_id)
    if mode == 'S_CAL' and cell_id == 'D':
        raise Refused('D reuses A calibration; duplicate calibration is not declared')
    if gpu not in (0, 1):
        raise Refused('physical GPU0 or1 required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise Refused('export physical CUDA_VISIBLE_DEVICES before numerical imports')
    protocol = load_protocol(protocol_path, protocol_sha)
    review, prep = read(inside(review_path)), read(inside(prep_path))
    if (review.get('status'), review.get('protocol_sha256'), review.get('module_sha256'), review.get('tests_sha256')) != (
            'PASS', protocol_sha, sha(__file__), sha(CODE / 'experiment_extensions/test_cold_dp_tuning.py')):
        raise Refused('exact independent source/protocol review required')
    if mode not in review.get('modes_allowed', []) or cell_id not in review.get('cells_allowed', []):
        raise Refused('independent review does not approve this cell/mode')
    if review.get('reviewer') in (None, '', '/root/independent_launch_review', 'independent_launch_review'):
        raise Refused('reviewer must be independent of this module author')
    if (prep.get('status') != 'PASS_PREPARED_NOT_TRAINED' or prep.get('protocol_sha256') != protocol_sha
            or prep.get('cell_id') != cell_id or prep.get('source_sha256') != sha(__file__)
            or prep.get('N_clients') != N or prep.get('N_decisions') != 646067
            or prep.get('full_client_examples_used') is not True):
        raise Refused('exact full CALIB provider/theta0 preparation required')
    if review.get('preparation_sha256') != sha(prep_path):
        raise Refused('independent review must bind exact full preparation receipt')
    clip, clip_binding = (None, None) if mode == 'S_CAL' else load_clip(clip_path, cell, protocol_sha)
    if mode == 'DP' and review.get('clip_sha256') != clip_binding['sha256']:
        raise Refused('DP review must bind this exact fresh clip record')
    directory = inside(out)
    if directory.is_symlink():
        raise Refused('symlink output refused')
    run_id = ('COLD_DP_SCAL_' if mode == 'S_CAL' else 'COLD_DP_TUNE_') + cell_id + '_CALIB_s2026'
    binding = {'protocol_sha256': protocol_sha, 'source_sha256': sha(__file__), 'cell_id': cell_id,
               'mode': mode, 'run_id': run_id, 'review': pin(review_path), 'preparation': pin(prep_path),
               'clip': clip_binding, 'output': str(directory)}
    private = directory / 'PRIVATE_STATE.json'
    if resume:
        state = read(private)
        if state['binding'] != binding or (directory / 'EXTENSION_COMPLETE.json').exists():
            raise Refused('only exact own incomplete private-state resume allowed')
    else:
        if directory.exists():
            raise Refused('fresh private output directory required')
        directory.mkdir(parents=True, mode=0o700)
        state = {'binding': binding, 'participation_seed': secrets.randbits(63), 'noise_seed': secrets.randbits(63)}
        while state['participation_seed'] == state['noise_seed']:
            state['noise_seed'] = secrets.randbits(63)
        write_private(private, state)
    os.chmod(directory, 0o700)
    if os.stat(private).st_mode & 0o077:
        raise Refused('private state permissions are too broad')
    spec = build_spec(cell, mode=mode, clip=clip)
    inputs = _inputs(cell, spec, gpu, device='cuda:0')
    if inputs['theta0'] != prep['theta0_digest'] or inputs['key'] != prep['comparison_key']:
        raise Refused('runtime inputs differ from reviewed preparation')
    import torch
    from v4int import launch_fl as F
    from v4train.recipe import RunConfig
    from v4train.trainer import CentralTrainer
    cfg = RunConfig(run_id=run_id, family=F.FAMILY, seed=2026, exposure_point='end',
        cuda_visible_devices=str(gpu), phase='CALIBRATION', population='COHORT', peak_lr=inputs['small']['peak_lr'],
        key=inputs['key'], test_overrides=False, determinism_mode='STRICT',
        manifest_registry=str(inputs['lock']['manifest_registry_path']), data_version=inputs['lock']['data_version'],
        schema_registry=str(inputs['lock']['schema_registry_path']), recipe_id=inputs['small']['recipe_id'],
        recipe_registry=inputs['small']['recipe_registry'], model_variant=inputs['small']['model_variant'],
        dropout=tuple(inputs['small']['dropout']), schedule=inputs['small']['schedule'], theta0_init_sha256=inputs['theta0'])
    harness = CentralTrainer(cfg, inputs['schema'], inputs['tp'], inputs['ep'], directory / 'private_eval_harness',
                             device='cuda:0', v1_root=str(DATA))
    def evaluate(mark):
        value = harness._evaluate(mark)
        return value.value, value.rows, value.ranks, value.result
    def log(message):
        with (directory / 'OPERATOR_PRIVATE.log').open('a') as f:
            f.write(str(message) + '\n')
    with execution_metadata(spec, binding):
        with private_randomness(participation_seed=state['participation_seed'], noise_seed=state['noise_seed']):
            result = F.execute(spec=spec, run_id=run_id, run_dir=directory, server_adapter=harness.adapter,
                init_sha256=inputs['theta0'], workers=[inputs['built'].adapter], keys=inputs['keys'],
                n_valid_by_key=inputs['n_valid'], get_client=inputs['get_client'],
                n_decisions=inputs['tp'].n_decisions, manifest_hash=inputs['tp'].fingerprint,
                evaluate=None if mode == 'S_CAL' else evaluate, resume=resume, ckpt_every=8, say=log)
    if mode == 'S_CAL':
        raw = F.load_s_record(directory / 'S_RECORD.json', smoke=False)
        record = {'format': 'PPSI_COLD_DP_OWN_CLIP_1', 'protocol_sha256': protocol_sha,
                  'S_CAL_group': cell['S_CAL_group'], 'source_record': pin(directory / 'S_RECORD.json'),
                  'source_binding': binding, 'theta0_digest': inputs['theta0'], 'base_S': raw['S'],
                  'private_tuning_accounted': False}
        record['self_sha256'] = canonical(record)
        write_private(directory / 'OWN_CLIP_BINDING.json', record)
    else:
        from fl.checkpoint import load_checkpoint
        checkpoint = load_checkpoint(directory / 'ckpt' / run_id / 'endpoint_6.0.pt')
        if checkpoint['run_id'] != run_id or checkpoint['config_digest'] != F.fl_run_config(spec, run_id, 8).digest():
            raise Refused('own endpoint checkpoint identity differs')
        release = directory / 'MODEL_ONLY'
        release.mkdir(mode=0o700)
        torch.save(model_only_payload(checkpoint['theta'], cell, protocol_sha), release / 'model.pt')
        os.chmod(release / 'model.pt', 0o600)
    load_protocol(protocol_path, protocol_sha)
    write_private(directory / 'EXTENSION_COMPLETE.json', {'status': 'COMPLETE_OWN_CALIBRATION', 'utc': utc(),
        'binding': binding, 'mode': mode, 'engine_status': result['status'], 'TEST_scored': False,
        'privacy_certified': False, 'private_tuning_accounted': False})
    return {'status': 'COMPLETE_OWN_CALIBRATION', 'cell_id': cell_id, 'mode': mode}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build-protocol'); build.add_argument('--out', required=True, type=Path)
    prep = sub.add_parser('prepare'); launch = sub.add_parser('run')
    for parser in (prep, launch):
        parser.add_argument('--protocol', required=True, type=Path); parser.add_argument('--protocol-sha256', required=True)
        parser.add_argument('--cell', required=True, choices=('A', 'B', 'C', 'D')); parser.add_argument('--out', required=True, type=Path)
        parser.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    launch.add_argument('--mode', choices=('S_CAL', 'DP'), required=True)
    launch.add_argument('--review', type=Path, required=True); launch.add_argument('--preparation', type=Path, required=True)
    launch.add_argument('--clip', type=Path); launch.add_argument('--resume', action='store_true')
    a = p.parse_args(argv)
    try:
        if a.command == 'build-protocol': result = build_protocol(a.out)
        elif a.command == 'prepare': result = prepare(a.protocol, a.protocol_sha256, a.cell, a.out, gpu=a.gpu)
        else: result = run(a.protocol, a.protocol_sha256, a.cell, a.mode, a.out, a.review, a.preparation,
                          gpu=a.gpu, clip_path=a.clip, resume=a.resume)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (Refused, OSError, ValueError, KeyError) as exc:
        print('COLD_DP_TUNING_REFUSED: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
