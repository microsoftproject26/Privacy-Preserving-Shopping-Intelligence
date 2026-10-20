"""CALIB-only matched-six-EFE local-pass experiments, with unchanged kernels.

Six new FA cells cover the missing PB100 P4 and cold P2/P4 experiments.
Two existing PB100 P2 cells are bound and reused. A mechanical CALIB-only
selection admits four subsequent FP/PF cells. Every numerical result is
post-hoc exploratory; no TEST or MAIN launch is accepted here.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import ppsi_paths

BASE = ppsi_paths.SERVER_BASE
EXT = ppsi_paths.EXTENSIONS_ROOT
WORK = EXT / 'ppsi_v4'
CODE = WORK / 'code'
DONOR = BASE / 'Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4'
DATA = BASE / 'shared_data/rees46_nextitem_v1'
TRAIN = 'CALIB_TRAIN_5_100_32768'
EVAL = 'CALIB_VALIDATION_5_100_32768'
LABEL = 'POST_HOC_EXPLORATORY'
FORMAT = 'PPSI_LOCAL_PASSES_CALIB_PROTOCOL_1'
SEED = 2026
PEAK = 0.0007114121116984968
BEST_DELTA = 0.0002
INIT = DONOR / 'runs_exp_pb100/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_PB100K_s2026/PRETRAIN_FOR_FL_INIT.json'
INIT_SHA = '22135cdf132eae1d405dd802acf1a756c9aba122e02c9a01f53876894bd58383'
INIT_DIGEST = '3e1c48efe79a1f1a1377719c144af9fe8e8610060ea7cc18b6bff545b61ce4e2'
LAUNCH_SHA = 'f4e56b8008e7f4a66d5a6a07cbc0b9d978c28d90d9f9596a3b166f5b01475734'
_ACTIVE = False


class Refused(ValueError):
    pass


def physical(path):
    s = str(path)
    for alias in ppsi_paths.LEGACY_ALIASES:
        if s == alias or s.startswith(alias + '/'):
            s = str(BASE) + s[len(alias):]
            break
    return Path(s)


def sha(path):
    h = hashlib.sha256()
    with physical(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    d = {k: v for k, v in obj.items() if k != 'self_sha256'}
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    def bad(v):
        raise Refused(f'nonfinite JSON {v}')
    return json.loads(physical(path).read_text(encoding='utf-8'), parse_constant=bad)


def pin(path):
    p = physical(path).resolve()
    if BASE not in p.parents or not p.is_file():
        raise Refused('bound input must be an existing file inside the EID copy')
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def check_pin(p):
    q = physical(p['path'])
    if q.stat().st_size != p['bytes'] or sha(q) != p['sha256']:
        raise Refused(f'pinned file changed: {p["path"]}')


def scoped(path, root=EXT):
    p, r = physical(path), physical(root).resolve()
    if p.resolve() == r or r not in p.resolve().parents:
        raise Refused('output must be strictly inside the fresh extension')
    if any(q.is_symlink() for q in [p, *p.parents] if q == r or r in q.parents):
        raise Refused('symlink output refused')
    return p


def write_once(path, obj):
    p = scoped(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=True, indent=1, allow_nan=False)
        f.write('\n')


def rounds(passes, n=32768):
    if passes not in (1, 2, 4) or n != 32768:
        raise Refused('only the fixed CALIB sweep population and P1/P2/P4 budgets')
    return -(-(60 * n) // (576 * passes))


def stage1_cells():
    rows = []
    for line, etas, local in [('pb100', (3e-5, 1e-4), (4,)),
                              ('cold', (3e-4, 1e-3), (2, 4))]:
        for eta in etas:
            for passes in local:
                name = f'{line}_FA_eta{eta:g}_P{passes}_E6_s2026'
                rows.append({'cell': name, 'line': line, 'method': 'FA', 'server_lr': eta,
                             'passes': passes, 'seed': SEED, 'stage': 'FA_SCREEN'})
    return rows


def fixed_declaration():
    return {'label': LABEL, 'role': 'CALIBRATION', 'seed': SEED,
        'train_manifest': TRAIN, 'eval_manifest': EVAL, 'n_clients': 32768,
        'train_decisions': 646067, 'validation_manifest_rows': 158135,
        'catalogue_K': 158486, 'history_policy': 'USER_HISTORY_L50',
        'endpoint_EFE': 6., 'marks': [u / 2 for u in range(1, 13)],
        'budget_formula': 'ceil(60*N/(576*passes)); consumed examples include repeated local passes',
        'solver': {'optimizer': 'sgd_m', 'momentum': .9, 'lr': .1, 'batch_size': 16,
                   'clip': 1., 'weight_decay': 0., 'item_lr_mult_pb100': .1,
                   'item_lr_mult_cold': 1., 'lr_shape': 'fl_const'},
        'server': {'name': 'fedadam', 'tau': 1e-6, 'beta1': .9, 'beta2': .99,
                   'v_init': 'tau_squared'},
        'plan': {'sampling': 'sweep', 'group_size': 64, 'dropout_p': .1, 'max_attempts': 2},
        'FP_mu': .01, 'PF_lambda': 1e-4, 'PF_clip': 1., 'PF_personal_lr': PEAK,
        'PF_lr_rule': 'unchanged SMALL peak times round factor, independently of shared SGD LR',
        'stage1_cells': stage1_cells(), 'selection_metric': 'CALIB E2E client-macro MRR@20 practical-best',
        'selection_marks': '0.5 through6.0, no mark0', 'best_delta': BEST_DELTA,
        'PB_retention_rule': 'chronological float score > retained_best + best_delta; initial retained_best=None; no initialization evaluation',
        'tie_band_inclusive': .001,
        'tie_order': ['lower passes', 'lower server_lr', 'lexicographic cell'],
        'stage2': 'FP and PF on mechanically chosen passes2/4 candidate per line; keep worse-thanP1 outcome',
        'MAIN_allowed': False, 'TEST_allowed': False,
        'schedule_disclosure': 'matched6EFE changes communication rounds, coverage, dropout cohorts and local trajectories; whole-recipe contrast',
        'CI_scope': 'one fixed FLscreen seed and pretrain2026; descriptive, repeated CALIB exposure, no privacy claim'}


def source_files():
    return sorted(p for p in CODE.rglob('*.py') if 'experiment_extensions' not in p.parts) + [
        CODE / 'experiment_extensions/__init__.py', CODE / 'experiment_extensions/localpasses.py',
        CODE / 'experiment_extensions/test_localpasses.py']


def numerical_binding():
    files = {str(physical(p).relative_to(CODE)): sha(p) for p in source_files()}
    return {'files': files, 'sha256': canonical(files)}


def data_files():
    reg = read(WORK / 'contracts/manifest_registry/REGISTRY_V2.json')
    paths = set()
    for mid in (TRAIN, EVAL):
        entry = reg['manifests'][mid]
        paths.update((WORK / entry['path'], WORK / entry['marker']))
        paths.update(WORK / table['file'] for table in entry['tables'].values())
    paths.add(WORK / 'data_v2/TRANSFORM_STATE_CLEAN_V2.json')
    # SHA-only feature reads span the store; only TRAIN/CALIB validation providers
    # parse selected rows. This is not an observation of a TEST rank or metric.
    paths.update(DATA.glob('model_ready/user_bucket=*/features.parquet'))
    for split in ('TRAIN', 'VALIDATION_SELECT', 'VALIDATION_CONFIRM'):
        paths.update((DATA / 'decisions_index' / f'split={split}').glob('user_bucket=*/*.parquet'))
    paths.update((DATA / 'mappings').glob('*.parquet'))
    paths.update((DATA / 'mappings').glob('*.json'))
    for name in ('VERSION.json', 'source_manifest.json', 'TRAINING_READINESS.json'):
        paths.add(DATA / name)
    return sorted(paths)


def prior_directories():
    p, c = DONOR / 'runs_x_rees46/PB100', DONOR / 'runs_x_rees46/COLD'
    return [p / f'{stage}/eta{eta}_c0.1_ilr0.1_tau1e-6_E6_P{passes}'
            for stage, passes in [('S1', 1), ('S2', 2)] for eta in ('3e-05', '0.0001')] + [
            c / f'G/FA_eta{eta}_tau1e-6_E6' for eta in ('0.0003', '0.001')]


def verify_registry(reg):
    if reg['roles']['CALIBRATION'] != {'train': TRAIN, 'eval': EVAL}:
        raise Refused('CALIB role pair changed')
    t, e = [reg['manifests'][mid] for mid in (TRAIN, EVAL)]
    if (t['n_users'], t['n_decisions'], e['n_decisions']) != (32768, 646067, 158135):
        raise Refused('CALIB population changed')


def build_protocol(out):
    scoped(out)
    if physical(out).exists():
        raise Refused('protocol output exists')
    verify_registry(read(WORK / 'contracts/manifest_registry/REGISTRY_V2.json'))
    refs = []
    for directory in prior_directories():
        side = read(directory / 'ARM_SPEC.json')
        spec, cfg = side['arm_spec'], side['arm_spec']['cfg']
        if spec['role'] != 'CALIBRATION' or cfg['seed'] != SEED or cfg['method'] != 'FA' or cfg['dp'] or cfg['pf']:
            raise Refused('historical reference is not the own nonDP CALIB FA recipe')
        if cfg['endpoint_efe'] != 6 or cfg['endpoint_rounds'] != rounds(cfg['solver']['passes']):
            raise Refused('historical CALIB budget differs')
        files = [directory / name for name in ('ARM_SPEC.json', 'STATUS.json', 'RUN_SUMMARY.json')]
        files += sorted((directory / 'evals').glob('eval_*.result.json'))
        files += sorted((directory / 'evals').glob('eval_*.ranks_int32.npy'))
        if len([p for p in files if p.name.endswith('.result.json')]) != 12:
            raise Refused('historical CALIB reference lacks twelve marks')
        summary = read(directory / 'RUN_SUMMARY.json')
        scores = {f'{u/2:.1f}': read(directory / 'evals' / f'eval_{u/2:.1f}.result.json')['E2E']['macro']['mrr@20']
                  for u in range(1, 13)}
        pb = retained_pb(summary, scores, n_rounds=rounds(cfg['solver']['passes']))
        checkpoint = verify_retained_checkpoint(directory, side, summary, pb,
            read(directory / 'evals' / f'eval_{pb["mark"]:.1f}.result.json'))
        files.append(Path(checkpoint['path']))
        refs.append({'directory': str(directory), 'line': 'pb100' if '/PB100/' in str(directory) else 'cold',
            'passes': cfg['solver']['passes'], 'server_lr': cfg['server_opt']['lr'],
            'run_id': side['run_id'], 'arm_spec_sha256': side['arm_spec_sha256'], 'files': [pin(p) for p in files],
            'wall_s_this_leg': summary['wall_s_this_leg'], 'retained_PB': pb,
            'retained_checkpoint': checkpoint})
    rec = read(INIT)
    checkpoint = INIT.parent / rec['fl_init']['path']
    if sha(checkpoint) != INIT_SHA:
        raise Refused('warm init is not PB100 seed2026 endpoint')
    protocol = {'format': FORMAT, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'declaration': fixed_declaration(), 'sources': [pin(p) for p in source_files()],
        'contracts': [pin(p) for p in sorted((WORK / 'contracts').rglob('*')) if p.is_file()],
        'data': [pin(p) for p in data_files()], 'warm_init': [pin(INIT), pin(checkpoint)],
        'historical_references': refs, 'numerical_code_sha256': numerical_binding()['sha256']}
    protocol['self_sha256'] = canonical(protocol)
    write_once(out, protocol)
    return protocol


def load_protocol(path, expected_sha=None):
    if expected_sha is not None and sha(path) != expected_sha:
        raise Refused('protocol SHA differs')
    p = read(path)
    if p.get('format') != FORMAT or p.get('self_sha256') != canonical(p) or p.get('declaration') != fixed_declaration():
        raise Refused('not the fixed CALIB-only protocol')
    if [row['path'] for row in p['sources']] != [str(physical(q).resolve()) for q in source_files()]:
        raise Refused('source inventory differs')
    if [row['path'] for row in p['data']] != [str(physical(q).resolve()) for q in data_files()]:
        raise Refused('data inventory differs')
    for group in ('sources', 'contracts', 'data', 'warm_init'):
        for row in p[group]:
            check_pin(row)
    for row in p['historical_references']:
        for item in row['files']:
            check_pin(item)
    if p['numerical_code_sha256'] != numerical_binding()['sha256']:
        raise Refused('current numerical binding differs')
    verify_registry(read(WORK / 'contracts/manifest_registry/REGISTRY_V2.json'))
    return p


def select_cell(name, *, role='CALIBRATION', method='FA', selection=None):
    if role != 'CALIBRATION' or method not in ('FA', 'FP', 'PF'):
        raise Refused('only declared FA/FP/PF CALIBRATION cells')
    if method == 'FA':
        hits = [row for row in stage1_cells() if row['cell'] == name]
        if len(hits) != 1:
            raise Refused('cell is not in the fixed six-cell missing grid')
        return dict(hits[0])
    if selection is None:
        raise Refused('FP/PF require the exact mechanical CALIB selection')
    hits = [row for row in selection['winners'].values() if row['cell'] == name]
    if len(hits) != 1:
        raise Refused('FP/PF require the exact mechanical CALIB selection')
    winner = hits[0]
    cell = {key: winner[key] for key in ('line', 'server_lr', 'passes', 'seed')}
    cell.update(method=method, stage='FP_PF_FOLLOW_UP', selected_FA_cell=name,
                cell=name.removesuffix('_HISTORICAL').replace('_FA_', '_' + method + '_'))
    return cell


def fixed_spec(cell, *, init_identity=None, peak_lr=PEAK):
    if peak_lr != PEAK or cell['seed'] != SEED or cell['passes'] not in (2, 4):
        raise Refused('fixed recipe/seed/passes differ')
    line, method, passes = cell['line'], cell['method'], cell['passes']
    if line not in ('cold', 'pb100') or method not in ('FA', 'FP', 'PF'):
        raise Refused('unknown declared line/method')
    expected = (3e-5, 1e-4) if line == 'pb100' else (3e-4, 1e-3)
    if cell['server_lr'] not in expected:
        raise Refused('server LR outside fixed per-line grid')
    if (line == 'pb100') != (init_identity is not None):
        raise Refused('PB100 needs its own endpoint; cold must have no init')
    if init_identity is not None and (init_identity.get('checkpoint_sha256'), init_identity.get('state_digest'),
        init_identity.get('seed')) != (INIT_SHA, INIT_DIGEST, SEED):
        raise Refused('warm init is not own PB100 seed2026 endpoint')
    solver = {'lr': .1, 'batch_size': 16, 'passes': passes, 'clip': 1., 'weight_decay': 0.,
              'fused': None, 'optimizer': 'sgd_m'}
    if line == 'pb100':
        solver['item_lr_mult'] = .1
    cfg = {'method': method, 'seed': SEED, 'lr_peak_local': .1, 'solver': solver, 'n_shards': 2,
        'max_attempts': 2, 'endpoint_efe': 6., 'exposure_point': 'end', 'mu': .01 if method == 'FP' else 0.,
        'pf': {'lr': PEAK, 'lam': 1e-4, 'clip': 1.} if method == 'PF' else None, 'dp': None,
        'dropout_p': .1, 'endpoint_rounds': rounds(passes), 'lr_shape': 'fl_const',
        'server_opt': {'name': 'fedadam', 'lr': cell['server_lr'], 'beta1': .9, 'beta2': .99,
                       'tau': 1e-6, 'v_init': 'tau_squared'}}
    spec = {'arm': f'EXT_LP_{line.upper()}_{method}', 'study': 'LOCAL_PASSES_CALIB_EXTENSION',
        'role': 'CALIBRATION', 'plan': {'group_size': 64, 'sampling': 'sweep'}, 'cfg': cfg,
        'max_rounds': rounds(passes), 'lr_mult': 1., 'lr_peak_base': PEAK, 'upload_codec': None,
        'server_opt': 'fedadam', 'server_lr': cell['server_lr'], 'server_tau': 1e-6,
        'client_opt': 'sgd_m', 'client_lr': .1, 'lr_shape': 'fl_const',
        'local_passes': passes, 'extension_cell': dict(cell),
        'exposure_budget': {'endpoint_efe': 6., 'n_clients': 32768, 'passes': passes,
            'group': 64, 'dropout_p': .1, 'rounds': rounds(passes), 'formula': 'ceil(60*N/(576*passes))'}}
    if init_identity is not None:
        spec['init_from'] = dict(init_identity)
    return spec


def paired_calib_harness(tr, provider, key_base):
    """Use the real CALIB manifest for PF, without modifying the MAIN harness."""
    if provider.manifest_id != EVAL or key_base.get('eval_manifest_id') != EVAL or key_base.get('train_manifest_id') != TRAIN:
        raise Refused('personal evaluator must use the own CALIB provider and keys')
    proxy = copy.copy(tr)
    proxy.evalp, proxy.key_base = provider, dict(key_base)
    old_labels = tr._row_labels
    proxy._row_labels = lambda: dict(old_labels(), manifest_role='CALIBRATION', run_role='CALIBRATION',
                                  train_manifest_id=TRAIN, eval_manifest_id=EVAL, comparison_status=LABEL)
    return proxy


@contextlib.contextmanager
def launcher_context(cell, binding):
    """Temporary own admissions/metadata; execute/client/server/evaluator unchanged."""
    global _ACTIVE
    if _ACTIVE:
        raise Refused('nested local-passes context')
    from v4int import launch_fl as F
    from v4int import launch_device as D
    if sha(F.__file__) != LAUNCH_SHA:
        raise Refused('inherited launch_fl changed')
    arm = f'EXT_LP_{cell["line"].upper()}_{cell["method"]}'
    original_base, original_warm = F.base_arm, F.is_warm_arm
    original_resolve, original_spec_check = F.resolve_spec, F.check_warm_spec
    original_tuning, original_role, original_write = F.spec_tuning, F.role_evaluator, F.write_json
    original_rows = F.fl_rows
    original_require, original_init, original_pf = F.require_lr_selection, F.load_pretrain_init, D.pf_evaluator
    captured = {}

    def require(args):
        if args.arm != arm:
            return original_require(args)
        if args.role != 'CALIBRATION' or args.seed != SEED or args.smoke:
            raise Refused('own launcher requires non-smoke fixed CALIB identity')

    def resolve(args, peak_lr, fused, init_identity=None):
        if args.arm != arm:
            return original_resolve(args, peak_lr, fused, init_identity)
        require(args)
        return fixed_spec(cell, init_identity=init_identity, peak_lr=peak_lr), {'localpasses_binding': binding}

    def check(spec):
        if spec['arm'] != arm:
            return original_spec_check(spec)
        if spec != fixed_spec(cell, init_identity=spec.get('init_from')):
            raise Refused('runtime spec is not the exact own fixed recipe')

    def tuning(spec):
        if spec['arm'] != arm:
            return original_tuning(spec)
        check(spec)
        return {'optimizer': 'fedadam', 'server_lr': cell['server_lr'], 'server_tau': 1e-6,
            'lr_mult': 1., 'lr_shape': 'fl_const', 'client_opt': 'sgd_m', 'client_lr': .1,
            'local_passes': cell['passes'], 'local_batch': 16}

    def init(path, *, model_variant):
        if physical(path).resolve() != INIT.resolve() or model_variant != 'SASREC_D64_B2':
            raise Refused('only PB100 own endpoint init allowed')
        state, ident, launch = original_init(physical(path), model_variant=model_variant)
        fixed_spec(cell, init_identity=ident)
        return state, ident, launch

    def role(tr, cep, kb, over):
        if cep.manifest_id != EVAL or kb.get('eval_manifest_id') != EVAL:
            raise Refused('CALIB role evaluator resolved another population')
        captured.update(tr=tr, provider=cep, key_base=kb)
        return original_role(tr, cep, kb, over)

    def pf(tr, _main_keys):
        if captured.get('tr') is not tr:
            raise Refused('CALIB provider must be captured before PF evaluation')
        from v4base.identity import eval_client_key_hex_by_eval_user_key
        from v4int.verify import DataPaths
        provider = captured['provider']
        proxy = paired_calib_harness(tr, provider, captured['key_base'])
        keys = eval_client_key_hex_by_eval_user_key(provider, manifest_dir=DataPaths.default(WORK).manifest_dir)
        return original_pf(proxy, keys)

    def write(path, obj):
        if isinstance(obj, dict):
            obj = dict(obj, exploratory_label=LABEL, localpasses_binding_sha256=canonical(binding),
                       manifest_role='CALIBRATION', run_role='CALIBRATION')
            if obj.get('status') == 'DONE_MAIN':
                obj['status'] = 'DONE_CALIBRATION'
            if physical(path).name.startswith('LAUNCH_'):
                obj.update(label=LABEL, task='Local-passes CALIB extension', localpasses_binding=binding)
        return original_write(path, obj)

    def rows(records, **kw):
        if kw.get('arm') != arm:
            return original_rows(records, **kw)
        kw['lifecycle'] = 'FRESH_SGDM_PER_VISIT'
        return [dict(r, run_label=LABEL, comparison_status=LABEL, exploratory_label=LABEL,
                     row_status='LOCAL_PASSES_CALIB', localpasses_cell=cell['cell'],
                     localpasses_binding_sha256=canonical(binding),
                     pf_private_optimizer_lifecycle='PERSISTENT_ADAMW_PER_CLIENT' if cell['method'] == 'PF' else None)
                for r in original_rows(records, **kw)]

    patches = {'ALL_ARMS': F.ALL_ARMS + (arm,), 'METHOD_CODE': dict(F.METHOD_CODE, **{arm: cell['method']}),
        'base_arm': lambda a: cell['method'] if a == arm else original_base(a),
        'is_warm_arm': lambda a: cell['line'] == 'pb100' if a == arm else original_warm(a),
        'require_lr_selection': require, 'resolve_spec': resolve, 'check_warm_spec': check,
        'spec_tuning': tuning, 'load_pretrain_init': init, 'role_evaluator': role, 'write_json': write,
        'fl_rows': rows,
        'fl_code_sha256': lambda: binding['numerical_code_sha256']}
    saved = {key: getattr(F, key) for key in patches}
    _ACTIVE = True
    try:
        for key, value in patches.items():
            setattr(F, key, value)
        D.pf_evaluator = pf
        yield F
    finally:
        for key, value in saved.items():
            setattr(F, key, value)
        D.pf_evaluator = original_pf
        _ACTIVE = False


def review_gate(path, protocol_path, numerical):
    r = read(path)
    if r.get('status') != 'PASS' or r.get('protocol_sha256') != sha(protocol_path) or \
            r.get('module_sha256') != sha(__file__) or r.get('tests_sha256') != sha(CODE / 'experiment_extensions/test_localpasses.py') or \
            r.get('numerical_code_sha256') != numerical['sha256'] or r.get('reviewer') in (None, '', '/root/ft_finalize'):
        raise Refused('independent review does not bind exact local-passes bytes')
    return pin(path)


def output_guard(out, binding, *, resume=False, root=EXT):
    p = scoped(out, root)
    marker = p / 'LOCALPASSES_BINDING.json'
    if p.exists():
        if not resume or not marker.is_file() or read(marker) != binding:
            raise Refused('existing output is not an exact own partial resume')
        if (p / 'LOCALPASSES_COMPLETE.json').exists() or \
                ((p / 'STATUS.json').exists() and read(p / 'STATUS.json').get('status') in ('DONE_MAIN', 'DONE_CALIBRATION')):
            raise Refused('completed CALIB experiment never reruns')
    elif resume:
        raise Refused('resume requires existing own partial binding')
    else:
        p.mkdir(parents=True)
        with marker.open('x') as f:
            json.dump(binding, f, sort_keys=True, allow_nan=False)
    return p


def choose(rows):
    """Inclusive decimal tie band, followed by deterministic cost/config order."""
    if not rows or len({r['cell'] for r in rows}) != len(rows) or \
            any(not math.isfinite(r['value']) for r in rows):
        raise Refused('selection pool absent, duplicate or nonfinite')
    maximum = max(Decimal(str(r['value'])) for r in rows)
    tied = [r for r in rows if maximum - Decimal(str(r['value'])) <= Decimal('0.001')]
    return min(tied, key=lambda r: (r['passes'], r['server_lr'], r['cell']))


def check_result_identity(result, manifest):
    from v4eval import evaluator_code_sha256
    from v4eval.frozen_config import load
    from v4eval.device_pin import load_pin
    expected = load().eval_config('adapter_stream', 'NVIDIA_A40-FP32-DET')
    expected.update(eval_stream='E2E_ROWS', device_pin_sha256=load_pin()['self_sha256'])
    if result['manifest_id'] != EVAL or result['hashes'] != dict(manifest.hashes) or result['n_rows'] != manifest.n or \
            result['evaluator_code_sha256'] != evaluator_code_sha256() or result['eval_config'] != expected:
        raise Refused('CALIB result population or exact frozen A40 evaluator differs')


def retained_pb(summary, scores, *, n_rounds):
    """Reproduce unchanged FLRun._evaluate's chronological retained-best rule.

    The float stored by the evaluator, rather than a rounded or rederived rank
    statistic, controls the inherited strict best_delta boundary.
    """
    marks = [f'{u/2:.1f}' for u in range(1, 13)]
    evaluations = summary.get('evals')
    if set(scores) != set(marks) or not isinstance(evaluations, list) or len(evaluations) != 12:
        raise Refused('retained PB requires exactly twelve positive marks; no initialization mark0')
    best, best_unit, best_round, previous_round = None, None, None, -1
    improvements = []
    for unit, evaluation in enumerate(evaluations, 1):
        if not isinstance(evaluation, list) or len(evaluation) != 5:
            raise Refused('invalid chronological PB evaluation record')
        u, r, value, recorded_improved, phase = evaluation
        stored = scores[f'{unit/2:.1f}']
        if type(u) is not int or u != unit or type(r) is not int or not previous_round < r < n_rounds or \
                phase != 'MAIN' or type(recorded_improved) is not bool or \
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or \
                isinstance(stored, bool) or not isinstance(stored, (int, float)) or not math.isfinite(stored) or \
                float(value) != float(stored):
            raise Refused('chronological PB marks/rounds/phase or exact evaluator score differ')
        value = float(value)
        improved = best is None or value > best + BEST_DELTA
        if recorded_improved != improved:
            raise Refused('retained PB improvement flag differs from strict inherited best_delta')
        if improved:
            best, best_unit, best_round = value, unit, r + 1
        improvements.append(improved)
        previous_round = r
    if type(summary.get('best_half_efe')) is not int or summary['best_half_efe'] != best_unit or \
            summary.get('best_metric') != best:
        raise Refused('summary retained PB mark/metric differs from chronological best_delta')
    return {'mark': best_unit / 2, 'value': best, 'round': best_round,
            'best_delta': BEST_DELTA, 'improvements': improvements}


def verify_retained_checkpoint(directory, side, summary, pb, result):
    """Bind the retained PB score to the actual engine-retained weights."""
    from fl.checkpoint import load_checkpoint
    from fl.numerics import state_digest
    from v4int import launch_fl as F
    path = physical(directory) / 'ckpt' / side['run_id'] / 'best.pt'
    if not path.is_file():
        raise Refused('actual retained PB checkpoint is missing')
    payload = load_checkpoint(path)
    cfg = F.fl_run_config(side['arm_spec'], side['run_id'], 64)
    if cfg.best_delta != BEST_DELTA or payload.get('kind') != 'weights' or \
            payload.get('run_id') != side['run_id'] or payload.get('config_digest') != cfg.digest() or \
            payload.get('phase') != 'MAIN' or type(payload.get('round')) is not int or payload['round'] != pb['round'] or \
            payload.get('lr_table_digest') != summary.get('lr_table_digest') or \
            isinstance(payload.get('efe'), bool) or not isinstance(payload.get('efe'), (int, float)) or \
            not math.isfinite(payload['efe']) or payload['efe'] <= 0 or \
            state_digest(payload['theta']) != result.get('model_sha256'):
        raise Refused('actual retained PB checkpoint identity/state/round differs')
    return dict(pin(path), state_digest=result['model_sha256'], round=payload['round'],
                actual_EFE=float(payload['efe']), config_digest=payload['config_digest'])


def read_curve(directory, entry, *, expected_binding=None, manifest=None):
    """Full rank/client-macro rederivation on the frozen CALIB metric population."""
    import numpy as np
    from v4int.providers import StoreEvalProvider
    from v4int.verify import DataPaths
    from v4eval.metrics import credits_from_ranks
    from v4eval.bootstrap import user_stats, point
    if manifest is None:
        ep = StoreEvalProvider(EVAL, paths=DataPaths.default(WORK), history_policy='USER_HISTORY_L50',
            manifest_registry=WORK / 'contracts/manifest_registry/REGISTRY_V2.json', registry_root=WORK, prefetch=0)
        manifest = ep.manifest
    m = manifest
    if m.manifest_id != EVAL:
        raise Refused('curve analysis needs the real CALIB manifest')
    directory = physical(directory)
    side, summary = read(directory / 'ARM_SPEC.json'), read(directory / 'RUN_SUMMARY.json')
    s = side['arm_spec']
    if s['role'] != 'CALIBRATION' or s['cfg']['seed'] != SEED or s['cfg']['method'] != entry['method'] or \
            s['cfg']['solver']['passes'] != entry['passes'] or s['cfg']['server_opt']['lr'] != entry['server_lr'] or \
            summary.get('done') is not True or summary['rounds'] != rounds(entry['passes']):
        raise Refused('selection source is not the own completed CALIB cell')
    if expected_binding is not None:
        done = read(directory / 'LOCALPASSES_COMPLETE.json')
        if done.get('binding') != expected_binding or done.get('status') != 'DONE_CALIBRATION':
            raise Refused('own completion identity differs')
    curve, rank_curve, files, results = {}, {}, [], {}
    for unit in range(1, 13):
        rp = directory / 'evals' / f'eval_{unit/2:.1f}.result.json'
        npy = directory / 'evals' / f'eval_{unit/2:.1f}.ranks_int32.npy'
        r, ranks = read(rp), np.load(npy, allow_pickle=False)
        check_result_identity(r, m)
        if ranks.shape != (m.n,) or not np.issubdtype(ranks.dtype, np.integer):
            raise Refused('selection ranks/population differ from full CALIB manifest')
        if np.any(ranks[m.rankable] < 1) or np.any(ranks[m.rankable] > m.catalogue_K) or np.any(ranks[~m.rankable] >= 1):
            raise Refused('invalid rankable/censored CALIB rank')
        credit = credits_from_ranks(ranks, (20,))['mrr@20']
        st = user_stats(credit, m.eval_mask, m.rankable, m.eval_user_key, entry['cell'])
        value = point(st, 'macro_e2e')
        if not math.isclose(value, r['E2E']['macro']['mrr@20'], abs_tol=1e-12, rel_tol=0):
            raise Refused('client-macro CALIB rank reconstruction differs')
        mark = f'{unit/2:.1f}'
        curve[mark] = r['E2E']['macro']['mrr@20']
        rank_curve[mark], results[mark] = value, r
        files += [pin(rp), pin(npy)]
    pb = retained_pb(summary, curve, n_rounds=rounds(entry['passes']))
    checkpoint = verify_retained_checkpoint(directory, side, summary, pb, results[f'{pb["mark"]:.1f}'])
    files += [pin(directory / 'ARM_SPEC.json'), pin(directory / 'RUN_SUMMARY.json'), pin(checkpoint['path'])]
    return dict(entry, value=pb['value'], best_mark=pb['mark'], endpoint=curve['6.0'], curve=curve,
                rank_rederived_curve=rank_curve, retained_PB=pb, retained_checkpoint=checkpoint,
                directory=str(directory), population_hashes=dict(m.hashes), sources=files)


def derive_selection(protocol_path):
    p = load_protocol(protocol_path)
    from v4int.providers import StoreEvalProvider
    from v4int.verify import DataPaths
    ep = StoreEvalProvider(EVAL, paths=DataPaths.default(WORK), history_policy='USER_HISTORY_L50',
        manifest_registry=WORK / 'contracts/manifest_registry/REGISTRY_V2.json', registry_root=WORK, prefetch=0)
    results = []
    for cell in stage1_cells():
        directory = EXT / 'runs/LOCAL_PASSES' / cell['cell']
        b = read(directory / 'LOCALPASSES_BINDING.json')
        if b['protocol_sha256'] != sha(protocol_path) or b['numerical_code_sha256'] != p['numerical_code_sha256'] or b['cell'] != cell:
            raise Refused('stage1 output binding is not this exact protocol')
        results.append(read_curve(directory, cell, expected_binding=b, manifest=ep.manifest))
    references = []
    for ref in p['historical_references']:
        cell = {'cell': f"{ref['line']}_FA_eta{ref['server_lr']:g}_P{ref['passes']}_E6_s2026_HISTORICAL",
                'line': ref['line'], 'method': 'FA', 'server_lr': ref['server_lr'], 'passes': ref['passes'],
                'seed': SEED, 'stage': 'SHA_BOUND_HISTORICAL_CALIB'}
        references.append(read_curve(ref['directory'], cell, manifest=ep.manifest))
    results += [r for r in references if r['passes'] == 2]
    winners = {line: choose([r for r in results if r['line'] == line]) for line in ('cold', 'pb100')}
    # Existing PB100 P2 winner is represented by its own source identity. Later
    # FP/PF admission must derive its identical recipe, rather than retrain FA.
    result = {'format': 'PPSI_LOCAL_PASSES_CALIB_SELECTION_1', 'label': LABEL, 'role': 'CALIBRATION',
        'protocol_sha256': sha(protocol_path), 'numerical_code_sha256': p['numerical_code_sha256'],
        'winners': winners, 'pool': results, 'P1_references': [r for r in references if r['passes'] == 1],
        'rule': fixed_declaration()['tie_order'], 'tie_band_inclusive': .001,
        'MAIN_allowed': False, 'TEST_allowed': False}
    result['self_sha256'] = canonical(result)
    return result


def make_selection(protocol_path, out):
    result = derive_selection(protocol_path)
    write_once(out, result)
    return result


def prepare_data(protocol_path, out):
    """CPU-only real TRAIN/client and exact CALIB E2E metric-population proof."""
    import numpy as np
    from v4int import launch_fl as F
    from v4int.providers import StoreTrainProvider, StoreEvalProvider, comparison_key_base
    from v4int.verify import DataPaths
    from v4base.identity import eval_client_key_hex_by_eval_user_key
    from fl.participation import ParticipationPlan
    p = load_protocol(protocol_path)
    paths = DataPaths.default(WORK)
    kw = dict(paths=paths, history_policy='USER_HISTORY_L50',
        manifest_registry=WORK / 'contracts/manifest_registry/REGISTRY_V2.json', registry_root=WORK, log=print)
    tp = StoreTrainProvider(TRAIN, **kw)
    ep = StoreEvalProvider(EVAL, prefetch=0, **kw)
    keys, positions = F.client_index(tp, paths.manifest_dir)
    if len(keys) != 32768 or tp.n_decisions != 646067 or sum(len(v) for v in positions.values()) != 646067:
        raise Refused('real CALIB TRAIN clients/valid supports differ')
    m = ep.manifest
    external = eval_client_key_hex_by_eval_user_key(ep, manifest_dir=paths.manifest_dir)
    eval_users = np.unique(m.eval_user_key[m.eval_mask])
    rankable_users = np.unique(m.eval_user_key[m.eval_mask & m.rankable])
    eval_clients = {str(external[u]) for u in eval_users}
    missing = sorted(eval_clients - set(keys))
    if missing or m.n != 158135 or m.catalogue_K != 158486:
        raise Refused('CALIB evaluation users do not have own TRAIN support/catalogue')
    proof = {'status': 'PASS', 'label': LABEL, 'role': 'CALIBRATION', 'protocol_sha256': sha(protocol_path),
        'numerical_code_sha256': p['numerical_code_sha256'], 'TRAIN_manifest': TRAIN, 'validation_manifest': EVAL,
        'TRAIN_clients': len(keys), 'TRAIN_valid_decisions': tp.n_decisions, 'validation_rows': m.n,
        'E2E_decisions': int(m.eval_mask.sum()), 'E2E_users': len(eval_users), 'rankable_users': len(rankable_users),
        'OOV_only_users': len(eval_users) - len(rankable_users), 'missing_SUPPORT_users': len(missing),
        'validation_hashes': dict(m.hashes), 'client_order_sha256': canonical({'keys': keys}),
        'participation_plan_sha256': ParticipationPlan.build(keys, seed=SEED, manifest_hash=tp.fingerprint,
            group_size=64, sampling='sweep').digest(), 'TEST_read': False, 'GPU_initialized': False}
    write_once(out, proof)
    return proof


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-protocol', type=Path)
    p.add_argument('--protocol', type=Path)
    p.add_argument('--protocol-sha256')
    p.add_argument('--cell')
    p.add_argument('--method', choices=('FA', 'FP', 'PF'), default='FA')
    p.add_argument('--selection', type=Path)
    p.add_argument('--make-selection', type=Path)
    p.add_argument('--prepare-data', type=Path)
    p.add_argument('--role', default='CALIBRATION')
    p.add_argument('--gpu', type=int)
    p.add_argument('--out', type=Path)
    p.add_argument('--review', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        if a.role != 'CALIBRATION':
            raise Refused('CALIBRATION only; MAIN/TEST never admitted')
        if a.build_protocol:
            p = build_protocol(a.build_protocol)
            print(json.dumps({'status': 'PROTOCOL_BUILT_NOT_TRAINED', 'protocol_sha256': sha(a.build_protocol),
                'sources': len(p['sources']), 'data': len(p['data'])}))
            return 0
        if a.protocol is None:
            raise Refused('exact protocol required')
        if a.make_selection:
            make_selection(a.protocol, a.make_selection)
            print('CALIB MECHANICAL SELECTION COMPLETE')
            return 0
        if a.prepare_data:
            prepare_data(a.protocol, a.prepare_data)
            print('CALIB PROVIDER PREPARATION COMPLETE')
            return 0
        if a.gpu not in (0, 1) or a.out is None or a.cell is None:
            raise Refused('declared cell/physical A40/fresh output required')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.gpu):
            raise Refused('export CUDA_VISIBLE_DEVICES=<--gpu> before Python starts')
        p = load_protocol(a.protocol, a.protocol_sha256)
        selection = None
        if a.selection:
            selection = read(a.selection)
            if selection.get('format') != 'PPSI_LOCAL_PASSES_CALIB_SELECTION_1' or selection.get('self_sha256') != canonical(selection) or \
                    selection.get('protocol_sha256') != sha(a.protocol) or selection.get('numerical_code_sha256') != p['numerical_code_sha256']:
                raise Refused('not the own exact mechanical CALIB selection')
            if selection != derive_selection(a.protocol):
                raise Refused('selection does not mechanically rederive from all exact completed CALIB cells')
        cell = select_cell(a.cell, role=a.role, method=a.method, selection=selection)
        numerical = numerical_binding()
        review = None if a.dry_run else review_gate(a.review, a.protocol, numerical) if a.review else None
        if not a.dry_run and review is None:
            raise Refused('independent exact review required before real training')
        b = {'format': 'PPSI_LOCAL_PASSES_BINDING_1', 'label': LABEL, 'role': a.role, 'cell': cell,
            'protocol_sha256': sha(a.protocol), 'numerical_code_sha256': numerical['sha256'],
            'module_sha256': sha(__file__), 'review': review, 'selection': None if not a.selection else pin(a.selection)}
        out = physical(a.out)
        expected_out = EXT / 'runs/LOCAL_PASSES' / cell['cell']
        if out.resolve() != expected_out.resolve():
            raise Refused('output is not the exact own declared cell directory')
        with launcher_context(cell, b) as F:
            run_id = 'EXT_LP_' + cell['cell']
            av = ['--arm', f'EXT_LP_{cell["line"].upper()}_{cell["method"]}', '--seed', str(SEED),
                '--gpu', str(a.gpu), '--role', 'CALIBRATION', '--run-id', run_id, '--run-dir', str(out),
                '--history-policy', 'USER_HISTORY_L50', '--n-threads', '4', '--checkpoint-every', '64']
            if cell['line'] == 'pb100':
                av += ['--init-from', str(INIT)]
            if a.resume:
                av += ['--resume']
            args = F.parse_args(av)
            from v4int.launch_central import export_environment, gpu_checks
            env = export_environment(a.gpu)
            ident = None
            if cell['line'] == 'pb100':
                state, ident, _ = F.load_pretrain_init(INIT, model_variant='SASREC_D64_B2')
                del state
            spec = fixed_spec(cell, init_identity=ident)
            if a.dry_run:
                scoped(out)
                if out.exists():
                    raise Refused('dry-run requires absent own output')
                print(json.dumps({'status': 'PASS_DRY_RUN', 'binding': b, 'arm_spec': spec,
                    'fl_config_digest': F.fl_run_config(spec, run_id, 64).digest(), 'output_created': False}, indent=1))
                return 0
            from v4sup.gpu import default_runner
            gpu = gpu_checks(args, default_runner, os.environ)
            output_guard(out, b, resume=a.resume)
            rc = F.run(args, {'localpasses_binding': b, 'gpu_check': gpu}, env)
            if rc == 0 and read(out / 'STATUS.json')['status'] == 'DONE_CALIBRATION':
                write_once(out / 'LOCALPASSES_COMPLETE.json', {'status': 'DONE_CALIBRATION', 'binding': b})
            return rc
    except Refused as e:
        print(f'LOCAL PASSES REFUSED: {e}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
