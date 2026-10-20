"""Fixed-six-EFE FULL_PRE seed2027, using the inherited numerical pretrain path.

Only this process admits the new seed and disables post-six-EFE continuation
for the exact extension run ID. No TEST role or arbitrary launcher argument is
accepted. Protocol -> preparation -> independent exact-byte review -> training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import ppsi_paths

BASE = ppsi_paths.SERVER_BASE
EXT = ppsi_paths.EXTENSIONS_ROOT
WORK = EXT / 'ppsi_v4'
CODE = WORK / 'code'
DATA = BASE / 'shared_data/rees46_nextitem_v1'
FORMAT = 'PPSI_FULL_PRE_SECOND_SEED_PROTOCOL_1'
LABEL = 'POST_HOC_EXPLORATORY'
RUN_ID = 'PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_FULLPRE_EXT_s2027'
RUN_DIR = WORK / 'new_runs/full_pre_seed2027' / RUN_ID
TRAIN = 'PRETRAIN_TRAIN_5_100_NONMAIN'
EVAL = 'UNEW_VAL_VALIDATION_32768'
SEED = 2027
N = 24089824
N_USERS = 1217610
THETA0 = 'a5d1e6839c4d446bb2b4f09fd2a6ef340cf596e60b8a035c2ef683d58f5e34c0'
REG_REL = 'contracts/manifest_registry/REGISTRY_V2_PRETRAIN_EXT.json'
RECIPE_REL = 'contracts/hp_registry/SMALL_RECIPE_SASREC_D64_B2.json'


class Refused(RuntimeError):
    pass


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def physical(path):
    """Normalize the starter's two historical absolute-path aliases."""
    s = str(path)
    for old in ppsi_paths.LEGACY_ALIASES:
        if s == old or s.startswith(old + '/'):
            s = str(BASE) + s[len(old):]
            break
    return Path(s).resolve()


def inside(path, root=EXT):
    p, root = physical(path), physical(root)
    if p == root or root not in p.parents:
        raise Refused(f'path escapes extension output root: {path}')
    return p


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(obj):
    return hashlib.sha256(json.dumps({k: v for k, v in obj.items() if k != 'self_sha256'},
        sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def strict(path):
    def bad(value):
        raise Refused(f'non-finite JSON constant {value}')
    obj = json.loads(Path(path).read_text(), parse_constant=bad)
    if not isinstance(obj, dict):
        raise Refused('expected a JSON object')
    return obj


def write_once(path, obj):
    p = inside(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    return p


def fresh_output(path):
    p = inside(path)
    if p.exists():
        raise Refused(f'output already exists; automatic resume/overwrite refused: {p}')
    return p


def binding(path):
    p = physical(path)
    if BASE not in p.parents or not p.is_file():
        raise Refused(f'input is outside the EID copy or absent: {path}')
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha_file(p)}


def fixed_declaration():
    return {'label': LABEL, 'run_id': RUN_ID, 'run_dir': str(RUN_DIR), 'seed': SEED,
        'family': 'SASREC', 'variant': 'SASREC_D64_B2', 'recipe_id': 'SMALL_SASREC_D64_B2',
        'train_manifest': TRAIN, 'eval_manifest': EVAL, 'n_users': N_USERS,
        'n_train_decisions': N, 'ranks': [131072, 1348682], 'history_policy': 'USER_HISTORY_L50',
        'peak_lr': 0.0007114121116984968, 'dropout': [0.1], 'effective_batch': 256,
        'schedule': 'S0', 'precision': 'FP32_STRICT', 'endpoint_EFE': 6.0,
        'endpoint_updates': 6 * math.ceil(N / 256), 'endpoint_exposures': 6 * N,
        'init_checkpoint': 'sealed/checkpoints/endpoint_6.0.pt', 'theta0_digest': THETA0,
        'post_endpoint_continuation': False, 'checkpoint_selection_for_init': False,
        'test_scoring': False, 'resume': False,
        'budget_exception': 'Exact new run only: RunConfig.effective_auto_continuation=False; '
            'all first-six-EFE scheduling and numerical code unchanged; cfg.auto_continuation=None.'}


def source_paths():
    # Other independent extension modules are not imported by this experiment.
    return sorted(p for p in CODE.rglob('*.py') if 'experiment_extensions' not in p.parts) + [
        CODE / 'experiment_extensions/pretrain_seed.py',
        CODE / 'experiment_extensions/test_pretrain_seed.py']


def contract_paths():
    return sorted(physical(p) for p in (WORK / 'contracts').rglob('*') if p.is_file())


def data_paths():
    reg = strict(WORK / REG_REL)
    paths = set()
    for mid in (TRAIN, EVAL, 'UNEW_VAL_SUPPORT_32768'):
        ent = reg['manifests'][mid]
        paths.add(WORK / ent['path'])
        paths.add(WORK / ent['marker'])
        paths.update(WORK / row['file'] for row in ent['tables'].values())
    paths.add(WORK / 'data_v2/TRANSFORM_STATE_CLEAN_V2.json')
    # Model-ready features contain all time bands; SHA-only reads here are not
    # a TEST observation. Providers later filter TRAIN / VALIDATION exclusively.
    paths.update(DATA.glob('model_ready/user_bucket=*/features.parquet'))
    for split in ('TRAIN', 'VALIDATION_SELECT', 'VALIDATION_CONFIRM'):
        paths.update((DATA / 'decisions_index' / f'split={split}').glob('user_bucket=*/*.parquet'))
    paths.update((DATA / 'mappings').glob('*.parquet'))
    paths.update((DATA / 'mappings').glob('*.json'))
    for name in ('VERSION.json', 'source_manifest.json', 'TRAINING_READINESS.json'):
        paths.add(DATA / name)
    return sorted(paths)


def verify_registry_design(reg):
    if reg.get('roles', {}).get('PRETRAIN') != {'train': TRAIN, 'eval': EVAL}:
        raise Refused('FULL_PRE role is not the fixed TRAIN/UNEW_VALIDATION pair')
    ent = reg.get('manifests', {}).get(TRAIN, {})
    band = reg.get('population_bands', {}).get('PRETRAIN_NONMAIN', {})
    if ent.get('n_users') != N_USERS or ent.get('n_decisions') != N or \
            band.get('rank_range') != [131072, 1348682]:
        raise Refused('FULL_PRE population identity/counts changed or PB100 substituted')


def build_protocol(out):
    fresh_output(out)
    verify_registry_design(strict(WORK / REG_REL))
    sources = [binding(p) for p in source_paths()]
    contracts = [binding(p) for p in contract_paths()]
    inputs = [binding(p) for p in data_paths()]
    obj = {'format': FORMAT, 'utc': utc(), 'declaration': fixed_declaration(),
        'source_files': sources, 'contract_files': contracts, 'data_files': inputs,
        'root': str(WORK), 'dataset_root': str(DATA),
        'parent_code_immutable': True, 'real_training_requires_exact_independent_review': True}
    obj['self_sha256'] = canonical_sha(obj)
    write_once(out, obj)
    return {'protocol': str(physical(out)), 'protocol_sha256': sha_file(out),
        'source_files': len(sources), 'contract_files': len(contracts), 'data_files': len(inputs)}


def load_protocol(path, expected_sha):
    p = inside(path)
    if sha_file(p) != expected_sha:
        raise Refused('protocol file hash mismatch')
    obj = strict(p)
    if obj.get('format') != FORMAT or obj.get('self_sha256') != canonical_sha(obj):
        raise Refused('protocol format/self hash mismatch')
    if obj.get('declaration') != fixed_declaration() or obj.get('root') != str(WORK) or \
            obj.get('dataset_root') != str(DATA):
        raise Refused('protocol differs from fixed predeclared experiment')
    if [row['path'] for row in obj['source_files']] != [str(physical(p)) for p in source_paths()]:
        raise Refused('source inventory differs from declared numerical tree')
    if [row['path'] for row in obj['data_files']] != [str(physical(p)) for p in data_paths()]:
        raise Refused('data inventory differs from declared inputs')
    current_contracts = [str(p) for p in contract_paths()]
    if [row['path'] for row in obj['contract_files']] != current_contracts:
        raise Refused('contract inventory changed')
    for group in ('source_files', 'contract_files', 'data_files'):
        for row in obj[group]:
            p = physical(row['path'])
            if BASE not in p.parents or p.stat().st_size != row['bytes'] or sha_file(p) != row['sha256']:
                raise Refused(f'pinned input changed: {row["path"]}')
    verify_registry_design(strict(WORK / REG_REL))
    return obj


def load_review(path, expected_sha, protocol_sha):
    p = inside(path)
    if sha_file(p) != expected_sha:
        raise Refused('independent review file hash mismatch')
    review = strict(p)
    if review.get('status') != 'PASS' or review.get('protocol_sha256') != protocol_sha or \
            review.get('module_sha256') != sha_file(CODE / 'experiment_extensions/pretrain_seed.py') or \
            review.get('tests_sha256') != sha_file(CODE / 'experiment_extensions/test_pretrain_seed.py'):
        raise Refused('review does not approve this exact protocol/module/test content')
    if review.get('reviewer') in (None, '', 'next_pretrain', '/root/next_pretrain'):
        raise Refused('independent reviewer identity absent or is implementer')
    return review


def install_seed_and_budget(S, RC, *, run_id=RUN_ID):
    """No inherited file edits and no effects on other configurations."""
    S.REQUIRED = dict(S.REQUIRED, seed=SEED)
    original = RC.RunConfig.effective_auto_continuation
    if getattr(original.fget, '_fullpre_extension', False):
        raise Refused('extension budget hook already installed in this process')

    def continuation(cfg):
        if cfg.run_id == run_id:
            if cfg.seed != SEED or cfg.phase != 'MAIN' or cfg.population != 'COHORT' or \
                    cfg.recipe_id != 'SMALL_SASREC_D64_B2' or cfg.test_overrides or cfg.auto_continuation is not None:
                raise Refused('fixed-budget hook received an inconsistent extension config')
            return False
        return original.fget(cfg)

    continuation._fullpre_extension = True
    RC.RunConfig.effective_auto_continuation = property(continuation, doc=original.__doc__)
    original_fields = S.pretrain_row_fields

    def rows():
        return dict(original_fields(), comparison_status=LABEL, extension_study='FULL_PRE_SECOND_SEED',
            pretrain_seed=SEED, controlled_init_EFE=6.0, adaptive_continuation=False)

    S.pretrain_row_fields = rows


def launch_argv(gpu):
    if gpu not in (0, 1):
        raise Refused('GPU must be physical A40 index 0 or 1')
    return ['--family', 'SASREC', '--seed', str(SEED), '--phase', 'MAIN', '--recipe-id',
        'SMALL_SASREC_D64_B2', '--gpu', str(gpu), '--run-dir', str(RUN_DIR), '--run-id', RUN_ID,
        '--history-policy', 'USER_HISTORY_L50', '--n-threads', '8', '--log-every', '500']


def install(gpu):
    # Must precede importing v4train/ppsi_env/torch. The original launcher repeats
    # and verifies this same environment export before constructing providers.
    os.environ['V4_DATASET_ROOT'] = str(DATA)
    from v4int import launch_central as LC
    LC.export_environment(gpu)
    from v4int import pretrain_small as S
    from v4train import recipe as RC
    install_seed_and_budget(S, RC)
    return S


def rebuild_theta0(cfg):
    import torch
    import v4models
    from fl.numerics import enforce_strict_fp32, state_digest
    from v4int import launch_central as LC
    enforce_strict_fp32()
    prod = strict(WORK / 'contracts/DATA_PROTOCOL_LOCK_V2.json')['features_and_transforms']['production_schema']
    schema = LC.load_production_schema(prod['schema_id'], prod['self_sha256'],
                                      physical(cfg['schema_registry']))
    before_rng = torch.get_rng_state().clone()
    built = v4models.build('SASREC', schema, SEED, 'cpu', v1_root=DATA,
        variant='SASREC_D64_B2', dropout=tuple(cfg['dropout']))
    got = state_digest(built.adapter.broadcast_state())
    if built.seed != SEED or built.K != 158486 or built.variant != 'SASREC_D64_B2' or \
            built.init_sha256 != THETA0 or got != THETA0 or built.schema_self_sha256 != prod['self_sha256'] or \
            built.recentered is not True:
        raise Refused('CPU rebuilt theta0 differs from exact registered seed2027/schema/catalogue identity')
    if not torch.equal(before_rng, torch.get_rng_state()):
        raise Refused('CPU theta0 preparation unexpectedly changed global Torch RNG')
    return {'status': 'PASS', 'seed': SEED, 'device': 'cpu', 'torch': torch.__version__,
        'registered_theta0_digest': THETA0, 'rebuilt_init_digest': built.init_sha256,
        'rebuilt_state_digest': got, 'raw_init_digest': built.raw_init_sha256,
        'parameter_manifest_digest': built.manifest_digest, 'catalogue_K': built.K,
        'schema_id': schema.schema_id, 'schema_self_sha256': schema.self_sha256,
        'schema_file': binding(physical(schema.path)), 'global_torch_rng_unchanged': True,
        'source': binding(CODE / 'v4models/build.py')}


def prepare(protocol_path, protocol_sha, out, gpu, *, no_providers=False):
    load_protocol(protocol_path, protocol_sha)
    fresh_output(RUN_DIR)
    fresh_output(out)
    S = install(gpu)
    raw = inside(str(out) + '.raw.json')
    fresh_output(raw)
    argv = ['--ext-registry', str(WORK / REG_REL), '--dry-run-out', str(raw)]
    if no_providers:
        argv.append('--no-providers')
    rc = S.main(argv + ['--'] + launch_argv(gpu))
    if rc != 0:
        raise Refused(f'inherited preparation refused with exit {rc}')
    result = strict(raw)
    if no_providers:
        if result['verified'][TRAIN]['n_decisions'] != N:
            raise Refused('manifest-only preparation resolved wrong TRAIN count')
    else:
        cfg = result['config']
        if result.get('effective_auto_continuation') is not False or cfg['seed'] != SEED or \
                cfg['theta0_init_sha256'] != THETA0 or cfg['key']['train_manifest_id'] != TRAIN or \
                cfg['key']['eval_manifest_id'] != EVAL or result['train_provider']['n_decisions'] != N:
            raise Refused('prepared run does not match fixed seed/data/theta0/six-EFE budget')
        from v4train.schedule import planned_lr_table
        table = planned_lr_table(cfg['peak_lr'], N, 256, 'MAIN', 'end', schedule='S0')
        if table['n_updates'] != 6 * math.ceil(N / 256) or table['final_exposures'] != 6 * N:
            raise Refused('original schedule does not reach the declared fixed six-EFE endpoint')
        result['extension_fixed6_schedule'] = table
        result['extension_actual_cpu_theta0'] = rebuild_theta0(cfg)
    proof = {'status': 'PASS', 'protocol_sha256': protocol_sha, 'utc': utc(),
        'preparation_kind': 'MANIFEST_ONLY' if no_providers else 'FULL_PROVIDERS_AND_REGISTERED_THETA0',
        'raw_preparation': binding(raw), 'run_directory_created': RUN_DIR.exists(),
        'fixed_declaration': fixed_declaration(), 'training_started': False}
    if not no_providers:
        proof['fixed6_planned_lr_table'] = result['extension_fixed6_schedule']
        proof['actual_cpu_theta0'] = result['extension_actual_cpu_theta0']
    if proof['run_directory_created']:
        raise Refused('preparation unexpectedly created run output directory')
    write_once(out, proof)
    return proof


def verify_endpoint(run_dir, *, require_real=True):
    """Check numerical checkpoint identity before releasing it as a warm init."""
    import torch
    from fl.numerics import state_digest
    p = Path(run_dir)
    rec = strict(p / 'PRETRAIN_FOR_FL_INIT.json')
    ck = p / 'sealed/checkpoints/endpoint_6.0.pt'
    if rec['run_id'] != RUN_ID or rec['train_manifest'] != TRAIN or rec['fl_init']['kind'] != 'endpoint_6.0' or \
            rec['fl_init']['path'] != 'sealed/checkpoints/endpoint_6.0.pt' or \
            rec['fl_init']['sha256'] != sha_file(ck) or rec['fl_init']['bytes'] != ck.stat().st_size:
        raise Refused('init record does not describe this exact fixed endpoint')
    obj = torch.load(ck, map_location='cpu', weights_only=True)
    meta = obj['meta']
    if obj['kind'] != 'endpoint_6.0' or meta['run_id'] != RUN_ID or meta['seed'] != SEED or \
            meta['efe'] != 6.0 or meta['model_variant'] != 'SASREC_D64_B2' or \
            meta['run_label'] != 'PRETRAIN_FOR_FL' or meta['state_digest'] != state_digest(obj['model']):
        raise Refused('endpoint checkpoint seed/run/EFE/variant/state identity mismatch')
    if require_real:
        if meta['updates'] != 6 * math.ceil(N / 256) or meta['exposures'] != 6 * N:
            raise Refused('endpoint checkpoint has wrong training steps/exposures')
        start, summ = strict(p / 'RUN_START.json'), strict(p / 'RUN_SUMMARY.json')
        if start['N'] != N or start['config']['seed'] != SEED or \
                start['theta0_init_sha256'] != THETA0 or summ['efe'] != 6.0 or \
                summ['phase'] != 'MAIN' or summ.get('continuation') is not None or summ['status'] != 'DONE_MAIN':
            raise Refused('real run config/summary mismatches the fixed six-EFE protocol')
    from v4int.launch_fl import load_pretrain_init
    _, identity, launch = load_pretrain_init(p / 'PRETRAIN_FOR_FL_INIT.json', model_variant='SASREC_D64_B2')
    if identity['seed'] != SEED:
        raise Refused('warm-init loader did not preserve actual pretraining seed')
    return {'identity': identity, 'init_record_sha256': launch['init_record_sha256'],
        'checkpoint_sha256': sha_file(ck), 'endpoint_updates': meta.get('updates'),
        'endpoint_exposures': meta.get('exposures'), 'endpoint_EFE': meta['efe']}


def train(protocol_path, protocol_sha, review_path, review_sha, preparation_path,
          preparation_sha, gpu):
    protocol = load_protocol(protocol_path, protocol_sha)
    review = load_review(review_path, review_sha, protocol_sha)
    pp = inside(preparation_path)
    if sha_file(pp) != preparation_sha:
        raise Refused('full provider preparation receipt hash mismatch')
    prep = strict(pp)
    if prep.get('status') != 'PASS' or prep.get('protocol_sha256') != protocol_sha or \
            prep.get('preparation_kind') != 'FULL_PROVIDERS_AND_REGISTERED_THETA0' or \
            prep.get('fixed_declaration') != fixed_declaration() or \
            prep.get('actual_cpu_theta0', {}).get('rebuilt_state_digest') != THETA0:
        raise Refused('exact full provider/theta0 preparation is required before training')
    if prep['raw_preparation'] != binding(prep['raw_preparation']['path']):
        raise Refused('raw provider preparation changed')
    fresh_output(RUN_DIR)
    if shutil.disk_usage(EXT).free < 20 * 2**30:
        raise Refused('less than 20 GiB free disk before pretraining')
    S = install(gpu)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    started = {'utc': utc(), 'label': LABEL, 'protocol_sha256': protocol_sha,
        'protocol_self_sha256': protocol['self_sha256'], 'review_path': str(physical(review_path)),
        'review_sha256': review_sha, 'reviewer': review['reviewer'],
        'preparation_sha256': preparation_sha, 'fixed_declaration': fixed_declaration(),
        'launch_argv': launch_argv(gpu), 'status': 'STARTED_WRITE_ONCE_NO_RESUME'}
    write_once(RUN_DIR / 'EXTENSION_TRAIN_STARTED.json', started)
    rc = S.main(['--ext-registry', str(WORK / REG_REL), '--'] + launch_argv(gpu))
    if rc != 0:
        raise Refused(f'inherited pretrainer exited {rc}; failed outputs preserved')
    endpoint = verify_endpoint(RUN_DIR)
    load_protocol(protocol_path, protocol_sha)
    result = {'status': 'COMPLETE_FIXED6_VALIDATION_DESCRIPTIVE', 'utc': utc(), 'label': LABEL,
        'protocol_sha256': protocol_sha, 'review_sha256': review_sha, 'endpoint': endpoint,
        'test_scored': False, 'run_directory': str(RUN_DIR)}
    write_once(RUN_DIR / 'EXTENSION_COMPLETE.json', result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build-protocol'); build.add_argument('--out', required=True, type=Path)
    prep = sub.add_parser('prepare')
    run = sub.add_parser('train')
    for parser in (prep, run):
        parser.add_argument('--protocol', required=True, type=Path)
        parser.add_argument('--protocol-sha256', required=True)
        parser.add_argument('--gpu', choices=(0, 1), type=int, default=0)
    prep.add_argument('--out', required=True, type=Path)
    prep.add_argument('--no-providers', action='store_true')
    run.add_argument('--review', required=True, type=Path); run.add_argument('--review-sha256', required=True)
    run.add_argument('--preparation', required=True, type=Path); run.add_argument('--preparation-sha256', required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == 'build-protocol':
            result = build_protocol(args.out)
        elif args.command == 'prepare':
            result = prepare(args.protocol, args.protocol_sha256, args.out, args.gpu,
                             no_providers=args.no_providers)
        else:
            result = train(args.protocol, args.protocol_sha256, args.review, args.review_sha256,
                           args.preparation, args.preparation_sha256, args.gpu)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (Refused, OSError, KeyError, ValueError) as exc:
        print(f'FULL_PRE_EXTENSION_REFUSED: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
