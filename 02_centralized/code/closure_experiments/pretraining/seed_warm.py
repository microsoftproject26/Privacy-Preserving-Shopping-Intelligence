"""Second FULL_PRE seed: exact transferred recipes and MAIN validation only.

The template cannot launch. A ready protocol requires the completed, real
seed2027 pretraining endpoint. Inherited modules and contracts are never edited;
small process-local admission/label contexts are restored on every exit.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import ppsi_paths

BASE = ppsi_paths.SERVER_BASE
EXT = ppsi_paths.EXTENSIONS_ROOT
WORK = EXT / 'ppsi_v4'
CODE = WORK / 'code'
DONOR = BASE / 'Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4'
DATA = BASE / 'shared_data/rees46_nextitem_v1'
LABEL = 'POST_HOC_EXPLORATORY'
FORMAT = 'PPSI_FULL_PRE_SEED_WARM_PROTOCOL_1'
PRE_RUN = 'PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_FULLPRE_EXT_s2027'
PRE_DIR = WORK / 'new_runs/full_pre_seed2027' / PRE_RUN
PRE_TRAIN = 'PRETRAIN_TRAIN_5_100_NONMAIN'
PRE_EVAL = 'UNEW_VAL_VALIDATION_32768'
PRE_N = 24089824
PRE_USERS = 1217610
PRE_THETA0 = 'a5d1e6839c4d446bb2b4f09fd2a6ef340cf596e60b8a035c2ef683d58f5e34c0'
PRE_PROTOCOL_SHA = '318b74df29c177e9ba6d79e7270d2ba0b978677ab60c52bdd22fb6314b8cf4e6'
PRE_REVIEW_SHA = '3615b0b5d733e477e1f519cf855ec89764993e280164fc9cc814e73e4031af41'
PRE_PREPARATION_SHA = 'db941bd382e395075d9877d470a1d5d4eaa81c080cf36e5da58161b33f7dadbe'
OLD_INIT_SHA = 'cf08588c11f019cc760f129ba7ba6b23806297660e0170f9938436a5cb88cf78'
OLD_INIT_STATE = 'ee33345be4f20b61a53cd80fa11e1af702031bfcd1425af490825cd0019dc54a'
MAIN_TRAIN = 'MAIN_TRAIN_5_100_131072'
MAIN_EVAL = 'MAIN_VALIDATION_5_100_131072'
VARIANT = 'SASREC_D64_B2'
RECIPE = 'SMALL_SASREC_D64_B2'
LINES = ('frozen', 'unfrozen')
SEEDS = (2026, 2027)
RULES = {'CB': 'CONTROLLED_BUDGET', 'PB': 'PRACTICAL_BEST'}
RULE_FILES = {'CB': 'endpoint_6.0.pt', 'PB': 'best.pt'}
FROZEN = ['item_embed.weight', 'output_embed', 'output_bias']
ROUNDS = 13654
_ACTIVE_TRAIN = False
_ACTIVE_REEVAL = False


class Refused(RuntimeError):
    pass


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def physical(path):
    s = str(path)
    for alias in ppsi_paths.LEGACY_ALIASES:
        if s == alias or s.startswith(alias + '/'):
            s = str(BASE) + s[len(alias):]
            break
    return Path(s).resolve()


def sha(path):
    h = hashlib.sha256()
    with physical(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canon(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
        allow_nan=False, ensure_ascii=True).encode()).hexdigest()


def read(path):
    def bad(value):
        raise Refused(f'nonfinite JSON {value}')
    obj = json.loads(physical(path).read_text(), parse_constant=bad)
    if not isinstance(obj, dict):
        raise Refused('expected JSON object')
    return obj


def output_path(path, *, fresh=False):
    # Reject symlink components as well as resolution outside the owned phase.
    raw = Path(str(path)).absolute()
    for alias in ppsi_paths.LEGACY_ALIASES:
        if str(raw).startswith(alias + '/'):
            raw = BASE / str(raw)[len(alias) + 1:]
            break
    p = physical(raw)
    if p == physical(EXT) or physical(EXT) not in p.parents:
        raise Refused('output escapes the new extension phase')
    if any(q.is_symlink() for q in (raw, *raw.parents) if q == EXT or EXT in q.parents):
        raise Refused('symlink output component refused')
    if fresh and p.exists():
        raise Refused('output exists; automatic overwrite/resume refused')
    return p


def write_once(path, obj):
    p = output_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def pin(path):
    p = physical(path)
    if physical(BASE) not in p.parents or not p.is_file():
        raise Refused('input is absent or outside the preserved physical EID copy')
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def check_pin(row):
    if pin(row['path']) != row:
        raise Refused(f'pinned input changed: {row["path"]}')


def pair(line, seed, *, role='MAIN', resume=False):
    if line not in LINES or seed not in SEEDS or role != 'MAIN' or resume:
        raise Refused('only the four fixed MAIN validation arms; no TEST/resume')
    return line, seed


def new_run_id(line, seed):
    pair(line, seed)
    return f'EXT_FULLPRE2027_{line.upper()}_FL_s{seed}'


def new_run_dir(line, seed):
    return WORK / 'new_runs/full_pre_seed_matrix' / new_run_id(line, seed)


def donor_dir(line, seed):
    pair(line, seed)
    if line == 'frozen':
        return DONOR / 'runs_ds3/FL_T' / f'FA_WARM_FROZEN_s{seed}'
    return DONOR / 'runs_x_rees46/MAIN' / f'X_FA_UNF_eta3e-06_tau1e-6_s{seed}'


def declaration():
    return {'label': LABEL, 'pretrain_seed': 2027, 'fl_seeds': list(SEEDS),
        'lines': list(LINES), 'role': 'MAIN', 'split': 'VALIDATION',
        'new_pretrain_run_id': PRE_RUN, 'new_pretrain_directory': str(PRE_DIR),
        'new_pretrain_protocol_sha256': PRE_PROTOCOL_SHA, 'new_pretrain_review_sha256': PRE_REVIEW_SHA,
        'new_init_rule': 'EXACT_OWN_FULL_PRE_ENDPOINT_6.0_NO_SELECTION',
        'main_train_manifest': MAIN_TRAIN, 'main_eval_manifest': MAIN_EVAL,
        'main_clients': 131072, 'main_train_decisions': 2587776,
        'catalogue_K': 158486, 'planned_endpoint_EFE': 6.0, 'endpoint_rounds': ROUNDS,
        'budget_basis': 'original round-count schedule normalized to6.0; report actual consumed EFE separately',
        'frozen': {'server_lr': .001, 'tau': .001, 'client_lr': .1,
            'items': 'frozen', 'item_lr_mult': None},
        'unfrozen': {'server_lr': 3e-6, 'tau': 1e-6, 'client_lr': .1,
            'items': 'unfrozen', 'item_lr_mult': .1},
        'local_passes': 1, 'local_batch': 16, 'clip': 1.0, 'dropout': .1,
        'weight_decay': 0.0, 'group': 64, 'sampling': 'sweep',
        'actual_warm_shared_optimizer_lifecycle': 'FRESH_SGDM_PER_VISIT',
        'lifecycle_metadata_rule': 'preserve inherited comparison-key fields; disclose actual optimizer in a separate non-key field',
        'history_policy': 'USER_HISTORY_L50', 'primary': 'CB', 'secondary': 'PB',
        'PB_rule': 'unchanged FLRun best_half_efe, first evaluated mark at least .5 EFE; fixed before new data',
        'selection_or_adaptive_budget': False, 'TEST_functionality': False,
        'CI': {'metric': 'E2E_CLIENT_MACRO_MRR_20', 'resamples': 1000,
            'bootstrap_seed': 20261014, 'pairing': 'same MAIN validation users across all eight arms',
            'seed_average': 'arithmetic mean over FL2026/2027, same user draw',
            'scope': 'conditional on two fixed PRE and two fixed FL seeds, descriptive, no multiplicity correction'},
        'historical_X_RNG_diagnostic': 'omitted; per-visit order/dropout RNG is derived and forked, tested independently',
        'numerics': 'unchanged inherited launch_fl.run/execute, no FedAdam-step diagnostic hook',
        'schedules': 'unchanged original full 13654-round schedule; new recipes never tune on MAIN validation',
        'guards': 'ready protocol + exact independent review + full provider/CPU-init preparation; exclusive outputs'}


def source_paths():
    # No other extension imports occur in this module.
    return sorted(p for p in CODE.rglob('*.py') if 'experiment_extensions' not in p.parts) + [
        CODE / 'experiment_extensions/seed_warm.py', CODE / 'experiment_extensions/test_seed_warm.py']


def contract_paths():
    return sorted(p for p in (WORK / 'contracts').rglob('*') if p.is_file())


def data_paths():
    paths = set()
    for rel, mids in [('contracts/manifest_registry/REGISTRY_V2.json', (MAIN_TRAIN, MAIN_EVAL)),
            ('contracts/manifest_registry/REGISTRY_V2_PRETRAIN_EXT.json', (PRE_TRAIN, PRE_EVAL))]:
        reg = read(WORK / rel)
        for mid in mids:
            ent = reg['manifests'][mid]
            paths.add(WORK / ent['path']); paths.add(WORK / ent['marker'])
            paths.update(WORK / row['file'] for row in ent['tables'].values())
    paths.add(WORK / 'data_v2/TRANSFORM_STATE_CLEAN_V2.json')
    # Integrity-only byte hashes of feature stores do not read TEST values.
    paths.update(DATA.glob('model_ready/user_bucket=*/features.parquet'))
    for split in ('TRAIN', 'VALIDATION_SELECT', 'VALIDATION_CONFIRM'):
        paths.update((DATA / 'decisions_index' / f'split={split}').glob('user_bucket=*/*.parquet'))
    paths.update((DATA / 'mappings').glob('*.parquet')); paths.update((DATA / 'mappings').glob('*.json'))
    for name in ('VERSION.json', 'source_manifest.json', 'TRAINING_READINESS.json'):
        paths.add(DATA / name)
    return sorted(paths)


@contextlib.contextmanager
def tau_context(line):
    """Only spec rederivation/admission changes; no training operation is wrapped."""
    from v4int import launch_fl as F
    original = F.p2_server_opt
    def opt(recipe):
        out = original(recipe)
        return dict(out, tau=1e-6) if line == 'unfrozen' and out and out['name'] == 'fedadam' else out
    try:
        F.p2_server_opt = opt
        yield F
    finally:
        F.p2_server_opt = original


def check_recipe(spec, line, seed, identity):
    """Exact recipe semantics, followed by inherited full spec reconstruction."""
    pair(line, seed)
    cfg = spec.get('cfg', {})
    so, sol = cfg.get('server_opt', {}), cfg.get('solver', {})
    expected_arm = 'FA_WARM_FROZEN' if line == 'frozen' else 'P2_W'
    common = {'arm': (spec.get('arm'), expected_arm), 'role': (spec.get('role'), 'MAIN'),
        'init': (spec.get('init_from'), identity), 'seed': (cfg.get('seed'), seed),
        'method': (cfg.get('method'), 'FA'), 'dp': (cfg.get('dp'), None), 'pf': (cfg.get('pf'), None),
        'mu': (cfg.get('mu'), 0.0), 'rounds': (cfg.get('endpoint_rounds'), ROUNDS),
        'max_rounds': (spec.get('max_rounds'), ROUNDS), 'EFE': (cfg.get('endpoint_efe'), 6.0),
        'plan': (spec.get('plan'), {'group_size': 64, 'sampling': 'sweep'}),
        'solver': (sol, dict(optimizer='sgd_m', lr=.1, batch_size=16, passes=1,
            weight_decay=0.0, clip=1.0, fused=None,
            **({'item_lr_mult': .1} if line == 'unfrozen' else {}))),
        'server': (so, {'name': 'fedadam', 'lr': .001 if line == 'frozen' else 3e-6,
            'beta1': .9, 'beta2': .99, 'tau': .001 if line == 'frozen' else 1e-6, 'v_init': 'tau_squared'}),
        'dropout': (cfg.get('dropout_p'), .1), 'lr_shape': (cfg.get('lr_shape'), 'fl_const'),
        'lr_peak_local': (cfg.get('lr_peak_local'), .1), 'n_shards': (cfg.get('n_shards'), 2),
        'max_attempts': (cfg.get('max_attempts'), 2), 'exposure_point': (cfg.get('exposure_point'), 'end'),
        'codec': (spec.get('upload_codec'), None),
        'frozen': (spec.get('frozen_item_tables'), FROZEN if line == 'frozen' else None)}
    bad = [k for k, (got, wanted) in common.items() if got != wanted]
    if bad:
        raise Refused(f'fixed recipe mismatch: {bad}')
    with tau_context(line) as F:
        F.check_warm_spec(spec) if line == 'frozen' else F.check_p2_spec(spec)


def check_donor(line, seed):
    """Read only the four historical MAIN validation donors, never TEST outputs."""
    import numpy as np
    from fl.checkpoint import load_checkpoint
    from fl.numerics import state_digest
    from v4int import launch_fl as F
    d = donor_dir(line, seed)
    side, launch, status, summ = [read(d / f) for f in ('ARM_SPEC.json', 'LAUNCH_0.json', 'STATUS.json', 'RUN_SUMMARY.json')]
    spec, rid = side['arm_spec'], side['run_id']
    if side['arm_spec_sha256'] != F.spec_digest(spec) or launch['arm_spec'] != spec or \
            launch['arm_spec_sha256'] != side['arm_spec_sha256'] or status.get('status') != 'DONE_MAIN' or \
            summ.get('status') != 'DONE_MAIN' or summ.get('rounds') != ROUNDS:
        raise Refused('historical donor launch/spec/completion identity mismatch')
    for obj in (launch, status, summ):
        if obj.get('run_id') != rid or obj.get('arm') != side['arm']:
            raise Refused('historical donor run/arm mismatch')
    identity = spec.get('init_from', {})
    if identity.get('seed') != 2026 or identity.get('checkpoint_sha256') != OLD_INIT_SHA or \
            identity.get('state_digest') != OLD_INIT_STATE or identity.get('train_manifest') != PRE_TRAIN:
        raise Refused('historical comparison is not FULL_PRE seed2026')
    check_recipe(spec, line, seed, identity)
    pins = [pin(d / f) for f in ('ARM_SPEC.json', 'LAUNCH_0.json', 'STATUS.json', 'RUN_SUMMARY.json',
                                'REGISTRY_V2_TRAINER_VIEW.json')]
    endpoints = {}
    for rule, filename in RULE_FILES.items():
        cp = d / 'ckpt' / rid / filename
        payload = load_checkpoint(cp)
        if payload.get('kind') != 'weights' or payload.get('run_id') != rid or \
                payload.get('config_digest') != launch['fl_config_digest'] or \
                payload.get('config_digest') != F.fl_run_config(spec, rid, 1).digest():
            raise Refused('historical checkpoint is not its own run/config')
        mark = 6.0 if rule == 'CB' else summ['best_half_efe'] / 2.0
        if payload.get('phase') != 'MAIN' or not math.isfinite(payload['efe']) or payload['efe'] <= 0 or \
                (rule == 'CB' and payload['round'] != ROUNDS) or mark < .5:
            raise Refused('historical checkpoint budget/positive best mark mismatch')
        result, rows_path = d / f'evals/eval_{mark:.1f}.result.json', d / f'evals/eval_{mark:.1f}.rows.json'
        ranks_path = d / f'evals/eval_{mark:.1f}.ranks_int32.npy'
        val, rows = read(result), read(rows_path)['rows']
        ranks = np.load(ranks_path, allow_pickle=False)
        state = state_digest(payload['theta'])
        if val['model_sha256'] != state or val['manifest_id'] != MAIN_EVAL or \
                val['catalogue_K'] != 158486 or ranks.dtype != np.int32 or ranks.ndim != 1 or \
                ranks.size != val['n_rows'] or not rows or any(r.get('split') != 'VALIDATION' or
                r.get('eval_manifest_id') != MAIN_EVAL or r.get('fl_run_id') != rid for r in rows):
            raise Refused('historical validation ranks/rows/state identity mismatch')
        pp = [pin(cp), pin(result), pin(rows_path), pin(ranks_path)]
        pins.extend(pp)
        endpoints[rule] = {'checkpoint': pp[0], 'result': pp[1], 'rows': pp[2], 'ranks': pp[3],
            'mark': mark, 'actual_EFE': payload['efe'], 'actual_round': payload['round'],
            'state_digest': state, 'n_rows': int(ranks.size), 'result_hashes': val['hashes']}
    return {'line': line, 'fl_seed': seed, 'pretrain_seed': 2026, 'directory': str(d),
        'run_id': rid, 'arm': side['arm'], 'arm_spec': spec, 'arm_spec_sha256': side['arm_spec_sha256'],
        'pins': pins, 'endpoints': endpoints}


def verify_new_init():
    """A real completed fixed-six-EFE own run is required; old init never substitutes."""
    import torch
    from fl.numerics import state_digest
    from v4int.launch_fl import load_pretrain_init
    d = physical(PRE_DIR)
    complete, started, start, summ = [read(d / f) for f in (
        'EXTENSION_COMPLETE.json', 'EXTENSION_TRAIN_STARTED.json', 'RUN_START.json', 'RUN_SUMMARY.json')]
    rec = read(d / 'PRETRAIN_FOR_FL_INIT.json')
    cp = d / 'sealed/checkpoints/endpoint_6.0.pt'
    checkpoint = pin(cp)
    if complete.get('status') != 'COMPLETE_FIXED6_VALIDATION_DESCRIPTIVE' or complete.get('label') != LABEL or \
            complete.get('test_scored') is not False or physical(complete.get('run_directory', '')) != d or \
            started.get('status') != 'STARTED_WRITE_ONCE_NO_RESUME' or started.get('label') != LABEL or \
            complete.get('protocol_sha256') != started.get('protocol_sha256') or \
            complete.get('review_sha256') != started.get('review_sha256') or \
            started.get('protocol_sha256') != PRE_PROTOCOL_SHA or started.get('review_sha256') != PRE_REVIEW_SHA or \
            started.get('preparation_sha256') != PRE_PREPARATION_SHA:
        raise Refused('real own PRE2027 completion/protocol/review sidecars required')
    dec = started.get('fixed_declaration', {})
    wanted = {'seed': 2027, 'run_id': PRE_RUN, 'n_train_decisions': PRE_N, 'n_users': PRE_USERS,
        'train_manifest': PRE_TRAIN, 'eval_manifest': PRE_EVAL, 'endpoint_EFE': 6.0,
        'endpoint_updates': 6 * math.ceil(PRE_N / 256), 'endpoint_exposures': 6 * PRE_N,
        'theta0_digest': PRE_THETA0, 'post_endpoint_continuation': False,
        'checkpoint_selection_for_init': False, 'test_scoring': False, 'resume': False}
    if any(dec.get(k) != v for k, v in wanted.items()):
        raise Refused('new PRE sidecar does not bind the fixed full population/seed/budget')
    cfg = start.get('config', {})
    if start.get('run_id') != PRE_RUN or cfg.get('run_id') != PRE_RUN or cfg.get('seed') != 2027 or \
            cfg.get('family') != 'SASREC' or cfg.get('model_variant') != VARIANT or cfg.get('recipe_id') != RECIPE or \
            cfg.get('phase') != 'MAIN' or cfg.get('test_overrides') or cfg.get('auto_continuation') is not None or \
            start.get('N') != PRE_N or start.get('theta0_init_sha256') != PRE_THETA0 or \
            summ.get('status') != 'DONE_MAIN' or summ.get('phase') != 'MAIN' or summ.get('efe') != 6.0 or \
            summ.get('continuation') is not None:
        raise Refused('new PRE RUN_START/RUN_SUMMARY does not describe the registered fixed run')
    if rec.get('run_id') != PRE_RUN or rec.get('train_manifest') != PRE_TRAIN or \
            rec.get('run_label') != 'PRETRAIN_FOR_FL' or rec.get('fl_init', {}).get('kind') != 'endpoint_6.0' or \
            rec.get('fl_init', {}).get('path') != 'sealed/checkpoints/endpoint_6.0.pt' or \
            rec['fl_init'].get('sha256') != checkpoint['sha256'] or rec['fl_init'].get('bytes') != checkpoint['bytes']:
        raise Refused('new PRE init record does not describe the exact own endpoint')
    obj = torch.load(cp, map_location='cpu', weights_only=True)
    meta = obj.get('meta', {})
    if obj.get('kind') != 'endpoint_6.0' or meta.get('run_id') != PRE_RUN or meta.get('seed') != 2027 or \
            meta.get('model_variant') != VARIANT or meta.get('run_label') != 'PRETRAIN_FOR_FL' or \
            meta.get('efe') != 6.0 or meta.get('mark') != 6.0 or \
            meta.get('updates') != 6 * math.ceil(PRE_N / 256) or meta.get('exposures') != 6 * PRE_N or \
            meta.get('state_digest') != state_digest(obj.get('model', {})):
        raise Refused('new PRE checkpoint seed/run/state/budget identity mismatch')
    state, ident, launch = load_pretrain_init(d / 'PRETRAIN_FOR_FL_INIT.json', model_variant=VARIANT)
    if ident.get('seed') != 2027 or ident.get('run_id') != PRE_RUN or ident['checkpoint_sha256'] != checkpoint['sha256'] or \
            ident['state_digest'] != state_digest(state) or ident['checkpoint_sha256'] == OLD_INIT_SHA:
        raise Refused('actual new FULL_PRE endpoint identity required, not historical seed2026')
    if complete.get('endpoint', {}).get('identity') != ident or \
            complete['endpoint'].get('checkpoint_sha256') != checkpoint['sha256'] or \
            complete['endpoint'].get('init_record_sha256') != launch['init_record_sha256']:
        raise Refused('new init identity differs from reviewed completion sidecar')
    files = ['EXTENSION_COMPLETE.json', 'EXTENSION_TRAIN_STARTED.json', 'RUN_START.json',
        'RUN_SUMMARY.json', 'PRETRAIN_FOR_FL_INIT.json', 'sealed/checkpoints/endpoint_6.0.pt']
    return {'identity': ident, 'init_record': pin(d / 'PRETRAIN_FOR_FL_INIT.json'),
        'checkpoint': checkpoint, 'pretrain_protocol_sha256': complete['protocol_sha256'],
        'pretrain_review_sha256': complete['review_sha256'], 'pins': [pin(d / f) for f in files]}


def transfer_spec(donor, new_identity):
    spec = copy.deepcopy(donor['arm_spec'])
    spec['init_from'] = dict(new_identity)
    check_recipe(spec, donor['line'], donor['fl_seed'], new_identity)
    # Prove that init alone changes the donor's complete numerical spec.
    if {k: v for k, v in spec.items() if k != 'init_from'} != \
            {k: v for k, v in donor['arm_spec'].items() if k != 'init_from'}:
        raise Refused('transfer changed a recipe/schedule beyond pretraining initialization')
    return spec


def build_protocol(out, *, template=False):
    output_path(out, fresh=True)
    donors = [check_donor(line, seed) for line in LINES for seed in SEEDS]
    obj = {'format': FORMAT, 'status': 'TEMPLATE_PENDING_REAL_PRE_ENDPOINT' if template else 'READY',
        'utc': utc(), 'declaration': declaration(), 'root': str(WORK), 'dataset_root': str(DATA),
        'source_files': [pin(p) for p in source_paths()], 'contract_files': [pin(p) for p in contract_paths()],
        'data_files': [pin(p) for p in data_paths()], 'historical_donors': donors,
        'new_init': None if template else verify_new_init()}
    obj['self_sha256'] = canon(obj)
    write_once(out, obj)
    return {'status': obj['status'], 'protocol': str(physical(out)), 'protocol_sha256': sha(out),
        'source_files': len(obj['source_files']), 'contract_files': len(obj['contract_files']),
        'data_files': len(obj['data_files']), 'historical_donors': len(donors)}


def load_protocol(path, expected_sha):
    p = output_path(path)
    if sha(p) != expected_sha:
        raise Refused('protocol file SHA mismatch')
    obj = read(p)
    if obj.get('format') != FORMAT or obj.get('status') != 'READY' or \
            obj.get('self_sha256') != canon({k: v for k, v in obj.items() if k != 'self_sha256'}) or \
            obj.get('declaration') != declaration() or obj.get('root') != str(WORK) or obj.get('dataset_root') != str(DATA):
        raise Refused('ready exact predeclared protocol required; template cannot launch')
    for group, paths in [('source_files', source_paths()), ('contract_files', contract_paths()), ('data_files', data_paths())]:
        if [r['path'] for r in obj[group]] != [str(physical(p)) for p in paths]:
            raise Refused(f'{group} inventory differs')
        for row in obj[group]:
            check_pin(row)
    if [(d['line'], d['fl_seed']) for d in obj['historical_donors']] != [(l, s) for l in LINES for s in SEEDS]:
        raise Refused('historical donor matrix changed')
    for donor in obj['historical_donors']:
        for row in donor['pins']:
            check_pin(row)
        check_recipe(donor['arm_spec'], donor['line'], donor['fl_seed'], donor['arm_spec']['init_from'])
    current = verify_new_init()
    if current != obj['new_init']:
        raise Refused('new pretraining init changed after protocol binding')
    return obj


def entry(protocol, line, seed):
    pair(line, seed)
    hits = [d for d in protocol['historical_donors'] if (d['line'], d['fl_seed']) == (line, seed)]
    if len(hits) != 1:
        raise Refused('donor recipe is not uniquely bound')
    return hits[0]


def review_gate(path, expected_sha, protocol_sha, *, preparation_sha=None):
    if sha(path) != expected_sha:
        raise Refused('review file SHA mismatch')
    review = read(path)
    if review.get('status') != 'PASS' or review.get('protocol_sha256') != protocol_sha or \
            review.get('module_sha256') != sha(CODE / 'experiment_extensions/seed_warm.py') or \
            review.get('tests_sha256') != sha(CODE / 'experiment_extensions/test_seed_warm.py') or \
            review.get('reviewer') in (None, '', 'next_pretrain', '/root/next_pretrain', 'pretrain_finalize', '/root/pretrain_finalize'):
        raise Refused('exact independent review required')
    if preparation_sha is not None and review.get('preparation_sha256') != preparation_sha:
        raise Refused('independent review must bind the full provider preparation')
    return review


def export(gpu):
    if gpu not in (0, 1):
        raise Refused('explicit physical GPU0 or1 required')
    os.environ['V4_DATASET_ROOT'] = str(DATA)
    from v4int.launch_central import export_environment
    return export_environment(gpu)


def require_early_cuda(gpu):
    """Numerical source verification imports Torch: require export before it."""
    if gpu not in (0, 1) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise Refused('export CUDA_VISIBLE_DEVICES=<physical GPU0 or1> before Python and before any input read')


def launch_args(F, line, seed, gpu, *, dry=False):
    pair(line, seed)
    argv = ['--arm', 'FA_WARM_FROZEN' if line == 'frozen' else 'P2_W', '--family', 'SASREC',
        '--seed', str(seed), '--gpu', str(gpu), '--role', 'MAIN', '--run-dir', str(new_run_dir(line, seed)),
        '--run-id', new_run_id(line, seed), '--init-from', str(PRE_DIR / 'PRETRAIN_FOR_FL_INIT.json'),
        '--history-policy', 'USER_HISTORY_L50', '--n-threads', '8', '--checkpoint-every', '64']
    if line == 'frozen':
        argv.append('--freeze-item-tables')
    if dry:
        argv.append('--dry-run')
    return F.parse_args(argv)


def assert_launch_args(F, args, line, seed, gpu, *, dry=False):
    wanted = launch_args(F, line, seed, gpu, dry=dry)
    if vars(args) != vars(wanted):
        raise Refused('runtime launcher arguments differ from the fixed extension')


@contextlib.contextmanager
def train_context(protocol, line, seed, gpu, binding, *, dry=False):
    global _ACTIVE_TRAIN
    if _ACTIVE_TRAIN:
        raise Refused('nested seed-warm training context')
    donor = entry(protocol, line, seed)
    identity = protocol['new_init']['identity']
    spec = transfer_spec(donor, identity)
    with tau_context(line) as F:
        saved = {k: getattr(F, k) for k in ('resolve_spec', 'p2_preflight', 'warm_preflight',
            'require_lr_selection', 'fl_rows', 'write_json', 'fl_code_sha256')}
        def admit(args):
            assert_launch_args(F, args, line, seed, gpu, dry=dry)
        def resolve(args, peak_lr, fused=None, init_identity=None):
            admit(args)
            if peak_lr != spec['lr_peak_base'] or init_identity != identity:
                raise Refused('runtime recipe or actual new init differs from the protocol')
            return copy.deepcopy(spec), {'seed_warm_transfer': binding,
                'historical_recipe_source': donor['pins'][0], 'historical_recipe_only': True}
        def rows(*args, **kwargs):
            out = saved['fl_rows'](*args, **kwargs)
            return [dict(r, comparison_status=LABEL, row_status='EXPLORATORY_ARM',
                study='FULL_PRE_SEED_MATRIX', pretrain_seed=2027, fl_seed=seed,
                actual_shared_optimizer_lifecycle='FRESH_SGDM_PER_VISIT',
                seed_warm_binding_sha256=canon(binding)) for r in out]
        def write(path, obj):
            if isinstance(obj, dict):
                obj = dict(obj, exploratory_label=LABEL, seed_warm_binding_sha256=canon(binding))
                if Path(path).name.startswith('LAUNCH_'):
                    obj.update(label=LABEL, task='FULL_PRE2027 fixed recipe transfer MAIN validation',
                        seed_warm_transfer=binding)
            return saved['write_json'](path, obj)
        patches = {'resolve_spec': resolve, 'p2_preflight': admit, 'warm_preflight': admit,
            'require_lr_selection': admit, 'fl_rows': rows, 'write_json': write,
            'fl_code_sha256': lambda: canon({'inherited': saved['fl_code_sha256'](),
                'extension_module': sha(__file__), 'binding': canon(binding)})}
        _ACTIVE_TRAIN = True
        try:
            for key, value in patches.items():
                setattr(F, key, value)
            yield F, spec
        finally:
            for key, value in saved.items():
                setattr(F, key, value)
            _ACTIVE_TRAIN = False


def prepare(protocol_path, protocol_sha, out, gpu):
    require_early_cuda(gpu)
    protocol = load_protocol(protocol_path, protocol_sha)
    output_path(out, fresh=True)
    env = export(gpu)
    import torch
    import ppsi_env
    from v4int import launch_fl as F
    from v4int.launch_central import bind_data_protocol_lock_v2, data_paths as default_paths, load_production_schema, _theta0_for
    from v4int.providers import StoreTrainProvider, StoreEvalProvider, comparison_key_base
    from v4models import build
    from fl.numerics import enforce_strict_fp32, state_digest
    from fl.participation import ParticipationPlan
    from fl.runtime import build_lr_table
    snap = ppsi_env.verify_snapshot(); ppsi_env.assert_isolated()
    torch.set_num_threads(8); enforce_strict_fp32()
    paths = default_paths(argparse.Namespace(manifest_dir=None, transform_state=None))
    lock = bind_data_protocol_lock_v2(paths, smoke=False)
    reg = read(lock['manifest_registry_path'])
    if reg['roles']['MAIN'] != {'train': MAIN_TRAIN, 'eval': MAIN_EVAL}:
        raise Refused('MAIN providers do not resolve the exact fixed manifests')
    kw = dict(history_policy='USER_HISTORY_L50', paths=paths,
        manifest_registry=lock['manifest_registry_path'], registry_root=WORK)
    tp, ep = StoreTrainProvider(MAIN_TRAIN, **kw), StoreEvalProvider(MAIN_EVAL, prefetch=0, **kw)
    keys, positions = F.client_index(tp, paths.manifest_dir)
    if len(keys) != 131072 or tp.n_decisions != 2587776:
        raise Refused('MAIN TRAIN provider count/client identity differs')
    recipe = F._small_recipe(argparse.Namespace(small_recipe=None, smoke=False))
    schema = load_production_schema(lock['schema_id'], lock['schema_self_sha256'], lock['schema_registry_path'])
    cpu = []
    for seed in SEEDS:
        before = torch.get_rng_state().clone()
        bm = build('SASREC', schema, seed, 'cpu', v1_root=DATA,
            variant=VARIANT, dropout=tuple(recipe['dropout']))
        theta = _theta0_for(argparse.Namespace(theta0_variants=None, smoke=False), lock['schema_id'],
            {'model_variant': VARIANT}, seed)
        if bm.init_sha256 != theta or state_digest(bm.adapter.broadcast_state()) != theta or \
                bm.K != 158486 or not torch.equal(before, torch.get_rng_state()):
            raise Refused('CPU rebuilt FL harness theta0/catalogue/RNG identity mismatch')
        F.apply_init_state(bm.adapter, load_init_state(protocol), protocol['new_init']['identity']['state_digest'])
        cpu.append({'seed': seed, 'theta0_digest': theta, 'loaded_new_init_digest': state_digest(bm.adapter.broadcast_state()),
            'K': bm.K, 'global_torch_rng_unchanged_before_init': True})
    records = []
    for line in LINES:
        for seed in SEEDS:
            output_path(new_run_dir(line, seed), fresh=True)
            binding = arm_binding(protocol, protocol_sha, line, seed)
            with train_context(protocol, line, seed, gpu, binding, dry=True) as (F, spec):
                args = launch_args(F, line, seed, gpu, dry=True)
                pre = F.preflight(args)
                actual, _ = F.resolve_spec(args, recipe['peak_lr'], fused=True,
                    init_identity=protocol['new_init']['identity'])
                if actual != spec:
                    raise Refused('prepared spec changed')
                cfg = F.fl_run_config(spec, new_run_id(line, seed), 64)
                plan = ParticipationPlan.build(keys, seed=seed, manifest_hash=tp.fingerprint,
                    group_size=64, sampling='sweep')
                lr = build_lr_table(plan, {k: int(v.size) for k, v in positions.items()}, tp.n_decisions, cfg)
                if lr['n_rounds'] != ROUNDS:
                    raise Refused('prepared full original LR schedule has wrong rounds')
                records.append({'line': line, 'fl_seed': seed, 'run_id': new_run_id(line, seed),
                    'spec_digest': F.spec_digest(spec), 'cfg_digest': cfg.digest(),
                    'plan_digest': plan.digest(), 'lr_table_digest': lr['digest'], 'n_rounds': lr['n_rounds'],
                    'preflight': pre, 'numerical_recipe_unchanged_except_init': True})
    proof = {'status': 'PASS', 'preparation_kind': 'FULL_MAIN_PROVIDERS_AND_CPU_REGISTERED_HARNESS_INIT',
        'utc': utc(), 'protocol_sha256': protocol_sha, 'declaration': declaration(), 'environment': env,
        'source_snapshot_verified_files': len(snap['files']), 'main_train': tp.describe(), 'main_eval': ep.describe(),
        'comparison_key': comparison_key_base(paths, tp, ep), 'registered_harness_init': cpu,
        'four_arm_preflight': records, 'training_started': False, 'TEST_read': False}
    write_once(out, proof)
    return {'status': 'PASS', 'proof': str(physical(out)), 'proof_sha256': sha(out), 'arms': len(records)}


def load_init_state(protocol):
    from v4int.launch_fl import load_pretrain_init
    state, identity, _ = load_pretrain_init(physical(protocol['new_init']['init_record']['path']), model_variant=VARIANT)
    if identity != protocol['new_init']['identity']:
        raise Refused('runtime init identity differs')
    return state


def arm_binding(protocol, protocol_sha, line, seed):
    donor = entry(protocol, line, seed)
    return {'format': 'PPSI_SEED_WARM_ARM_BINDING_1', 'label': LABEL, 'role': 'MAIN', 'split': 'VALIDATION',
        'line': line, 'fl_seed': seed, 'pretrain_seed': 2027, 'run_id': new_run_id(line, seed),
        'protocol_sha256': protocol_sha, 'new_init': protocol['new_init'],
        'historical_recipe': donor['pins'][0], 'historical_run_id': donor['run_id'],
        'numerical_spec': transfer_spec(donor, protocol['new_init']['identity']),
        'module_sha256': sha(CODE / 'experiment_extensions/seed_warm.py')}


def train(protocol_path, protocol_sha, review_path, review_sha, preparation_path, preparation_sha, line, seed, gpu):
    pair(line, seed)
    require_early_cuda(gpu)
    protocol = load_protocol(protocol_path, protocol_sha)
    review = review_gate(review_path, review_sha, protocol_sha, preparation_sha=preparation_sha)
    if sha(preparation_path) != preparation_sha:
        raise Refused('provider preparation SHA mismatch')
    prep = read(preparation_path)
    if prep.get('status') != 'PASS' or prep.get('protocol_sha256') != protocol_sha or \
            prep.get('preparation_kind') != 'FULL_MAIN_PROVIDERS_AND_CPU_REGISTERED_HARNESS_INIT' or \
            prep.get('declaration') != declaration() or prep.get('training_started') is not False or \
            prep.get('TEST_read') is not False:
        raise Refused('exact full MAIN provider/CPU-init preparation required')
    out = output_path(new_run_dir(line, seed), fresh=True)
    if shutil.disk_usage(EXT).free < 20 * 2**30:
        raise Refused('less than20GiB disk before new FL training')
    env = export(gpu)
    binding = arm_binding(protocol, protocol_sha, line, seed)
    with train_context(protocol, line, seed, gpu, binding) as (F, spec):
        args = launch_args(F, line, seed, gpu)
        pre = F.preflight(args)
        prepared = [r for r in prep.get('four_arm_preflight', []) if (r.get('line'), r.get('fl_seed')) == (line, seed)]
        if len(prepared) != 1 or prepared[0].get('spec_digest') != F.spec_digest(spec) or \
                prepared[0].get('cfg_digest') != F.fl_run_config(spec, new_run_id(line, seed), 64).digest() or \
                prepared[0].get('n_rounds') != ROUNDS:
            raise Refused('preparation does not bind this exact new arm/config/schedule')
        out.mkdir(parents=True, exist_ok=False)
        write_once(out / 'SEED_WARM_TRAIN_STARTED.json', {'status': 'STARTED_WRITE_ONCE_NO_RESUME',
            'utc': utc(), 'binding': binding, 'review_path': str(physical(review_path)),
            'review_sha256': review_sha, 'reviewer': review['reviewer'], 'preparation_sha256': preparation_sha,
            'prepared_arm': prepared[0]})
        rc = F.run(args, pre, env)
        if rc != 0 or read(out / 'STATUS.json').get('status') != 'DONE_MAIN':
            raise Refused('inherited new FL run incomplete; outputs retained without retry')
    # Completion is released only after all current source/data/init pins recheck.
    load_protocol(protocol_path, protocol_sha)
    for rule in RULES:
        own_fl_checkpoint(protocol, protocol_sha, line, seed, rule)
    result = {'status': 'COMPLETE_MAIN_VALIDATION_DESCRIPTIVE', 'utc': utc(), 'binding': binding,
        'review_sha256': review_sha, 'TEST_read': False, 'output_directory': str(out)}
    write_once(out / 'SEED_WARM_COMPLETE.json', result)
    return {'status': result['status'], 'line': line, 'fl_seed': seed, 'directory': str(out)}


def check_fl_payload(payload, launch, *, filename):
    from fl.numerics import state_digest
    from v4int import launch_fl as F
    if payload.get('kind') != 'weights' or payload.get('phase') != 'MAIN' or not isinstance(payload.get('theta'), dict) or \
            payload.get('run_id') != launch['run_id'] or payload.get('config_digest') != launch['fl_config_digest'] or \
            launch['fl_config_digest'] != F.fl_run_config(launch['arm_spec'], launch['run_id'], 64).digest():
        raise Refused('own FL checkpoint kind/run/config identity mismatch')
    efe = payload.get('efe')
    if not isinstance(efe, (int, float)) or not math.isfinite(efe) or efe <= 0 or \
            not isinstance(payload.get('round'), int) or not 1 <= payload['round'] <= ROUNDS:
        raise Refused('own FL checkpoint is not a positive best mark/fixed6 endpoint')
    if filename == 'endpoint_6.0.pt' and payload.get('round') != ROUNDS:
        raise Refused('own CB checkpoint does not have the exact fixed rounds')
    if 'p2_lora' in payload or 'store' in payload:
        raise Refused('unexpected LoRA/personal checkpoint component')
    return state_digest(payload['theta'])


def own_fl_checkpoint(protocol, protocol_sha, line, seed, rule):
    pair(line, seed)
    if rule not in RULES:
        raise Refused('CB primary/PB secondary only')
    from fl.checkpoint import load_checkpoint
    from v4int import launch_fl as F
    out, rid = new_run_dir(line, seed), new_run_id(line, seed)
    binding = arm_binding(protocol, protocol_sha, line, seed)
    started = read(out / 'SEED_WARM_TRAIN_STARTED.json')
    if started.get('binding') != binding or started.get('status') != 'STARTED_WRITE_ONCE_NO_RESUME':
        raise Refused('own new FL training sidecar/protocol/init identity mismatch')
    status, side, summ = [read(out / f) for f in ('STATUS.json', 'ARM_SPEC.json', 'RUN_SUMMARY.json')]
    if status.get('status') != 'DONE_MAIN' or summ.get('status') != 'DONE_MAIN' or summ.get('rounds') != ROUNDS or \
            side.get('run_id') != rid or side.get('arm_spec') != binding['numerical_spec'] or \
            side.get('arm_spec_sha256') != F.spec_digest(binding['numerical_spec']) or \
            summ.get('arm_spec') != binding['numerical_spec']:
        raise Refused('own new FL completion/spec/budget identity mismatch')
    launches = sorted(out.glob('LAUNCH_*.json'))
    if len(launches) != 1:
        raise Refused('fresh no-resume new FL must have exactly one launch')
    launch = read(launches[0])
    if launch.get('label') != LABEL or launch.get('run_id') != rid or launch.get('seed_warm_transfer') != binding or \
            launch.get('arm_spec') != binding['numerical_spec'] or \
            launch.get('arm_spec_sha256') != side['arm_spec_sha256'] or \
            launch.get('manifests') != {'train': MAIN_TRAIN, 'eval': MAIN_EVAL} or \
            launch.get('fl05', {}).get('role') != 'MAIN' or \
            launch.get('small_recipe', {}).get('model_variant') != VARIANT or \
            launch['small_recipe'].get('recipe_id') != RECIPE:
        raise Refused('own new FL launch/role/recipe/init provenance mismatch')
    cp = out / 'ckpt' / rid / RULE_FILES[rule]
    payload = load_checkpoint(cp)
    state = check_fl_payload(payload, launch, filename=cp.name)
    prepared = started.get('prepared_arm', {})
    if payload.get('plan_digest') != prepared.get('plan_digest') or \
            payload.get('lr_table_digest') != prepared.get('lr_table_digest') or \
            payload.get('lr_table_digest') != summ.get('lr_table_digest'):
        raise Refused('own checkpoint plan/LR schedule differs from full MAIN provider preparation')
    mark = 6.0 if rule == 'CB' else summ['best_half_efe'] / 2.0
    if mark < .5 or mark > 6:
        raise Refused('own best checkpoint is not an eligible normalized validation mark')
    selected = [e for e in summ.get('evals', []) if e[0] == int(mark * 2)]
    if len(selected) != 1 or selected[0][1] + 1 != payload['round'] or selected[0][4] != 'MAIN':
        raise Refused('own checkpoint round does not match its declared original normalized evaluation mark')
    result = read(out / f'evals/eval_{mark:.1f}.result.json')
    if result.get('model_sha256') != state or result.get('manifest_id') != MAIN_EVAL or result.get('catalogue_K') != 158486:
        raise Refused('own checkpoint does not match its own original MAIN validation state')
    if line == 'frozen':
        init = load_init_state(protocol)
        import torch
        if any(not torch.equal(payload['theta'][k], init[k]) for k in FROZEN):
            raise Refused('new frozen item table changed from actual PRE2027 init')
    return {'checkpoint': pin(cp), 'state_digest': state, 'mark': mark,
        'actual_EFE': payload['efe'], 'actual_round': payload['round'], 'launch': launch,
        'source_files': [pin(p) for p in (out / 'SEED_WARM_TRAIN_STARTED.json', out / 'ARM_SPEC.json',
            out / 'LAUNCH_0.json', out / 'RUN_SUMMARY.json', out / 'STATUS.json')],
        'result': pin(out / f'evals/eval_{mark:.1f}.result.json'),
        'ranks': pin(out / f'evals/eval_{mark:.1f}.ranks_int32.npy')}


def reeval_args(R, protocol, protocol_sha, target, line, seed, rule, gpu, out, *, dry=False):
    if target not in ('pre', 'warm') or rule not in RULES:
        raise Refused('own PRE or warm endpoints, CB/PB only')
    if target == 'pre':
        if line is not None or seed != 2027 or rule != 'CB':
            raise Refused('PRE2027 fixed6 endpoint is scored once on MAIN validation, CB label only')
        cp = protocol['new_init']['checkpoint']; kind = 'central'; label = 'PRE'
    else:
        pair(line, seed)
        cp = own_fl_checkpoint(protocol, protocol_sha, line, seed, rule)['checkpoint']
        kind = 'fl'; label = 'FA_WARM_FROZEN' if line == 'frozen' else 'EXT_WARM_UNF_PRE2027'
    av = ['--ckpt', cp['path'], '--ckpt-kind', kind, '--expect-sha256', cp['sha256'],
        '--variant', VARIANT, '--recipe-id', RECIPE, '--seed', str(seed), '--role', 'MAIN',
        '--arm-label', label, '--selection-rule', RULES[rule], '--checkpoint-id', f'{target}:{line}:s{seed}:{rule}',
        '--gpu', str(gpu), '--out-dir', str(out), '--history-policy', 'USER_HISTORY_L50']
    if dry:
        av.append('--dry-run')
    return R.parse_args(av)


@contextlib.contextmanager
def reeval_context(protocol, protocol_sha, target, line, seed, rule):
    """Own admissions; original MAIN source/data/harness/ranking/device path retained."""
    global _ACTIVE_REEVAL
    if _ACTIVE_REEVAL:
        raise Refused('nested seed-warm reeval context')
    from v4int import reeval as R
    saved = {k: getattr(R, k) for k in ('CENTRAL_SOURCES', 'source_central', 'source_fl',
        'check_warm_line', 'check_fl_payload', 'label_rows', 'write_outputs')}
    metadata = {'label': LABEL, 'study': 'FULL_PRE_SEED_MATRIX', 'pretrain_seed': 2027,
        'fl_seed': None if target == 'pre' else seed, 'line': line, 'protocol_sha256': protocol_sha,
        'own_new_init_identity': protocol['new_init']['identity'], 'role': 'MAIN', 'split': 'VALIDATION',
        'actual_shared_optimizer_lifecycle': 'PERSISTENT_ADAMW_CENTRAL' if target == 'pre' else 'FRESH_SGDM_PER_VISIT'}
    def source_central(cp, args):
        if target != 'pre' or physical(cp) != physical(protocol['new_init']['checkpoint']['path']) or \
                args.role != 'MAIN' or args.seed != 2027 or args.arm_label != 'PRE':
            raise Refused('only exact own PRE2027 MAIN source admitted')
        verify_new_init()
        return saved['source_central'](cp, args)
    def source_fl(cp, args):
        if target != 'warm' or args.role != 'MAIN' or args.seed != seed:
            raise Refused('only exact own warm MAIN source admitted')
        own = own_fl_checkpoint(protocol, protocol_sha, line, seed, rule)
        if physical(cp) != physical(own['checkpoint']['path']) or args.expect_sha256 != own['checkpoint']['sha256'] or \
                args.arm_label != ('FA_WARM_FROZEN' if line == 'frozen' else 'EXT_WARM_UNF_PRE2027'):
            raise Refused('warm scorer path/hash/label differs from own protocol')
        return {'kind': 'fl', 'run_dir': physical(new_run_dir(line, seed)), 'launch': own['launch'],
            'launch_files': {Path(p['path']).name: p['sha256'] for p in own['source_files'] if Path(p['path']).name.startswith('LAUNCH_')},
            'run_id': new_run_id(line, seed), 'arm': own['launch']['arm']}
    def warm(args, src):
        if target == 'pre':
            return saved['check_warm_line'](args, src)
        if line == 'frozen':
            return saved['check_warm_line'](args, src)
        if args.arm_label != 'EXT_WARM_UNF_PRE2027' or src['launch']['arm_spec']['init_from'] != protocol['new_init']['identity']:
            raise Refused('own unfrozen line/new init mismatch')
        check_recipe(src['launch']['arm_spec'], line, seed, protocol['new_init']['identity'])
        return {'variant': 'FULL_PRE2027_TAU_FIXED_UNFROZEN', 'own_protocol_sha256': protocol_sha}
    def payload(obj, src):
        saved['check_fl_payload'](obj, src)
        check_fl_payload(obj, src['launch'], filename=RULE_FILES[rule])
    def rows(rows, args, **kwargs):
        from v4eval.rows import COMPARISON_KEY
        if any(r.get('split') != 'VALIDATION' or r.get('eval_manifest_id') != MAIN_EVAL for r in rows):
            raise Refused('reeval rows do not belong to exact MAIN validation')
        row_metadata = {k: v for k, v in metadata.items() if k not in COMPARISON_KEY}
        kwargs['extra'] = dict(kwargs.get('extra', {}), comparison_status=LABEL,
            row_status='EXPLORATORY_REEVAL', **row_metadata)
        return saved['label_rows'](rows, args, **kwargs)
    def outputs(out, *, record, **kwargs):
        return saved['write_outputs'](out, record=dict(record, seed_warm_reeval=metadata), **kwargs)
    patches = {'source_central': source_central, 'source_fl': source_fl, 'check_warm_line': warm,
        'check_fl_payload': payload, 'label_rows': rows, 'write_outputs': outputs}
    if target == 'pre':
        patches['CENTRAL_SOURCES'] = dict(R.CENTRAL_SOURCES,
            PRE=dict(R.CENTRAL_SOURCES['PRE'], seed=2027, descriptive_seeds=()))
    _ACTIVE_REEVAL = True
    try:
        for k, v in patches.items():
            setattr(R, k, v)
        yield R
    finally:
        for k, v in saved.items():
            setattr(R, k, v)
        _ACTIVE_REEVAL = False


def reeval(protocol_path, protocol_sha, review_path, review_sha, target, line, seed, rule, gpu, out, *, dry=False):
    # Role is fixed before a checkpoint is read; CLI has no TEST/registry-ext/role override.
    if target == 'pre':
        if line is not None or seed != 2027 or rule != 'CB':
            raise Refused('PRE2027 fixed6 on MAIN validation only')
    else:
        pair(line, seed)
    require_early_cuda(gpu)
    protocol = load_protocol(protocol_path, protocol_sha)
    review_gate(review_path, review_sha, protocol_sha)
    out = output_path(out, fresh=True)
    export(gpu)
    with reeval_context(protocol, protocol_sha, target, line, seed, rule) as R:
        args = reeval_args(R, protocol, protocol_sha, target, line, seed, rule, gpu, out, dry=dry)
        result = R.run(args)
    load_protocol(protocol_path, protocol_sha)
    if not dry:
        write_once(out / 'SEED_WARM_REEVAL_COMPLETE.json', {'status': 'COMPLETE_MAIN_VALIDATION_DESCRIPTIVE',
            'utc': utc(), 'protocol_sha256': protocol_sha, 'review_sha256': review_sha,
            'target': target, 'line': line, 'fl_seed': seed, 'pretrain_seed': 2027, 'rule': rule, 'TEST_read': False})
    return {'status': 'PASS_DRY_RUN' if dry else 'COMPLETE_MAIN_VALIDATION_DESCRIPTIVE', 'target': target,
        'line': line, 'fl_seed': seed, 'rule': rule, 'output': str(out)}


def validation_stats(manifest, ranks, label):
    """Exact full E2E client denominator, retaining clients with no rankable rows."""
    import numpy as np
    from v4eval.bootstrap import user_stats
    from v4eval.metrics import credits_from_ranks
    ranks = np.asarray(ranks)
    if ranks.dtype != np.int32 or ranks.shape != (manifest.n,):
        raise Refused('validation ranks must be full-manifest int32')
    if np.any(ranks[manifest.rankable] < 1) or np.any(ranks[manifest.rankable] > manifest.catalogue_K) or \
            np.any(ranks[~manifest.rankable] >= 1):
        raise Refused('invalid full-catalogue rank or positive padding/OOV rank')
    return user_stats(credits_from_ranks(ranks, (20,))['mrr@20'],
        manifest.eval_mask, manifest.rankable, manifest.eval_user_key, label)


def analyze(protocol_path, protocol_sha, review_path, review_sha, out):
    """Fixed 2PRE x2FL x2recipe matrix, original CB primary/PB secondary only."""
    import numpy as np
    from v4eval import bootstrap as B
    from v4base.identity import eval_client_key_hex_by_eval_user_key
    from v4int.providers import StoreEvalProvider
    from v4int.verify import DataPaths
    protocol = load_protocol(protocol_path, protocol_sha)
    review_gate(review_path, review_sha, protocol_sha)
    output_path(out, fresh=True)
    paths = DataPaths.default(WORK)
    ep = StoreEvalProvider(MAIN_EVAL, paths=paths, history_policy='USER_HISTORY_L50',
        manifest_registry=WORK / 'contracts/manifest_registry/REGISTRY_V2.json', registry_root=WORK, prefetch=0)
    manifest = ep.manifest
    external = eval_client_key_hex_by_eval_user_key(ep, manifest_dir=paths.manifest_dir)
    stats, points, pins, evaluator = {}, {}, [], None
    for line in LINES:
        for seed in SEEDS:
            complete = read(new_run_dir(line, seed) / 'SEED_WARM_COMPLETE.json')
            if complete.get('status') != 'COMPLETE_MAIN_VALIDATION_DESCRIPTIVE' or \
                    complete.get('binding') != arm_binding(protocol, protocol_sha, line, seed) or complete.get('TEST_read') is not False:
                raise Refused('analysis requires all four exact completed own validation runs')
            donor = entry(protocol, line, seed)
            for rule in RULES:
                own = own_fl_checkpoint(protocol, protocol_sha, line, seed, rule)
                for preseed, rec in [(2026, donor['endpoints'][rule]), (2027, own)]:
                    for key in ('result', 'ranks'):
                        check_pin(rec[key]); pins.append(rec[key])
                    result = read(rec['result']['path'])
                    if result.get('manifest_id') != MAIN_EVAL or result.get('n_rows') != manifest.n or \
                            result.get('hashes') != dict(manifest.hashes) or result.get('catalogue_K') != manifest.catalogue_K or \
                            result.get('model_sha256') != rec['state_digest']:
                        raise Refused('matrix result does not bind the exact aligned validation manifest/state')
                    identity = {'evaluator_code_sha256': result['evaluator_code_sha256'], 'eval_config': result['eval_config']}
                    if evaluator is None:
                        evaluator = identity
                    elif identity != evaluator:
                        raise Refused('historical/new evaluator source or frozen runtime differs')
                    ranks = np.load(physical(rec['ranks']['path']), allow_pickle=False)
                    label = f'{line}_PRE{preseed}_FL{seed}_{rule}'
                    st = validation_stats(manifest, ranks, label)
                    value = B.point(st, 'macro_e2e')
                    if not math.isclose(value, result['E2E']['macro']['mrr@20'], rel_tol=0, abs_tol=1e-12) or \
                            len(st.user_keys) != 40764 or st.n_decisions != 519714 or np.count_nonzero(st.NR == 0) != 46:
                        raise Refused('matrix client-macro reconstruction/population differs from own scored result')
                    stats[(line, seed, preseed, rule)] = st
                    points[label] = {'MRR20': value, 'HR10': result['E2E']['macro']['hr@10'],
                        'NDCG10': result['E2E']['macro']['ndcg@10'], 'n_users': len(st.user_keys),
                        'n_decisions': st.n_decisions, 'checkpoint': rec['checkpoint'], 'ranks': rec['ranks'],
                        'normalized_mark': rec['mark'], 'actual_EFE': rec['actual_EFE'], 'actual_round': rec['actual_round']}
    first = next(iter(stats.values()))
    keys = [str(external[k]) for k in first.user_keys]
    if len(set(keys)) != len(keys) or any(not np.array_equal(s.user_keys, first.user_keys) for s in stats.values()):
        raise Refused('matrix users/external identities are not exactly paired')
    plan = B.make_plan(first.user_keys, seed=20261014, n_resamples=1000, client_key_hex=keys)
    contrasts, interaction = {}, {}
    for rule in RULES:
        differences = {}
        for line in LINES:
            old = {s: stats[(line, s, 2026, rule)] for s in SEEDS}
            new = {s: stats[(line, s, 2027, rule)] for s in SEEDS}
            d = B.seed_contrast(plan, new, old, 'macro_e2e', f'{line}_{rule}_PRE2027-PRE2026')
            rb = np.mean([B.resampled(plan, old[s], 'macro_e2e') for s in SEEDS], axis=0)
            rn = np.mean([B.resampled(plan, new[s], 'macro_e2e') for s in SEEDS], axis=0)
            if np.any(rb <= 0):
                raise Refused('paired retention has zero historical baseline draw')
            before = float(np.mean([B.point(old[s], 'macro_e2e') for s in SEEDS]))
            after = float(np.mean([B.point(new[s], 'macro_e2e') for s in SEEDS]))
            contrasts[f'{line}_{rule}'] = {'delta': d, 'seed_mean_old': before, 'seed_mean_new': after,
                'retention': {'point_percent': 100 * after / before, 'ci95_percent': B.ci95(100 * rn / rb)},
                'per_FL_seed_delta': {str(s): B.point(new[s], 'macro_e2e') - B.point(old[s], 'macro_e2e') for s in SEEDS}}
            differences[line] = (after - before, rn - rb)
        interaction[rule] = {'point': differences['unfrozen'][0] - differences['frozen'][0],
            'ci95': B.ci95(differences['unfrozen'][1] - differences['frozen'][1]),
            'meaning': '(unfrozen PRE2027-PRE2026) minus (frozen PRE2027-PRE2026), paired same users and FLseeds'}
    result = {'format': 'PPSI_FULL_PRE_SEED_MATRIX_VALIDATION_ANALYSIS_1', 'status': 'COMPLETE_DESCRIPTIVE',
        'utc': utc(), 'label': LABEL, 'role': 'MAIN', 'split': 'VALIDATION', 'TEST_read': False,
        'protocol_sha256': protocol_sha, 'review_sha256': review_sha, 'points': points, 'contrasts': contrasts,
        'recipe_interaction': interaction, 'primary': 'CB', 'secondary': 'PB', 'declaration': declaration(),
        'pairing': {'manifest_hashes': dict(manifest.hashes), 'user_set_sha256': plan.user_set_sha256,
            'bootstrap_W_sha256': plan.W_sha256, 'external_client_keys_unique': True},
        'source_outputs': pins, 'evaluator_identity': evaluator,
        'limitations': ['Previously observed MAIN validation and fixed transferred recipes; posthoc descriptive.',
            'Two fixed PRE and two fixed FL realizations do not estimate a population distribution of seed effects.',
            'User-bootstrap CIs are conditional on these fitted models; no multiplicity correction.',
            'Historical unfrozen X RNG diagnostic omitted; per-visit order/dropout independence tested.',
            'Historical confirmatory/TEST results remain unchanged; no privacy or deployment certification.']}
    write_once(out, result)
    return {'status': result['status'], 'analysis': str(physical(out)), 'sha256': sha(out), 'points': len(points)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build-protocol'); b.add_argument('--out', required=True, type=Path)
    b.add_argument('--template', action='store_true')
    p = sub.add_parser('prepare'); t = sub.add_parser('train'); r = sub.add_parser('reeval'); a = sub.add_parser('analyze')
    for parser in (p, t, r, a):
        parser.add_argument('--protocol', required=True, type=Path)
        parser.add_argument('--protocol-sha256', required=True)
    for parser in (p, t, r):
        parser.add_argument('--gpu', required=True, choices=(0, 1), type=int)
    p.add_argument('--out', required=True, type=Path)
    for parser in (t, r):
        parser.add_argument('--review', required=True, type=Path); parser.add_argument('--review-sha256', required=True)
        parser.add_argument('--line', choices=LINES); parser.add_argument('--seed', required=True, choices=SEEDS, type=int)
    t.add_argument('--preparation', required=True, type=Path); t.add_argument('--preparation-sha256', required=True)
    r.add_argument('--target', required=True, choices=('pre', 'warm')); r.add_argument('--rule', choices=tuple(RULES), default='CB')
    r.add_argument('--out', required=True, type=Path); r.add_argument('--dry-run', action='store_true')
    a.add_argument('--review', required=True, type=Path); a.add_argument('--review-sha256', required=True)
    a.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        if args.command == 'build-protocol':
            result = build_protocol(args.out, template=args.template)
        elif args.command == 'prepare':
            result = prepare(args.protocol, args.protocol_sha256, args.out, args.gpu)
        elif args.command == 'train':
            result = train(args.protocol, args.protocol_sha256, args.review, args.review_sha256,
                args.preparation, args.preparation_sha256, args.line, args.seed, args.gpu)
        elif args.command == 'reeval':
            result = reeval(args.protocol, args.protocol_sha256, args.review, args.review_sha256,
                args.target, args.line, args.seed, args.rule, args.gpu, args.out, dry=args.dry_run)
        else:
            result = analyze(args.protocol, args.protocol_sha256, args.review, args.review_sha256, args.out)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (Refused, OSError, ValueError, KeyError) as exc:
        print(f'SEED_WARM_REFUSED: {type(exc).__name__}: {exc}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
