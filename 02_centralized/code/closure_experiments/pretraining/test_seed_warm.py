"""Identity, scope and real numerical controls for the second-init warm matrix."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import ppsi_env  # noqa: F401,E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from experiment_extensions.pretraining import seed_warm as S  # noqa: E402
from fl.numerics import enforce_strict_fp32, state_digest  # noqa: E402
from v4int import launch_fl as F  # noqa: E402
from v4int import reeval as R  # noqa: E402


@pytest.fixture(autouse=True)
def strict_cpu():
    torch.set_num_threads(1)
    enforce_strict_fp32()


def donor(line, seed=2026):
    d = S.donor_dir(line, seed)
    rec = S.read(d / 'ARM_SPEC.json')
    return {'line': line, 'fl_seed': seed, 'arm_spec': rec['arm_spec'],
        'run_id': rec['run_id'], 'pins': [S.pin(d / 'ARM_SPEC.json')]}


def new_identity():
    spec = donor('frozen')['arm_spec']
    return dict(spec['init_from'], run_id=S.PRE_RUN, seed=2027,
        checkpoint_sha256='1' * 64, state_digest='2' * 64)


def protocol():
    return {'historical_donors': [donor(l, s) for l in S.LINES for s in S.SEEDS],
        'new_init': {'identity': new_identity()}}


@pytest.mark.parametrize('line', S.LINES)
@pytest.mark.parametrize('seed', S.SEEDS)
def test_real_donor_transfer_changes_only_actual_pre_init(line, seed):
    d = donor(line, seed); before = copy.deepcopy(d)
    changed = S.transfer_spec(d, new_identity())
    assert d == before
    assert changed['cfg'] == before['arm_spec']['cfg']
    assert changed['init_from']['seed'] == 2027 and changed['cfg']['seed'] == seed
    assert {k: v for k, v in changed.items() if k != 'init_from'} == {
        k: v for k, v in d['arm_spec'].items() if k != 'init_from'}


@pytest.mark.parametrize('field,value', [('seed', 999), ('method', 'PF'), ('dp', {'z': 1}),
    ('pf', {}), ('endpoint_efe', 12), ('endpoint_rounds', 6827), ('dropout_p', .2), ('n_shards', 1)])
def test_hidden_numerical_recipe_changes_refused(field, value):
    d = donor('frozen'); d['arm_spec']['cfg'][field] = value
    with pytest.raises(S.Refused, match='recipe'):
        S.transfer_spec(d, new_identity())


@pytest.mark.parametrize('role', ['MAIN_TEST', 'TEST2', 'CALIBRATION', 'UNSEEN_VAL'])
def test_no_test_or_hidden_population_role(role):
    with pytest.raises(S.Refused): S.pair('frozen', 2027, role=role)


def test_wrong_seed_line_and_resume_refused():
    for args in [('PB100', 2027), ('frozen', 2028), ('cold', 2026)]:
        with pytest.raises(S.Refused): S.pair(*args)
    with pytest.raises(S.Refused): S.pair('frozen', 2027, resume=True)


@pytest.mark.parametrize('line', S.LINES)
def test_scoped_admissions_restore_and_do_not_wrap_fedadam_step(line):
    import fl.server as server
    step = server.FedAdam.step
    names = ('resolve_spec', 'p2_preflight', 'warm_preflight', 'require_lr_selection',
        'fl_rows', 'write_json', 'fl_code_sha256', 'p2_server_opt')
    old = {n: getattr(F, n) for n in names}
    p = protocol()
    with pytest.raises(RuntimeError, match='injected'):
        with S.train_context(p, line, 2026, 0, {'toy': True}, dry=True) as (mod, spec):
            args = S.launch_args(mod, line, 2026, 0, dry=True)
            got, _ = mod.resolve_spec(args, spec['lr_peak_base'], fused=True, init_identity=new_identity())
            assert got == spec
            assert server.FedAdam.step is step
            raise RuntimeError('injected')
    assert all(getattr(F, n) is old[n] for n in names)
    assert not S._ACTIVE_TRAIN and server.FedAdam.step is step


@pytest.mark.parametrize('field,value', [('seed', 2027), ('role', 'MAIN_TEST'), ('resume', True),
    ('smoke', True), ('history_policy', 'SESSION_L50'), ('run_id', 'other'), ('n_threads', 96)])
def test_runtime_argument_scope_rejects_hidden_override(field, value):
    p = protocol()
    with S.train_context(p, 'frozen', 2026, 0, {'toy': True}, dry=True) as (mod, _):
        args = S.launch_args(mod, 'frozen', 2026, 0, dry=True)
        setattr(args, field, value)
        with pytest.raises(S.Refused, match='arguments'):
            mod.warm_preflight(args)


def test_original_historical_warm_guard_still_refuses_new_identity():
    old_func = F.bind_warm_selection
    args = SimpleNamespace(arm='FA_WARM_FROZEN', warm_selection=F.WARM_SELECTION_PATH['S8'])
    binding = {'path': str(F.FL_TUNING_PATH), 'sha256': S.sha(F.FL_TUNING_PATH)}
    with pytest.raises(F.LaunchRefused, match='theta_pre identity differs'):
        F.bind_warm_selection(args, new_identity(), binding)
    with S.train_context(protocol(), 'frozen', 2026, 0, {'toy': True}, dry=True):
        assert F.bind_warm_selection is old_func
        with pytest.raises(F.LaunchRefused, match='theta_pre identity differs'):
            F.bind_warm_selection(args, new_identity(), binding)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True))


def real_identity_fixture(tmp_path, monkeypatch):
    # The tiny payload is exclusively a synthetic guard test, not real evidence.
    root = tmp_path / 'owned'; root.mkdir()
    d = root / S.PRE_RUN
    cp = d / 'sealed/checkpoints/endpoint_6.0.pt'; cp.parent.mkdir(parents=True)
    monkeypatch.setattr(S, 'BASE', root); monkeypatch.setattr(S, 'PRE_DIR', d)
    model = {'w': torch.arange(8, dtype=torch.float32).reshape(2, 4)}
    meta = {'run_id': S.PRE_RUN, 'seed': 2027, 'family': 'SASREC', 'efe': 6.0, 'mark': 6.0,
        'model_variant': S.VARIANT, 'run_label': 'PRETRAIN_FOR_FL',
        'manifest_registry': {'train_manifest': S.PRE_TRAIN}, 'state_digest': state_digest(model),
        'updates': 6 * ((S.PRE_N + 255) // 256), 'exposures': 6 * S.PRE_N}
    torch.save({'kind': 'endpoint_6.0', 'meta': meta, 'model': model}, cp)
    rec = {'run_id': S.PRE_RUN, 'run_label': 'PRETRAIN_FOR_FL', 'train_manifest': S.PRE_TRAIN,
        'fl_init': {'kind': 'endpoint_6.0', 'path': 'sealed/checkpoints/endpoint_6.0.pt',
            'sha256': S.sha(cp), 'bytes': cp.stat().st_size}}
    dump(d / 'PRETRAIN_FOR_FL_INIT.json', rec)
    _, ident, launch = F.load_pretrain_init(d / 'PRETRAIN_FOR_FL_INIT.json', model_variant=S.VARIANT)
    dec = {'seed': 2027, 'run_id': S.PRE_RUN, 'n_train_decisions': S.PRE_N, 'n_users': S.PRE_USERS,
        'train_manifest': S.PRE_TRAIN, 'eval_manifest': S.PRE_EVAL, 'endpoint_EFE': 6.0,
        'endpoint_updates': meta['updates'], 'endpoint_exposures': meta['exposures'],
        'theta0_digest': S.PRE_THETA0, 'post_endpoint_continuation': False,
        'checkpoint_selection_for_init': False, 'test_scoring': False, 'resume': False}
    dump(d / 'EXTENSION_TRAIN_STARTED.json', {'status': 'STARTED_WRITE_ONCE_NO_RESUME', 'label': S.LABEL,
        'protocol_sha256': S.PRE_PROTOCOL_SHA, 'review_sha256': S.PRE_REVIEW_SHA,
        'preparation_sha256': S.PRE_PREPARATION_SHA, 'fixed_declaration': dec})
    dump(d / 'RUN_START.json', {'run_id': S.PRE_RUN, 'N': S.PRE_N, 'theta0_init_sha256': S.PRE_THETA0,
        'config': {'run_id': S.PRE_RUN, 'seed': 2027, 'family': 'SASREC', 'model_variant': S.VARIANT,
            'recipe_id': S.RECIPE, 'phase': 'MAIN', 'test_overrides': False, 'auto_continuation': None}})
    dump(d / 'RUN_SUMMARY.json', {'status': 'DONE_MAIN', 'phase': 'MAIN', 'efe': 6.0, 'continuation': None})
    dump(d / 'EXTENSION_COMPLETE.json', {'status': 'COMPLETE_FIXED6_VALIDATION_DESCRIPTIVE', 'label': S.LABEL,
        'test_scored': False, 'run_directory': str(d), 'protocol_sha256': S.PRE_PROTOCOL_SHA, 'review_sha256': S.PRE_REVIEW_SHA,
        'endpoint': {'identity': ident, 'checkpoint_sha256': S.sha(cp), 'init_record_sha256': launch['init_record_sha256']}})
    return d, cp


def test_complete_own_init_identity_roundtrip(tmp_path, monkeypatch):
    d, cp = real_identity_fixture(tmp_path, monkeypatch)
    result = S.verify_new_init()
    assert result['identity']['seed'] == 2027 and result['checkpoint']['sha256'] == S.sha(cp)
    assert result['identity']['train_manifest'] == S.PRE_TRAIN


@pytest.mark.parametrize('mutation', ['seed', 'runid', 'kind', 'variant', 'state', 'round_budget',
    'train_manifest', 'main_phase', 'old_init', 'continuation', 'completion_protocol'])
def test_own_pre_wrong_identity_and_hidden_budget_refused(tmp_path, monkeypatch, mutation):
    d, cp = real_identity_fixture(tmp_path, monkeypatch)
    if mutation in ('main_phase', 'continuation', 'completion_protocol'):
        file = d / ('RUN_START.json' if mutation == 'main_phase' else 'RUN_SUMMARY.json' if mutation == 'continuation'
                    else 'EXTENSION_COMPLETE.json')
        obj = S.read(file)
        if mutation == 'main_phase': obj['config']['phase'] = 'CALIBRATION'
        elif mutation == 'continuation': obj['continuation'] = {'added': 1}
        else: obj['protocol_sha256'] = '5' * 64
        dump(file, obj)
    else:
        obj = torch.load(cp, weights_only=True)
        if mutation == 'seed': obj['meta']['seed'] = 2026
        elif mutation == 'runid': obj['meta']['run_id'] = 'PRETRAIN_FOR_FL_OTHER'
        elif mutation == 'kind': obj['kind'] = 'best'
        elif mutation == 'variant': obj['meta']['model_variant'] = 'SASREC_D256_B3'
        elif mutation == 'state': obj['model']['w'][0, 0] += 1
        elif mutation == 'round_budget': obj['meta']['updates'] -= 1
        elif mutation == 'train_manifest': obj['meta']['manifest_registry']['train_manifest'] = S.MAIN_TRAIN
        elif mutation == 'old_init': obj['meta']['seed'] = 2026
        torch.save(obj, cp)
        rec = S.read(d / 'PRETRAIN_FOR_FL_INIT.json')
        rec['fl_init'].update(sha256=S.sha(cp), bytes=cp.stat().st_size)
        dump(d / 'PRETRAIN_FOR_FL_INIT.json', rec)
    with pytest.raises((S.Refused, F.LaunchRefused)):
        S.verify_new_init()


def test_template_cannot_launch_even_when_self_hash_is_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'EXT', tmp_path)
    p = tmp_path / 'template.json'
    obj = {'format': S.FORMAT, 'status': 'TEMPLATE_PENDING_REAL_PRE_ENDPOINT'}
    obj['self_sha256'] = S.canon(obj); dump(p, obj)
    with pytest.raises(S.Refused, match='template'):
        S.load_protocol(p, S.sha(p))


def test_exclusive_outputs_symlink_escape_and_existing_refused(tmp_path, monkeypatch):
    allowed = tmp_path / 'allowed'; allowed.mkdir()
    monkeypatch.setattr(S, 'EXT', allowed)
    outside = tmp_path / 'other'; outside.mkdir()
    for path in (allowed, outside / 'x'):
        with pytest.raises(S.Refused): S.output_path(path)
    (allowed / 'link').symlink_to(allowed / 'real', target_is_directory=True)
    with pytest.raises(S.Refused, match='symlink'): S.output_path(allowed / 'link/new')
    monkeypatch.chdir(allowed)
    with pytest.raises(S.Refused, match='symlink'): S.output_path(Path('link/relative'))
    p = allowed / 'new.json'; S.write_once(p, {'a': 1})
    with pytest.raises(FileExistsError): S.write_once(p, {'a': 2})
    with pytest.raises(S.Refused, match='exists'): S.output_path(p, fresh=True)
    assert json.loads(p.read_text()) == {'a': 1}


@pytest.mark.parametrize('mutation', ['run', 'config', 'kind', 'EFE', 'round', 'lora', 'personal', 'phase'])
def test_own_fl_payload_refuses_hidden_identity_and_zero_mark(mutation):
    spec = S.transfer_spec(donor('frozen'), new_identity())
    rid = S.new_run_id('frozen', 2026)
    launch = {'run_id': rid, 'arm_spec': spec, 'fl_config_digest': F.fl_run_config(spec, rid, 64).digest()}
    obj = {'kind': 'weights', 'theta': {'w': torch.tensor([1.0])}, 'run_id': rid,
        'config_digest': launch['fl_config_digest'], 'efe': 6.001, 'round': S.ROUNDS, 'phase': 'MAIN'}
    if mutation == 'run': obj['run_id'] = 'another'
    elif mutation == 'config': obj['config_digest'] = '0' * 64
    elif mutation == 'kind': obj['kind'] = 'latest'
    elif mutation == 'EFE': obj['efe'] = 0
    elif mutation == 'round': obj['round'] -= 1
    elif mutation == 'lora': obj['p2_lora'] = {}
    elif mutation == 'personal': obj['store'] = {}
    else: obj['phase'] = 'CONTINUATION'
    with pytest.raises(S.Refused): S.check_fl_payload(obj, launch, filename='endpoint_6.0.pt')


def test_reeval_context_restores_all_guards_after_exception():
    old = {k: getattr(R, k) for k in ('CENTRAL_SOURCES', 'source_central', 'source_fl',
        'check_warm_line', 'check_fl_payload', 'label_rows', 'write_outputs')}
    with pytest.raises(RuntimeError):
        with S.reeval_context(protocol(), '1' * 64, 'pre', None, 2027, 'CB') as mod:
            assert mod.CENTRAL_SOURCES['PRE']['seed'] == 2027
            assert mod.run is R.run and mod.build_fl_harness is R.build_fl_harness
            raise RuntimeError('injected')
    assert all(getattr(R, k) is v for k, v in old.items()) and not S._ACTIVE_REEVAL


def test_reeval_labels_preserve_all_comparison_keys_and_reject_test_rows():
    from v4eval.rows import COMPARISON_KEY
    args = SimpleNamespace(role='MAIN', arm_label='PRE', selection_rule='CONTROLLED_BUDGET',
        checkpoint_id='toy', ckpt_kind='central')
    row = {k: 'toy-' + k for k in COMPARISON_KEY}
    row.update(split='VALIDATION', eval_manifest_id=S.MAIN_EVAL)
    with S.reeval_context(protocol(), '1' * 64, 'pre', None, 2027, 'CB') as mod:
        got = mod.label_rows([row], args, checkpoint_sha256='2' * 64,
            source_run_id=S.PRE_RUN, efe=6.0, extra={})
        assert {k: got[0][k] for k in COMPARISON_KEY} == {k: row[k] for k in COMPARISON_KEY}
        assert got[0]['comparison_status'] == S.LABEL and got[0]['pretrain_seed'] == 2027
        assert got[0]['actual_shared_optimizer_lifecycle'] == 'PERSISTENT_ADAMW_CENTRAL'
        row['split'] = 'RETRO_TEST'
        with pytest.raises(S.Refused, match='validation'):
            mod.label_rows([row], args, checkpoint_sha256='2' * 64,
                source_run_id=S.PRE_RUN, efe=6.0, extra={})


def test_actual_sgdm_lifecycle_disclosed_without_changing_historical_comparison_key():
    from v4eval.rows import COMPARISON_KEY
    assert 'actual_shared_optimizer_lifecycle' not in COMPARISON_KEY
    with S.train_context(protocol(), 'frozen', 2026, 0, {'toy': True}, dry=True) as (mod, _):
        row = mod.fl_rows([{'optimizer_lifecycle': 'OLD_KEY'}], arm='FA_WARM_FROZEN',
            run_id='toy', efe_raw=1.0, mark=1.0, digest='0' * 64, role='MAIN')[0]
        assert row['optimizer_lifecycle'] == 'FRESH_ADAMW_PER_VISIT'
        assert row['actual_shared_optimizer_lifecycle'] == 'FRESH_SGDM_PER_VISIT'


def test_round_based_fixed_budget_admits_real_seed2027_actual_efe_below_six():
    spec = S.transfer_spec(donor('frozen', 2027), new_identity())
    rid = S.new_run_id('frozen', 2027)
    launch = {'run_id': rid, 'arm_spec': spec, 'fl_config_digest': F.fl_run_config(spec, rid, 64).digest()}
    obj = {'kind': 'weights', 'theta': {'w': torch.tensor([1.0])}, 'run_id': rid,
        'config_digest': launch['fl_config_digest'], 'efe': 5.999705152223377, 'round': S.ROUNDS, 'phase': 'MAIN'}
    assert S.check_fl_payload(obj, launch, filename='endpoint_6.0.pt') == state_digest(obj['theta'])
    obj['round'] -= 1
    with pytest.raises(S.Refused, match='rounds'):
        S.check_fl_payload(obj, launch, filename='endpoint_6.0.pt')


def tiny_run(*, frozen, context=False, ambient=False):
    from fl.client import solver_from_dict
    from fl.runtime import FLRun, RunConfig, freeze_item_tables
    from fl.server import Server, ServerOptConfig
    from fl.synthetic import make_clients, TinyRec, TinyAdapter
    from fl.participation import ParticipationPlan
    server_adapter = TinyAdapter(TinyRec(seed=11, tied=False, K=12, d=4))
    worker = TinyAdapter(TinyRec(seed=12, tied=False, K=12, d=4))
    theta = server_adapter.broadcast_state(clone=True)
    if frozen:
        freeze_item_tables(server_adapter)
        freeze_item_tables(worker, source=theta)
    clients = make_clients(32, K=12, L=4, seed=21, sizes=[3 + i % 4 for i in range(32)], invalid_frac=0)
    by_key = {c.key: c for c in clients}
    cfg = RunConfig(run_id='SEED_WARM_TOY', method='FA', seed=2027, lr_peak_local=.1,
        solver=solver_from_dict(dict(lr=.1, passes=1, batch_size=3, weight_decay=0, optimizer='sgd_m',
            item_lr_mult=None if frozen else .1)), n_shards=2, exposure_point='end', endpoint_rounds=8,
        server_opt=ServerOptConfig('fedadam', lr=.001 if frozen else 3e-6, tau=.001 if frozen else 1e-6),
        lr_shape='fl_const')
    plan = ParticipationPlan.build(list(by_key), seed=cfg.seed, manifest_hash='SYNTHETIC_MAIN', group_size=4)
    run = FLRun(cfg, plan, Server(server_adapter, init_sha256=state_digest(theta)),
        by_key.__getitem__, sum(len(c.examples['target_class']) for c in clients), workers=[worker])
    values = []
    for _ in range(4):
        if ambient: _ = torch.randperm(123)
        result = run.step()
        values.append({'state': run.server.digest(), 'opt': run.server.server_opt.digest(),
            'weight': result['weight'], 'lr': result['lr'], 'counters': asdict(run.counters),
            'ledger': asdict(run.ledger)})
    if frozen:
        assert all(torch.equal(run.server.adapter.broadcast_state()[k], theta[k]) for k in S.FROZEN)
    assert torch.equal(run.server.adapter.module.item_embed.weight[0], torch.zeros(4))
    def plain(value):
        if isinstance(value, (np.ndarray, torch.Tensor)): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [plain(v) for v in value]
        return value
    return plain(values)


@pytest.mark.parametrize('frozen', [False, True])
def test_real_four_round_frozen_unfrozen_numerics_and_pervisit_rng_unchanged(frozen):
    line = 'frozen' if frozen else 'unfrozen'
    original = tiny_run(frozen=frozen)
    with S.train_context(protocol(), line, 2027, 0, {'toy': True}, dry=True):
        changed = tiny_run(frozen=frozen)
        ambient = tiny_run(frozen=frozen, ambient=True)
    assert changed == original == ambient


def test_command_line_never_exposes_test_or_recipe_override():
    # Unknown options fail in argparse before protocol/checkpoint file access.
    for extra in ('--role', '--resume', '--registry-ext', '--server-tau'):
        with pytest.raises(SystemExit):
            S.main(['train', '--protocol', '/does/not/exist', '--protocol-sha256', '0' * 64,
                '--gpu', '0', '--review', '/does/not/exist', '--review-sha256', '0' * 64,
                '--preparation', '/does/not/exist', '--preparation-sha256', '0' * 64,
                '--seed', '2027', '--line', 'frozen', extra, 'MAIN_TEST'])


def test_validation_full_client_macro_keeps_oov_only_clients():
    from v4eval import bootstrap as B
    m = SimpleNamespace(n=5, catalogue_K=5, rankable=np.array([1, 1, 1, 0, 0], dtype=bool),
        eval_mask=np.ones(5, dtype=bool), eval_user_key=np.array([0, 0, 1, 2, 2], dtype=np.int64))
    stats = S.validation_stats(m, np.array([1, 5, 2, -1, -1], dtype=np.int32), 'toy')
    assert stats.n_decisions == 5 and len(stats.user_keys) == 3
    assert np.count_nonzero(stats.NR == 0) == 1
    assert B.point(stats, 'macro_e2e') == pytest.approx((.6 + .5 + 0) / 3)
    assert B.point(stats, 'micro_e2e') == pytest.approx(1.7 / 5)
    assert B.point(stats, 'macro_e2e') != B.point(stats, 'micro_e2e')


@pytest.mark.parametrize('mutation', ['positive_oov', 'missing_rank', 'too_large', 'wrong_shape', 'float_rank'])
def test_invalid_validation_rank_and_padding_denominator_refused(mutation):
    m = SimpleNamespace(n=3, catalogue_K=5, rankable=np.array([1, 1, 0], dtype=bool),
        eval_mask=np.ones(3, dtype=bool), eval_user_key=np.array([0, 1, 2]))
    ranks = np.array([1, 5, -1], dtype=np.int32)
    if mutation == 'positive_oov': ranks[-1] = 1
    elif mutation == 'missing_rank': ranks[0] = 0
    elif mutation == 'too_large': ranks[1] = 6
    elif mutation == 'wrong_shape': ranks = ranks[:2]
    else: ranks = ranks.astype(np.float32)
    with pytest.raises(S.Refused): S.validation_stats(m, ranks, 'toy')


def test_exact_review_rejects_wrong_protocol_source_tests_or_implementer(tmp_path, monkeypatch):
    root = tmp_path / 'code/experiment_extensions'; root.mkdir(parents=True)
    module, tests = root / 'seed_warm.py', root / 'test_seed_warm.py'
    module.write_text('module-v1'); tests.write_text('test-v1')
    monkeypatch.setattr(S, 'CODE', root.parent)
    file = tmp_path / 'review.json'
    good = {'status': 'PASS', 'protocol_sha256': '1' * 64, 'module_sha256': S.sha(module),
        'tests_sha256': S.sha(tests), 'reviewer': 'independent_launch_review', 'preparation_sha256': '2' * 64}
    dump(file, good)
    assert S.review_gate(file, S.sha(file), '1' * 64, preparation_sha='2' * 64)['status'] == 'PASS'
    for key, bad in [('status', 'FAIL'), ('protocol_sha256', '0' * 64), ('module_sha256', '0' * 64),
                     ('tests_sha256', '0' * 64), ('reviewer', '/root/pretrain_finalize'), ('preparation_sha256', '0' * 64)]:
        dump(file, dict(good, **{key: bad}))
        with pytest.raises(S.Refused): S.review_gate(file, S.sha(file), '1' * 64, preparation_sha='2' * 64)
    dump(file, good)
    with pytest.raises(S.Refused, match='SHA'): S.review_gate(file, '0' * 64, '1' * 64)


def gpu_command(command):
    if command == 'prepare': return lambda: S.prepare(Path('never'), '0' * 64, Path('never'), 1)
    if command == 'train': return lambda: S.train(Path('never'), '0' * 64, Path('never'), '0' * 64,
        Path('never'), '0' * 64, 'frozen', 2026, 1)
    return lambda: S.reeval(Path('never'), '0' * 64, Path('never'), '0' * 64,
        'pre', None, 2027, 'CB', 1, Path('never'), dry=True)


@pytest.mark.parametrize('command', ['prepare', 'train', 'reeval'])
@pytest.mark.parametrize('visible', [None, '0', '0,1', 'cuda:1'])
def test_gpu1_wrong_or_missing_export_refused_before_any_input_read(command, visible, monkeypatch):
    if visible is None: monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else: monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setattr(S, 'load_protocol', lambda *a: (_ for _ in ()).throw(AssertionError('input read started')))
    with pytest.raises(S.Refused, match='before Python'):
        gpu_command(command)()


@pytest.mark.parametrize('command', ['prepare', 'train', 'reeval'])
def test_gpu1_correct_export_reaches_protocol_validation(command, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setattr(S, 'load_protocol', lambda *a: (_ for _ in ()).throw(RuntimeError('protocol reached after guard')))
    with pytest.raises(RuntimeError, match='protocol reached after guard'):
        gpu_command(command)()
