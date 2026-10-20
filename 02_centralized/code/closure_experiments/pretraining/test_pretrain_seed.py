"""Controls for the second FULL_PRE seed and fixed six-EFE budget."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import ppsi_paths

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import ppsi_env  # noqa: E402,F401
import pytest  # noqa: E402
import torch  # noqa: E402

from experiment_extensions.pretraining import pretrain_seed as P  # noqa: E402
from fl.numerics import enforce_strict_fp32, state_digest  # noqa: E402
from v4int import pretrain_small as S  # noqa: E402
from v4int import launch_fl as F  # noqa: E402
from v4train import recipe as RC  # noqa: E402
from v4train.schedule import planned_lr_table, eval_marks  # noqa: E402


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    torch.set_num_threads(1)
    enforce_strict_fp32()
    monkeypatch.setattr(S, 'REQUIRED', dict(S.REQUIRED))
    monkeypatch.setattr(S, 'pretrain_row_fields', S.pretrain_row_fields)
    monkeypatch.setattr(RC.RunConfig, 'effective_auto_continuation', RC.RunConfig.effective_auto_continuation)


def registry():
    return {'roles': {'PRETRAIN': {'train': P.TRAIN, 'eval': P.EVAL}},
        'manifests': {P.TRAIN: {'n_users': P.N_USERS, 'n_decisions': P.N}},
        'population_bands': {'PRETRAIN_NONMAIN': {'rank_range': [131072, 1348682]}}}


@pytest.mark.parametrize('mutant', ['PB100', 'MAIN', 'wrong_decisions', 'wrong_eval'])
def test_full_pre_band_substitution_refused(mutant):
    r = registry()
    if mutant == 'PB100':
        r['manifests'][P.TRAIN]['n_users'] = 100000
        r['population_bands']['PRETRAIN_NONMAIN']['rank_range'] = [131072, 231072]
    elif mutant == 'MAIN':
        r['roles']['PRETRAIN']['train'] = 'MAIN_TRAIN_5_100_131072'
    elif mutant == 'wrong_decisions':
        r['manifests'][P.TRAIN]['n_decisions'] -= 1
    else:
        r['roles']['PRETRAIN']['eval'] = 'MAIN_TEST_5_100_131072'
    with pytest.raises(P.Refused):
        P.verify_registry_design(r)


def test_registered_seed2027_only_same_launcher_constraints():
    from v4int import launch_central as LC
    old = LC.parse_args(P.launch_argv(0))
    with pytest.raises(S.PretrainRefused, match='seed'):
        S.check_launch_args(old)
    P.install_seed_and_budget(S, RC)
    assert S.check_launch_args(old) == P.RUN_ID
    old.seed = 2026
    with pytest.raises(S.PretrainRefused, match='seed'):
        S.check_launch_args(old)


def config(run_id=P.RUN_ID, **kw):
    args = dict(run_id=run_id, family='SASREC', seed=P.SEED, exposure_point='end',
        cuda_visible_devices='0', phase='MAIN', population='COHORT',
        recipe_id='SMALL_SASREC_D64_B2', auto_continuation=None)
    args.update(kw)
    return RC.RunConfig(**args)


def test_fixed_budget_property_has_exact_scope_and_config_hash_unchanged():
    own, other = config(), config('HISTORICAL_PRE_2027')
    before = own.sha256()
    original = RC.RunConfig.effective_auto_continuation
    assert own.effective_auto_continuation is True
    expected_other = original.fget(other)
    P.install_seed_and_budget(S, RC)
    assert own.effective_auto_continuation is False
    assert other.effective_auto_continuation is expected_other
    assert own.auto_continuation is None and own.sha256() == before
    assert S.pretrain_row_fields()['comparison_status'] == P.LABEL


@pytest.mark.parametrize('field,value', [('seed', 2026), ('phase', 'CALIBRATION'),
    ('population', 'FULL_TRAIN'), ('recipe_id', '02B_SASREC'), ('test_overrides', True),
    ('auto_continuation', True)])
def test_fixed_budget_scope_rejects_inconsistent_configuration(field, value):
    P.install_seed_and_budget(S, RC)
    with pytest.raises(P.Refused):
        _ = config(**{field: value}).effective_auto_continuation


def test_fixed6_does_not_change_real_first6_schedule_or_marks():
    table_before = planned_lr_table(0.0007114121116984968, P.N, 256, 'MAIN', 'end', schedule='S0')
    marks_before = eval_marks('MAIN')
    P.install_seed_and_budget(S, RC)
    assert planned_lr_table(0.0007114121116984968, P.N, 256, 'MAIN', 'end', schedule='S0') == table_before
    assert table_before['n_updates'] == 564606
    assert table_before['final_exposures'] == 144538944
    assert table_before['sha256'] == '03d4605f1c943339a45194ebd2e0f388ba9c0f120fc37904a99d669c472abb3f'
    assert eval_marks('MAIN') == marks_before == [i / 2 for i in range(1, 13)]


def test_actual_trainer_first6_weights_equal_when_only_continuation_disabled(tmp_path):
    # Use the inherited small synthetic providers and real AdamW/CE/trainer.
    # The real config hook supplies the bool that CentralTrainer stores as
    # auto_continue. With identical synthetic seed/input, compare fixed6 with
    # original C3's retained endpoint, not its post6 weights.
    train_tests = ppsi_paths.code_dir('tests_train')
    sys.path.insert(0, str(train_tests))
    import trainkit as K
    values = {m: 0.1 + m / 100 for m in eval_marks('MAIN') + eval_marks('CONTINUATION')}
    a = K.trainer(tmp_path / 'original', K.cfg('synthetic_original', 'SASREC', auto_continuation=True),
                  eval_override=K.scripted(values))
    sa = a.run()
    P.install_seed_and_budget(S, RC)
    b = K.trainer(tmp_path / 'fixed', K.cfg('synthetic_fixed', 'SASREC', auto_continuation=True),
                  eval_override=K.scripted(values))
    b.auto_continue = config().effective_auto_continuation
    sb = b.run()
    assert sa['status'] == 'DONE_CONTINUATION' and sb['status'] == 'DONE_MAIN'
    ca = torch.load(tmp_path / 'original/sealed/checkpoints/endpoint_6.0.pt', weights_only=True)
    cb = torch.load(tmp_path / 'fixed/sealed/checkpoints/endpoint_6.0.pt', weights_only=True)
    assert state_digest(ca['model']) == state_digest(cb['model'])
    for k in ca['model']:
        assert torch.equal(ca['model'][k], cb['model'][k]), k
    assert ca['meta']['updates'] == cb['meta']['updates']
    assert ca['meta']['exposures'] == cb['meta']['exposures']
    assert cb['meta']['efe'] == 6.0
    assert not (tmp_path / 'fixed/sealed/RESULT_ROWS_EXTENDED.json').exists()


def make_checkpoint(tmp_path):
    run = tmp_path / 'run'
    ck = run / 'sealed/checkpoints/endpoint_6.0.pt'
    ck.parent.mkdir(parents=True)
    model = {'w': torch.arange(8, dtype=torch.float32).reshape(2, 4)}
    meta = {'run_id': P.RUN_ID, 'seed': 2027, 'family': 'SASREC', 'efe': 6.0,
        'model_variant': 'SASREC_D64_B2', 'run_label': 'PRETRAIN_FOR_FL',
        'manifest_registry': {'train_manifest': P.TRAIN}, 'state_digest': state_digest(model),
        'updates': 564606, 'exposures': 144538944}
    torch.save({'kind': 'endpoint_6.0', 'meta': meta, 'model': model}, ck)
    S.fl_init_record(run, P.RUN_ID)
    return run, ck


def test_init_record_roundtrip_preserves_new_seed_and_old_warm_guard_refuses(tmp_path):
    run, _ = make_checkpoint(tmp_path)
    endpoint = P.verify_endpoint(run, require_real=False)
    assert endpoint['identity']['seed'] == 2027
    assert endpoint['identity']['run_id'] == P.RUN_ID
    args = SimpleNamespace(arm='FA_WARM_FROZEN', warm_selection=F.WARM_SELECTION_PATH['S8'])
    tuning = F.FL_TUNING_PATH
    with pytest.raises(F.LaunchRefused, match='theta_pre identity differs'):
        F.bind_warm_selection(args, endpoint['identity'], {'path': str(tuning), 'sha256': P.sha_file(tuning)})


@pytest.mark.parametrize('mutant', ['seed', 'kind', 'EFE', 'state', 'train'])
def test_bad_endpoint_identity_rejected_before_warm_use(tmp_path, mutant):
    run, ck = make_checkpoint(tmp_path)
    obj = torch.load(ck, weights_only=True)
    if mutant == 'seed': obj['meta']['seed'] = 2026
    elif mutant == 'kind': obj['kind'] = 'best'
    elif mutant == 'EFE': obj['meta']['efe'] = 5.5
    elif mutant == 'state': obj['model']['w'][0, 0] += 1
    else: obj['meta']['manifest_registry']['train_manifest'] = 'MAIN_TRAIN_5_100_131072'
    torch.save(obj, ck)
    rec = P.strict(run / 'PRETRAIN_FOR_FL_INIT.json')
    rec['fl_init']['sha256'] = P.sha_file(ck); rec['fl_init']['bytes'] = ck.stat().st_size
    (run / 'PRETRAIN_FOR_FL_INIT.json').write_text(json.dumps(rec))
    with pytest.raises((P.Refused, F.LaunchRefused)):
        P.verify_endpoint(run, require_real=False)


def test_output_escape_existing_and_symlink_refused(tmp_path, monkeypatch):
    root = tmp_path / 'allowed'; root.mkdir()
    monkeypatch.setattr(P, 'EXT', root)
    # Default argument binds old EXT, so explicitly exercise containment with
    # the same check for a temporary fixture root.
    with pytest.raises(P.Refused): P.inside(root / '../escape', root)
    (root / 'link').symlink_to(tmp_path / 'outside', target_is_directory=True)
    with pytest.raises(P.Refused): P.inside(root / 'link/output', root)
    p = P.EXT / 'existing'
    p.mkdir()
    monkeypatch.setattr(P, 'inside', lambda path: P.physical(path))
    with pytest.raises(P.Refused, match='already exists'): P.fresh_output(p)


def test_review_exact_bytes_required(tmp_path, monkeypatch):
    module, tests = tmp_path / 'pretrain_seed.py', tmp_path / 'test_pretrain_seed.py'
    module.write_text('module'); tests.write_text('tests')
    fakecode = tmp_path / 'code'; (fakecode / 'experiment_extensions').mkdir(parents=True)
    (fakecode / 'experiment_extensions/pretrain_seed.py').write_text('module')
    (fakecode / 'experiment_extensions/test_pretrain_seed.py').write_text('tests')
    monkeypatch.setattr(P, 'CODE', fakecode)
    monkeypatch.setattr(P, 'inside', lambda path: Path(path))
    review = {'status': 'PASS', 'protocol_sha256': 'a' * 64, 'module_sha256': P.sha_file(module),
        'tests_sha256': P.sha_file(tests), 'reviewer': '/root/independent_reviewer'}
    p = tmp_path / 'review.json'; p.write_text(json.dumps(review))
    assert P.load_review(p, P.sha_file(p), 'a' * 64)['status'] == 'PASS'
    module2 = fakecode / 'experiment_extensions/pretrain_seed.py'; module2.write_text('changed')
    with pytest.raises(P.Refused): P.load_review(p, P.sha_file(p), 'a' * 64)
    module2.write_text('module'); review['reviewer'] = '/root/next_pretrain'; p.write_text(json.dumps(review))
    with pytest.raises(P.Refused): P.load_review(p, P.sha_file(p), 'a' * 64)


@pytest.mark.parametrize('flag', ['--resume', '--role', '--seed', '--init-from', '--registry-ext'])
def test_no_arbitrary_launcher_or_test_argument_admitted(flag):
    with pytest.raises(SystemExit):
        P.main(['prepare', '--protocol', '/tmp/example', '--protocol-sha256', 'a' * 64,
                '--out', '/tmp/example2', flag, 'MAIN_TEST'])


def test_exclusive_receipt_write_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'inside', lambda p: Path(p))
    p = tmp_path / 'receipt.json'
    P.write_once(p, {'status': 'FIRST'})
    original = p.read_bytes()
    with pytest.raises(FileExistsError): P.write_once(p, {'status': 'SECOND'})
    assert p.read_bytes() == original


@pytest.mark.parametrize('group', ['source', 'data', 'contract', 'protocol_declaration'])
def test_input_and_declared_protocol_tamper_refused(tmp_path, monkeypatch, group):
    work = P.physical(tmp_path / 'fixture'); work.mkdir()
    (work / 'contracts/manifest_registry').mkdir(parents=True)
    reg = work / P.REG_REL; reg.write_text(json.dumps(registry()))
    source, data, contract = work / 'module.py', work / 'train.parquet', work / 'contracts/recipe.json'
    source.write_text('original source'); data.write_text('original data'); contract.write_text('original recipe')
    monkeypatch.setattr(P, 'WORK', work)
    monkeypatch.setattr(P, 'source_paths', lambda: [source])
    monkeypatch.setattr(P, 'data_paths', lambda: [data])
    pp = work / 'protocol.json'
    P.build_protocol(pp)
    original_sha = P.sha_file(pp)
    assert P.load_protocol(pp, original_sha)['declaration']['seed'] == 2027
    if group == 'protocol_declaration':
        obj = P.strict(pp); obj['declaration']['seed'] = 2026
        obj['self_sha256'] = P.canonical_sha(obj); pp.write_text(json.dumps(obj))
        expected = P.sha_file(pp)
    else:
        {'source': source, 'data': data, 'contract': contract}[group].write_text('changed')
        expected = original_sha
    with pytest.raises(P.Refused): P.load_protocol(pp, expected)


def theta_cfg():
    return {'schema_registry': str(P.WORK / 'contracts/derived_schemas/REGISTRY.json'), 'dropout': [0.1]}


def test_actual_full_catalogue_cpu_theta0_equals_registered_seed2027():
    proof = P.rebuild_theta0(theta_cfg())
    assert proof['rebuilt_state_digest'] == P.THETA0
    assert proof['raw_init_digest'] == 'b4298a79d95d24d108a6abc9ba34bb2f7ec7da167dfee0bd8bff0b72808ca78a'
    assert proof['parameter_manifest_digest'] == 'df21f357a21effbce96692f6b1c21d8aa1fed4b051c8caecba5299ff9e08312c'
    assert proof['global_torch_rng_unchanged'] is True


def test_actual_cpu_theta0_wrong_rebuild_seed_fails_closed(monkeypatch):
    import v4models
    original = v4models.build
    def wrong_seed(family, schema, seed, device, **kw):
        return original(family, schema, 2026, device, **kw)
    monkeypatch.setattr(v4models, 'build', wrong_seed)
    with pytest.raises(P.Refused, match='CPU rebuilt theta0 differs'):
        P.rebuild_theta0(theta_cfg())
