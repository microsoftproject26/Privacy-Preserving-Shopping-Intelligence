"""Recovery identity refusal and unchanged actual AdamW/RNG resume controls."""
import copy
import json
from pathlib import Path
import sys
import ppsi_env  # noqa: F401
import pytest
import torch
from experiment_extensions.pretraining import pretrain_resume as R
from experiment_extensions.pretraining import pretrain_seed as P
from fl.numerics import enforce_strict_fp32
import ppsi_paths


@pytest.fixture(autouse=True)
def strict():
    enforce_strict_fp32()
    torch.set_num_threads(1)


def records():
    determinism = {'mode': 'STRICT', 'cublas_workspace_config': ':4096:8'}
    schema = {'schema_id': 'toy', 'schema_self_sha256': 'a'*64}
    start = dict(config=dict(run_id=P.RUN_ID, seed=P.SEED), N=P.N, theta0_init_sha256=P.THETA0,
                 config_sha256='b'*64, device_pin_sha256='c'*64, determinism=determinism, schema=schema)
    meta = dict(run_id=P.RUN_ID, seed=P.SEED, init_sha256=P.THETA0, run_role='MAIN',
                model_variant='SASREC_D64_B2', config_sha256='b'*64, device_pin_sha256='c'*64,
                updates=94101, exposures=P.N, efe=1., attempt=0, determinism=determinism, schema=schema)
    return start, dict(meta=meta, file='checkpoints/latest.pt'), dict(status='RUNNING')


def test_verified_fixed1EFE_recovery_metadata_admitted():
    R.verify_meta(*records())


def test_actual_style_rich_START_schema_provenance_compares_same_identity():
    a, b, c = copy.deepcopy(records())
    a['schema'] = dict(a['schema'], file='/copied/schema.json', file_sha256='d'*64,
                       kind='DERIVED', registry={'file_sha256': 'e'*64})
    R.verify_meta(a, b, c)
    b['meta']['schema']['schema_self_sha256'] = 'different'
    with pytest.raises(P.Refused):
        R.verify_meta(a, b, c)


@pytest.mark.parametrize('key,value', [('seed', 2026), ('run_id', 'old-run'), ('run_role', 'CALIBRATION'),
                                     ('model_variant', 'SASREC_D256_B2'), ('init_sha256', 'wrong'),
                                     ('config_sha256', 'wrong'), ('device_pin_sha256', 'wrong'),
                                     ('updates', 104000), ('exposures', P.N+256), ('efe', 1.1), ('attempt', 1)])
def test_drifted_latest_identity_and_uncheckpointed_progress_refused(key, value):
    a, b, c = records()
    b['meta'][key] = value
    with pytest.raises(P.Refused):
        R.verify_meta(a, b, c)


def test_finished_status_and_schema_or_determinism_change_refused():
    for mutation in ('done', 'schema', 'determinism', 'best'):
        a, b, c = copy.deepcopy(records())
        if mutation == 'done': c['status'] = 'DONE_MAIN'
        elif mutation == 'schema': b['meta']['schema'] = {'schema_id': 'other'}
        elif mutation == 'determinism': b['meta']['determinism'] = {'mode': 'WARN'}
        else: b['file'] = 'checkpoints/best.pt'
        with pytest.raises(P.Refused):
            R.verify_meta(a, b, c)


def test_wrong_GPU_refuses_before_loading_any_protocol_or_checkpoint(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setattr(R, 'load_amendment', lambda *a: (_ for _ in ()).throw(AssertionError('input read')))
    with pytest.raises(P.Refused, match='physicalGPU0'):
        R.resume('never', 'never', 'never', 'never')


def test_no_arbitrary_run_seed_role_device_or_continue_flags_admitted():
    for flag in ('--gpu', '--seed', '--role', '--run-dir', '--continue-run', '--endpoint'):
        with pytest.raises(SystemExit):
            R.main(['resume', '--amendment', 'never', '--amendment-sha256', 'never',
                    '--review', 'never', '--review-sha256', 'never', flag, 'MAIN_TEST'])


def test_actual_SASRec_AdamW_checkpoint_resume_bitwise_endpoint_and_completed_marks(tmp_path):
    sys.path.insert(0, str(ppsi_paths.code_dir('tests_train')))
    import trainkit as K
    from v4train.trainer import InjectedInterruption
    from v4train.checkpoint import sha256_file
    cfg = K.cfg('fixed6-recovery-control', 'SASREC', seed=2027, auto_continuation=False, checkpoint_every=4)
    full = K.trainer(tmp_path / 'full', cfg)
    a = full.run()
    cut = K.trainer(tmp_path / 'cut', cfg, fault=K.interrupt_after(9))
    with pytest.raises(InjectedInterruption):
        cut.run()
    index = K.read(K.art(tmp_path / 'cut') / 'CHECKPOINTS.json')
    assert index['latest']['meta']['updates'] == 8
    old_evals = {p.name: P.sha_file(p) for p in (K.art(tmp_path / 'cut') / 'evals').iterdir()}
    start = P.sha_file(tmp_path / 'cut/RUN_START.json')
    torch.manual_seed(918234)  # foreign dropout stream, restored from checkpoint
    b = K.trainer(tmp_path / 'cut', cfg).resume()
    assert a['status'] == b['status'] == 'DONE_MAIN' and a['efe'] == b['efe'] == 6.
    for field in ('final_state_digest', 'final_rng_sha256', 'updates', 'exposures'):
        assert a[field] == b[field]
    assert [(e['mark'], e['value']) for e in a['evals']] == [(e['mark'], e['value']) for e in b['evals']]
    for kind in ('best', 'endpoint_6.0'):
        require = lambda root: sha256_file(K.art(root) / f'checkpoints/{kind}.pt')
        assert require(tmp_path / 'full') == require(tmp_path / 'cut')
    assert P.sha_file(tmp_path / 'cut/RUN_START.json') == start
    for name, digest in old_evals.items():
        assert P.sha_file(K.art(tmp_path / 'cut') / 'evals' / name) == digest


def test_orphan_validation_refuses_before_checkpoint_payload_loading(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    (run / 'sealed/evals').mkdir(parents=True)
    a, b, c = records()
    (run / 'RUN_START.json').write_text(json.dumps(a))
    (run / 'STATUS.json').write_text(json.dumps(c))
    for key in ('latest', 'best'):
        p = run / 'sealed/checkpoints' / (key+'.pt')
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(key.encode())
    entry = dict(b, bytes=6, sha256=P.sha_file(run / 'sealed/checkpoints/latest.pt'))
    best = dict(file='checkpoints/best.pt', bytes=4, sha256=P.sha_file(run / 'sealed/checkpoints/best.pt'))
    (run / 'sealed/CHECKPOINTS.json').write_text(json.dumps(dict(latest=entry, best=best)))
    for mark in ('0.5', '1.0', '1.5'):
        for suffix in ('ranks_int32.npy', 'result.json', 'rows.json'):
            (run / 'sealed/evals' / f'eval_{mark}.{suffix}').touch()
    monkeypatch.setattr(P, 'RUN_DIR', run)
    with pytest.raises(P.Refused, match='orphan'):
        R.partial_gate()

