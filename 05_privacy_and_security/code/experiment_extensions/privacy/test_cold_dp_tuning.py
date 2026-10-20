"""Fixed-grid, private-randomness and own-source/release-boundary controls."""
import os
import ppsi_paths
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import ppsi_env  # noqa: F401,E402
from collections import OrderedDict
import copy
import json
import math
from pathlib import Path
import sys
import pytest
import torch

from . import cold_dp_tuning as C


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)
    from fl.numerics import enforce_strict_fp32
    enforce_strict_fp32()


def test_fixed_four_cells_equal_nominal_budget_and_new_registered_accountant():
    from fl import accountant as A
    for c in C.load_grid()['cells']:
        assert c['T'] * c['q'] * c['passes'] == 6
        assert A.epsilon(c['planned_m'], c['N'], c['T'], c['accountant']['z'], 5e-6) <= 8
        assert A.epsilon(c['planned_m'], c['N'], c['T'], .5642, 5e-6) > 8
        assert c['S_CAL_group'] == ('A' if c['cell_id'] == 'D' else c['cell_id'])


@pytest.mark.parametrize('role', ['MAIN', 'MAIN_TEST', 'TRUE_COLD_TEST', 'TEST2', 'VALIDATION_CONFIRM'])
def test_bad_role_refuses_before_grid_access(monkeypatch, role):
    monkeypatch.setattr(C, 'load_grid', lambda: (_ for _ in ()).throw(AssertionError('read before refusal')))
    with pytest.raises(C.Refused):
        C.select_cell('A', role=role)


def test_exact_cold_source_spec_has_full_recipe_budget_and_no_fake_confirmation():
    from v4int import launch_fl as F
    for cell in C.load_grid()['cells']:
        spec = C.build_spec(cell, mode='DP', clip=.2 * cell['S_multiplier'])
        assert spec['role'] == 'CALIBRATION' and spec['extension_label'] == C.LABEL
        cfg = F.fl_run_config(spec, cell['run_id'], 8)
        assert cfg.endpoint_rounds == cell['T'] and cfg.solver.passes == cell['passes']
        assert cfg.solver.lr == cell['client_lr'] and cfg.dropout_p == 0
        assert spec['plan'] == {'group_size':1024, 'sampling':'poisson'}
        assert cfg.dp.denominator == 1024 and cfg.dp.noise_multiplier == cell['accountant']['z']
        with C.execution_metadata(spec, {'unit':True}):
            labels = F.row_labels(F.spec_tuning(spec))
            assert labels['fl_client_lr'] == cell['client_lr']
            assert F.poisson_row_labels(spec)['fl_poisson_local_rule'] == 'OWN_COLD_DP_GRID_01'


def test_mutant_cell_or_missing_clip_refused_before_numerical_setup():
    cell = copy.deepcopy(C.select_cell('A'))
    cell['T'] = 384
    with pytest.raises(C.Refused, match='fixed recipe'):
        C.build_spec(cell, mode='DP', clip=.2)
    for clip in [None, 0., math.nan, math.inf, -1.]:
        with pytest.raises(C.Refused):
            C.build_spec(C.select_cell('A'), mode='DP', clip=clip)


def test_private_noise_is_original_gaussian_law_with_private_entropy_only():
    from fl import dp as D
    template = OrderedDict([('b', torch.zeros(4)), ('a', torch.zeros(7))])
    original = D.noise_like
    before = torch.get_rng_state().clone()
    expected = original(list(template), template, seed=881, round_idx=3, std=.4)
    public = original(list(template), template, seed=2026, round_idx=3, std=.4)
    with C.private_randomness(participation_seed=771, noise_seed=881):
        actual = D.noise_like(list(template), template, seed=2026, round_idx=3, std=.4)
        other_public_seed = D.noise_like(list(template), template, seed=2027, round_idx=3, std=.4)
        assert all(torch.equal(expected[k], actual[k]) and torch.equal(actual[k], other_public_seed[k]) for k in template)
        assert any(not torch.equal(public[k], actual[k]) for k in template)
    assert D.noise_like is original and torch.equal(before, torch.get_rng_state())


def test_private_poisson_stream_exact_original_law_and_same_resume_keys():
    from fl.participation import ParticipationPlan as P
    keys = [f'u{i:05}' for i in range(32768)]
    original = P.__dict__['build']
    expected = P.build(keys, seed=771, manifest_hash='owned-test', group_size=1024, sampling='poisson')
    with C.private_randomness(participation_seed=771, noise_seed=881):
        actual = P.build(keys, seed=2026, manifest_hash='owned-test', group_size=1024, sampling='poisson')
        repeat = P.build(keys, seed=2027, manifest_hash='owned-test', group_size=1024, sampling='poisson')
        assert actual.inclusion_probability == .03125
        for r in [0,1,17]:
            assert actual.round_keys(r) == expected.round_keys(r) == repeat.round_keys(r)
        with pytest.raises(C.Refused):
            P.build(keys, seed=2026, manifest_hash='x', group_size=1024, sampling='uniform')
    assert P.__dict__['build'] is original


def test_private_context_restores_both_hooks_after_exception_and_refuses_nesting():
    from fl import dp as D
    from fl.participation import ParticipationPlan as P
    saved = (D.noise_like, P.__dict__['build'])
    with pytest.raises(RuntimeError):
        with C.private_randomness(participation_seed=7, noise_seed=8):
            with pytest.raises(C.Refused, match='nested'):
                with C.private_randomness(participation_seed=9, noise_seed=10):
                    pass
            raise RuntimeError('owned control')
    assert D.noise_like is saved[0] and P.__dict__['build'] is saved[1]


@pytest.mark.parametrize('p,n', [(1,1), (-1,2), (True,2), (2,2**63), (1.5,2)])
def test_private_entropy_invalid_or_shared_refused(p,n):
    with pytest.raises(C.Refused):
        with C.private_randomness(participation_seed=p, noise_seed=n):
            pass


def test_own_metadata_restores_and_refuses_other_recipe():
    from v4int import launch_fl as F
    spec = C.build_spec(C.select_cell('A'), mode='S_CAL')
    saved = {k:getattr(F,k) for k in ['write_json','spec_tuning','poisson_row_labels']}
    with pytest.raises(RuntimeError):
        with C.execution_metadata(spec, {'unit':True}):
            mutant = copy.deepcopy(spec);mutant['cfg']['solver']['lr'] = .03
            with pytest.raises(C.Refused):F.spec_tuning(mutant)
            with pytest.raises(C.Refused):
                with C.execution_metadata(spec, {}):pass
            raise RuntimeError('owned control')
    assert all(getattr(F,k) is v for k,v in saved.items())


def test_release_whitelist_does_not_accept_private_checkpoint_metadata_or_alias_weights():
    weights = OrderedDict(w=torch.arange(8, dtype=torch.float32))
    payload = C.model_only_payload(weights, C.select_cell('A'), 'a'*64)
    assert set(payload)=={'format','theta','cell_id','accountant_fixed_cell_only','private_tuning_accounted'}
    assert 'noise_seed' not in payload and 'participation_seed' not in payload
    assert 'protocol_sha256' not in payload
    weights['w'].fill_(math.nan)
    assert torch.isfinite(payload['theta']['w']).all()
    assert payload['private_tuning_accounted'] is False
    with pytest.raises(C.Refused):C.model_only_payload(weights, C.select_cell('A'), 'a'*64)


def test_private_record_exclusive_and_mode600(tmp_path):
    p=tmp_path/'private.json';C.write_private(p,{'seed':'owned-synthetic'})
    assert p.stat().st_mode & 0o077 == 0
    before=p.read_bytes()
    with pytest.raises(FileExistsError):C.write_private(p,{'seed':'different'})
    assert p.read_bytes()==before


@pytest.mark.parametrize('flag', ['--role','--seed','--train-manifest','--epsilon','--init-from'])
def test_arbitrary_test_or_recipe_cli_unavailable(flag):
    with pytest.raises(SystemExit):
        C.main(['prepare','--protocol','/tmp/unused','--protocol-sha256','a'*64,'--cell','A','--out','/tmp/unused2',flag,'MAIN_TEST'])


def test_source_binding_excludes_other_independently_owned_extensions():
    sources=C.numerical_sources()
    own=[p.name for p in sources if 'experiment_extensions' in p.parts]
    assert own==['cold_dp_tuning.py','test_cold_dp_tuning.py']
    assert all(p.is_file() for p in sources)


def test_real_cpu_solver_consumes_all_client_rows_with_own_poisson_and_noise():
    # Synthetic only. Reuse the unchanged solver/runtime, not a wrapper-shaped
    # mock, and verify every sampled client's full eligible rows are consumed.
    sys.path.insert(0, str(ppsi_paths.code_dir('tests_fl')))
    from fl_testkit import tiny, one_client
    from fl.participation import ParticipationPlan
    from fl.runtime import FLRun
    from fl.server import Server
    from fl.numerics import state_digest
    from v4int import launch_fl as F
    spec=C.build_spec(C.select_cell('A'),mode='DP',clip=.2)
    cfg=F.fl_run_config(spec,'OWN_SYNTHETIC_COLD_CONTROL',8)
    adapter=tiny(3,tied=False)
    server=Server(adapter,init_sha256=state_digest(adapter.broadcast_state()))
    keys=[f'u{i:05}' for i in range(C.N)]
    reads=[]
    def get_client(k):
        value=one_client(k,13,seed=11,invalid_frac=0)
        assert len(value.examples['target_class'])==13
        reads.append(k)
        return value
    with C.private_randomness(participation_seed=771,noise_seed=881):
        plan=ParticipationPlan.build(keys,seed=2026,manifest_hash='synthetic-only',group_size=1024,sampling='poisson')
        run=FLRun(cfg,plan,server,get_client,C.N*13,workers=[tiny(9,tied=False)],
                  n_valid_by_key={k:13 for k in keys})
        result=run.step()
    assert len(reads)==len(plan.round_keys(0)) and len(reads)==len(set(reads))
    assert run.counters.exposures==len(reads)*13*2
    assert result['round']==0 and cfg.dp.denominator==1024
    assert all(torch.isfinite(t).all() for t in server.broadcast().values())
