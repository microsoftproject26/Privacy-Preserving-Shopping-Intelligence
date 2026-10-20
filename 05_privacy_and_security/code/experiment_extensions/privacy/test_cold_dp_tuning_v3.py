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

from . import cold_dp_tuning_v3 as C


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
    assert own==['cold_dp_tuning_v3.py','test_cold_dp_tuning_v3.py']
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


@pytest.fixture(scope='module')
def real_calib_inputs():
    """Actual TRAIN/CALIB providers + production SMALL cold initializer; no scoring."""
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    cell=C.select_cell('A')
    value=C._inputs(cell,C.build_spec(cell,mode='S_CAL'),-1,device='cpu')
    assert not torch.cuda.is_initialized()
    return value


@pytest.fixture(scope='module')
def real_calib_harness(real_calib_inputs,tmp_path_factory):
    return C.CalibrationInferenceHarness(real_calib_inputs,gpu=-1,device='cpu',
        run_dir=tmp_path_factory.mktemp('true_calib_inference_view'))


def legacy_invalid_config(inputs):
    from v4train.recipe import RunConfig
    from v4int import launch_fl as F
    small=inputs['small'];lock=inputs['lock']
    return RunConfig(run_id='OWN_CPU_REPRODUCE_INVALID_OLD_CALIB',family=F.FAMILY,seed=2026,
        exposure_point='end',cuda_visible_devices='-1',phase='CALIBRATION',population='COHORT',
        peak_lr=small['peak_lr'],key=inputs['key'],test_overrides=False,determinism_mode='STRICT',
        manifest_registry=str(lock['manifest_registry_path']),data_version=lock['data_version'],
        schema_registry=str(lock['schema_registry_path']),recipe_id=small['recipe_id'],
        recipe_registry=small['recipe_registry'],model_variant=small['model_variant'],
        dropout=tuple(small['dropout']),schedule=small['schedule'],theta0_init_sha256=inputs['theta0'])


def test_actual_production_RunConfig_and_CentralTrainer_reproduce_old_failure_before_cuda(real_calib_inputs,tmp_path):
    from v4train.recipe import RecipeError
    from v4train.trainer import CentralTrainer
    cfg=legacy_invalid_config(real_calib_inputs)
    with pytest.raises(RecipeError,match='MAIN-phase COHORT'):
        cfg.validate()
    with pytest.raises(RecipeError,match='MAIN-phase COHORT'):
        CentralTrainer(cfg,real_calib_inputs['schema'],real_calib_inputs['tp'],real_calib_inputs['ep'],
                       tmp_path/'not_created',device='cpu',v1_root=str(C.DATA))
    assert not (tmp_path/'not_created').exists() and not torch.cuda.is_initialized()


def test_null_recipe_alone_is_not_a_repair(real_calib_inputs):
    from dataclasses import replace
    from v4train.recipe import RecipeError
    cfg=replace(legacy_invalid_config(real_calib_inputs),recipe_id=None,recipe_registry=None)
    with pytest.raises(RecipeError,match='peak LR|model variant|dropout'):
        cfg.validate()


def test_true_actual_CALIB_inference_contract_and_independent_server_storage(real_calib_harness,real_calib_inputs):
    from fl.numerics import state_digest
    h=real_calib_harness; worker=real_calib_inputs['built']
    assert h.train.manifest_id==C.TRAIN and h.evalp.manifest_id==C.EVAL
    assert len(real_calib_inputs['keys'])==32768 and h.N==646067
    assert h.schema.self_sha256==real_calib_inputs['lock']['schema_self_sha256']
    assert h.adapter is not worker.adapter and h.module is not worker.module
    assert state_digest(h.adapter.broadcast_state())==state_digest(worker.adapter.broadcast_state())==real_calib_inputs['theta0']
    for (sn,server),(wn,client) in zip(h.module.named_parameters(),worker.module.named_parameters()):
        assert sn==wn and torch.equal(server,client) and server.data_ptr()!=client.data_ptr()
    assert h.cfg.recipe_id is None and h.cfg.recipe_registry is None
    assert not hasattr(h,'opt') and not hasattr(h,'planned') and not hasattr(h,'run')
    assert h.key_base['train_manifest_id']==C.TRAIN and h.key_base['eval_manifest_id']==C.EVAL
    assert h.key_base['run_role']==h.key_base['manifest_role']=='CALIBRATION'
    assert h.key_base['training_paradigm']=='FEDERATED'
    assert h.key_base['optimizer_lifecycle']!='PERSISTENT_ADAMW'
    assert h.key_base['source_small_recipe_id']==real_calib_inputs['small']['recipe_id']
    assert h.registry_source==C.pin(real_calib_inputs['lock']['manifest_registry_path'])
    assert h.registry_view==C.pin(h.registry_view['path'])
    assert h.registry.self_sha256!=h.registry_source['sha256']
    original=C.read(h.registry_source['path']);view=C.read(h.registry_view['path'])
    assert view['roles']['CALIBRATION']==original['roles']['CALIBRATION']
    assert all(view['manifests'][mid]==original['manifests'][mid] for mid in view['manifests'])
    assert view['derived_from']['self_sha256']==original['self_sha256']
    assert view['derived_from']['file_sha256']==h.registry_source['sha256']
    assert h.frozen.score_block==158486 and h.frozen.class_chunk==32768 and h.frozen.eval_batch==1024
    assert h.eval_device_class=='CPU-FP32-DET' and not torch.cuda.is_initialized()


@pytest.mark.parametrize('mutation',['comparison_key','small_provenance','schema_pin','visibility'])
def test_real_CALIB_metadata_mutants_fail_closed_before_new_model(real_calib_inputs,mutation,tmp_path):
    value=dict(real_calib_inputs)
    if mutation=='comparison_key':value['key']=dict(value['key'],split='TEST')
    if mutation=='small_provenance':value['small']=dict(value['small'],dropout=[0.2])
    if mutation=='schema_pin':value['lock']=dict(value['lock'],schema_self_sha256='0'*64)
    gpu=0 if mutation=='visibility' else -1
    with pytest.raises(C.Refused):C.CalibrationInferenceHarness(value,gpu=gpu,device='cpu',run_dir=tmp_path/'view')
    assert not torch.cuda.is_initialized()


def test_real_model_frozen_inference_equals_source_CentralTrainer_evaluation_and_fullK_oracle(real_calib_harness,monkeypatch):
    """Synthetic six-row manifest; never score actual CALIB/MAIN/TEST decisions.

    The reference calls the inherited _evaluate function on evaluation-only state.
    It does NOT claim that the incompatible central constructor was admitted.
    """
    import numpy as np
    from types import SimpleNamespace
    from v4models.synthetic import synthetic_examples_v2
    from v4eval.manifest import build_eval_manifest
    from v4train.trainer import CentralTrainer
    from v4int import launch_fl as F
    from fl.numerics import state_digest
    h=real_calib_harness
    batch=synthetic_examples_v2(6,158486,h.adapter.input_widths,torch.Generator().manual_seed(417),
        max_len=3,min_len=1,with_names=True,with_session_offset=True,with_session_intent=True)
    ids=np.arange(6,dtype=np.int64)+100
    truth=np.array([0,1,2,-1,-1,-1],dtype=np.int64)
    manifest=build_eval_manifest({'manifest_id':'SYNTHETIC_CALIB_REPAIR_ONLY','v4_split':'VALIDATION','catalogue_K':158486},{
        'decision_id':ids,'eval_user_key':np.array([10,10,20,20,30,40]),
        'eval_mask':np.array([1,1,1,1,1,0],bool),'target_class':truth,
        'target_oov':np.array([0,0,0,1,1,0],bool),'censor_class':np.array(['NONE']*5+['CEN_NO_NEXT'])})
    batch['decision_id']=torch.as_tensor(ids)
    batch['target_class']=torch.as_tensor(truth);batch['label']='SYNTHETIC_TRUTH_NEVER_QUERY'
    query={k:v for k,v in batch.items() if k not in {'target','targets','label','labels','target_class','target_oov','loss_mask'}}
    original=h.adapter.query
    seen=[]
    def spy(value):
        assert not set(value)&{'target','targets','label','labels','target_class','target_oov','loss_mask'}
        seen.append(True);return original(value)
    monkeypatch.setattr(h.adapter,'query',spy)
    digest_before=state_digest(h.adapter.broadcast_state())
    for force_ties in (False,True):
        if force_ties:
            W=h.adapter.head_weight();bias=h.adapter.head_bias()
            saved_W=W.detach().clone();saved_bias=bias.detach().clone()
            with torch.no_grad():W.zero_();bias.zero_()
        try:
            h.module.eval()
            with torch.no_grad():
                logits=h.adapter.logits(h.adapter.query(query))
                assert logits.shape==(6,158486)
                classes=torch.arange(158486)
                dense=[1+int((logits[i]>logits[i,t]).sum())+int(((logits[i]==logits[i,t])&(classes<t)).sum()) for i,t in enumerate(truth[:3])]
            monkeypatch.setattr(h,'evalp',SimpleNamespace(manifest=manifest,batches=lambda n:iter([batch])))
            out=h._evaluate(0.5)
            monkeypatch.setattr(h,'evalp',SimpleNamespace(manifest=manifest,batches=lambda n:iter([query])))
            # Explicit evaluation-only reference state. The original central
            # constructor remains correctly rejected; no optimizer/run exists.
            reference_state=copy.copy(h)
            reference_state._eval_override=None
            reference=CentralTrainer._evaluate(reference_state,0.5)
            assert np.array_equal(out.ranks,reference.ranks) and out.result==reference.result and out.rows==reference.rows
            assert out.ranks[:3].tolist()==dense and out.ranks[3:].tolist()==[-1,-1,-1]
            rr=[1/r if r<=20 else 0 for r in dense]
            assert out.value==pytest.approx(((rr[0]+rr[1])/2+rr[2]/2+0)/3)
            assert out.result['E2E']['n_decisions']==5 and out.result['RANKABLE']['n_decisions']==3 and out.result['oov']['oov_count']==2
            assert all(r['status']=='REHEARSAL_NON_FROZEN_DEVICE' and r['selection_rule'] is None for r in out.rows)
            assert all(r['run_role']=='CALIBRATION' and r['row_status']=='CALIBRATION_INFERENCE_ONLY' and not r['c_matched_eligible'] for r in out.rows)
            converted=F.fl_rows(out.rows,arm='DP',run_id='SYNTHETIC_ONLY',efe_raw=.5,mark=.5,digest='a'*64,role='CALIBRATION')
            assert all(r['run_role']=='CALIBRATION' and r['row_status']=='FL_LR_SCREEN' and r['selection_rule'] is None and r['status']=='REHEARSAL_NON_FROZEN_DEVICE' for r in converted)
            spec=C.build_spec(C.select_cell('A'),mode='DP',clip=.2)
            with C.execution_metadata(spec,{'synthetic_only':True}):
                labels=dict(F.row_labels(F.spec_tuning(spec)),**F.poisson_row_labels(spec))
                final_rows=F.fl_rows(out.rows,arm='DP',run_id='SYNTHETIC_ONLY',efe_raw=.5,mark=.5,
                                     digest='a'*64,role='CALIBRATION',tuning_labels=labels)
            assert all(r['fl_client_opt']=='sgd_m' and r['optimizer_lifecycle']=='FRESH_SGD_M_PER_VISIT' for r in final_rows)
            if force_ties:
                assert dense==[1,2,3] and out.result['tie_diagnostic']['max_n_tied']==158485
        finally:
            if force_ties:
                with torch.no_grad():W.copy_(saved_W);bias.copy_(saved_bias)
    assert seen and state_digest(h.adapter.broadcast_state())==digest_before
    assert not torch.cuda.is_initialized()


def test_all_fixed_specs_and_private_math_stay_identical_to_frozen_original():
    from . import cold_dp_tuning as original
    assert C.load_grid()==original.load_grid()
    for cell in C.load_grid()['cells']:
        assert C.build_spec(cell,mode='S_CAL')==original.build_spec(cell,mode='S_CAL')
        assert C.build_spec(cell,mode='DP',clip=.2)==original.build_spec(cell,mode='DP',clip=.2)
    import ast
    old=ast.parse(Path(original.__file__).read_text());new=ast.parse(Path(C.__file__).read_text())
    for name in ('build_spec','private_randomness','_inputs','model_only_payload','load_clip'):
        of=next(n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        nf=next(n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        assert ast.dump(of,include_attributes=False)==ast.dump(nf,include_attributes=False)


def test_old_protocol_and_preparations_are_not_v3_admission():
    import inspect
    assert C.inference_contract()['central_training'] is False
    assert C.inference_contract()['central_recipe_id'] is None
    assert 'PPSI_COLD_DP_CALIB_PROTOCOL_3' in inspect.getsource(C.load_protocol)
    assert 'fresh v3 true CALIB inference-only preparation required' in inspect.getsource(C.run)
    assert 'test_cold_dp_tuning_v3.py' in inspect.getsource(C.run)
    for version in ('v1','v2'):
        path=C.EXT/f'protocols/COLD_DP_CALIB_PROTOCOL_{version}.json'
        with pytest.raises(C.Refused,match='self hash/format'):
            C.load_protocol(path,C.sha(path))
