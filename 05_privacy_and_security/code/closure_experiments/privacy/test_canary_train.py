"""Own actual candidate numerical and private identity controls, CPU only."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import ppsi_env  # noqa: F401,E402
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from . import canary_train as C
from .canary_replay import PrivateEntropy
from fl.client import ClientData, ItemLRSolver
from fl.numerics import enforce_strict_fp32, state_digest
from fl.synthetic import client_examples, make_tiny_adapter
from fl.participation import ParticipationPlan
from fl.runtime import FLRun, RunConfig
from fl.server import Server, ServerOptConfig


@pytest.fixture(autouse=True)
def strict():
    torch.set_num_threads(1)
    enforce_strict_fp32(warn_only=False)
    yield
    assert not torch.cuda.is_initialized()


def fixture(*, zero=True, dp=False):
    plan = ParticipationPlan.build(['toy::0','toy::1'], seed=2026,
        manifest_hash='EID_CANARY_SYNTHETIC_CONTROL', group_size=1, sampling='sweep')
    first, second = plan.round_keys(0)[0], plan.round_keys(1)[0]
    sizes = {first:3, second:0 if zero else 5}
    clients = {k:ClientData(k,client_examples(n,12,8,np.random.default_rng(31),invalid_frac=0)) for k,n in sizes.items()}
    from fl.dp import DPConfig
    cfg = RunConfig('CANARY_CANDIDATE_CPU_CONTROL', method='FA', seed=2026,
        lr_peak_local=.1, solver=ItemLRSolver(lr=.1,optimizer='sgd_m',passes=1,
            item_lr_mult=.1,weight_decay=0.,clip=1.,batch_size=16),
        exposure_point='end', endpoint_rounds=2, lr_shape='fl_const',
        dp=DPConfig(.2,.4,1) if dp else None,
        server_opt=ServerOptConfig('fedadam',lr=3e-5,tau=1e-6))
    adapter=make_tiny_adapter(K=12,d=4,tied=False,seed=11)
    worker=make_tiny_adapter(K=12,d=4,tied=False,seed=21)
    return cfg,plan,adapter,[worker],clients.__getitem__,sizes,13


def construct(values):
    cfg,plan,adapter,workers,get,counts,normalizer=values
    return FLRun(cfg,plan,Server(adapter,init_sha256=state_digest(adapter.broadcast_state())),get,
                 normalizer,workers=workers,n_valid_by_key=counts)


def test_actual_zero_round_preserves_nonzero_FedAdam_model_moments_and_RNG_advances_clock():
    run=construct(fixture())
    with C.non_dp_zero_policy():
        run.step()  # produce nonzero momentum using the original positive path
        assert run.server.server_opt.steps==1
        theta=run.server.adapter.broadcast_state(clone=True)
        moments=run.server.server_opt.state_dict()
        ambient=torch.get_rng_state().clone()
        before=run.counters.exposures
        result=run.step()
    assert run.state.cursor==2 and run.state.done and len(run.lr_log)==2
    assert run.counters.exposures==before==3 and run.counters.noop_visits==1
    assert all(torch.equal(theta[k],v) for k,v in run.server.adapter.broadcast_state().items())
    assert run.server.server_opt.steps==moments['steps']==1
    for name in ('m','v'):
        assert all(torch.equal(v,getattr(run.server.server_opt,name)[k]) for k,v in moments[name].items())
    assert torch.equal(ambient,torch.get_rng_state())
    assert result['weight']==0 and result['train_loss']['n_steps']==0
    assert run.ledger.messages==4 and run.ledger.by_round[1]['messages']==2
    assert run.ledger.by_round[1]['up']>0 and run.ledger.by_round[1]['down']>0


def test_actual_bitflip_zeros_are_consumed_zeros_same_fixed_plan_T_and_LR():
    absent=C.run_kernel(*fixture(zero=True))
    included=C.run_kernel(*fixture(zero=False))
    assert absent.plan.digest()==included.plan.digest() and absent.cfg.digest()==included.cfg.digest()
    assert absent.state.cursor==included.state.cursor==2 and absent.lr_log==included.lr_log
    assert [r[4] for r in absent.lr_table['rows']]==[r[4] for r in included.lr_table['rows']]
    assert absent.n_decisions_total==included.n_decisions_total==13
    assert absent.counters.exposures==3 and included.counters.exposures==8
    assert state_digest(absent.server.broadcast())!=state_digest(included.server.broadcast())


def test_positive_numerics_bitwise_equal_to_unchanged_original_path():
    original=construct(fixture(zero=False))
    original.run()
    candidate=C.run_kernel(*fixture(zero=False))
    assert state_digest(original.server.broadcast())==state_digest(candidate.server.broadcast())
    assert original.server.server_opt.digest()==candidate.server.server_opt.digest()
    assert original.server.server_opt.steps==candidate.server.server_opt.steps==2
    assert original.lr_log==candidate.lr_log and original.counters.summary()==candidate.counters.summary()


def test_zero_policy_uses_original_client_visits_and_virtual_uploads(monkeypatch):
    from fl import server as S
    original=S.visit_with_retry
    seen=[]
    def visit(*args,**kwargs):
        result=original(*args,**kwargs)
        seen.append((result.key,result.n_valid,result.n_consumed,result.steps))
        return result
    monkeypatch.setattr(S,'visit_with_retry',visit)
    C.run_kernel(*fixture())
    assert len(seen)==2 and seen[0][1:]==(3,3,1) and seen[1][1:]==(0,0,0)


def test_context_restores_exact_runtime_finalize_and_instance_apply_on_success_and_failure(monkeypatch):
    from fl import runtime,server as S
    rt,finalize=runtime.run_round,S.finalize
    run=construct(fixture())
    with C.non_dp_zero_policy():
        run.step();run.step()
    assert runtime.run_round is rt and S.finalize is finalize and 'apply' not in run.server.__dict__
    run=construct(fixture())
    real=S.visit_with_retry
    def fail(*a,**kw): raise RuntimeError('injected original-driver failure')
    with C.non_dp_zero_policy():
        run.step()
        monkeypatch.setattr(S,'visit_with_retry',fail)
        with pytest.raises(RuntimeError):run.step()
        monkeypatch.setattr(S,'visit_with_retry',real)
    assert runtime.run_round is rt and S.finalize is finalize and 'apply' not in run.server.__dict__


def test_nested_or_external_runtime_patch_refused(monkeypatch):
    from fl import runtime
    with C.non_dp_zero_policy():
        with pytest.raises(C.Refused):
            with C.non_dp_zero_policy():pass
    monkeypatch.setattr(runtime,'run_round',lambda *a,**k:None)
    with pytest.raises(C.Refused):
        with C.non_dp_zero_policy():pass


@pytest.mark.parametrize('field,value',[('mu',.1),('pf',object()),('store',object()),
                                       ('upload_observer',object()),('fault',object())])
def test_zero_extension_cannot_silently_admit_personal_observed_or_faulty_round(field,value):
    from fl import runtime
    run=construct(fixture())
    zero=[run.get_client(run.plan.round_keys(1)[0])]
    with C.non_dp_zero_policy():
        with pytest.raises(C.Refused):
            runtime.run_round(run.server,run.workers,zero,run.cfg.solver,round_idx=1,seed=2026,**{field:value})


def test_DP_zero_clients_and_empty_cohorts_never_use_nonDP_skip(monkeypatch):
    from fl import runtime
    from fl.dp import DPConfig
    values=list(fixture(zero=True))
    cfg,plan,adapter,workers,get,counts,normalizer=values
    from dataclasses import replace
    cfg=replace(cfg,endpoint_rounds=32,dp=DPConfig(.2,.4,1))
    plan=ParticipationPlan.build(plan.keys,seed=771,manifest_hash='EID_CANARY_SYNTHETIC_CONTROL',
                                 group_size=1,sampling='poisson')
    original_step=FLRun.step
    zero=[]
    def capture(run):
        before=run.server.digest();out=original_step(run)
        if out['weight']==0:zero.append((out,before,run.server.digest()))
        return out
    monkeypatch.setattr(FLRun,'step',capture)
    run=C.run_kernel(cfg,plan,adapter,workers,get,counts,normalizer,entropy=PrivateEntropy(771,881))
    assert zero and any(x[0]['dp'].get('noise_only') for x in zero)
    assert any(x[0]['dp'].get('n_noop',0)>0 for x in zero)
    assert all(x[1]!=x[2] and x[0]['dp']['denominator']==1 for x in zero)
    assert run.server.server_opt.steps==32 and run.state.done


def test_zero_round_preserves_python_numpy_and_torch_rng():
    import random
    from fl.checkpoint import rng_state,payload_digest
    run=construct(fixture())
    with C.non_dp_zero_policy():
        run.step()
        random.seed(9181);np.random.seed(775);torch.manual_seed(41)
        before=payload_digest(rng_state())
        run.step()
    assert payload_digest(rng_state())==before


def test_concurrent_thread_cannot_enter_scoped_zero_policy():
    import threading
    from fl import runtime
    run=construct(fixture())
    zero=[run.get_client(run.plan.round_keys(1)[0])]
    failures=[]
    def call():
        try:runtime.run_round(run.server,run.workers,zero,run.cfg.solver,round_idx=1,seed=2026)
        except Exception as error:failures.append(error)
    with C.non_dp_zero_policy():
        thread=threading.Thread(target=call);thread.start();thread.join()
    assert len(failures)==1 and isinstance(failures[0],C.Refused)
    assert run.server.server_opt.steps==0


def test_frozen_real_kernel_keeps_fixed_tables_but_trains_encoder():
    from fl.runtime import freeze_item_tables,FROZEN_ITEM_KEYS
    cfg,plan,adapter,workers,get,counts,normalizer=fixture(zero=False)
    from dataclasses import replace
    from fl.client import LocalSolver
    cfg=replace(cfg,solver=LocalSolver(lr=.1,optimizer='sgd_m',passes=1,clip=1.,weight_decay=0.,batch_size=16))
    freeze_item_tables(adapter)
    before=adapter.broadcast_state(clone=True)
    for worker in workers:freeze_item_tables(worker,source=before)
    run=C.run_kernel(cfg,plan,adapter,workers,get,counts,normalizer)
    after=adapter.broadcast_state()
    assert all(torch.equal(before[k],after[k]) for k in FROZEN_ITEM_KEYS)
    assert any(not torch.equal(before[k],after[k]) for k in adapter.manifest.shared_keys)
    assert all(k not in adapter.manifest.shared_keys for k in FROZEN_ITEM_KEYS)
    assert run.state.done


def test_private_checkpoints_and_monitor_do_not_change_model_or_rng(tmp_path):
    from fl.checkpoint import load_checkpoint,rng_state,payload_digest
    original=C.run_kernel(*fixture(zero=False))
    with C.private_umask():
        checkpoint,progress=C.private_monitor(tmp_path,'UNFROZEN')
        actual=C.run_kernel(*fixture(zero=False),checkpoint=checkpoint,progress=progress)
    assert state_digest(original.server.broadcast())==state_digest(actual.server.broadcast())
    assert original.server.server_opt.digest()==actual.server.server_opt.digest()
    assert original.lr_log==actual.lr_log and original.counters.summary()==actual.counters.summary()
    latest=load_checkpoint(tmp_path/'PRIVATE_CHECKPOINT/latest.pt')
    assert latest['cursor']==2
    assert (tmp_path/'PRIVATE_PROGRESS.jsonl').stat().st_mode&0o777==0o600
    assert (tmp_path/'PRIVATE_CHECKPOINT/latest.pt').stat().st_mode&0o777==0o600


@pytest.mark.parametrize('mode',C.MODES)
def test_model_only_release_is_complete_independent_copy_without_private_metadata(tmp_path,mode):
    adapter=make_tiny_adapter(K=12,d=4,tied=False,seed=28)
    payload=C.model_only_payload(adapter,mode,{'epsilon':7.9,'N':33768})
    assert set(payload)=={'format','theta','mechanism','private_tuning_accounted','privacy_certified'}
    assert set(payload['mechanism'])=={'mode','scope','rounds','local_passes','lr_shape','fixed_population',
        'fixed_denominator','fixed_cell_accountant'}
    assert all(v.device.type=='cpu' and v.data_ptr()!=adapter.broadcast_state()[k].data_ptr() for k,v in payload['theta'].items())
    path=tmp_path/'model.pt';torch.save(payload,path)
    other=make_tiny_adapter(K=12,d=4,tied=False,seed=47)
    C.load_model_only(path,mode,other,{'epsilon':7.9,'N':33768})
    assert state_digest(other.broadcast_state())==state_digest(adapter.broadcast_state())
    payload['private_seed']=2026;torch.save(payload,path)
    with pytest.raises(C.Refused):C.load_model_only(path,mode,other,{'epsilon':7.9,'N':33768})


@pytest.mark.parametrize('corruption',['nan','fp64','shape','missing','mode','accountant'])
def test_bad_model_tensor_or_mechanism_cannot_be_scored(tmp_path,corruption):
    adapter=make_tiny_adapter(K=12,d=4,tied=False,seed=28)
    payload=C.model_only_payload(adapter,'DP',{'epsilon':7.9})
    key=next(iter(payload['theta']))
    if corruption=='nan':payload['theta'][key].flatten()[0]=float('nan')
    elif corruption=='fp64':payload['theta'][key]=payload['theta'][key].double()
    elif corruption=='shape':payload['theta'][key]=payload['theta'][key].flatten()[:1]
    elif corruption=='missing':del payload['theta'][key]
    elif corruption=='mode':payload['mechanism']['mode']='UNFROZEN'
    else:payload['mechanism']['fixed_cell_accountant']['epsilon']=9.
    path=tmp_path/'bad.pt';torch.save(payload,path)
    with pytest.raises(C.Refused):C.load_model_only(path,'DP',adapter,{'epsilon':7.9})


@pytest.mark.parametrize('role',['MAIN','TEST','CALIBRATION','CALIB_VAL_5_100_8192','MAIN_TRAIN_5_100_131072'])
def test_no_nonTRAIN_CALIB_role_admission(role):
    with pytest.raises(C.Refused):C.exact_role(role)


def test_actual_configs_use_new_fixed_denominators_and_independent_calibration():
    from .canary_replay import dp_accountant
    a=C.make_config('FROZEN');b=C.make_config('UNFROZEN');d=C.make_config('DP',clip=.137)
    c=C.make_config('S_CAL')
    assert a.endpoint_rounds==b.endpoint_rounds==3518==int(np.ceil(60*33768/576))
    assert a.solver.passes==b.solver.passes==1 and a.dropout_p==b.dropout_p==.1
    assert type(a.solver).__name__=='LocalSolver' and type(b.solver).__name__=='ItemLRSolver'
    assert d.solver.passes==c.solver.passes==2 and d.endpoint_rounds==c.endpoint_rounds==384
    assert d.dp.denominator==1024 and d.dp.noise_multiplier==dp_accountant()['z']
    assert c.dp.clip_norm is None and c.dp.noise_multiplier==0
    assert dp_accountant()['N']==33768 and dp_accountant()['q']==1024/33768
    assert dp_accountant()['achieved_epsilon']<=8 and dp_accountant()['conversion']=='standard (Mironov 2017 Prop. 3)'


@pytest.mark.parametrize('clip',[None,0,-1,float('nan'),float('inf'),True])
def test_DP_cannot_reuse_missing_or_nonpositive_clip(clip):
    with pytest.raises(C.Refused):C.make_config('DP',clip=clip)


def test_tail_audit_is_lower_bound_with_full_nine_test_family():
    scores=np.arange(1000,dtype=float)
    truth=np.where(np.arange(1000)<500,-1,1).astype(np.int8)
    result=C.audit_fixed_scores(scores,truth)
    assert result['bound_type']=='LOWER_BOUND_ONLY' and result['simultaneous_family']==9
    assert result['privacy_certified'] is False
    assert [(r['k_positive'],r['k_negative']) for r in result['results']]==list(C.TAIL_RULES)
    assert all(r['interpretation'] in ('DETECTED_CONTRADICTION_AT_EPS8','NO_DETECTED_CONTRADICTION_NOT_PRIVACY_PASS') for r in result['results'])
    assert all(r['n_predeclared_tests']==9 for r in result['results'])


def test_review_separately_admits_modes_and_exact_fresh_DP_calibration(tmp_path,monkeypatch):
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    prep=tmp_path/'prep.json';prep.write_text('{}')
    clip=tmp_path/'clip.json';clip.write_text('{"S":1}')
    monkeypatch.setattr(C,'pin',lambda path:{'path':str(path),'sha256':C.sha(path)})
    record=dict(status='PASS',scope='CANARY_CALIB_ACTUAL_TRAIN',reviewer='/root/independent_reviewer',
        protocol=C.pin(protocol),preparation=C.pin(prep),module_sha256=C.sha(C.__file__),
        tests_sha256=C.sha(__file__),GPU_control='PASS',real_training_allowed=True,allowed_modes=['FROZEN','S_CAL'])
    review=tmp_path/'review.json';review.write_text(C.json.dumps(record))
    C.review_gate(review,protocol,prep,mode='FROZEN')
    with pytest.raises(C.Refused):C.review_gate(review,protocol,prep,mode='DP',clip_path=clip)
    record['allowed_modes'].append('DP');review.write_text(C.json.dumps(record))
    with pytest.raises(C.Refused):C.review_gate(review,protocol,prep,mode='DP',clip_path=clip)
    record['fresh_calibration']=C.pin(clip);review.write_text(C.json.dumps(record))
    C.review_gate(review,protocol,prep,mode='DP',clip_path=clip)
    clip.write_text('{"S":2}')
    with pytest.raises(C.Refused):C.review_gate(review,protocol,prep,mode='DP',clip_path=clip)
    for author in ('/root/pretrain_finalize','/root/canary_finish'):
        record['reviewer']=author;review.write_text(C.json.dumps(record))
        with pytest.raises(C.Refused):C.review_gate(review,protocol,prep,mode='FROZEN')


def test_calibration_entropy_file_cannot_contain_membership_bits(tmp_path,monkeypatch):
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    prep=tmp_path/'prep.json';prep.write_text('{}')
    monkeypatch.setattr(C,'pin',lambda path:{'path':str(path),'sha256':C.sha(path)})
    record=dict(format='EID_CANARY_CALIBRATION_ENTROPY_1',protocol=C.pin(protocol),preparation=C.pin(prep),
        entropy=dict(participation_seed=31,noise_seed=42))
    path=tmp_path/'calibration.json';path.write_text(C.json.dumps(record));path.chmod(0o600)
    assert C.load_calibration_state(path,protocol,prep)==PrivateEntropy(31,42)
    record['signs']=[1,-1];path.write_text(C.json.dumps(record))
    with pytest.raises(C.Refused):C.load_calibration_state(path,protocol,prep)
    del record['signs'];path.write_text(C.json.dumps(record));path.chmod(0o644)
    with pytest.raises(C.Refused):C.load_calibration_state(path,protocol,prep)


@pytest.fixture
def rich_adapter():
    from v4models import build
    from v4models.derived import load_derived_schema
    schema=load_derived_schema(C.WORK/'contracts/derived_schemas/L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7.json')
    return build('SASREC',schema,2026,'cpu',K=12,variant=C.VARIANT,dropout=(.1,)).adapter


def definitions():
    from .canary_data import CanaryDefinitions
    return CanaryDefinitions(np.array([[0,1,2,3],[4,5,6,7]],dtype=np.int64),np.array([2,8],dtype=np.int64),12,50)


def test_actual_rich_architecture_requires_declared_context_and_accepts_completed_forward(rich_adapter):
    from v4models.schema import check_batch_layout
    b=definitions().batch(np.array([0,1]))
    with pytest.raises(ValueError):rich_adapter.scores(b)
    full=C.complete_synthetic_context(b,rich_adapter)
    check_batch_layout(full,rich_adapter.input_widths)
    projected=rich_adapter.project(full)
    assert projected['event_numeric_features'].shape==(2,50,7)
    assert projected['event_quality_flags'].shape==(2,50,2)
    assert projected['user_context'].shape==(2,6) and projected['user_context_masks'].shape==(2,4)
    scores=C.rich_full_catalogue_logprob(rich_adapter,full)
    assert scores.shape==(2,) and torch.isfinite(scores).all()
    assert rich_adapter.module.cfg.rich and rich_adapter.module.side_enabled
    assert rich_adapter.module.has_side_branch and rich_adapter.module.has_user_branch


def test_rich_context_distinguishes_missing_unknown_and_padding_without_using_target(rich_adapter):
    b=definitions().batch(np.array([0,1]))
    out=C.complete_synthetic_context(b,rich_adapter)
    changed=dict(b,target_class=b['target_class'].flip(0))
    other=C.complete_synthetic_context(changed,rich_adapter)
    for key in out:
        if key!='target_class':
            assert torch.equal(out[key],other[key]) if isinstance(out[key],torch.Tensor) else out[key]==other[key]
    mask=b['attention_mask'].bool()
    for key in ('category_tokens','brand_tokens','session_mode_main_category_tokens','session_offset_tokens'):
        assert torch.all(out[key][mask]==1) and torch.all(out[key][~mask]==0)
    for flag in C.synthetic_context_declaration()['valid_missing_flags']:
        col=out['event_quality_flags'][...,rich_adapter.input_widths.flag_names.index(flag)]
        assert torch.equal(col,mask.float())
    assert not torch.any(out['event_numeric_features']) and not torch.any(out['user_context_masks'])
    assert set(b)==set(definitions().batch(np.array([0,1])))  # input was not enriched in place


def test_rich_scorer_full_catalogue_denominator_and_no_label_forward(rich_adapter,monkeypatch):
    from fl.checkpoint import rng_state,payload_digest
    b=C.complete_synthetic_context(definitions().batch(np.array([0,1])),rich_adapter)
    flags=[m.training for m in rich_adapter.module.modules()]
    before=state_digest(rich_adapter.broadcast_state()),payload_digest(rng_state())
    original=rich_adapter.scores
    calls=[]
    def capture(inputs):
        assert not {'target_class','loss_mask','signs','included','private_signs'}&set(inputs)
        calls.append(set(inputs))
        return original(inputs)
    monkeypatch.setattr(rich_adapter,'scores',capture)
    actual=C.rich_full_catalogue_logprob(rich_adapter,b)
    rich_adapter.module.eval()
    with torch.inference_mode():
        logits=original({k:v for k,v in b.items() if k not in ('target_class','loss_mask')})
        expected=torch.log_softmax(logits,dim=1).gather(1,b['target_class'][:,None]).squeeze(1)
    for module,flag in zip(rich_adapter.module.modules(),flags):module.training=flag
    assert calls and torch.allclose(actual,expected,atol=1e-6,rtol=0)
    assert before==(state_digest(rich_adapter.broadcast_state()),payload_digest(rng_state()))
    assert flags==[m.training for m in rich_adapter.module.modules()]


def test_rich_completion_shared_between_actual_synthetic_visit_and_scoring(rich_adapter):
    from .canary_data import FixedRosterCanaries
    from fl.client import client_update
    from fl.runtime import freeze_item_tables
    raw=FixedRosterCanaries([],lambda _:None,definitions(),np.array([1,-1],dtype=np.int8),
        train_manifest_id=C.TRAIN,repeats=10)
    roster=C.RichFixedRoster(raw,rich_adapter,real_keys=[])
    included=roster.get_client(roster.keys[0]);absent=roster.get_client(roster.keys[1])
    from fl.client import valid_rows
    assert valid_rows(included.examples).numel()==10 and valid_rows(absent.examples).numel()==0
    before=rich_adapter.broadcast_state(clone=True)
    result=client_update(rich_adapter,before,included,C.make_config('UNFROZEN').solver,round_idx=0,seed=2026)
    assert result.n_valid==10 and result.n_consumed==10 and result.steps==1
    assert rich_adapter.module.cfg.rich and rich_adapter.module.side_enabled


def _all_correct_tail_scalar(epsilon,m,r,delta):
    # Independent scalar recurrence, no scipy CDF/tail or production statistic.
    import math
    p=1/(1+math.exp(-epsilon))
    probabilities=[0.]*(r+1)
    probabilities[r]=p**r
    for k in range(r,0,-1):probabilities[k-1]=probabilities[k]*k/(r-k+1)*(1-p)/p
    mass=0.;maximum=0.
    for i in range(1,r+1):
        mass+=probabilities[r-i]
        maximum=max(maximum,2*mass/i)
    return min(1.,probabilities[r]+delta*m*maximum)


@pytest.mark.parametrize('row_index',range(3))
def test_ceiling_matches_independent_binomial_recurrence_and_bracket(row_index):
    from .canary_statistics import audit_tail_bound
    row=C.attainable_bound_ceiling()['rows'][row_index]
    eps=row['maximum_attainable_epsilon_lower_bound'];r=row['guesses'];threshold=(1-.95)/9
    assert abs(_all_correct_tail_scalar(eps,1000,r,5e-6)-threshold)<1e-12
    assert _all_correct_tail_scalar(eps-1e-6,1000,r,5e-6)<threshold
    assert _all_correct_tail_scalar(eps+1e-6,1000,r,5e-6)>threshold
    assert abs(_all_correct_tail_scalar(8.,1000,r,5e-6)-row['all_correct_null_tail_at_eps8'])<1e-12
    assert row['all_correct_null_tail_at_eps8']>threshold and eps<8


@pytest.mark.parametrize('epsilon',[0.,1.,4.,8.])
@pytest.mark.parametrize('r',[200,500,1000])
def test_null_bound_is_monotone_over_every_possible_correct_count(epsilon,r):
    from .canary_statistics import audit_tail_bound
    bounds=[audit_tail_bound(epsilon,5e-6,1000,r,v) for v in range(r+1)]
    assert all(first+1e-12>=second for first,second in zip(bounds,bounds[1:]))


def test_ceiling_receipt_is_preoutcome_and_cannot_be_mutated_via_cached_dict():
    first=C.attainable_bound_ceiling();first['rows'][0]['maximum_attainable_epsilon_lower_bound']=999
    again=C.attainable_bound_ceiling()
    assert again['rows'][0]['maximum_attainable_epsilon_lower_bound']<8
    assert again['eps8_rejection_possible'] is False
    assert again['n_canaries']==1000 and again['simultaneous_tests']==9
    assert C.declaration()['statistical_power']==again
