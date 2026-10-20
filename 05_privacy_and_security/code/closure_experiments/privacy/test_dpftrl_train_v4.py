"""Synthetic-only full client/server integration and release boundary controls."""
import copy
import math
import os
from pathlib import Path
import tempfile

import ppsi_env  # noqa: F401
import numpy as np
import pytest
import torch
from fl.client import ClientData, client_update
from fl.numerics import enforce_strict_fp32
from fl.synthetic import make_tiny_adapter, make_clients
from . import dpftrl_train_v4 as D
from . import dpftrl_tree as T


@pytest.fixture(autouse=True)
def strict():
    torch.set_num_threads(2)
    enforce_strict_fp32(warn_only=False)


def server(*, mechanism='tree', z=0., orders=None, post=None, theta=None, group=2):
    return D.PublicSweepServer(torch.zeros(2) if theta is None else theta, roster=('a','b','c'),
        orders=orders or [('a','b','c')]*2, mechanism=mechanism, eta=.1, noise_seed=927 if mechanism == 'tree' else None,
        noise_multiplier=z, group_size=group, postprocess=post)


def advance(server, visit):
    if server.mechanism == 'tree':
        return server.step(visit)
    messages = {}
    for key in server.next_group():
        client = D.LocalClientRandomizer(server.theta.shape, slot=key,
            private_key=1000 + server.roster.index(key), clip_norm=server.clip, noise_multiplier=server.z)
        messages[key] = client.randomize(visit(key), sweep=server.sweep)
    return server.step_messages(messages)


def test_predeclared_recipe_fixedS_no_private_calibration_and_P1():
    p=D.declaration()
    assert (p['flat_client_delta_clip'],p['clip_selection'],p['passes'],p['sweeps'],p['group_size'])==(1.,'FIXED_PUBLIC_NO_S_CAL',1,6,64)
    s=D.solver()
    assert (s.lr,s.optimizer,s.passes,s.batch_size,s.clip,s.item_lr_mult,s.weight_decay)==(.1,'sgd_m',1,16,1.,.1,0.)
    assert not p['TEST_allowed'] and not p['resume'] and not p['automatic_retry']


@pytest.mark.parametrize('mechanism,eta,role',[('tree',.3,'CALIBRATION'),('local',True,'CALIBRATION'),('fedadam',.1,'CALIBRATION'),('tree',.1,'MAIN'),('local',1.,'TEST')])
def test_arbitrary_cell_refused_before_data(mechanism,eta,role):
    with pytest.raises(D.Refused):D.select_cell(mechanism,eta,role=role)


def test_tree_vs_separate_local_six_visit_accountant_exact_oracle():
    from .dpftrl_statistics import rho_for_epsilon
    tree,local=D.budget('tree'),D.budget('local')
    rho=rho_for_epsilon(8.,5e-6)
    assert tree['rounds_per_sweep']==512 and tree['tree_depth_sum']==60
    assert tree['z']==pytest.approx(math.sqrt(60/(2*rho)),rel=1e-15)
    assert local['z']==pytest.approx(math.sqrt(6/(2*rho)),rel=1e-15)
    for b in (tree,local):
        assert b['rho']==pytest.approx(rho,rel=1e-15) and b['epsilon']==pytest.approx(8,abs=1e-13)
        assert b['replacement_rho_if_same_noise']==4*b['rho'] and not b['amplification_used'] and not b['privacy_certified']
    assert tree['z']/local['z']==pytest.approx(math.sqrt(10))


@pytest.mark.parametrize('n,group,sweeps',[(3,2,6),(64,64,6),(65,64,1),(1,64,6)])
def test_public_geometry_and_partial_groups_accounted(n,group,sweeps):
    b=D.budget('tree',n=n,group_size=group,sweeps=sweeps)
    assert b['rounds_per_sweep']==math.ceil(n/group)
    assert b['tree_depth_sum']==math.ceil(n/group).bit_length()*sweeps


def test_public_permutations_six_exact_roster_and_no_global_RNG():
    before=torch.get_rng_state().clone();npbefore=np.random.get_state()
    a=D.public_orders(tuple(str(i)for i in range(37)))
    assert a==D.public_orders(tuple(str(i)for i in range(37))) and len(a)==6
    assert all(len(o)==37 and len(set(o))==37 for o in a)
    assert torch.equal(before,torch.get_rng_state())
    assert npbefore[0]==np.random.get_state()[0] and np.array_equal(npbefore[1],np.random.get_state()[1])


@pytest.mark.parametrize('mechanism',['tree','local'])
def test_zero_noise_hand_prefix_fixed_tail_denominator_and_sweep_anchor(mechanism):
    s=server(mechanism=mechanism)
    values={'a':torch.tensor([3.,4.]),'b':None,'c':torch.tensor([-2.,0.])}
    torch.testing.assert_close(advance(s, values.get),torch.tensor([.03,.04]),rtol=0,atol=2e-9)
    torch.testing.assert_close(advance(s, values.get),torch.tensor([-.02,.04]),rtol=0,atol=4e-9)
    with pytest.raises(D.Refused):s.next_group()
    s.begin_sweep()
    torch.testing.assert_close(advance(s, values.get),torch.tensor([.01,.08]),rtol=0,atol=5e-9)


@pytest.mark.parametrize('mechanism',['tree','local'])
def test_zero_slot_and_actual_zero_upload_have_same_shaped_noise_law(mechanism):
    a,b=server(mechanism=mechanism,z=1.),server(mechanism=mechanism,z=1.)
    before=torch.get_rng_state().clone()
    for _ in range(2):
        torch.testing.assert_close(advance(a, lambda k:None),advance(b, lambda k:torch.zeros(2)),rtol=0,atol=0)
    assert torch.equal(before,torch.get_rng_state())
    assert not torch.equal(a.weights(),torch.zeros(2))


def test_streaming_tree_matches_frozen_independently_reviewed_nonstreaming_primitive():
    roster=tuple(str(i)for i in range(7));orders=D.public_orders(roster,sweeps=2)
    a=D.PublicSweepServer(torch.zeros(3),roster=roster,orders=orders,mechanism='tree',eta=.1,
        noise_seed=908,noise_multiplier=.7,group_size=3)
    b=T.ClientUpdateFTRL(torch.zeros(3),public_roster=roster,sweep_orders=orders,clip_norm=1.,
        noise_multiplier=.7,noise_seed=908,eta=.1,group_size=3)
    values={k:torch.tensor([float(i),2.,-1.])for i,k in enumerate(roster)}
    for epoch in range(2):
        if epoch:a.begin_sweep();b.begin_sweep()
        for _ in range(3):
            group=a.next_group()
            torch.testing.assert_close(a.step(values.get),b.step({k:values[k]for k in group}),rtol=0,atol=0)


@pytest.mark.parametrize('mechanism',['tree','local'])
@pytest.mark.parametrize('bad',[torch.tensor([1.]),torch.tensor([math.nan,0.]),torch.tensor([1,2]),torch.ones(2,2)])
def test_invalid_private_direction_no_broadcast_or_commit(mechanism,bad):
    s=server(mechanism=mechanism,z=1.);before=s.weights()
    with pytest.raises((ValueError,D.Refused,RuntimeError)):advance(s, lambda k:bad)
    assert s.round==0 and torch.equal(before,s.weights())
    if s.tree is not None:assert s.tree.count==0


def test_callback_failure_keeps_server_prefix_cursor_weights_unchanged():
    s=server(z=1.,post=lambda x:(_ for _ in ()).throw(RuntimeError('postfail')))
    before=s.weights()
    with pytest.raises(RuntimeError,match='postfail'):s.step(lambda k:torch.ones(2))
    assert s.round==0 and s.tree.count==0 and torch.equal(s.weights(),before)


def test_local_noise_is_independent_by_client_and_sweep_and_has_no_server_seed():
    shape = (32768,)
    ca = D.LocalClientRandomizer(shape, slot='a', private_key=981, clip_norm=1., noise_multiplier=1.)
    cb = D.LocalClientRandomizer(shape, slot='b', private_key=982, clip_norm=1., noise_multiplier=1.)
    a, b = ca.randomize(None, sweep=0), cb.randomize(None, sweep=0)
    assert not torch.equal(a,b)
    assert abs(torch.stack((a,b)).cov()[0,1].item()) < .03
    assert (a+b).var().item() == pytest.approx(2., abs=.06)
    assert not torch.equal(a, ca.randomize(None, sweep=1))
    s = server(mechanism='local', z=1.)
    assert not hasattr(s, 'seed') and not hasattr(s, 'local_noise')


@pytest.mark.parametrize('tied',[True,False])
def test_shared_vector_roundtrip_aliases_buffers_and_full_dtype(tied):
    adapter=make_tiny_adapter(K=7,d=4,tied=tied,dropout=0.,seed=8)
    layout=D.SharedVector(adapter);state=adapter.broadcast_state(clone=True)
    flat=layout.flatten(state);roundtrip=layout.state(flat)
    assert flat.dtype==torch.float32 and flat.numel()==adapter.manifest.shared_numel
    assert set(layout.keys).isdisjoint(adapter.manifest.buffer_keys)
    assert set(layout.keys).isdisjoint(adapter.manifest.alias_keys)
    for k in state:assert torch.equal(state[k],roundtrip[k])
    roundtrip[layout.keys[0]].fill_(123)
    assert not torch.equal(roundtrip[layout.keys[0]],state[layout.keys[0]])


@pytest.mark.parametrize('mechanism',['tree','local'])
def test_actual_client_kernel_tail_rows_private_counters_and_zero_noise_oracle(mechanism):
    clients=make_clients(3,K=7,L=3,seed=91,sizes=[17,5,0],invalid_frac=0.)
    roster=tuple(c.key for c in clients)+('absent',);lookup={c.key:c for c in clients}
    orders=[roster]*2
    a=make_tiny_adapter(K=7,d=4,tied=False,dropout=0.,seed=15)
    b=make_tiny_adapter(K=7,d=4,tied=False,dropout=0.,seed=15)
    layout=D.SharedVector(a);reference=D.SharedVector(b)
    before=torch.get_rng_state().clone();records=[]
    actual,counters=D.run_clients(a,layout,roster=roster,get_client=lookup.get,mechanism=mechanism,
        eta=.1,private_seed=1001 if mechanism == 'tree' else None,noise_multiplier=0.,orders=orders,group_size=3,
        local_client_seeds={k:200+i for i,k in enumerate(roster)} if mechanism == 'local' else None,
        on_round=lambda r,p,c:records.append((r,p,c)))
    server_ref=T.ClientUpdateFTRL(reference.flatten(b.broadcast_state()),public_roster=roster,sweep_orders=orders,
        clip_norm=1.,noise_multiplier=0.,noise_seed=1001,eta=.1,group_size=3,
        post_recenter=lambda v:recenter(b,reference,v))
    idx=0
    for epoch in range(2):
        if epoch:server_ref.begin_sweep()
        for _ in range(server_ref.rounds_per_sweep):
            theta=reference.state(server_ref.weights());anchor=reference.flatten(theta,dtype=torch.float64);idx+=1;updates={}
            for key in server_ref.next_group():
                client=lookup.get(key)
                if client is None:continue
                result=client_update(b,theta,client,D.solver(),round_idx=idx,seed=2026,clone_upload=True)
                if result.n_valid:updates[key]=reference.flatten(result.upload,dtype=torch.float64)-anchor
            server_ref.step(updates)
    torch.testing.assert_close(layout.flatten(actual),server_ref.weights(),rtol=0,atol=0)
    assert counters==dict(valid_exposures=44,optimizer_steps=6,present_visits=4,noop_visits=4)
    assert torch.equal(before,torch.get_rng_state())
    assert len(records)==4 and all(r[1]['planned_denominator']==3 for r in records)


def recenter(adapter,layout,v):
    adapter.load_state_(layout.state(v));adapter.server_post_aggregate()
    return layout.flatten(adapter.broadcast_state())


def test_wrong_client_callback_key_refused():
    a=make_tiny_adapter(K=7,d=4,dropout=0.)
    c=make_clients(1,K=7,L=3,sizes=[3])[0]
    with pytest.raises(D.Refused,match='key differs'):
        D.run_clients(a,D.SharedVector(a),roster=['different'],get_client=lambda k:c,
            mechanism='tree',eta=.1,private_seed=7,noise_multiplier=0.,orders=[['different']])


def test_model_only_export_whitelist_excludes_fixed_data_buffers_and_private_identifiers():
    a=make_tiny_adapter(K=7,d=4,tied=True,dropout=0.);layout=D.SharedVector(a)
    theta=dict(a.broadcast_state(clone=True),PRIVATE_SEED=torch.tensor(123))
    cell=D.select_cell('local',.1);out=D.model_only_payload(theta,layout,cell,D.budget('local'))
    assert set(out)=={'format','shared_theta','public_cell','conditional_fixed_recipe_accountant','pretraining_preprocessing_tuning_accounted','privacy_certified','execution_scope','transport_isolation_implemented'}
    assert set(out['shared_theta'])==set(a.manifest.shared_keys)
    assert not set(out['shared_theta'])&set(a.manifest.buffer_keys)
    assert 'PRIVATE_SEED' not in out['shared_theta'] and not out['privacy_certified']
    assert out['execution_scope']=='SINGLE_PROCESS_ROLE_BOUNDARY_SIMULATOR_NOT_DEPLOYED_TRANSPORT'
    assert out['transport_isolation_implemented'] is False
    with pytest.raises(D.Refused):D.model_only_payload(theta,layout,cell,dict(D.budget('local'),private_seed=123))


@pytest.mark.parametrize('export',[None,'','0,1','1','-1'])
def test_early_CUDA_export_guard(export,monkeypatch):
    if export is None:monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    else:monkeypatch.setenv('CUDA_VISIBLE_DEVICES',export)
    with pytest.raises(D.Refused):D.require_early_cuda(0)


def test_private_output600_no_overwrite_and_symlink_refusal():
    with tempfile.TemporaryDirectory(dir=D.EXT/'scratch')as tmp:
        p=Path(tmp)/'own.json';D.write_private(p,{'private':1})
        assert p.stat().st_mode&0o777==0o600
        with pytest.raises(D.Refused):D.write_private(p,{'private':2})
        alias=Path(tmp)/'alias';alias.symlink_to(Path(tmp),target_is_directory=True)
        with pytest.raises(D.Refused):D.write_private(alias/'bad.json',{})
        assert D.read(p)=={'private':1}


def test_unreviewed_training_refuses_before_model_data_or_CUDA_allocation(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(D,'load_protocol',lambda *a: {'declaration':D.declaration()})
    monkeypatch.setattr(D,'actual_inputs',lambda *a,**k:(_ for _ in ()).throw(AssertionError('actual inputs read')))
    with tempfile.TemporaryDirectory(dir=D.EXT/'scratch')as tmp:
        p=Path(tmp)/'review.json';p.write_text('{}')
        with pytest.raises(D.Refused,match='independent'):
            D.train('protocol','sha','prep','sha',p,D.sha(p),'tree',.1,Path(tmp)/'never',0)
        assert not (Path(tmp)/'never').exists() and not torch.cuda.is_initialized()


@pytest.mark.parametrize('flag',['--role','--seed','--epsilon','--resume','--TEST'])
def test_unregistered_CLI_options_refused_before_input_access(flag):
    with pytest.raises(SystemExit):D.main(['build-protocol','--out','/never',flag,'TEST'])


def test_registered_functional_dropout_configuration_is_required_for_actual_models():
    a = make_tiny_adapter(K=7, d=4, dropout=0.)
    with pytest.raises(D.Refused, match='registered model cfg.dropout'):
        D.require_zero_model_dropout(a, require_registered_cfg=True)
    result = D.require_zero_model_dropout(a)
    assert result == dict(cfg_dropout=None, nn_dropout_sites=1, functional_p_sites=0,
                         all_zero=True, registered_cfg_required=False)


@pytest.mark.parametrize('probability', [.1, -0.1, math.nan, math.inf, False, '0'])
def test_functional_cfg_dropout_refused_even_without_nonzero_nn_dropout(probability):
    from types import SimpleNamespace
    a = make_tiny_adapter(K=7, d=4, dropout=0.)
    a.module.cfg = SimpleNamespace(dropout=probability)
    assert all(m.p == 0 for m in a.module.modules() if isinstance(m, torch.nn.Dropout))
    with pytest.raises(D.Refused, match='cfg.dropout'):
        D.require_zero_model_dropout(a)


def test_nonzero_nn_dropout_refused_before_any_client_kernel():
    a = make_tiny_adapter(K=7, d=4, dropout=.1)
    with pytest.raises(D.Refused, match='nn.Dropout'):
        D.run_clients(a, D.SharedVector(a), roster=['a'],
            get_client=lambda key: (_ for _ in ()).throw(AssertionError('client data read')),
            mechanism='tree', eta=.1, private_seed=8, noise_multiplier=0., orders=[['a']])


def test_actual_registered_SMALL_single_site_zero_dropout_and_PB100_init(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from v4int import launch_central as LC, launch_fl as F
    from v4int.verify import DataPaths
    from v4models import build, variants
    from fl.numerics import state_digest
    monkeypatch.setenv('V4_DATASET_ROOT', str(D.DATA))
    paths = DataPaths.default(D.WORK)
    lock = LC.bind_data_protocol_lock_v2(paths, smoke=False)
    schema = LC.load_production_schema(lock['schema_id'], lock['schema_self_sha256'], lock['schema_registry_path'])
    recipe = F._small_recipe(SimpleNamespace(small_recipe=None, smoke=False))
    assert recipe['model_variant'] == 'SASREC_D64_B2'
    assert variants.dropout_of('SASREC', recipe['model_variant'], (0.,)) == (0.,)
    with pytest.raises(variants.VariantError, match='must be 1 value'):
        variants.dropout_of('SASREC', recipe['model_variant'], (0., 0.))
    before = torch.get_rng_state().clone()
    # Actual production schema, full registered K and mappings; CPU model only.
    bm = build(F.FAMILY, schema, D.SEED, 'cpu', v1_root=D.DATA,
        variant=recipe['model_variant'], dropout=(0.,))
    assert bm.K == 158486 and bm.variant == 'SASREC_D64_B2' and bm.dropout == (0.,)
    result = D.require_zero_model_dropout(bm.adapter, require_registered_cfg=True)
    assert result['cfg_dropout'] == 0. and result['functional_p_sites'] == 2 and result['all_zero']
    state, identity, _ = F.load_pretrain_init(D.INIT, model_variant=recipe['model_variant'])
    assert identity['checkpoint_sha256'] == D.INIT_CHECKPOINT_SHA and identity['state_digest'] == D.INIT_DIGEST
    F.apply_init_state(bm.adapter, state, D.INIT_DIGEST)
    assert state_digest(bm.adapter.broadcast_state()) == D.INIT_DIGEST
    assert D.require_zero_model_dropout(bm.adapter, require_registered_cfg=True) == result
    # SASRec uses functional F.dropout, so nn.Dropout-only checks miss this.
    original = bm.module.cfg
    bm.module.cfg = replace(original, dropout=.1)
    assert all(m.p == 0 for m in bm.module.modules() if isinstance(m, torch.nn.Dropout))
    with pytest.raises(D.Refused, match='cfg.dropout'):
        D.require_zero_model_dropout(bm.adapter, require_registered_cfg=True)
    bm.module.cfg = original
    for block in bm.module.blocks:
        saved = block.p; block.p = .1
        with pytest.raises(D.Refused, match='functional dropout probability'):
            D.require_zero_model_dropout(bm.adapter, require_registered_cfg=True)
        block.p = saved
    assert D.require_zero_model_dropout(bm.adapter, require_registered_cfg=True) == result
    assert torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized()


def test_local_server_has_no_private_key_and_refuses_raw_callback_before_read():
    s = server(mechanism='local', z=.7)
    assert not hasattr(s, 'seed') and not hasattr(s, 'private_key') and not hasattr(s, 'local_noise')
    with pytest.raises(D.Refused, match='never raw callbacks'):
        s.step(lambda key: (_ for _ in ()).throw(AssertionError('server executed raw-client callback')))
    with pytest.raises(D.Refused, match='must not receive'):
        D.PublicSweepServer(torch.zeros(2), roster=('a',), orders=[('a',)], mechanism='local',
            eta=.1, noise_seed=123, noise_multiplier=.7)
    assert s.round == 0 and torch.equal(s.weights(), torch.zeros(2))


def test_client_clip_plus_Gaussian_oracle_and_fixed_noop_message_shape():
    from .dpftrl_tree import clip_l2
    shape=(2,); raw=torch.tensor([3.,4.]); secret=91723
    a=D.LocalClientRandomizer(shape,slot='a',private_key=secret,clip_norm=1.,noise_multiplier=.7)
    before=torch.get_rng_state().clone()
    message=a.randomize(raw,sweep=0)
    tag=D.canonical(['PPSI_CLIENT_LOCAL_MESSAGE_NOISE_2',secret,0,'a'])
    generator=torch.Generator(device='cpu').manual_seed(int(tag[:16],16))
    noise=torch.randn(shape,generator=generator,dtype=torch.float64)*.7
    torch.testing.assert_close(message,clip_l2(raw,1.)+noise,rtol=0,atol=0)
    assert message.dtype==torch.float64 and tuple(message.shape)==shape and not torch.equal(message,clip_l2(raw,1.))
    for direction in (None,torch.zeros(2)):
        c=D.LocalClientRandomizer(shape,slot='a',private_key=secret,clip_norm=1.,noise_multiplier=.7)
        torch.testing.assert_close(c.randomize(direction,sweep=0),noise,rtol=0,atol=0)
    assert torch.equal(before,torch.get_rng_state())
    with pytest.raises(D.Refused,match='one fresh local message'):
        a.randomize(raw,sweep=0)


def test_local_server_sums_noisy_messages_without_clipping_or_rng(monkeypatch):
    s=server(mechanism='local',z=.7)
    messages={'a':torch.tensor([20.,30.],dtype=torch.float64),'b':torch.tensor([-4.,7.],dtype=torch.float64)}
    before=torch.get_rng_state().clone()
    monkeypatch.setattr(T,'clip_l2',lambda *a:(_ for _ in ()).throw(AssertionError('server clipped a local message')))
    torch.testing.assert_close(s.step_messages(messages),torch.tensor([.8,1.85]),rtol=0,atol=1e-7)
    assert torch.equal(before,torch.get_rng_state())
    assert all('seed' not in key and 'private_key' not in key and 'client' not in key for key in vars(s))


@pytest.mark.parametrize('bad',[None,torch.zeros(2),torch.zeros(3,dtype=torch.float64),
    torch.tensor([math.nan,0.],dtype=torch.float64),torch.tensor([math.inf,0.],dtype=torch.float64), {'private_seed':123}])
def test_malformed_local_message_refused_without_state_commit(bad):
    s=server(mechanism='local',z=.7);before=s.weights()
    with pytest.raises((D.Refused,ValueError)):
        s.step_messages({'a':bad,'b':torch.zeros(2,dtype=torch.float64)})
    assert s.round==0 and torch.equal(before,s.weights()) and not s.prefix.any()


@pytest.mark.parametrize('messages',[{}, {'a':torch.zeros(2,dtype=torch.float64)},
    {'a':torch.zeros(2,dtype=torch.float64),'b':torch.zeros(2,dtype=torch.float64),'private_seed':123}])
def test_local_message_roster_and_payload_field_whitelist(messages):
    s=server(mechanism='local',z=.7)
    with pytest.raises(D.Refused,match='exact fixed public group'):
        s.step_messages(messages)
    assert s.round==0


def test_actual_local_client_kernel_delivers_only_noisy_tensors_to_seedless_server(monkeypatch):
    clients=make_clients(2,K=7,L=3,seed=77,sizes=[17,0],invalid_frac=0.)
    roster=tuple(c.key for c in clients)+('absent',);lookup={c.key:c for c in clients}
    a=make_tiny_adapter(K=7,d=4,tied=False,dropout=0.,seed=15)
    captured=[];original=D.PublicSweepServer.step_messages
    def inspect_boundary(server,messages):
        assert server.mechanism=='local' and not hasattr(server,'seed')
        assert set(messages)==set(server.next_group())
        assert all(isinstance(v,torch.Tensor) and v.dtype==torch.float64 and v.shape==server.theta.shape for v in messages.values())
        captured.extend(v.clone() for v in messages.values())
        return original(server,messages)
    monkeypatch.setattr(D.PublicSweepServer,'step_messages',inspect_boundary)
    monkeypatch.setattr(D.PublicSweepServer,'step',lambda *args:(_ for _ in ()).throw(AssertionError('raw callback crossed server API')))
    before=torch.get_rng_state().clone()
    _,counters=D.run_clients(a,D.SharedVector(a),roster=roster,get_client=lookup.get,
        mechanism='local',eta=.1,private_seed=None,noise_multiplier=.1,
        local_client_seeds={k:200+i for i,k in enumerate(roster)},orders=[roster],group_size=2)
    assert len(captured)==3 and all(torch.isfinite(v).all() and v.any() for v in captured)
    assert counters==dict(valid_exposures=17,optimizer_steps=2,present_visits=1,noop_visits=2)
    assert torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('kind',['master_on_server','duplicate_keys','missing_key','tree_keys'])
def test_local_entropy_admission_refuses_wrong_scope_before_client_read(kind):
    a=make_tiny_adapter(K=7,d=4,dropout=0.)
    kwargs=dict(private_seed=None,local_client_seeds={'a':1,'b':2},mechanism='local')
    if kind=='master_on_server':kwargs['private_seed']=2026
    elif kind=='duplicate_keys':kwargs['local_client_seeds']={'a':1,'b':1}
    elif kind=='missing_key':kwargs['local_client_seeds']={'a':1}
    else:kwargs.update(mechanism='tree',private_seed=20)
    with pytest.raises(D.Refused):
        D.run_clients(a,D.SharedVector(a),roster=['a','b'],
            get_client=lambda key:(_ for _ in ()).throw(AssertionError('client read before entropy admission')),
            eta=.1,noise_multiplier=.7,orders=[['a','b']],**kwargs)


def test_independent_client_entropy_is_not_a_public_training_seed(monkeypatch):
    draws=iter([101,101,202]);monkeypatch.setattr(D.secrets,'randbits',lambda bits:next(draws))
    assert D.private_client_keys(['a','b'])=={'a':101,'b':202}
    assert not D.declaration()['transport_isolation_implemented']
    assert D.declaration()['execution_scope']=='SINGLE_PROCESS_ROLE_BOUNDARY_SIMULATOR_NOT_DEPLOYED_TRANSPORT'


@pytest.mark.parametrize('approved,selected,admit',[(['tree'],'tree',True),(['local'],'local',True),
    (['tree','local'],'tree',True),(['tree','local'],'local',True),(['tree'],'local',False),
    (['local'],'tree',False),([],'tree',False),(['tree','tree'],'tree',False),(['tree','other'],'tree',False)])
def test_independent_review_can_approve_only_one_selected_mechanism(approved,selected,admit):
    with tempfile.TemporaryDirectory(dir=D.EXT/'scratch') as tmp:
        p=Path(tmp)/'mechanism_review.json'
        body=dict(status='PASS',reviewer='/root/independent_review',protocol_sha256='fixedprotocol',
            preparation_sha256='fixedpreparation',module_sha256=D.sha(D.__file__),
            tests_sha256=D.sha(D.CODE/'experiment_extensions/test_dpftrl_train_v4.py'),
            run_accountant_concurrence=True,modes=approved,fixed_public_S=1.,TEST_approved=False)
        p.write_text(__import__('json').dumps(body))
        if admit:
            assert D.review_gate(p,D.sha(p),'fixedprotocol','fixedpreparation',selected)==body
        else:
            with pytest.raises(D.Refused,match='independent'):
                D.review_gate(p,D.sha(p),'fixedprotocol','fixedpreparation',selected)


@pytest.mark.parametrize('case', ['idle', 'busy', 'foreign_memory', 'smi_failure', 'cuda_mismatch'])
def test_actual_train_entrypoint_inherited_preflight_true_runner(monkeypatch, tmp_path, case):
    """Reach real inherited gpu_checks via train; only subprocess observations are simulated."""
    import subprocess
    from types import SimpleNamespace
    from v4int import launch_central as LC
    from v4sup.gpu import default_runner
    assert D.default_runner is default_runner
    assert not hasattr(LC, 'default_runner')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0' if case != 'cuda_mismatch' else '1')
    monkeypatch.setattr(D, 'require_early_cuda', lambda gpu: None)
    monkeypatch.setattr(D, 'load_protocol', lambda *a: {'declaration': D.declaration()})
    monkeypatch.setattr(D, 'review_gate', lambda *a: None)
    proof = {'status': 'PASS_PREPARED_NOT_TRAINED'}
    prep = dict(proof, protocol_sha256='p', declaration=D.declaration(), module_sha256='m', tests_sha256='m', GPU_initialized=False)
    monkeypatch.setattr(D, 'sha', lambda path: 'prep' if path == 'prep-path' else 'm')
    monkeypatch.setattr(D, 'read', lambda path: prep)
    monkeypatch.setattr(D, 'fresh', lambda path: tmp_path / 'NEVER_CREATED')
    monkeypatch.setattr(D, 'actual_inputs', lambda *a, **kw: {'proof': proof})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    calls = []
    def observed(argv, **kw):
        calls.append(argv)
        assert kw == {'capture_output': True, 'text': True, 'timeout': 60}
        if case == 'smi_failure':
            return subprocess.CompletedProcess(argv, 1, '', 'diagnostic')
        if '--query-gpu=index,pci.bus_id' in argv:
            out = '0, 00000000:01:00.0\n1, 00000000:02:00.0\n'
        elif '--query-compute-apps=gpu_bus_id,pid' in argv:
            out = '00000000:01:00.0, 123\n' if case == 'busy' else ''
        else:
            assert '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu' in argv
            out = '0, GPU-test-0, A40, %s, 46068, 0\n1, GPU-test-1, A40, 0, 46068, 0\n' % (4096 if case == 'foreign_memory' else 0)
        return subprocess.CompletedProcess(argv, 0, out, '')
    monkeypatch.setattr(subprocess, 'run', observed)
    error = D.Refused if case == 'idle' else LC.LaunchRefused
    with pytest.raises(error, match='physical GPU unavailable' if case == 'idle' else None):
        D.train('p-path', 'p', 'prep-path', 'prep', 'review', 'r', 'tree', .1, tmp_path/'unused', 0)
    assert len(calls) == (0 if case == 'cuda_mismatch' else 3 if case in ('idle', 'foreign_memory') else 2)
    assert not (tmp_path/'NEVER_CREATED').exists()
    assert not torch.cuda.is_initialized()
