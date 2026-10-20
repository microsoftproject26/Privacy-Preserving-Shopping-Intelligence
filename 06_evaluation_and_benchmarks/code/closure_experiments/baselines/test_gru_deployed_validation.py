"""CPU-only actual saved-model interfaces, dense rank oracle and fixed admission."""
import ppsi_env
import copy
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from . import gru_deployed_validation as G

@pytest.mark.parametrize('index',range(8))
def test_fixed_matching_MAIN_checkpoint_provenance(index):
    cell=G.cells()[index];m=cell['expected_meta'];s=G.read(cell['start']['path'])
    assert m['family']==cell['family'] and m['model_variant']==cell['variant']
    assert m['schema']['schema_self_sha256']==G.SCHEMA
    assert s['train_provider']['manifest_id']==G.TRAIN and s['eval_provider']['manifest_id']==G.EVAL
    assert cell['checkpoint']==G.pin(cell['checkpoint']['path'])
    assert cell['rule'] in ('PRACTICAL_BEST','CONTROLLED_BUDGET')

@pytest.mark.parametrize('family',['GRU','SASREC'])
def test_actual_CPU_saved_checkpoint_adapter_frozen_predict_dense_oracle(family):
    from v4models.synthetic import synthetic_examples_v2
    from v4eval.manifest import build_eval_manifest
    from v4eval.predict import predict_adapter
    from v4eval.evaluate import evaluate
    from v4eval import evaluator_code_sha256
    cell=next(c for c in G.cells() if c['family']==family and c['seed']==2026)
    _,fz,paths,lock,schema,bm,start,identity=G.build_loaded(cell)
    assert not torch.cuda.is_initialized()
    bm.module.eval()
    batch=synthetic_examples_v2(6,158486,bm.adapter.input_widths,torch.Generator().manual_seed(417),
        max_len=3,min_len=1,with_names=True,with_session_offset=True,with_session_intent=True)
    target=np.array([0,1,2,-1,-1,-1],dtype=np.int64)
    m=build_eval_manifest({'manifest_id':'SYNTHETIC_L4_CPU_VALIDATION','v4_split':'VALIDATION','catalogue_K':158486},{
        'decision_id':np.arange(6,dtype=np.int64)+100,'eval_user_key':np.array([10,10,20,20,30,40]),
        'eval_mask':np.array([1,1,1,1,1,0],bool),'target_class':target,
        'target_oov':np.array([0,0,0,1,1,0],bool),'censor_class':np.array(['NONE']*5+['CEN_NO_NEXT'])})
    batch['decision_id']=torch.arange(6)+100
    batch['target_class']=torch.as_tensor(target)
    query_batch=next(G.truth_free_batches([batch]));assert not set(query_batch)&G.TRUTH_KEYS
    with torch.no_grad():
        q=bm.adapter.query(query_batch);W=bm.adapter.head_weight();b=bm.adapter.head_bias()
        logits=q@W.T+b
        classes=torch.arange(158486)
        dense=[]
        for i,t in enumerate(target[:3]):
            sc=logits[i,t];dense.append(1+int((logits[i]>sc).sum())+int(((logits[i]==sc)&(classes<t)).sum()))
    art=predict_adapter(bm.adapter,[query_batch],m,config=fz,model_sha256=identity['state_digest'])
    assert art.ranks[:3].tolist()==dense and art.ranks[3:].tolist()==[-1,-1,-1]
    res=evaluate(m,art,config=fz,evaluator_code_sha256=evaluator_code_sha256())
    credits=[1/r if r<=20 else 0 for r in dense]
    assert res['E2E']['macro']['mrr@20']==pytest.approx(((credits[0]+credits[1])/2+credits[2]/2+0)/3)
    assert str(res['eval_config']['eval_device_class']).startswith('CPU')
    assert not torch.cuda.is_initialized()
    if family=='GRU':
        from fl.numerics import state_digest
        with torch.no_grad():W.zero_();b.zero_()
        tied=predict_adapter(bm.adapter,[query_batch],m,config=fz,model_sha256=state_digest(bm.adapter.broadcast_state()))
        assert tied.ranks[:3].tolist()==[1,2,3]
        assert tied.n_tied[:3].tolist()==[158485]*3
        assert tied.eq_before[:3].tolist()==[0,1,2]
        assert tied.pessimistic_ranks[:3].tolist()==[158486]*3

@pytest.mark.parametrize('keys',[['target_class','label'],['target','targets','target_oov'],['labels','loss_mask']])
def test_truth_never_reaches_model_query(keys):
    batch=dict(decision_id=[1],item_tokens=[[1]])
    batch.update({k:'secret_synthetic_truth' for k in keys})
    out=list(G.truth_free_batches([batch]))[0]
    assert set(out)=={'decision_id','item_tokens'} and set(batch)-set(out)==set(keys)

@pytest.mark.parametrize('value',['0','1',''])
def test_CPU_early_visibility_refused(monkeypatch,value):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',value)
    with pytest.raises(ValueError,match='CPU-only'):G.runtime()

def test_declared_full_resource_bound_and_fresh_identity():
    s=G.specification()
    assert s['parallel_jobs']==8 and s['threads_per_job']==11 and 8*11+2*4==s['max_training_and_CPU_threads']==96
    assert s['full_catalogue_K']==158486 and s['eval_batch']==1024 and s['score_block']==158486
    assert all(s[k] is False for k in ('TEST','training','retry','resume','claim_original_A40_block'))
    assert not G.RUN_ROOT.exists()

def test_protocol_never_reads_TEST_data():
    paths=G.input_paths()
    assert paths and not any('split=TEST' in str(p) or '/test/' in str(p).lower() for p in paths)
    assert len(G.cells())==8


def test_completion_refuses_zero_exit_without_artifacts(monkeypatch,tmp_path):
    monkeypatch.setattr(G,'RUN_ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):G.verified_completion({'cell_id':'fake'},tmp_path/'approval.json')

@pytest.mark.parametrize('failure',[False,True])
def test_owned_queue_tracks_eight_children_verifies_completion_and_never_retries(monkeypatch,tmp_path,failure):
    root=tmp_path/'fresh_queue';monkeypatch.setattr(G,'RUN_ROOT',root)
    def fresh(path):
        path=Path(path)
        assert not path.exists()
        return path
    monkeypatch.setattr(G,'fresh',fresh)
    selected=[{'cell_id':f'CELL{i}'} for i in range(8)]
    monkeypatch.setattr(G,'approval_gate',lambda *a:{'cells':selected})
    calls=[]
    class Process:
        pid=12345
        def wait(self):return 0
    def popen(cmd,**kw):
        calls.append(cmd)
        assert cmd[-1].startswith('CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=11 MKL_NUM_THREADS=11 OPENBLAS_NUM_THREADS=11 ')
        return Process()
    monkeypatch.setattr(G.subprocess,'Popen',popen)
    def verified(c,a):
        if failure and c['cell_id']=='CELL3':raise ValueError('synthetic missing own artifact')
        return {'path':'synthetic_verified_'+c['cell_id']}
    monkeypatch.setattr(G,'verified_completion',verified)
    assert G.queue('p',tmp_path/'approval','a')==(3 if failure else 0)
    state=G.read(root/'STATUS.json')
    assert len(calls)==8 and len(state['records'])==8 and state['active_jobs']=={}
    assert state['status']==('STOPPED_CPU_CELL_FAILURE_NO_RETRY' if failure else G.SUCCESS)
    assert state['automatic_retry'] is False and state['training_resume'] is False
