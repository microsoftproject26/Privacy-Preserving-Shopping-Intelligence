"""Fixed CPU MAIN VALIDATION deployed-size GRU/C_SMALL comparison; no training/TEST.

Two families x two original seeds x registered PB/CB checkpoints. The frozen
predictor/evaluator is unchanged; CPU rows retain REHEARSAL_NON_FROZEN_DEVICE.
This is a descriptive CPU extension of C11, not the original registered A40 block.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import ppsi_paths

BASE=ppsi_paths.SERVER_BASE
X=ppsi_paths.EXTENSIONS_ROOT
W=X/'ppsi_v4';CODE=W/'code'
V=BASE/'Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4'
DATA=BASE/'shared_data/rees46_nextitem_v1'
SOURCE=CODE/'experiment_extensions/gru_deployed_validation.py'
TESTS=CODE/'experiment_extensions/test_gru_deployed_validation.py'
PROTOCOL=X/'protocols/GRU_DEPLOYED_CPU_VALIDATION_PROTOCOL_01.json'
RUN_ROOT=X/'runs/GRU_DEPLOYED_CPU_VALIDATION_01'
TRAIN='MAIN_TRAIN_5_100_131072';EVAL='MAIN_VALIDATION_5_100_131072'
SCHEMA='c53d47fe2555dd537e27e11d45a4e8b23739566e6b98179dcf521852aa918f87'
SUCCESS='EIGHT_FIXED_GRU_CSMALL_CPU_VALIDATION_CELLS_COMPLETE'
TRUTH_KEYS=frozenset(('target','targets','label','labels','target_class','target_oov','loss_mask'))


def require(ok,msg='fixed CPU deployed-size study refused'):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def pin(p):
    p=Path(p).resolve();require(BASE in p.parents and p.is_file());return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def fresh(p):
    p=Path(p).absolute();require(X in p.parents and p==p.resolve() and not p.exists(),'fresh exact physical extension path required');return p
def write(p,obj):
    p=fresh(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def specification():
    return dict(label='DESCRIPTIVE_POST_HOC_EXPLORATORY_CPU_VALIDATION',role='MAIN',split='VALIDATION',
        train_manifest=TRAIN,eval_manifest=EVAL,seeds=[2026,2027],families=['GRU','SASREC'],
        variants=['GRU_D128','SASREC_D64_B2'],selection_rules=['PRACTICAL_BEST','CONTROLLED_BUDGET'],
        checkpoint_selection='ORIGINAL_SEALED_BEST_AND_ENDPOINT_6.0_NO_NEW_SELECTION',
        cells=8,parallel_jobs=8,threads_per_job=11,max_training_and_CPU_threads=96,
        device='cpu',early_cuda_visible_devices='-1',class_label='ACTUAL_CPU_REHEARSAL_NON_FROZEN_DEVICE',
        full_catalogue_K=158486,eval_batch=1024,score_block=158486,class_chunk=32768,
        inherited_numerical_sources_unchanged=True,truth_excluded_from_model_query=True,
        TEST=False,training=False,retry=False,resume=False,claim_original_A40_block=False,
        privacy_certified=False,run_root=str(RUN_ROOT))

def cells():
    out=[]
    for family in ('GRU','SASREC'):
        for seed in (2026,2027):
            directory=V/(f'runs_p3/GRU/02B_GRU_s{seed}' if family=='GRU' else f'runs_ds/C_SMALL_s{seed}')
            index=read(directory/'sealed/CHECKPOINTS.json');start=read(directory/'RUN_START.json')
            for rule,key in [('PRACTICAL_BEST','best'),('CONTROLLED_BUDGET','endpoint_6.0')]:
                row=index[key];meta=row['meta'];variant='GRU_D128' if family=='GRU' else 'SASREC_D64_B2'
                require(meta['family']==family and meta['model_variant']==variant and meta['seed']==seed)
                require(meta['manifest_registry']['train_manifest']==TRAIN and meta['manifest_registry']['eval_manifest']==EVAL
                    and meta['schema']['schema_self_sha256']==SCHEMA and meta['run_role']=='MAIN')
                require(start['train_provider']['manifest_id']==TRAIN and start['eval_provider']['manifest_id']==EVAL)
                if rule=='CONTROLLED_BUDGET':require(meta['efe']==6.0 and meta['exposures']==15526656)
                checkpoint=directory/'sealed'/row['file'];require(pin(checkpoint)['sha256']==row['sha256'])
                out.append(dict(cell_id=f'{family}_s{seed}_{key}',family=family,variant=variant,seed=seed,
                    rule=rule,checkpoint=pin(checkpoint),index=pin(directory/'sealed/CHECKPOINTS.json'),
                    start=pin(directory/'RUN_START.json'),registry=pin(directory/'REGISTRY_V2_TRAINER_VIEW.json'),
                    expected_meta=meta))
    require(len(out)==8 and len({c['cell_id'] for c in out})==8);return out

def input_paths():
    paths={p.resolve() for p in (W/'contracts').rglob('*') if p.is_file()}
    # Providers bind their actual manifest/table/marker paths; data are TRAIN/VALIDATION only.
    reg=read(W/'contracts/manifest_registry/REGISTRY_V2.json')
    paths.update([W/'data_v2/TRANSFORM_STATE_CLEAN_V2.json',DATA/'FILES.json'])
    paths.update((W/'data_v3/receipts').glob('DATA06*.json'))
    for mid in (TRAIN,EVAL):
        entry=reg['manifests'][mid]
        paths.update((W/entry[k]).resolve() for k in ('path','marker'))
        paths.update((W/r['file']).resolve() for r in entry['tables'].values())
    paths.update(p.resolve() for p in (W/'contracts/derived_schemas').glob('*.json'))
    paths.update(p.resolve() for p in (W/'contracts').glob('*SCHEMA*.json'))
    paths.update(p.resolve() for p in DATA.glob('model_ready/user_bucket=*/features.parquet'))
    for split in ('TRAIN','VALIDATION_SELECT','VALIDATION_CONFIRM'):
        paths.update(p.resolve() for p in (DATA/'decisions_index'/('split='+split)).glob('user_bucket=*/*.parquet'))
    paths.update(p.resolve() for p in (DATA/'mappings').glob('*') if p.is_file())
    return sorted(paths)

def source_paths():return sorted(p for p in CODE.rglob('*.py') if 'experiment_extensions' not in p.parts)+[CODE/'experiment_extensions/__init__.py',SOURCE,TESTS]

def build_protocol(out):
    from v4eval import evaluator_code_sha256
    obj=dict(status='PROSPECTIVE_FIXED_CPU_VALIDATION_NOT_SCORED',utc=utc(),specification=specification(),
        cells=cells(),evaluator_code_sha256=evaluator_code_sha256(),sources=[pin(p) for p in source_paths()],inputs=[pin(p) for p in input_paths()])
    write(out,obj);return pin(out)

def load_protocol(path,digest):
    require(Path(path)==PROTOCOL and sha(path)==digest,'exact fixed protocol required');p=read(path)
    from v4eval import evaluator_code_sha256
    require(p['specification']==specification() and p['cells']==cells() and p['evaluator_code_sha256']==evaluator_code_sha256(),'fixed checkpoint/selection/evaluator changed')
    for key,paths in [('sources',source_paths()),('inputs',input_paths())]:require(p[key]==[pin(v) for v in paths],key+' inventory changed')
    return p

def approval_gate(path,digest,protocol_sha):
    require(sha(path)==digest);a=read(path)
    require(a.get('status')=='APPROVED_FIXED_GRU_CSMALL_CPU_VALIDATION_01' and a.get('reviewer')=='/root'
        and a.get('source_sha256')==sha(SOURCE) and a.get('tests_sha256')==sha(TESTS)
        and a.get('protocol')==pin(PROTOCOL) and a.get('decision_log_recorded') is True
        and a.get('specification')==specification(),'fresh exact root CPU approval required')
    r=a['independent_review'];require(pin(r['path'])==r);r=read(r['path'])
    require(r.get('status')=='PASS' and r.get('reviewer') not in (None,'','/root/dpftrl_entrypoint_repair04')
        and r.get('source_sha256')==sha(SOURCE) and r.get('tests_sha256')==sha(TESTS),'independent exact source review required')
    return load_protocol(PROTOCOL,protocol_sha)

def runtime():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','CPU-only early CUDA visibility required')
    require(all(os.environ.get(k)=='11' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')), 'exact 11-thread worker environment required')
    os.environ['V4_DATASET_ROOT']=str(DATA)
    import torch,ppsi_env
    ppsi_env.verify_snapshot();ppsi_env.assert_isolated()
    require(not torch.cuda.is_initialized() and not torch.cuda.is_available(),'CPU worker must never initialize CUDA')
    from fl.numerics import enforce_strict_fp32
    from v4eval import frozen_config as FC
    enforce_strict_fp32(warn_only=False);fz=FC.load();FC.apply_runtime_flags(fz)
    torch.set_num_threads(11);return torch,fz

def build_loaded(cell):
    torch,fz=runtime()
    from v4int import launch_central as LC,reeval as R
    from v4int.verify import DataPaths
    from v4models import build
    from v4data.common import dataset_root
    paths=DataPaths.default(W);lock=LC.bind_data_protocol_lock_v2(paths,smoke=False)
    schema=LC.load_production_schema(lock['schema_id'],lock['schema_self_sha256'],lock['schema_registry_path'])
    require(schema.self_sha256==SCHEMA)
    start=read(cell['start']['path']);cfg=start['config']
    bm=build(cell['family'],schema,cell['seed'],'cpu',v1_root=dataset_root(),variant=cell['variant'],dropout=tuple(cfg['dropout']))
    obj=torch.load(cell['checkpoint']['path'],map_location='cpu',weights_only=False)
    require(obj['meta']==cell['expected_meta'],'exact saved checkpoint meta required')
    identity=R.load_central(bm,obj,start,SimpleNamespace(variant=cell['variant'],seed=cell['seed']))
    require(identity['state_digest']==cell['expected_meta']['state_digest'])
    return torch,fz,paths,lock,schema,bm,start,identity

def truth_free_batches(batches):
    for batch in batches:yield {k:v for k,v in batch.items() if k not in TRUTH_KEYS}

def score(cell_id,protocol_sha,approval,approval_sha):
    protocol=approval_gate(approval,approval_sha,protocol_sha)
    cell=next((c for c in protocol['cells'] if c['cell_id']==cell_id),None);require(cell is not None)
    out=fresh(RUN_ROOT/cell_id)
    torch,fz,paths,lock,schema,bm,start,identity=build_loaded(cell)
    from v4int.providers import StoreEvalProvider
    from v4eval.predict import predict_adapter
    from v4eval.evaluate import evaluate
    from v4eval.rows import rows_from_result
    from v4eval import evaluator_code_sha256
    from v4eval.values import to_jsonable
    ep=StoreEvalProvider(EVAL,prefetch=0,paths=paths,history_policy='USER_HISTORY_L50',
        manifest_registry=lock['manifest_registry_path'],registry_root=W,log=lambda *a:None)
    require(ep.fingerprint==start['eval_provider']['fingerprint'],'actual MAIN provider differs from source training')
    require(ep.manifest.catalogue_K==158486)
    out.mkdir();write(out/'STARTED.json',dict(status='CPU_FIXED_FULL_K_SCORING',cell=cell,protocol=pin(PROTOCOL),pid=os.getpid()))
    t=time.monotonic()
    art=predict_adapter(bm.adapter,truth_free_batches(ep.batches(fz.eval_batch)),ep.manifest,config=fz,model_sha256=identity['state_digest'])
    result=evaluate(ep.manifest,art,config=fz,evaluator_code_sha256=evaluator_code_sha256())
    require(str(result['eval_config']['eval_device_class']).startswith('CPU'),'actual CPU class must be preserved')
    rows=rows_from_result(result,start['key_base'],method='C',seed=cell['seed'],selection_rule=cell['rule'],efe=cell['expected_meta']['efe'])
    require(all(r['status']=='REHEARSAL_NON_FROZEN_DEVICE' for r in rows))
    for r in rows:r.update(study='L4_DESCRIPTIVE_CPU_DEPLOYED_SIZE',family=cell['family'],variant=cell['variant'],label='POST_HOC_EXPLORATORY',checkpoint_sha256=cell['checkpoint']['sha256'])
    import numpy as np
    np.save(out/'ranks.npy',art.ranks,allow_pickle=False)
    write(out/'RESULT.json',to_jsonable(result));write(out/'ROWS.json',to_jsonable(rows))
    require(not torch.cuda.is_initialized());load_protocol(PROTOCOL,protocol_sha)
    write(out/'COMPLETE.json',dict(status='FIXED_CPU_MAIN_VALIDATION_COMPLETE',cell=cell,protocol=pin(PROTOCOL),
        source=pin(SOURCE),approval=pin(approval),provider_fingerprint=ep.fingerprint,manifest_binding=ep.manifest_binding,
        population=dict(rows=ep.manifest.n,E2E_decisions=int(ep.manifest.eval_mask.sum())),
        device_class=result['eval_config']['eval_device_class'],elapsed_seconds=time.monotonic()-t,
        outputs=[pin(out/n) for n in ('ranks.npy','RESULT.json','ROWS.json')],GPU_initialized=False,TEST_read=False,trained=False))
    return 0

def verified_completion(cell,approval):
    out=RUN_ROOT/cell['cell_id'];record=read(out/'COMPLETE.json')
    require(record.get('status')=='FIXED_CPU_MAIN_VALIDATION_COMPLETE' and record.get('cell')==cell
        and record.get('protocol')==pin(PROTOCOL) and record.get('source')==pin(SOURCE)
        and record.get('approval')==pin(approval) and record.get('GPU_initialized') is False
        and record.get('TEST_read') is False and record.get('trained') is False
        and str(record.get('device_class','')).startswith('CPU'),'exact own CPU completion required')
    require(record.get('outputs')==[pin(out/n) for n in ('ranks.npy','RESULT.json','ROWS.json')], 'own artifacts differ')
    return pin(out/'COMPLETE.json')

def queue(protocol_sha,approval,approval_sha):
    protocol=approval_gate(approval,approval_sha,protocol_sha);fresh(RUN_ROOT);RUN_ROOT.mkdir()
    records=[];active={};mutex=threading.Lock()
    def status(value):
        with mutex:
            row=dict(utc=utc(),pid=os.getpid(),status=value,operator_sha256=sha(SOURCE),approval_sha256=approval_sha,
                records=copy.deepcopy(records),active_jobs=copy.deepcopy(active),automatic_retry=False,training_resume=False,TEST_scoring=False)
            tmp=RUN_ROOT/'STATUS.pending.json';tmp.write_text(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');tmp.replace(RUN_ROOT/'STATUS.json')
    def stage(c):
        name=c['cell_id'];out=RUN_ROOT/(name+'.log')
        args=['python','-B','-u','-m','experiment_extensions.baselines.gru_deployed_validation','score','--cell',name,
            '--protocol-sha256',protocol_sha,'--approval',str(approval),'--approval-sha256',approval_sha]
        cmd=['nice','-n','19','bash',str(X/'START_EXTENSIONS.sh'),'-c',
            'CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=11 MKL_NUM_THREADS=11 OPENBLAS_NUM_THREADS=11 '+shlex.join(args)]
        with mutex:active[name]=dict(started=utc())
        status('RUNNING_FIXED_CPU_CELLS')
        completion=None
        try:
            with out.open('x') as log:
                process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                with mutex:active[name].update(pid=process.pid,command=cmd)
                status('RUNNING_FIXED_CPU_CELLS')
                code=process.wait()
            if code==0:completion=verified_completion(c,approval)
        except Exception as error:
            code=3
            write(RUN_ROOT/(name+'.OPERATIONAL_FAILURE.json'),dict(error_type=type(error).__name__,no_retry=True))
        with mutex:active.pop(name);records.append(dict(stage=name,returncode=code,completion=completion))
        status('RUNNING_FIXED_CPU_CELLS');return code
    status('STARTING_EIGHT_FIXED_CPU_CELLS')
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes=[f.result() for f in as_completed([pool.submit(stage,c) for c in protocol['cells']])]
    require(len(records)==8 and len({r['stage'] for r in records})==8)
    status(SUCCESS if all(v==0 for v in codes) else 'STOPPED_CPU_CELL_FAILURE_NO_RETRY')
    return 0 if all(v==0 for v in codes) else 3

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build-protocol');b.add_argument('--out',type=Path,required=True)
    for name in ('queue','score'):
        q=sub.add_parser(name);q.add_argument('--protocol-sha256',required=True);q.add_argument('--approval',type=Path,required=True);q.add_argument('--approval-sha256',required=True)
        if name=='score':q.add_argument('--cell',required=True)
    a=p.parse_args(argv)
    if a.action=='build-protocol':print(json.dumps(build_protocol(a.out)));return 0
    if a.action=='queue':return queue(a.protocol_sha256,a.approval,a.approval_sha256)
    return score(a.cell,a.protocol_sha256,a.approval,a.approval_sha256)
if __name__=='__main__':sys.exit(main())
