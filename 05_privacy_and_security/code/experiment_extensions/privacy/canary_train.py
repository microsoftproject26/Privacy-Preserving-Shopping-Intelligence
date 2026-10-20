"""Own actual-CALIB canary trainer candidate; no inherited source is edited.

Raw protocol/preparation/checkpoint/truth artifacts are private. The release
boundary is model-only; private tuning/clip calibration are not accounted.
Real execution requires exact root-coordinated independent source review.
"""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import hashlib
import json
import math
import os
import threading
from functools import lru_cache
from pathlib import Path
import ppsi_paths

BASE = ppsi_paths.SERVER_BASE
EXT = ppsi_paths.EXTENSIONS_ROOT
WORK = EXT / 'ppsi_v4'
CODE = WORK / 'code'
DONOR = BASE / 'Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4'
DATA = BASE / 'shared_data/rees46_nextitem_v1'
TRAIN = 'CALIB_TRAIN_5_100_32768'
INIT = DONOR / 'runs_exp_pb100/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_PB100K_s2026/PRETRAIN_FOR_FL_INIT.json'
INIT_SHA = '22135cdf132eae1d405dd802acf1a756c9aba122e02c9a01f53876894bd58383'
INIT_DIGEST = '3e1c48efe79a1f1a1377719c144af9fe8e8610060ea7cc18b6bff545b61ce4e2'
SOURCE_ARM = DONOR / 'runs_x_rees46/PB100/S1/eta3e-05_c0.1_ilr0.1_tau1e-6_E6_P1/ARM_SPEC.json'
VARIANT = 'SASREC_D64_B2'
FORMAT = 'EID_CANARY_CALIB_TRAIN_CANDIDATE_2'
LABEL = 'POST_HOC_EXPLORATORY_LOWER_BOUND_ONLY'
MODES = ('FROZEN', 'UNFROZEN', 'DP')
TAIL_RULES = ((250, 250), (500, 500), (100, 100))
_ZERO_ACTIVE = False


class Refused(ValueError):
    pass


def _tensor_equal(first, second):
    import torch
    return list(first) == list(second) and all(torch.equal(first[k], second[k]) for k in first)


@contextlib.contextmanager
def non_dp_zero_policy():
    """Original driver/visits/traffic; skip only an all-zero FA apply.

    Positive and every DP call use the original function immediately. A zero
    call has no optimizer/client RNG step; its zero-weight finalization is
    narrowly admitted and its own server's apply is skipped so stale FedAdam
    momentum cannot move the model. Unchanged FLRun advances its clock/counters.
    """
    global _ZERO_ACTIVE
    from fl import runtime, server as driver
    if _ZERO_ACTIVE or runtime.run_round is not driver.run_round:
        raise Refused('nested or externally patched zero-round policy refused')
    original = runtime.run_round
    owner_thread = threading.get_ident()

    def run_round(server, workers, clients, solver, **kwargs):
        from fl.client import valid_rows
        from fl.checkpoint import rng_state, payload_digest
        import torch
        if threading.get_ident() != owner_thread:
            raise Refused('concurrent zero-policy use outside its owner thread')
        if kwargs.get('dp') is not None or not clients or any(
                int(valid_rows(c.examples, server.adapter.target_key, server.adapter.loss_mask_key).numel())
                for c in clients):
            return original(server, workers, clients, solver, **kwargs)
        if kwargs.get('mu', 0.) != 0 or kwargs.get('pf') is not None or kwargs.get('store') is not None or \
                kwargs.get('upload_observer') is not None or kwargs.get('fault') is not None or \
                getattr(server, 'observer', None) is not None:
            raise Refused('zero-round policy only admits own nonpersonal FA without observers/faults')
        keys = [c.key for c in clients]
        if len(keys) != len(set(keys)):
            raise Refused('zero cohort client keys must remain unique')
        theta = server.adapter.broadcast_state(clone=True)
        optimizer = server.server_opt
        moments = None if optimizer is None else (optimizer.digest(), optimizer.steps)
        ambient_rng_digest = payload_digest(rng_state())
        saved_finalize = driver.finalize
        absent = object()
        saved_instance_apply = server.__dict__.get('apply', absent)
        applied = [0]

        def finalize(aggregate, manifest, server_state, *, expected_keys=None):
            if aggregate.weight != 0 or aggregate.n_noop != len(keys) or expected_keys != keys or \
                    [a[1] for a in aggregate.added] != keys or any(a[2] != 0 for a in aggregate.added) or \
                    manifest is not server.manifest or not _tensor_equal(theta, server_state) or \
                    any(torch.count_nonzero(t).item() != 0 for t in aggregate.total.values()):
                raise Refused('zero finalization differs from own exact zero FA cohort')
            return server_state

        def skip_apply(new_state):
            if not _tensor_equal(theta, new_state):
                raise Refused('zero cohort tried to apply a changed model')
            applied[0] += 1

        try:
            driver.finalize, server.apply = finalize, skip_apply
            result = original(server, workers, clients, solver, **kwargs)
        finally:
            driver.finalize = saved_finalize
            if saved_instance_apply is absent:
                del server.apply
            else:
                server.apply = saved_instance_apply
        if applied[0] != 1 or result.weight != 0 or result.n_noop != len(keys) or \
                any(v.n_valid or v.n_consumed or v.steps or v.loss_sum for v in result.visits) or \
                not _tensor_equal(theta, server.adapter.broadcast_state()) or \
                server.server_opt is not optimizer or \
                (optimizer is not None and (optimizer.digest(), optimizer.steps) != moments) or \
                payload_digest(rng_state()) != ambient_rng_digest:
            raise Refused('zero cohort changed model, optimizer, RNG or actual zero visit records')
        return result

    _ZERO_ACTIVE = True
    try:
        runtime.run_round = run_round
        yield
    finally:
        runtime.run_round = original
        _ZERO_ACTIVE = False


def run_kernel(config, plan, adapter, workers, get_client, actual_counts, normalizer, *,
               entropy=None, stop_after=None, checkpoint=None, progress=None):
    """Private unchanged FLRun numerical path, shared by real and CPU controls."""
    from fl.numerics import state_digest
    from fl.runtime import FLRun
    from fl.server import Server
    from .canary_replay import PrivateEntropy, private_gaussian
    if not config.round_endpoint or config.lr_shape != 'fl_const' or config.method != 'FA' or config.pf or config.mu:
        raise Refused('candidate requires fixed-clock FL_CONST nonpersonal FA')
    if set(actual_counts) != set(plan.keys) or any(type(v) is not int or v < 0 for v in actual_counts.values()):
        raise Refused('actual private valid counts must cover the full potential roster')
    if config.dp is not None and config.dp.noise_multiplier > 0 and not isinstance(entropy, PrivateEntropy):
        raise Refused('DP requires independent private Gaussian/participation entropy')
    count = config.endpoint_rounds if stop_after is None else stop_after
    if type(count) is not int or not 1 <= count <= config.endpoint_rounds:
        raise Refused('only an exact fixed endpoint or declared calibration prefix')
    server = Server(adapter, init_sha256=state_digest(adapter.broadcast_state()))
    run = FLRun(config, plan, server, get_client, normalizer, workers=workers,
        n_valid_by_key=actual_counts, ckpt=checkpoint, eval_fn=None)
    noise = private_gaussian(entropy) if config.dp is not None and config.dp.noise_multiplier > 0 else contextlib.nullcontext()
    with noise, non_dp_zero_policy():
        for _ in range(count):
            record = run.step()
            if progress is not None:
                progress(run, record)
    if checkpoint is not None and not run.state.done:
        checkpoint.save_latest(run.state_dict())
    if run.state.cursor != count or run.state.evals or (count == config.endpoint_rounds and not run.state.done):
        raise Refused('fixed training endpoint or no-evaluation policy differs')
    return run


def physical(path):
    p = Path(path).absolute()
    s = str(p)
    for alias in ppsi_paths.LEGACY_ALIASES:
        if s == alias or s.startswith(alias + '/'):
            s = str(BASE) + s[len(alias):]
            break
    return Path(s)


def scoped(path, root=EXT):
    p, r = physical(path), physical(root).resolve()
    if p.resolve() == r or r not in p.resolve().parents or any(
            q.is_symlink() for q in (p, *p.parents) if q == r or r in q.parents):
        raise Refused('private output must be a fresh nonsymlink file inside this extension')
    return p


def sha(path):
    with physical(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps({k:v for k,v in obj.items() if k!='self_sha256'},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    def bad(v): raise Refused('nonfinite private JSON')
    return json.loads(physical(path).read_text(),parse_constant=bad)


def pin(path):
    p=physical(path).resolve()
    if BASE not in p.parents or not p.is_file(): raise Refused('input outside the preserved EID copy')
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def check_pin(record):
    if pin(record['path'])!=record: raise Refused('bound input bytes changed')


@contextlib.contextmanager
def private_umask():
    previous=os.umask(0o077)
    try:yield
    finally:os.umask(previous)


def write_private(path,obj,*,root=EXT):
    p=scoped(path,root)
    with private_umask():
        p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:
            json.dump(obj,f,sort_keys=True,indent=1,allow_nan=False);f.write('\n')


def exact_role(role):
    if role!=TRAIN:raise Refused('exact CALIB TRAIN only, no held-out/MAIN/TEST provider')


def source_files():
    own=['__init__.py','canary_data.py','test_canary_data.py','canary_statistics.py','test_canary_statistics.py',
         'canary_replay.py','test_canary_replay.py','canary_train.py','test_canary_train.py']
    return sorted(p for p in CODE.rglob('*.py') if 'experiment_extensions' not in p.parts)+[
        CODE/'experiment_extensions'/name for name in own]


def data_files():
    registry=read(WORK/'contracts/manifest_registry/REGISTRY_V2.json')
    row=registry['manifests'][TRAIN]
    if (row['n_users'],row['n_decisions'])!=(32768,646067):raise Refused('CALIB TRAIN population changed')
    files={WORK/row['path'],WORK/row['marker'],WORK/'data_v2/TRANSFORM_STATE_CLEAN_V2.json'}
    files.update(WORK/v['file'] for v in row['tables'].values())
    files.update(DATA.glob('model_ready/user_bucket=*/features.parquet'))
    files.update((DATA/'decisions_index/split=TRAIN').glob('user_bucket=*/*.parquet'))
    files.update(p for p in (DATA/'mappings').glob('*') if p.is_file())
    files.update(DATA/name for name in ('VERSION.json','source_manifest.json','TRAINING_READINESS.json'))
    return sorted(files)


def declaration():
    return dict(label=LABEL,role=TRAIN,real_clients=32768,potential_canaries=1000,
        potential_population=33768,real_decisions=646067,potential_normalizer=656067,
        train_history='USER_HISTORY_L50',catalogue_K=158486,model_variant=VARIANT,dropout=[.1],
        model_context='ACTUAL_REGISTERED_RICH_SCHEMA_ALL_BRANCHES_ENABLED',
        synthetic_context=synthetic_context_declaration(),statistical_power=attainable_bound_ceiling(),
        public_definition_seed=20261014,canary_history_length=20,canary_repeats=10,reference_count=10000,
        long_tail_fraction=.2,public_optimizer_seed=2026,
        nonDP=dict(passes=1,rounds=3518,formula='ceil(60*33768/576)',sampling='sweep',group_size=64,dropout_p=.1),
        DP=dict(passes=2,rounds=384,sampling='PRIVATE_POISSON',m=1024,N=33768,q=1024/33768,
                delta=5e-6,epsilon_target=8.,conversion='STANDARD'),
        client=dict(optimizer='sgd_m',momentum=.9,lr=.1,batch_size=16,clip=1.,weight_decay=0.,item_lr_mult_unfrozen=.1),
        server=dict(name='fedadam',lr=3e-5,tau=1e-6,beta1=.9,beta2=.99,v_init='tau_squared'),
        private_checkpoint_every_rounds=64,resume_supported=False,
        S_CAL=dict(N=32768,m=1024,source_round_clock=384,prefix_rounds=20,passes=2,seed=2026,
            sampling='PRIVATE_POISSON',canaries=False,rule='fresh real CALIB PB100 SGD momentum delta median'),
        LR='FL_CONST_BY_FIXED_ROUND',eval_callback=False,continuation=False,
        calibration_entropy_separate_private_file=True,
        independent_membership_vectors=list(MODES),guess_rules=[list(v) for v in TAIL_RULES],
        simultaneous_tests=9,confidence=.95,bound_type='LOWER_BOUND_ONLY',
        release='FINAL_MODEL_ONLY_FIXED_METADATA; AUDIT_SEPARATE_NOT_PART_OF_DP_RELEASE',
        comparison='WHOLE_SCHEDULE_P1_NONDP_VS_P2_DP_NOT_MATCHED_NOISE',
        private_tuning_accounted=False,privacy_certified=False,MAIN_allowed=False,TEST_allowed=False)


def verify_source_arm():
    from v4int import launch_fl as F
    record=read(SOURCE_ARM);s=record['arm_spec'];cfg=s['cfg']
    if record['arm_spec_sha256']!=F.spec_digest(s) or s['role']!='CALIBRATION' or \
            cfg['method']!='FA' or cfg['seed']!=2026 or cfg['dp'] or cfg['pf'] or cfg['mu'] or \
            cfg['solver']!={'batch_size':16,'clip':1.,'fused':None,'item_lr_mult':.1,'lr':.1,
                           'optimizer':'sgd_m','passes':1,'weight_decay':0.} or \
            cfg['server_opt']!=declaration()['server'] or cfg['lr_shape']!='fl_const' or \
            cfg['dropout_p']!=.1 or s.get('upload_codec') or s['p2_recipe']['lora_rank'] or \
            s['p2_recipe']['personal_head'] or s['init_from']['checkpoint_sha256']!=INIT_SHA:
        raise Refused('not the exact selected PB100 CALIB P1 source recipe')
    return record


def make_config(mode,*,clip=None):
    if mode not in (*MODES,'S_CAL'):raise Refused('only declared canary mechanisms or separate calibration')
    from fl.client import LocalSolver,ItemLRSolver
    from fl.runtime import RunConfig
    from fl.server import ServerOptConfig
    from fl.dp import DPConfig
    frozen=mode=='FROZEN';is_dp=mode in ('DP','S_CAL')
    if mode=='DP' and (isinstance(clip,bool) or not isinstance(clip,(int,float)) or not math.isfinite(clip) or clip<=0):
        raise Refused('DP requires an own fresh finite positive calibration')
    from .canary_replay import dp_accountant
    solver=(LocalSolver if frozen else ItemLRSolver)(lr=.1,optimizer='sgd_m',passes=2 if is_dp else 1,
        batch_size=16,clip=1.,weight_decay=0.,**({} if frozen else {'item_lr_mult':.1}))
    dp=DPConfig(None if mode=='S_CAL' else clip,0. if mode=='S_CAL' else dp_accountant()['z'],1024) if is_dp else None
    return RunConfig('EID_CANARY_CALIB_'+mode+'_s2026',method='FA',seed=2026,lr_peak_local=.1,solver=solver,
        exposure_point='end',endpoint_rounds=384 if is_dp else 3518,dropout_p=0. if is_dp else .1,
        lr_shape='fl_const',server_opt=ServerOptConfig(**declaration()['server']),dp=dp)


def build_protocol(out,*,role=TRAIN):
    exact_role(role)
    verify_source_arm()
    from v4int import launch_fl as F
    from .canary_replay import dp_accountant
    record=read(INIT);checkpoint=INIT.parent/record['fl_init']['path']
    _,identity,_=F.load_pretrain_init(INIT,model_variant=VARIANT)
    if identity['checkpoint_sha256']!=INIT_SHA or identity['state_digest']!=INIT_DIGEST or identity['seed']!=2026:
        raise Refused('not the actual own PB100 seed2026 endpoint')
    p=dict(format=FORMAT,status='CANDIDATE_NOT_EXECUTION_APPROVAL',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        declaration=declaration(),sources=[pin(q) for q in source_files()],
        contracts=[pin(q) for q in sorted((WORK/'contracts').rglob('*')) if q.is_file()],
        data=[pin(q) for q in data_files()],init=[pin(INIT),pin(checkpoint)],source_arm=pin(SOURCE_ARM),
        init_identity=identity,accountant=dp_accountant(),private_protocol=True,
        real_run_requires_independent_exact_review=True)
    p['self_sha256']=canonical(p)
    write_private(out,p)
    return p


def load_protocol(path,expected_sha=None,*,role=TRAIN):
    exact_role(role)
    if expected_sha is not None and sha(path)!=expected_sha:raise Refused('protocol SHA differs')
    p=read(path)
    if p.get('format')!=FORMAT or p.get('self_sha256')!=canonical(p) or p.get('declaration')!=declaration():
        raise Refused('not the fixed private canary CALIB candidate')
    for group,paths in [('sources',source_files()),('contracts',sorted(q for q in (WORK/'contracts').rglob('*') if q.is_file())),
                       ('data',data_files())]:
        if [r['path'] for r in p[group]]!=[str(physical(q).resolve()) for q in paths]:raise Refused('input inventory changed')
        for row in p[group]:check_pin(row)
    for row in p['init']:check_pin(row)
    check_pin(p['source_arm']);verify_source_arm()
    from .canary_replay import dp_accountant
    if p['accountant']!=dp_accountant():raise Refused('new population STANDARD accountant differs')
    return p


def real_inputs(protocol):
    from v4int.providers import StoreTrainProvider
    from v4int.verify import DataPaths
    from v4int import launch_fl as F
    paths=DataPaths.default(WORK)
    tp=StoreTrainProvider(TRAIN,paths=paths,history_policy='USER_HISTORY_L50',
        manifest_registry=WORK/'contracts/manifest_registry/REGISTRY_V2.json',registry_root=WORK,log=lambda _:None)
    if tp.manifest_id!=TRAIN or tp.n_decisions!=646067:raise Refused('actual CALIB TRAIN provider changed')
    keys,positions=F.client_index(tp,paths.manifest_dir)
    if len(keys)!=32768 or sum(v.size for v in positions.values())!=646067:
        raise Refused('actual real CALIB client population changed')
    lock=F.bind_data_protocol_lock_v2(paths,smoke=False)
    schema=F.load_production_schema(lock['schema_id'],lock['schema_self_sha256'],lock['schema_registry_path'])
    return tp,keys,positions,schema


def build_initialized_models(protocol,schema,*,device='cpu'):
    from v4models import build
    from v4int import launch_fl as F
    from fl.numerics import state_digest
    state,identity,_=F.load_pretrain_init(INIT,model_variant=VARIANT)
    if identity!=protocol['init_identity'] or state_digest(state)!=INIT_DIGEST:raise Refused('initializer identity differs')
    adapters=[]
    for _ in range(2):
        built=build('SASREC',schema,2026,device,v1_root=DATA,variant=VARIANT,dropout=(.1,))
        if built.K!=158486 or built.adapter.query_dim!=64 or not built.module.cfg.rich or \
                built.variant!=VARIANT or tuple(built.dropout)!=(.1,):raise Refused('actual source model/schema differs')
        built.adapter.load_state_(state)
        if state_digest(built.adapter.broadcast_state())!=INIT_DIGEST:raise Refused('loaded PB100 model state differs')
        adapters.append(built.adapter)
    for row in protocol['init']:check_pin(row)
    return adapters[0],adapters[1:]


def prepare(protocol_path,out,*,role=TRAIN):
    """Actual CPU CALIB/provider/init proof, no training or private membership."""
    exact_role(role)
    import numpy as np
    import torch
    from fl.client import valid_rows
    from fl.numerics import enforce_strict_fp32
    from . import canary_data as D
    enforce_strict_fp32(warn_only=False);torch.set_num_threads(4)
    p=load_protocol(protocol_path);tp,keys,positions,schema=real_inputs(p)
    counts=np.zeros(158486,dtype=np.int64)
    for start in range(0,646067,1024):
        batch=tp.batch(np.arange(start,min(start+1024,646067),dtype=np.int64))
        if int(valid_rows(batch).numel())!=len(batch['target_class']):raise Refused('TRAIN eligible examples changed')
        targets=batch['target_class'].numpy()
        if np.any(targets<0) or np.any(targets>=158486):raise Refused('TRAIN target class outside known catalogue')
        counts+=np.bincount(targets,minlength=158486)
    if counts.sum()!=646067:raise Refused('actual TRAIN popularity census differs')
    pool=D.long_tail_pool(counts,.2)
    definitions,references=D.public_definitions(pool,158486,n_canaries=1000,n_references=10000,
        history_length=20,max_len=50,seed=20261014)
    adapter,workers=build_initialized_models(p,schema)
    output=scoped(out)
    if output.exists():raise Refused('fresh private preparation directory required')
    with private_umask():
        output.mkdir(parents=True,mode=0o700)
        fd=os.open(output/'PUBLIC_DEFINITIONS.npz',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:
            np.savez(f,canary_histories=definitions.histories,canary_targets=definitions.targets,
                reference_histories=references.histories,reference_targets=references.targets)
    from fl.participation import ParticipationPlan
    from fl.runtime import fl_const_factor,round_survivors
    roster=tuple(keys)+tuple(D.PREFIX+f'{i:06d}' for i in range(1000))
    plan=ParticipationPlan.build(roster,seed=2026,manifest_hash='EID_E2_PUBLIC_ROSTER:'+definitions.commitment(),
        group_size=64,sampling='sweep')
    survival=[len(round_survivors(plan,make_config('UNFROZEN'),r)[1]) for r in range(3518)]
    # Actual architecture receives the exact same declared synthetic context
    # as future visits/scoring. No real example or synthetic example optimizes.
    synthetic_batch=complete_synthetic_context(definitions.batch(np.array([0],dtype=np.int64),repeats=10),adapter)
    empty_batch=complete_synthetic_context(definitions.batch(np.array([],dtype=np.int64),repeats=10),adapter)
    if int(valid_rows(synthetic_batch).numel())!=10 or int(valid_rows(empty_batch).numel())!=0:
        raise Refused('potential canary actual valid counts differ')
    logprob=rich_full_catalogue_logprob(adapter,synthetic_batch)
    if logprob.shape!=(10,) or not torch.isfinite(logprob).all():raise Refused('actual rich full-K forward failed')
    proof=dict(status='PREPARED_NOT_TRAINED',protocol=pin(protocol_path),module_sha256=sha(__file__),
        real_clients=32768,real_decisions=646067,potential_clients=len(roster),normalizer=656067,
        train_manifest=TRAIN,train_provider_fingerprint=tp.fingerprint,
        definition_file=pin(output/'PUBLIC_DEFINITIONS.npz'),definitions_sha256=definitions.commitment(),
        references_sha256=references.commitment(),schema_id=schema.schema_id,schema_sha256=schema.self_sha256,
        init_state_digest=INIT_DIGEST,model_K=adapter.K,model_query_dim=adapter.query_dim,
        private_nonDP_public_plan_sha256=plan.digest(),nonDP_rounds=3518,
        public_nonDP_min_survivors=min(survival),public_nonDP_empty_cohorts=sum(v==0 for v in survival),
        model_context='RICH_ALL_REGISTERED_BRANCHES_ENABLED',synthetic_actual_included_n_valid=10,
        synthetic_actual_absent_n_valid=0,actual_rich_full_K_forward=True,
        synthetic_context=synthetic_context_declaration(),statistical_power=attainable_bound_ceiling(),
        clock_LR_sha256=hashlib.sha256(json.dumps([.1*fl_const_factor(r,3518) for r in range(3518)]).encode()).hexdigest(),
        private_membership_drawn=False,DP_entropy_drawn=False,optimization_steps=0,GPU_initialized=torch.cuda.is_initialized(),
        MAIN_read=False,TEST_read=False)
    if proof['GPU_initialized'] or len(roster)!=33768:raise Refused('CPU preparation or fixed roster changed')
    write_private(output/'PREPARATION.json',proof)
    return proof


def load_preparation(path,protocol_path):
    p=read(path)
    if p.get('status')!='PREPARED_NOT_TRAINED' or p.get('protocol')!=pin(protocol_path) or \
            p.get('module_sha256')!=sha(__file__) or p.get('train_manifest')!=TRAIN or \
            (p.get('real_clients'),p.get('real_decisions'),p.get('potential_clients'),p.get('normalizer'))!=(32768,646067,33768,656067) or \
            p.get('private_membership_drawn') is not False or p.get('DP_entropy_drawn') is not False or \
            p.get('optimization_steps')!=0 or p.get('GPU_initialized') is not False or \
            p.get('model_context')!='RICH_ALL_REGISTERED_BRANCHES_ENABLED' or p.get('actual_rich_full_K_forward') is not True or \
            p.get('synthetic_context')!=synthetic_context_declaration() or p.get('statistical_power')!=attainable_bound_ceiling():
        raise Refused('not the exact source-bound CPU preparation')
    check_pin(p['definition_file'])
    return p


def load_definitions(preparation):
    import numpy as np
    from .canary_data import CanaryDefinitions
    check_pin(preparation['definition_file'])
    with np.load(preparation['definition_file']['path'],allow_pickle=False) as arrays:
        if set(arrays.files)!={'canary_histories','canary_targets','reference_histories','reference_targets'}:
            raise Refused('definition archive fields differ')
        definitions=CanaryDefinitions(arrays['canary_histories'],arrays['canary_targets'],158486,50)
        references=CanaryDefinitions(arrays['reference_histories'],arrays['reference_targets'],158486,50)
    if definitions.count!=1000 or references.count!=10000 or definitions.histories.shape[1]!=20 or \
            references.histories.shape[1]!=20 or definitions.commitment()!=preparation['definitions_sha256'] or \
            references.commitment()!=preparation['references_sha256']:
        raise Refused('frozen public canary/reference definitions differ')
    return definitions,references


def initialize_private_state(out,protocol_path,preparation_path):
    """Future operator action; draws no randomness before both commitments pass."""
    p=load_protocol(protocol_path);prep=load_preparation(preparation_path,protocol_path)
    load_definitions(prep)
    target=scoped(out)
    if target.exists():raise Refused('private membership must not be redrawn or overwritten')
    from .canary_data import private_fair_signs
    from .canary_replay import PrivateEntropy
    vectors={mode:private_fair_signs(1000).tolist() for mode in MODES}
    dp_entropy=PrivateEntropy.draw()
    # Independent draws; vectors can happen to coincide and must not be
    # rejection-sampled to force a truth-dependent distinction.
    obj=dict(format='EID_CANARY_PRIVATE_STATE_1',protocol=pin(protocol_path),preparation=pin(preparation_path),
        definitions_sha256=prep['definitions_sha256'],signs=vectors,
        dp_entropy=dict(participation_seed=dp_entropy.participation_seed,noise_seed=dp_entropy.noise_seed))
    write_private(target,obj)
    return obj


def load_private_state(path,protocol_path,preparation_path):
    import numpy as np
    from .canary_replay import PrivateEntropy
    target=physical(path)
    if target.stat().st_mode&0o077:raise Refused('private entropy/truth permissions too broad')
    state=read(target);prep=load_preparation(preparation_path,protocol_path)
    if state.get('format')!='EID_CANARY_PRIVATE_STATE_1' or state.get('protocol')!=pin(protocol_path) or \
            state.get('preparation')!=pin(preparation_path) or state.get('definitions_sha256')!=prep['definitions_sha256'] or \
            set(state.get('signs',{}))!=set(MODES):raise Refused('private state commitment differs')
    signs={}
    for mode in MODES:
        a=np.asarray(state['signs'][mode])
        if a.dtype.kind!='i' or a.shape!=(1000,) or not np.isin(a,[-1,1]).all():
            raise Refused('invalid independent fair membership vector')
        signs[mode]=a.astype(np.int8)
    if set(state)!={'format','protocol','preparation','definitions_sha256','signs','dp_entropy'}:
        raise Refused('membership entropy file whitelist differs')
    return signs,PrivateEntropy(**state['dp_entropy'])


def initialize_calibration_state(out,protocol_path,preparation_path):
    load_protocol(protocol_path);load_preparation(preparation_path,protocol_path)
    from .canary_replay import PrivateEntropy
    target=scoped(out)
    if target.exists():raise Refused('calibration entropy must not be redrawn or overwritten')
    entropy=PrivateEntropy.draw()
    obj=dict(format='EID_CANARY_CALIBRATION_ENTROPY_1',protocol=pin(protocol_path),preparation=pin(preparation_path),
        entropy=dict(participation_seed=entropy.participation_seed,noise_seed=entropy.noise_seed))
    write_private(target,obj)
    return obj


def load_calibration_state(path,protocol_path,preparation_path):
    from .canary_replay import PrivateEntropy
    target=physical(path)
    if target.stat().st_mode&0o077:raise Refused('calibration entropy permissions too broad')
    state=read(target)
    if set(state)!={'format','protocol','preparation','entropy'} or \
            state['format']!='EID_CANARY_CALIBRATION_ENTROPY_1' or state['protocol']!=pin(protocol_path) or \
            state['preparation']!=pin(preparation_path):
        raise Refused('separate calibration entropy binding differs; membership is not an accepted input')
    return PrivateEntropy(**state['entropy'])


def review_gate(review,protocol_path,preparation_path,*,mode=None,clip_path=None):
    r=read(review)
    if r.get('status')!='PASS' or r.get('scope')!='CANARY_CALIB_ACTUAL_TRAIN' or \
            r.get('reviewer') in (None,'','/root/pretrain_finalize','/root/canary_finish') or r.get('protocol')!=pin(protocol_path) or \
            r.get('preparation')!=pin(preparation_path) or r.get('module_sha256')!=sha(__file__) or \
            r.get('tests_sha256')!=sha(CODE/'experiment_extensions/test_canary_train.py') or \
            r.get('GPU_control')!='PASS' or r.get('real_training_allowed') is not True:
        raise Refused('exact independent full-trainer review and root-assigned GPU control required')
    if mode is not None and mode not in r.get('allowed_modes',[]):
        raise Refused('mechanism not admitted by the exact independent review')
    if mode=='DP' and (clip_path is None or r.get('fresh_calibration')!=pin(clip_path)):
        raise Refused('DP requires independent review of its exact fresh calibration')
    return r


def early_GPU(gpu):
    if gpu not in (0,1) or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):
        raise Refused('export exact physical CUDA_VISIBLE_DEVICES before any input/numerical import')


def fresh_run_directory(path,mode):
    p=scoped(path)
    expected=EXT/'runs/CANARY_AUDIT_01'/mode
    if mode not in (*MODES,'S_CAL') or p.resolve()!=expected.resolve() or p.exists():
        raise Refused('only fresh exact own private mechanism directory')
    with private_umask():p.mkdir(parents=True,mode=0o700)
    return p


def actual_calibration(protocol_path,preparation_path,calibration_state_path,review,out,*,gpu,role=TRAIN):
    exact_role(role);early_GPU(gpu)
    initialize_runtime(gpu)
    p=load_protocol(protocol_path);load_preparation(preparation_path,protocol_path)
    review_gate(review,protocol_path,preparation_path,mode='S_CAL')
    entropy=load_calibration_state(calibration_state_path,protocol_path,preparation_path)
    from fl.client import ClientData
    from fl.dp import median_clip_norm
    from .canary_replay import private_poisson_plan
    tp,keys,positions,schema=real_inputs(p)
    adapter,workers=build_initialized_models(p,schema,device='cuda:0')
    directory=fresh_run_directory(out,'S_CAL')
    plan=private_poisson_plan(keys,group_size=1024,entropy=entropy,public_roster_id='EID_E2_PUBLIC_ROSTER:SCAL_REAL_ONLY')
    get=lambda key:ClientData(key,tp.batch(positions[key]))
    with private_umask():
        checkpoint,progress=private_monitor(directory,'S_CAL')
        run=run_kernel(make_config('S_CAL'),plan,adapter,workers,get,{k:int(v.size) for k,v in positions.items()},646067,
            stop_after=20,checkpoint=checkpoint,progress=progress)
    clip=median_clip_norm(run.dp_norm_log)
    record=dict(format='EID_CANARY_FRESH_CALIBRATION_1',protocol=pin(protocol_path),preparation=pin(preparation_path),
        calibration_state=pin(calibration_state_path),init_checkpoint_sha256=INIT_SHA,init_state_digest=INIT_DIGEST,
        role=TRAIN,real_clients=32768,canaries=False,rounds=20,clock_rounds=384,passes=2,
        config_digest=run.cfg.digest(),private_plan_digest=run.plan.digest(),clip=clip,
        private_tuning_accounted=False,privacy_certified=False)
    write_private(directory/'CLIP_CALIBRATION.json',record)
    return record


def load_clip(path,protocol_path,preparation_path):
    r=read(path)
    if r.get('format')!='EID_CANARY_FRESH_CALIBRATION_1' or r.get('protocol')!=pin(protocol_path) or \
            r.get('preparation')!=pin(preparation_path) or \
            r.get('role')!=TRAIN or r.get('canaries') is not False or r.get('real_clients')!=32768 or \
            (r.get('rounds'),r.get('clock_rounds'),r.get('passes'))!=(20,384,2) or \
            r.get('init_checkpoint_sha256')!=INIT_SHA or r.get('init_state_digest')!=INIT_DIGEST or \
            r.get('config_digest')!=make_config('S_CAL').digest():
        raise Refused('not the exact own no-canary CALIB calibration source')
    check_pin(r['calibration_state'])
    load_calibration_state(r['calibration_state']['path'],protocol_path,preparation_path)
    clip=r['clip']['S']
    if isinstance(clip,bool) or not isinstance(clip,(int,float)) or not math.isfinite(clip) or clip<=0 or \
            r['clip']['rounds_0based']!=list(range(20)):
        raise Refused('fresh calibration median is invalid')
    return float(clip)


def model_only_payload(adapter,mode,accountant):
    """No state/protocol/data/model hashes, counts, entropy, clip or raw meta."""
    import torch
    from collections import OrderedDict
    if mode not in MODES:raise Refused('only final mechanisms can export model tensors')
    theta=adapter.broadcast_state(clone=True)
    if any(t.is_floating_point() and (t.dtype!=torch.float32 or not torch.isfinite(t).all()) for t in theta.values()):
        raise Refused('final model must be finite strict FP32')
    return dict(format='EID_CANARY_FINAL_MODEL_ONLY_1',theta=OrderedDict((k,v.detach().cpu().clone()) for k,v in theta.items()),
        mechanism=dict(mode=mode,scope='SYNTHETIC_INCLUSION_CONDITIONAL_FIXED_BACKGROUND_FIXED_CLIP',
            rounds=384 if mode=='DP' else 3518,local_passes=2 if mode=='DP' else 1,lr_shape='fl_const',
            fixed_population=33768,fixed_denominator=1024 if mode=='DP' else None,
            fixed_cell_accountant=copy.deepcopy(accountant) if mode=='DP' else None),
        private_tuning_accounted=False,privacy_certified=False)


def actual_train(mode,protocol_path,preparation_path,state_path,review,out,*,gpu,clip_path=None,role=TRAIN):
    exact_role(role);early_GPU(gpu)
    if mode not in MODES:raise Refused('only three declared canary mechanisms')
    initialize_runtime(gpu)
    p=load_protocol(protocol_path);prep=load_preparation(preparation_path,protocol_path)
    review_gate(review,protocol_path,preparation_path,mode=mode,clip_path=clip_path)
    signs,entropy=load_private_state(state_path,protocol_path,preparation_path)
    definitions,_=load_definitions(prep)
    clip=load_clip(clip_path,protocol_path,preparation_path) if mode=='DP' and clip_path else None
    config=make_config(mode,clip=clip)
    from fl.client import ClientData,valid_rows
    from fl.participation import ParticipationPlan
    from fl.runtime import freeze_item_tables,FROZEN_ITEM_KEYS
    from .canary_data import FixedRosterCanaries
    from .canary_replay import private_poisson_plan
    tp,keys,positions,schema=real_inputs(p)
    if tp.fingerprint!=prep['train_provider_fingerprint']:raise Refused('actual provider identity changed')
    adapter,workers=build_initialized_models(p,schema,device='cuda:0')
    real_get=lambda key:ClientData(key,tp.batch(positions[key]))
    raw_roster=FixedRosterCanaries(keys,real_get,definitions,signs[mode],train_manifest_id=TRAIN,repeats=10)
    roster=RichFixedRoster(raw_roster,adapter,real_keys=keys)
    if len(roster.keys)!=33768:raise Refused('potential client denominator changed with membership')
    counts={k:int(v.size) for k,v in positions.items()}
    counts.update({k:int(valid_rows(roster.get_client(k).examples).numel()) for k in roster.keys if k not in positions})
    if mode=='DP':
        plan=private_poisson_plan(roster.keys,group_size=1024,entropy=entropy,
            public_roster_id='EID_E2_PUBLIC_ROSTER:'+definitions.commitment())
    else:
        plan=ParticipationPlan.build(roster.keys,seed=2026,manifest_hash='EID_E2_PUBLIC_ROSTER:'+definitions.commitment(),
            group_size=64,sampling='sweep')
        if plan.digest()!=prep['private_nonDP_public_plan_sha256']:raise Refused('public non-DP schedule changed')
    if mode=='FROZEN':
        freeze_item_tables(adapter,FROZEN_ITEM_KEYS)
        theta=adapter.broadcast_state(clone=True)
        for worker in workers:freeze_item_tables(worker,FROZEN_ITEM_KEYS,source=theta)
    directory=fresh_run_directory(out,mode)
    with private_umask():
        checkpoint,progress=private_monitor(directory,mode)
        run=run_kernel(config,plan,adapter,workers,roster.get_client,counts,656067,
            entropy=entropy if mode=='DP' else None,checkpoint=checkpoint,progress=progress)
        payload=model_only_payload(adapter,mode,p['accountant'])
        import torch
        target=directory/'FINAL_MODEL_ONLY.pt'
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:torch.save(payload,f)
    completion=dict(status='COMPLETE_OWN_CANARY_MECHANISM',role=TRAIN,mode=mode,protocol=pin(protocol_path),
        preparation=pin(preparation_path),private_state=pin(state_path),model_file=pin(target),
        review=pin(review),source_init_state_digest=INIT_DIGEST,
        config_digest=run.cfg.digest(),private_actual_counters=run.counters.summary(),
        private_plan_digest=plan.digest(),private_LR_table_digest=run.lr_table['digest'],
        final_only_release=True,privacy_certified=False,private_tuning_accounted=False)
    write_private(directory/'COMPLETE_PRIVATE.json',completion)
    return completion


def audit_fixed_scores(scores,private_signs):
    """Separate private-truth audit; nine predeclared tests, lower bounds only."""
    import numpy as np
    from .canary_statistics import fixed_tail_guesses,correct_guesses,empirical_epsilon_lower_bound
    values=np.asarray(scores,dtype=np.float64);truth=np.asarray(private_signs)
    if values.shape!=(1000,) or truth.dtype.kind!='i' or truth.shape!=(1000,) or not np.isin(truth,[-1,1]).all():
        raise Refused('exact committed1000 public scores and fair private signs required')
    results=[]
    for kp,kn in TAIL_RULES:
        guesses=fixed_tail_guesses(values,kp,kn)
        n_correct=correct_guesses(guesses,truth)
        result=empirical_epsilon_lower_bound(1000,kp+kn,n_correct,delta=5e-6,confidence=.95,n_predeclared_tests=9)
        results.append(dict(result,k_positive=kp,k_negative=kn,
            interpretation='DETECTED_CONTRADICTION_AT_EPS8' if result['empirical_epsilon_lower_bound']>8 else 'NO_DETECTED_CONTRADICTION_NOT_PRIVACY_PASS'))
    return dict(scope='SEPARATE_TRUTH_AUDIT_OUTSIDE_MODEL_DP_RELEASE',results=results,
        simultaneous_family=9,bound_type='LOWER_BOUND_ONLY',privacy_certified=False,
        attainable_bound_ceiling=attainable_bound_ceiling(),
        eps8_rejection_possible=False,interpretation='DESCRIPTIVE_LEAKAGE_STUDY_NOT_CAPABLE_OF_REJECTING_EPS8')


def initialize_runtime(gpu):
    early_GPU(gpu)
    import torch
    from fl.numerics import enforce_strict_fp32
    enforce_strict_fp32(warn_only=False)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise Refused('one root-assigned physical GPU must be visible')


def private_monitor(directory,mode):
    """Private recovery evidence only. No resume CLI or public diagnostics."""
    from fl.checkpoint import CheckpointManager
    checkpoint=CheckpointManager(directory,'PRIVATE_CHECKPOINT',retain_weights=True)
    path=directory/'PRIVATE_PROGRESS.jsonl'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.close(fd)
    def progress(run,record):
        loss=record.get('train_loss',{})
        if loss.get('finite') is False:
            raise Refused('nonfinite private training loss')
        row=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),mode=mode,
            rounds_completed=run.state.cursor,fixed_endpoint=run.cfg.endpoint_rounds,
            private_actual_counters=run.counters.summary(),private_loss=loss)
        # File is already exclusively created mode600. No secrets go to stdout.
        with path.open('a') as f:
            json.dump(row,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush()
    return checkpoint,progress


def load_model_only(path,mode,adapter,accountant):
    """Only own exact finite model whitelist; no arbitrary checkpoint loader."""
    import torch
    obj=torch.load(physical(path),map_location='cpu',weights_only=True)
    expected=model_only_payload(adapter,mode,accountant)
    if not isinstance(obj,dict) or set(obj)!=set(expected) or any(
            obj.get(k)!=expected[k] for k in expected if k!='theta'):
        raise Refused('not the exact model-only export whitelist')
    theta=obj['theta'];base=adapter.broadcast_state()
    if not isinstance(theta,dict) or set(theta)!=set(base) or any(
            not isinstance(theta[k],torch.Tensor) or theta[k].shape!=base[k].shape or
            theta[k].dtype!=base[k].dtype or not torch.isfinite(theta[k]).all() for k in base):
        raise Refused('final model tensor schema or finiteness differs')
    adapter.load_state_(theta)
    return obj


def actual_score(mode,protocol_path,preparation_path,state_path,review,completion_path,out,*,gpu,role=TRAIN,clip_path=None):
    """Score public definitions first, then open truth for separate private audit."""
    exact_role(role);early_GPU(gpu)
    if mode not in MODES:raise Refused('only declared fixed audit mechanisms')
    initialize_runtime(gpu)
    p=load_protocol(protocol_path);prep=load_preparation(preparation_path,protocol_path)
    review_gate(review,protocol_path,preparation_path,mode=mode,clip_path=clip_path)
    expected_directory=EXT/'runs/CANARY_AUDIT_01'/mode
    if physical(completion_path).resolve()!=(expected_directory/'COMPLETE_PRIVATE.json').resolve():
        raise Refused('only exact own completed canary mechanism can be scored')
    completion=read(completion_path)
    if completion.get('status')!='COMPLETE_OWN_CANARY_MECHANISM' or completion.get('mode')!=mode or \
            completion.get('role')!=TRAIN or completion.get('protocol')!=pin(protocol_path) or \
            completion.get('preparation')!=pin(preparation_path) or completion.get('private_state')!=pin(state_path) or \
            completion.get('review')!=pin(review) or completion.get('source_init_state_digest')!=INIT_DIGEST or \
            physical(completion['model_file']['path']).resolve()!=(expected_directory/'FINAL_MODEL_ONLY.pt').resolve():
        raise Refused('completed model, private commitments or source init differ')
    check_pin(completion['model_file'])
    definitions,references=load_definitions(prep)
    _,_,_,schema=real_inputs(p)
    initializer,others=build_initialized_models(p,schema,device='cuda:0')
    final_model=others[0]
    load_model_only(completion['model_file']['path'],mode,final_model,p['accountant'])
    from fl.numerics import state_digest
    from fl.checkpoint import rng_state,payload_digest
    from .canary_data import finite_reference_exposure
    import numpy as np
    import torch
    before=(state_digest(initializer.broadcast_state()),state_digest(final_model.broadcast_state()),payload_digest(rng_state()))
    # Scores cannot access the private membership bits. Definitions and tail
    # counts were committed before the separate private-state action.
    canary_scores=rich_drift_scores(initializer,final_model,definitions,device='cuda:0',batch_size=32)
    reference_scores=rich_drift_scores(initializer,final_model,references,device='cuda:0',batch_size=32)
    exposure=finite_reference_exposure(canary_scores,reference_scores)
    after=(state_digest(initializer.broadcast_state()),state_digest(final_model.broadcast_state()),payload_digest(rng_state()))
    if before!=after:raise Refused('audit scorer changed model or ambient RNG')
    signs,_=load_private_state(state_path,protocol_path,preparation_path)
    audit=audit_fixed_scores(canary_scores,signs[mode])
    target=scoped(out)
    if target.exists():raise Refused('fresh private audit directory required')
    with private_umask():
        target.mkdir(parents=True,mode=0o700)
        fd=os.open(target/'SCORES_PRIVATE.npz',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:
            np.savez(f,canary_scores=canary_scores,reference_scores=reference_scores,finite_reference_exposure=exposure)
    result=dict(status='COMPLETE_SEPARATE_PRIVATE_TRUTH_AUDIT',mode=mode,role=TRAIN,
        protocol=pin(protocol_path),preparation=pin(preparation_path),private_state=pin(state_path),
        review=pin(review),completion=pin(completion_path),scores=pin(target/'SCORES_PRIVATE.npz'),
        definitions_sha256=prep['definitions_sha256'],references_sha256=prep['references_sha256'],
        score='STRICT_FP32_FULL_K_LOGPROB_FINAL_MINUS_INIT',catalogue_K=158486,
        canary_count=1000,reference_count=10000,finite_reference_exposure_max=math.log2(10001),
        finite_reference_exposure_median=float(np.median(exposure)),finite_reference_exposure_max_observed=float(exposure.max()),
        model_and_RNG_unchanged=True,private_truth_deleted=False,audit=audit,
        private_tuning_accounted=False,privacy_certified=False)
    write_private(target/'AUDIT_PRIVATE.json',result)
    return result



def synthetic_context_declaration():
    return dict(distribution='PREDECLARED_NEUTRAL_MISSING_CONTEXT_IN_STANDARDIZED_FEATURE_SPACE',
        reconstruction_of_real_raw_metadata=False,base_categorical_valid_token=1,base_categorical_padding_token=0,
        base_categorical_valid_semantics='MISSING',session_offset_valid_token=1,session_offset_semantics='CURRENT_SESSION',
        session_mode_valid_token=1,session_mode_semantics='UNKNOWN',float_context_value=0.,unsupported_user_masks=0.,
        valid_missing_flags=['price_missing','brand_missing','category_missing','category_code_missing',
            'main_category_missing','prev_gap_unavailable','item_age_unavailable'],
        all_other_flags=0.,padding_float_value=0.,label_or_membership_features=False,
        same_context_train_canary_score_reference_score=True,
        branches_dropped=False,model_cfg_mutated=False,source_frozen_item_only_scorer_used=False)


@lru_cache(maxsize=1)
def _attainable_bound_ceiling():
    from .canary_statistics import empirical_epsilon_lower_bound,audit_tail_bound,REFERENCE
    rows=[]
    for kp,kn in TAIL_RULES:
        r=kp+kn
        bound=empirical_epsilon_lower_bound(1000,r,r,delta=5e-6,confidence=.95,n_predeclared_tests=9)
        rows.append(dict(k_positive=kp,k_negative=kn,guesses=r,all_correct=r,
            maximum_attainable_epsilon_lower_bound=bound['empirical_epsilon_lower_bound'],
            all_correct_null_tail_at_eps8=audit_tail_bound(8.,5e-6,1000,r,r)))
    return dict(n_canaries=1000,confidence=.95,delta=5e-6,simultaneous_tests=9,
        significance_per_test=(1.-.95)/9,reference=REFERENCE,rows=rows,
        eps8_rejection_possible=False,
        proof='A(v+1)<=A(v)+2*P(W=v); tail(v+1)=tail(v)-P(W=v); 2*m*delta=.01<1, so null bound decreases with correct count. Inversion is maximized by all-correct guesses.',
        interpretation='FINITE_DESIGN_CEILING_BEFORE_MEMBERSHIP_DRAWS; DESCRIPTIVE_LOWER_BOUNDS_ONLY')


def attainable_bound_ceiling():
    return copy.deepcopy(_attainable_bound_ceiling())


def complete_synthetic_context(batch,adapter):
    """Fixed neutral missing context; never infer from target, truth or membership.

    The actual rich model and original projection stay unchanged. Context is
    an explicit synthetic distribution in standardized feature space, not a
    reconstruction of real raw metadata. Padding and missing sentinels differ.
    """
    import torch
    from v4models.schema import names_sequence,check_batch_layout
    from v4hist.windows import MAIN_UNKNOWN
    if MAIN_UNKNOWN!=1:raise Refused('registered session-mode UNKNOWN sentinel changed')
    widths=adapter.input_widths
    if not adapter.module.cfg.rich:raise Refused('own synthetic context is for actual rich models')
    out=dict(batch)
    valid=batch['attention_mask'].to(torch.bool)
    B,T=valid.shape
    for name in widths.categorical:
        if name!='item_tokens':out[name]=valid.to(torch.long)
    device=valid.device
    out['event_numeric_features']=torch.zeros((B,T,widths.numeric_dim),dtype=torch.float32,device=device)
    flags=torch.zeros((B,T,widths.flags_dim),dtype=torch.float32,device=device)
    for name in synthetic_context_declaration()['valid_missing_flags']:
        if name in widths.flag_names:flags[...,widths.flag_names.index(name)]=valid.to(torch.float32)
    out['event_quality_flags']=flags
    out['user_context']=torch.zeros((B,widths.user_context_dim),dtype=torch.float32,device=device)
    out['user_context_masks']=torch.zeros((B,widths.user_mask_dim),dtype=torch.float32,device=device)
    out['session_offset_tokens']=valid.to(torch.long)
    out['session_mode_main_category_tokens']=valid.to(torch.long)*MAIN_UNKNOWN
    for name in ('current_matches_session_mode','session_mode_share','current_matches_session_mode_available'):
        out[name]=torch.zeros((B,),dtype=torch.float32,device=device)
    out['feature_view']=widths.view_id
    out.update(names_sequence(widths))
    check_batch_layout(out,widths)
    return out


class RichFixedRoster:
    """Own context wrapper; unchanged real clients and immutable fixed roster."""
    def __init__(self,raw_roster,adapter,*,real_keys):
        self.raw,self.adapter,self.keys=raw_roster,adapter,raw_roster.keys
        self.real=frozenset(real_keys)
    def get_client(self,key):
        from fl.client import ClientData
        client=self.raw.get_client(key)
        return client if key in self.real else ClientData(key,complete_synthetic_context(client.examples,self.adapter))


def rich_full_catalogue_logprob(adapter,batch):
    """Explicit owned rich-model scorer; no target/truth passed to query."""
    import torch
    from fl.numerics import assert_strict_fp32
    assert_strict_fp32()
    if not adapter.module.cfg.rich or torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu') or \
            not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled() or \
            not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark or \
            any(t.is_floating_point() and t.dtype!=torch.float32 for t in adapter.module.state_dict().values()):
        raise Refused('actual strict FP32 deterministic rich model required')
    target=batch['target_class']
    if target.dtype!=torch.long or target.ndim!=1 or torch.any((target<0)|(target>=adapter.K)):
        raise Refused('known int64 target classes required')
    # Adapter/project/query receives context and input tensors only. Outcomes
    # remain local to gather after the full-K forward and denominator.
    inputs={k:v for k,v in batch.items() if k not in ('target_class','loss_mask','included','signs','private_signs')}
    flags=[(m,m.training) for m in adapter.module.modules()]
    try:
        adapter.module.eval()
        with torch.inference_mode():
            logits=adapter.scores(inputs)
            if logits.dtype!=torch.float32 or logits.shape!=(len(target),adapter.K) or not torch.isfinite(logits).all():
                raise Refused('finite strict FP32 full-catalogue rich logits required')
            return logits.gather(1,target[:,None]).squeeze(1)-torch.logsumexp(logits,dim=1)
    finally:
        for module,flag in flags:module.training=flag


def rich_drift_scores(initializer,final_model,definitions,*,device='cpu',batch_size=32):
    import numpy as np
    import torch
    if type(batch_size) is not int or batch_size<1 or initializer.K!=final_model.K or initializer.K!=definitions.catalogue_K:
        raise Refused('fixed positive batch and matching rich catalogue required')
    scores=np.empty(definitions.count,dtype=np.float64)
    for start in range(0,definitions.count,batch_size):
        end=min(start+batch_size,definitions.count)
        batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in definitions.batch(np.arange(start,end)).items()}
        first=complete_synthetic_context(batch,initializer)
        second=complete_synthetic_context(batch,final_model)
        a=rich_full_catalogue_logprob(initializer,first).double()
        b=rich_full_catalogue_logprob(final_model,second).double()
        scores[start:end]=(b-a).cpu().numpy()
    if not np.isfinite(scores).all():raise Refused('nonfinite rich canary drift')
    return scores

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build-protocol');b.add_argument('--out',required=True)
    for name in ('prepare','draw-private-state','draw-calibration-state','calibrate','train','score'):
        q=sub.add_parser(name)
        q.add_argument('--protocol',required=True);q.add_argument('--protocol-sha256',required=True)
        q.add_argument('--out',required=True)
        if name!='prepare':
            q.add_argument('--preparation',required=True);q.add_argument('--preparation-sha256',required=True)
            q.add_argument('--review',required=True);q.add_argument('--review-sha256',required=True)
        if name in ('calibrate','train','score'):
            state_name='calibration-state' if name=='calibrate' else 'private-state'
            q.add_argument('--'+state_name,required=True);q.add_argument('--'+state_name+'-sha256',required=True)
            q.add_argument('--gpu',type=int,choices=(0,1),required=True)
        if name in ('train','score'):
            q.add_argument('--mode',choices=MODES,required=True)
            q.add_argument('--clip');q.add_argument('--clip-sha256')
        if name=='score':q.add_argument('--completion',required=True);q.add_argument('--completion-sha256',required=True)
    a=parser.parse_args(argv)
    # Exact byte guards happen before any training/data/model loading.
    if a.command=='build-protocol':
        build_protocol(a.out)
    else:
        if hasattr(a,'gpu'):early_GPU(a.gpu)
        for name in ('protocol','preparation','review','private_state','calibration_state','clip','completion'):
            path=getattr(a,name,None)
            expected=getattr(a,name+'_sha256',None)
            if (path is None)!=(expected is None):raise Refused('path and exact SHA must be supplied together')
            if path is not None and sha(path)!=expected:raise Refused(name+' exact SHA differs')
        if a.command=='prepare':prepare(a.protocol,a.out)
        elif a.command in ('draw-private-state','draw-calibration-state'):
            review_gate(a.review,a.protocol,a.preparation,mode='S_CAL' if a.command=='draw-calibration-state' else None)
            action=initialize_calibration_state if a.command=='draw-calibration-state' else initialize_private_state
            action(a.out,a.protocol,a.preparation)
        elif a.command=='calibrate':actual_calibration(a.protocol,a.preparation,a.calibration_state,a.review,a.out,gpu=a.gpu)
        elif a.command=='train':actual_train(a.mode,a.protocol,a.preparation,a.private_state,a.review,a.out,
            gpu=a.gpu,clip_path=a.clip)
        else:actual_score(a.mode,a.protocol,a.preparation,a.private_state,a.review,a.completion,a.out,
            gpu=a.gpu,clip_path=a.clip)
    print(json.dumps(dict(status='COMPLETED_OWN_OPERATION',command=a.command,output=str(physical(a.out)))))


if __name__=='__main__':main()
