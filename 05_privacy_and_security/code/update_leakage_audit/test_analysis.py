"""Scientific controls for the new E3 analysis, independent of real uploads."""
import numpy as np
import pytest
from collections import OrderedDict

from .analysis import (CountSketch, ground_truth, set_metrics, cohort_mean_ci, point_rule,
                       _validate_split, category_attack, item_attack)
from .secagg_features import SecAggProjector, SecAggValidator, secagg_factory


def population():
    records, cohorts = [], {}
    for c in range(30):
        split = 'train' if c < 20 else 'holdout'
        keys = [f'{c:03d}-{i:02d}' for i in range(10)]
        cohorts[c] = dict(round=c, split=split, planned_keys=keys, observed_keys=keys,
                          input_row_ids=np.arange(3,103), encoder_delta=np.zeros(8))
        for i,key in enumerate(keys):
            feature = np.zeros(8); feature[0] = 1 if i%2 else -1
            item = 3+c*10+i
            records.append(dict(key=key, split=split, cohort_id=c, input_table_status='SHARED',
                                input_row_ids=np.array([item]), encoder_delta=feature,
                                truth=dict(input_items=[item], category=3+i%2)))
    return records, cohorts


def test_truth_excludes_masked_rows_padding_unknown_and_future_targets():
    examples = dict(target_class=np.array([8,9,-1,11]), loss_mask=np.array([1,0,1,1]),
                    item_tokens=np.array([[3,4,0],[7,8,0],[20,0,0],[1,2,6]]),
                    attention_mask=np.array([[1,1,0],[1,1,0],[1,0,0],[1,1,1]]),
                    main_category_tokens=np.array([[3,4,0],[15,15,0],[15,0,0],[1,2,3]]))
    truth = ground_truth(examples)
    assert truth['input_items']==[3,4,6] and truth['target_items']==[11,14]
    assert truth['category']==3 and truth['valid_rows']==2


def test_unknown_category_and_tie_are_explicit():
    ex=dict(target_class=np.array([1]),item_tokens=np.array([[3,4]]),main_category_tokens=np.array([[2,1]]))
    assert ground_truth(ex)['category']==-1
    ex['main_category_tokens']=np.array([[5,3]])
    assert ground_truth(ex)['category']==3


def test_projection_is_linear_deterministic_and_preserves_global_rng():
    np.random.seed(7); before=np.random.get_state()
    a=CountSketch(800); b=CountSketch(800)
    x=np.arange(800)/800; y=np.arange(800)[::-1]/800
    np.testing.assert_allclose(a.transform(2*x+3*y),2*a.transform(x)+3*a.transform(y),atol=1e-12)
    np.testing.assert_array_equal(a.transform(x),b.transform(x))
    after=np.random.get_state();np.testing.assert_array_equal(before[1],after[1])
    assert before[2:]==after[2:] and a.describe()==b.describe()


@pytest.mark.parametrize('x',[np.zeros(2),np.full(4,np.nan)])
def test_projection_refuses_invalid_features(x):
    with pytest.raises(ValueError):CountSketch(4).transform(x)


def test_set_recovery_is_against_singleton_not_union():
    result=set_metrics(range(100),[1,2])
    assert result['recall']==1 and result['precision']==.02
    assert result['f1']==pytest.approx(4/102)
    with pytest.raises(ValueError):set_metrics([],[])


def test_bootstrap_resamples_whole_cohorts_not_correlated_users():
    a=cohort_mean_ci(np.arange(10),np.arange(10))
    b=cohort_mean_ci(np.repeat(np.arange(10),50),np.repeat(np.arange(10),50))
    assert a['mean']==b['mean'] and a['ci95']==b['ci95']
    with pytest.raises(ValueError):cohort_mean_ci([1,2],[0,1])


@pytest.mark.parametrize('single,aggregate,expected',[(.11,.02,'SECAGG_NEEDED'),
        (.1,0,'NO_MATERIAL_SINGLE_ADVANTAGE_BY_POINT_RULE'),(.11,.021,'AGGREGATE_ATTACK_REMAINS')])
def test_prespecified_point_threshold_boundaries(single,aggregate,expected):
    assert point_rule(single,aggregate)==expected


def test_user_cohort_and_unsampled_peer_leakage_are_refused():
    records,cohorts=population();_validate_split(records,cohorts)
    records[-1]['key']=records[0]['key']
    with pytest.raises(ValueError,match='duplicate users'):_validate_split(records,cohorts)
    records,cohorts=population();cohorts[20]['planned_keys'].append(records[0]['key'])
    with pytest.raises(ValueError,match='unsampled peer'):_validate_split(records,cohorts)
    records,cohorts=population();cohorts[20]['split']='train'
    with pytest.raises(ValueError,match='cohort split'):_validate_split(records,cohorts)


def test_item_attack_requires_real_training_baseline():
    records,cohorts=population()
    for r in records:
        if r['split']=='train':r['truth']['input_items']=[]
    result=item_attack(records,cohorts,catalogue_known_count=400)
    assert result['status']=='INCOMPLETE' and result['n_train']==0


def test_input_recovery_and_matching_cardinality_baseline():
    records,cohorts=population()
    for c in cohorts:
        cohorts[c]['input_row_ids']=np.arange(3+c*10,13+c*10)
    result=item_attack(records,cohorts,catalogue_known_count=400)
    assert result['status']=='VALID_DESCRIPTIVE'
    assert result['metrics']['single_f1']['mean']==1
    assert result['metrics']['single_baseline_f1']['mean']==0
    assert result['metrics']['aggregate_recall']['mean']==1
    assert result['metrics']['aggregate_f1']['mean']==pytest.approx(2/11)
    assert result['support_cardinality']['aggregate']['mean']==10


def test_linear_probe_recovers_signal_and_aggregate_cannot_identify_balanced_singletons():
    records,cohorts=population();result=category_attack(records,cohorts)
    assert result['status']=='VALID_DESCRIPTIVE'
    assert result['metrics']['single_accuracy']['mean']==1
    assert result['metrics']['majority_baseline_accuracy']['mean']==.5
    assert result['metrics']['aggregate_accuracy']['mean']==.5
    assert result['threshold_met'] and not result['population_privacy_certified']


def test_missing_category_probe_is_incomplete():
    records,cohorts=population()
    for r in records:r['truth']['category']=-1
    assert category_attack(records,cohorts)['status']=='INCOMPLETE'


def test_nonconvergence_is_incomplete_before_any_holdout_prediction(monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    fit=LogisticRegression.fit
    def warned_fit(self,*args,**kwargs):
        result=fit(self,*args,**kwargs)
        warnings.warn('forced TRAIN convergence warning',ConvergenceWarning)
        return result
    def forbidden_prediction(*args,**kwargs):
        raise AssertionError('holdout prediction must not run after nonconverged TRAIN fit')
    monkeypatch.setattr(LogisticRegression,'fit',warned_fit)
    monkeypatch.setattr(LogisticRegression,'predict',forbidden_prediction)
    records,cohorts=population();result=category_attack(records,cohorts)
    assert result['status']=='INCOMPLETE'
    assert result['failed_probe']=='single'
    assert result['optimizer_diagnostics']['single']['convergence_warning_count']==1
    assert result['optimizer_diagnostics']['single']['n_iter']
    assert result['optimizer_diagnostics']['aggregate']['status']=='NOT_FITTED'


def test_frozen_input_rows_are_not_a_favorable_privacy_result():
    records,cohorts=population()
    for r in records:r['input_table_status']='FROZEN_NOT_UPLOADED'
    result=item_attack(records,cohorts,catalogue_known_count=400)
    assert result['status']=='NOT_APPLICABLE' and 'verdict' not in result


def test_limited_coordinate_secagg_identity_with_real_dropout_and_weights():
    state=OrderedDict([('item_embed.weight',np.zeros((12,2))),('encoder',np.zeros(8))])
    projector=SecAggProjector(state,raw_coordinates=16,sketch_dimensions=8)
    cohort=dict(round=0,planned_keys=['a','b','c','d'],observed_keys=['a','b','c'])
    contributions=[]
    for i,key in enumerate(cohort['observed_keys']):
        delta=OrderedDict((k,np.full(v.shape,(i+1)*.001)) for k,v in state.items())
        contributions.append((key,projector(delta,delta['encoder']),i+1))
    receipt=SecAggValidator(projector)(cohort,contributions)
    assert receipt['bit_identical'] and receipt['n_dropped']==1
    assert receipt['nonzero_encoded_words']>0 and not receipt['production_crypto']


def test_secagg_refuses_silent_clipping_trivial_identity_or_missing_peer():
    state=OrderedDict(encoder=np.zeros(8));p=SecAggProjector(state,raw_coordinates=4,sketch_dimensions=4)
    cohort=dict(round=0,planned_keys=['a','b','c'],observed_keys=['a','b','c'])
    validator=SecAggValidator(p)
    with pytest.raises(ValueError,match='trivial'):
        validator(cohort,[(k,np.zeros(8),1) for k in cohort['observed_keys']])
    with pytest.raises(ValueError,match='bound'):
        validator(cohort,[(k,np.full(8,100),1) for k in cohort['observed_keys']])
    with pytest.raises(ValueError,match='survivor'):
        validator(cohort,[('a',np.zeros(8),1)])


@pytest.mark.parametrize('frozen',[False,True])
def test_secagg_factory_uses_observer_keys_and_pins_before_training(frozen):
    from types import SimpleNamespace
    theta=OrderedDict([('output_bias',np.zeros(10)),('encoder',np.zeros(5000)),
                       ('item_embed.weight',np.zeros((12,2))),('output_embed',np.zeros((12,2)))])
    shared=[k for k in theta if not(frozen and k in {'item_embed.weight','output_bias','output_embed'})]
    server=SimpleNamespace(manifest=SimpleNamespace(shared_keys=shared),
                           adapter=SimpleNamespace(broadcast_state=lambda:theta))
    state={'server':server,'provider':'not_a_tensor'}
    validator,projector=secagg_factory(state)
    assert projector.keys == (('encoder',) if frozen else ('item_embed.weight','encoder'))
    assert 'self_sha256' in state['secagg_binding']
    assert state['secagg_binding']['raw_coordinates']==4096
    assert state['secagg_binding']['encoder_sketch']['dimensions']==512
