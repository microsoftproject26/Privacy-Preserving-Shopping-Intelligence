"""Predeclared CALIB-only E3 attacks; only aggregate metrics leave this module.

Input-set support F1 is the primary item metric. The popularity baseline gets
the same observable prediction cardinality, never the oracle true-set size.
Category labels count known categories in repeated valid TRAIN input windows,
not unique events. Linear CountSketch and train-only scaling precede a fixed
L2 logistic probe. Holdout uncertainty resamples whole cohorts.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np

PROTOCOL = {
    'format': 'PPSI_E3_ATTACK_PROTOCOL_2', 'label': 'POST_HOC_EXPLORATORY',
    'role': 'CALIBRATION', 'item_primary': 'set_F1_exact_nonzero_known_input_rows',
    'item_baseline': 'train_user_item_presence_popularity_matched_prediction_cardinality',
    'category_target': 'dominant_known_main_category_in_repeated_valid_TRAIN_input_windows',
    'category_unknown': -1, 'category_tie': 'minimum_token_id',
    'encoder_projection': 'linear_CountSketch_512', 'projection_seed': 20261013,
    'probe': 'L2_LogisticRegression_C1_lbfgs_train_only_StandardScaler',
    'probe_optimizer': {'solver': 'lbfgs', 'C': 1.0, 'tol': 1e-7, 'max_iter': 20000},
    'numerical_amendment': 'fixed_iteration_cap_increased_from_2000_after_TRAIN_nonconvergence_only; objective_and_holdout_protocol_unchanged',
    'bootstrap': '1000_holdout_cohort_resamples_conditional_on_fitted_probe',
    'bootstrap_seed': 20261014, 'single_threshold': 0.10, 'aggregate_threshold': 0.02,
    'interpretation': 'DESCRIPTIVE_point_rule_and_CI; not_population_privacy_certification',
    'secagg_identity_control': '4096_pinned_raw_coordinates_plus_512_linear_encoder_sketch',
    'secagg_fixedpoint_clip': 1.0, 'secagg_graph_k': 32, 'secagg_seed': 20261015,
    'category_minimum_known_users': {'train': 100, 'holdout': 50},
    'item_minimum_known_users': {'train': 100, 'holdout': 50},
    'aggregate_release': 'ideal_FP64_consumed_weighted_mean_of_observed_FP32_deltas; distinct_from_server_FP32_full_weight_release',
}


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def ground_truth(examples):
    target = _np(examples['target_class']).astype(np.int64)
    valid = target >= 0
    if 'loss_mask' in examples:
        valid &= _np(examples['loss_mask']).astype(bool)
    items = _np(examples['item_tokens']).astype(np.int64)
    if items.ndim != 2 or items.shape[0] != target.size:
        raise ValueError('TRAIN item windows have an invalid shape')
    mask = _np(examples.get('attention_mask', items != 0)).astype(bool)
    if mask.shape != items.shape:
        raise ValueError('TRAIN attention mask has an invalid shape')
    mask &= valid[:, None]
    input_items = np.unique(items[mask & (items >= 3)]).tolist()
    category = -1
    known_positions = 0
    if 'main_category_tokens' in examples:
        cats = _np(examples['main_category_tokens']).astype(np.int64)
        if cats.shape != items.shape:
            raise ValueError('TRAIN category windows have an invalid shape')
        known = cats[mask & (cats >= 3)]
        known_positions = int(known.size)
        if known.size:
            values, counts = np.unique(known, return_counts=True)
            category = int(values[np.flatnonzero(counts == counts.max())[0]])
    return dict(input_items=input_items, target_items=np.unique(target[valid] + 3).tolist(),
                category=category, category_known=category >= 3, valid_rows=int(valid.sum()),
                known_category_positions=known_positions,
                category_definition=PROTOCOL['category_target'])


class CountSketch:
    """A fixed, seeded linear transform independent of data and global RNG."""
    def __init__(self, n_features, dimensions=512, seed=20261013):
        if n_features < 1 or dimensions < 1:
            raise ValueError('projection dimensions must be positive')
        rng = np.random.Generator(np.random.PCG64(seed))
        self.bins = rng.integers(dimensions, size=n_features, dtype=np.int64)
        self.signs = (2 * rng.integers(2, size=n_features, dtype=np.int8) - 1).astype(np.float64)
        self.n_features, self.dimensions, self.seed = n_features, dimensions, seed

    def transform(self, values):
        x = np.asarray(_np(values), dtype=np.float64).reshape(-1)
        if x.size != self.n_features or not np.isfinite(x).all():
            raise ValueError('encoder delta is invalid for the pinned linear projection')
        return np.bincount(self.bins, weights=x * self.signs, minlength=self.dimensions)

    def describe(self):
        sha = hashlib.sha256(self.bins.astype('<i8').tobytes() + self.signs.astype('<f8').tobytes()).hexdigest()
        return dict(kind='linear_CountSketch', seed=self.seed, n_features=self.n_features,
                    dimensions=self.dimensions, map_sha256=sha)


def set_metrics(predicted, truth):
    p, t = set(predicted), set(truth)
    if not t:
        raise ValueError('empty known input truth cannot provide set-recovery evidence')
    overlap = len(p & t)
    precision = overlap / len(p) if p else 0.0
    recall = overlap / len(t)
    return dict(precision=precision, recall=recall,
                f1=2 * overlap / (len(p) + len(t)), predicted=len(p), truth=len(t))


def cohort_mean_ci(values, cohort_ids, *, seed=20261014, repetitions=1000):
    x, ids = np.asarray(values, dtype=np.float64), np.asarray(cohort_ids)
    if x.ndim != 1 or x.size != ids.size or not np.isfinite(x).all() or not x.size:
        raise ValueError('invalid paired cohort scores')
    clusters = np.unique(ids)
    if clusters.size < 5:
        raise ValueError('fewer than five holdout cohorts: uncertainty analysis is incomplete')
    totals = np.array([x[ids == c].sum() for c in clusters])
    counts = np.array([(ids == c).sum() for c in clusters])
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = rng.integers(clusters.size, size=(repetitions, clusters.size))
    draws = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return dict(mean=float(x.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                n_users=int(x.size), n_cohorts=int(clusters.size), repetitions=repetitions,
                scope='holdout_cohort_bootstrap_conditional_on_fixed_attack_training')


def point_rule(single, aggregate):
    if single > .10:
        return 'SECAGG_NEEDED' if aggregate <= .02 else 'AGGREGATE_ATTACK_REMAINS'
    return 'NO_MATERIAL_SINGLE_ADVANTAGE_BY_POINT_RULE'


def _paired_rule(single, aggregate, ids):
    a = cohort_mean_ci(single, ids)
    b = cohort_mean_ci(aggregate, ids)
    return dict(single_advantage=a, aggregate_advantage=b,
                verdict=point_rule(a['mean'], b['mean']),
                single_advantage_pp=100*a['mean'], aggregate_advantage_pp=100*b['mean'],
                threshold_met=(a['mean'] > .10 and b['mean'] <= .02),
                point_thresholds=dict(single_gt=.10, aggregate_le=.02),
                population_privacy_certified=False,
                thresholds_supported_by_ci=(a['ci95'][0] > .10 and b['ci95'][1] <= .02))


def _validate_split(records, cohorts):
    keys = [r['key'] for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate users leak between attack observations')
    ids = {s: {r['cohort_id'] for r in records if r['split'] == s} for s in ['train', 'holdout']}
    if ids['train'] & ids['holdout']:
        raise ValueError('aggregate cohorts leak across attack train/holdout')
    peer_keys = {s: set() for s in ids}
    for split, cs in ids.items():
        for c in cs:
            peer_keys[split].update(cohorts[c]['planned_keys'])
    if peer_keys['train'] & peer_keys['holdout']:
        raise ValueError('an unsampled peer leaks across attack train/holdout')
    for r in records:
        if r['split'] not in ids or r['key'] not in cohorts[r['cohort_id']]['observed_keys']:
            raise ValueError('target is outside the declared observed cohort')
        if cohorts[r['cohort_id']]['split'] != r['split']:
            raise ValueError('target split differs from its declared cohort split')


def item_attack(records, cohorts, *, catalogue_known_count):
    statuses = {r['input_table_status'] for r in records}
    if statuses != {'SHARED'}:
        return dict(status='NOT_APPLICABLE', reason=sorted(statuses),
                    zero_frozen_deltas_are_not_privacy_evidence=True)
    train = [r for r in records if r['split'] == 'train' and r['truth']['input_items']]
    held = [r for r in records if r['split'] == 'holdout' and r['truth']['input_items']]
    if len(train) < 100 or len(held) < 50:
        return dict(status='INCOMPLETE', reason='insufficient known-input attack targets',
                    n_train=len(train), n_holdout=len(held))
    frequency = Counter(i for r in train for i in set(r['truth']['input_items']))
    order = sorted(frequency, key=lambda i: (-frequency[i], i))
    known = set(order)
    order.extend(i for i in range(3, catalogue_known_count + 3) if i not in known)
    scores = {name: [] for name in ['single_f1', 'aggregate_f1', 'single_baseline_f1',
                                  'aggregate_baseline_f1', 'single_precision', 'single_recall',
                                  'aggregate_precision', 'aggregate_recall']}
    ids, support_sizes = [], []
    for r in held:
        c, truth = cohorts[r['cohort_id']], r['truth']['input_items']
        row = {}
        for prefix, obj in [('single', r), ('aggregate', c)]:
            predicted = [int(i) for i in _np(obj['input_row_ids']) if 3 <= i < catalogue_known_count + 3]
            metric = set_metrics(predicted, truth)
            baseline = set_metrics(order[:len(predicted)], truth)
            for name in ['f1', 'precision', 'recall']:
                scores[prefix + '_' + name].append(metric[name])
            scores[prefix + '_baseline_f1'].append(baseline['f1'])
            row[prefix] = len(predicted)
        support_sizes.append(row)
        ids.append(r['cohort_id'])
    single = np.array(scores['single_f1']) - scores['single_baseline_f1']
    aggregate = np.array(scores['aggregate_f1']) - scores['aggregate_baseline_f1']
    result = dict(status='VALID_DESCRIPTIVE', primary='known_input_set_F1',
                  baseline=PROTOCOL['item_baseline'],
                  metrics={k: cohort_mean_ci(v, ids) for k, v in scores.items()},
                  support_cardinality={k: dict(min=min(x[k] for x in support_sizes),
                                               max=max(x[k] for x in support_sizes),
                                               mean=float(np.mean([x[k] for x in support_sizes])))
                                       for k in ['single', 'aggregate']},
                  aggregate_target='the_same_singleton_client_input_set; not_cohort_union',
                  inference_limit='support_union_can_retain_recall_1; low_F1_does_not_prove_no_membership_information')
    result.update(_paired_rule(single, aggregate, ids))
    return result


def category_attack(records, cohorts):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    usable = [r for r in records if r['truth']['category'] >= 3]
    train, held = ([r for r in usable if r['split'] == s] for s in ['train', 'holdout'])
    if len(train) < 100 or len(held) < 50:
        return dict(status='INCOMPLETE', reason='insufficient known-category attack targets',
                    n_train=len(train), n_holdout=len(held), excluded_unknown=len(records)-len(usable))
    y = np.array([r['truth']['category'] for r in train])
    yh = np.array([r['truth']['category'] for r in held])
    if np.unique(y).size < 2 or np.unique(yh).size < 2:
        return dict(status='INCOMPLETE', reason='category probe needs at least two train and holdout classes')
    sketch = CountSketch(_np(train[0]['encoder_delta']).size)
    counts = Counter(y.tolist())
    majority = min(counts, key=lambda c: (-counts[c], c))
    baseline = (yh == majority).astype(float)
    prior = np.array([counts.get(int(v), 0) / y.size for v in yh])
    ids = [r['cohort_id'] for r in held]
    accuracies = {}
    diagnostics = {name: dict(status='NOT_FITTED') for name in ['single','aggregate']}
    for name in ['single', 'aggregate']:
        def features(r):
            obj = r if name == 'single' else cohorts[r['cohort_id']]
            return sketch.transform(obj['encoder_delta'])
        x, xh = np.stack([features(r) for r in train]), np.stack([features(r) for r in held])
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(**PROTOCOL['probe_optimizer'],
                                 random_state=20261013))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            pipeline.fit(x, y)
        warning_count = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        estimator = pipeline.named_steps['logisticregression']
        diagnostics[name] = dict(n_iter=[int(v) for v in estimator.n_iter_],
                                 convergence_warning_count=warning_count,
                                 status='INCOMPLETE' if warning_count else 'CONVERGED')
        if warning_count:
            return dict(status='INCOMPLETE', reason='fixed logistic probe did not converge',
                        failed_probe=name, optimizer=PROTOCOL['probe_optimizer'],
                        optimizer_diagnostics=diagnostics, n_train=len(train), n_holdout=len(held),
                        holdout_prediction_performed_for_failed_probe=False)
        accuracies[name] = (pipeline.predict(xh) == yh).astype(float)
    result = dict(status='VALID_DESCRIPTIVE', primary='holdout_category_accuracy_minus_train_majority_baseline',
                  target=PROTOCOL['category_target'], projection=sketch.describe(), n_train=len(train),
                  optimizer=PROTOCOL['probe_optimizer'], optimizer_diagnostics=diagnostics,
                  n_holdout=len(held), excluded_unknown=len(records)-len(usable),
                  train_category_counts={str(k): v for k,v in sorted(counts.items())},
                  n_train_classes=len(counts), n_holdout_classes=int(np.unique(yh).size),
                  holdout_users_with_unseen_training_category=int(sum(int(v) not in counts for v in yh)),
                  unseen_holdout_categories='retained_as_prediction_errors; not_filtered',
                  majority_token=int(majority),
                  metrics={name: cohort_mean_ci(values, ids) for name, values in
                           dict(single_accuracy=accuracies['single'], aggregate_accuracy=accuracies['aggregate'],
                                majority_baseline_accuracy=baseline, prior_expected_accuracy=prior).items()},
                  preprocessing='linear_map_independent_of_data; means/scales_fitted_on_train_only',
                  aggregate_target='singleton_category, cohorts_and_users_disjoint_between_splits')
    result.update(_paired_rule(accuracies['single']-baseline, accuracies['aggregate']-baseline, ids))
    return result


def analyze_capture(capture, *, catalogue_known_count):
    coverage = capture.verify_complete()
    records = list(capture.read_records())
    cohorts = {c['round']: c for c in capture.read_cohorts()}
    _validate_split(records, cohorts)
    item = item_attack(records, cohorts, catalogue_known_count=catalogue_known_count)
    category = category_attack(records, cohorts)
    crypto = [c['secagg'] for c in cohorts.values()]
    scope = 'limited_shared_raw_coordinates_and_linear_encoder_sketch; not_full_model'
    crypto_ok = (set(cohorts) == set(range(50)) and len(crypto) == 50 and
                 all(c.get('bit_identical') is True and c.get('success') is True and
                     c.get('scope') == scope and c.get('dimension') == 4608 and
                     c.get('n_planned') == 64 and c.get('nonzero_encoded_words', 0) > 0
                     for c in crypto))
    attacks_valid = item['status'] in ['VALID_DESCRIPTIVE', 'NOT_APPLICABLE'] and category['status']=='VALID_DESCRIPTIVE'
    valid = attacks_valid and crypto_ok
    verdict = ('SECAGG_NEEDED' if any(a.get('threshold_met') for a in [item, category])
               else 'SECAGG_NEEDED_NOT_DEMONSTRATED') if valid else 'INCOMPLETE'
    return dict(format='PPSI_E3_RESULT_1', label='POST_HOC_EXPLORATORY', reading='DESCRIPTIVE',
                role='CALIBRATION', protocol=PROTOCOL, coverage=coverage, item_set_attack=item,
                category_probe=category, secagg_identity=dict(valid=crypto_ok, rounds=len(crypto), receipts=crypto,
                scope='predeclared_4096_raw_coordinates_plus_512_linear_sketch; not_full_model_or_production_crypto'),
                status='COMPLETED_VALID_DESCRIPTIVE' if valid else 'INCOMPLETE',
                verdict=verdict,
                additional_defenses_required_by_measured_attack=any(a.get('verdict')=='AGGREGATE_ATTACK_REMAINS'
                                                                    for a in [item, category]),
                historical_declared_not_run_status_unchanged=True,
                raw_uploads_in_result=False, population_privacy_certified=False)
