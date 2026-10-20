"""Fixed published-Beauty formula transfer to REES46 TRAIN decisions and full-K ranks.

This is POST_HOC_EXPLORATORY / DESCRIPTIVE, not a published benchmark reproduction.
TRAIN statistics count each eligible decision's source-at-offset -> target once.
Unknown sources retain their original ages; targets never enter query construction.
REES46 class-ID ties/no seen filtering replace the published benchmark conventions.
The CLI deliberately cannot read TEST. Output identities are write-once.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse

LABEL = 'POST_HOC_EXPLORATORY_DESCRIPTIVE'
PATH = 'rees46_training_free_exact_sparse_v1'
METHODS = ('MC', 'SeqRules', 'PCTM')
MAX_OFFSET = 10
REGISTRY_SHA256 = '56dd0840cb50e89250a8b9a54dc1d4f621e9e6f9500ec44441cf07cb472ef740'
ROLE_IDS = {
    'CALIBRATION': ('CALIB_TRAIN_5_100_32768', 'CALIB_VALIDATION_5_100_32768'),
    'MAIN': ('MAIN_TRAIN_5_100_131072', 'MAIN_VALIDATION_5_100_131072'),
}
RECIPES = {
    'MC': {'distance': 1},
    'SeqRules': {'steps': 10, 'weighting': 'log', 'pruning': 100,
                 'idf_weight': False, 'history_length': 10, 'history_weighting': 'quadratic'},
    'PCTM': {'counts': 'inv:10:0.5', 'tau': 11250.0, 'kernel': 'exp:0.6', 'pop_boost': -0.04},
}
PROTOCOL = {
    'document': 'REES46_FIXED_TRAINING_FREE_V1', 'label': LABEL,
    'recipe_source': 'capacity_probes/configs/table1.json Beauty; fixed transfer, no REES46 selection',
    'fit_scope': 'eligible TRAIN decisions; one source-at-original-offset to target pair per decision',
    'popularity_scope': 'TRAIN target frequency, matching REES46 POP; not raw-event frequency',
    'history_policy': 'USER_HISTORY_L50', 'max_offset': MAX_OFFSET,
    'ranking': 'all frozen output classes, descending FP32 scores, ascending class ID ties; no seen filter',
    'prediction_path': PATH, 'recipes': RECIPES, 'selection': 'NONE', 'test_supported': False,
    'registry_file_sha256': REGISTRY_SHA256, 'role_ids': {k: list(v) for k, v in ROLE_IDS.items()},
}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    from v4eval.values import to_jsonable
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(to_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def refuse_phase(phase):
    if phase not in ('CALIBRATION', 'MAIN'):
        raise ValueError('Only CALIBRATION or MAIN validation is supported; TEST is refused before data access')


def _array(value, dtype=None):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def causal_input(batch, K):
    tokens = _array(batch['item_tokens'], np.int64)
    lengths = _array(batch['lengths'], np.int64)
    if tokens.ndim != 2 or lengths.shape != (tokens.shape[0],):
        raise ValueError('item_tokens must be [B,L], lengths [B]')
    if np.any(lengths < 0) or np.any(lengths > tokens.shape[1]):
        raise ValueError('length outside right-padded history')
    classes = tokens - 3
    valid = (tokens >= 3) & (classes < int(K))
    return classes, lengths, valid


def causal_pair_keys(batch, K, max_offset=MAX_OFFSET):
    """Offset-specific source*K+target keys; each eligible decision counted once."""
    classes, lengths, valid = causal_input(batch, K)
    target = _array(batch['target_class'], np.int64)
    loss = _array(batch['loss_mask'], bool)
    B = lengths.size
    if target.shape != (B,) or loss.shape != (B,) or not loss.all():
        raise ValueError('Every fit row must be an eligible TRAIN decision')
    if np.any(target < 0) or np.any(target >= K):
        raise ValueError('TRAIN target outside catalogue')
    result = []
    for distance in range(1, int(max_offset) + 1):
        rows = np.flatnonzero(lengths >= distance)
        pos = lengths[rows] - distance
        keep = valid[rows, pos]
        rows, pos = rows[keep], pos[keep]
        result.append(classes[rows, pos] * np.int64(K) + target[rows])
    return result


def offset_matrix(keys, K):
    ids, counts = np.unique(np.asarray(keys, dtype=np.int64), return_counts=True)
    result = sparse.coo_matrix((counts.astype(np.float32), (ids // K, ids % K)), shape=(K, K)).tocsr()
    result.sort_indices()
    return result


def rule_weights(steps=10, weighting='log'):
    positions = np.arange(1, int(steps) + 1, dtype=np.float32)
    if weighting == 'log':
        return np.float32(1) / np.log10(positions + np.float32(1.7))
    if weighting == 'div':
        return np.float32(1) / positions
    if weighting == 'quadratic':
        return np.float32(1) / np.square(positions)
    if weighting == 'same':
        return np.ones(int(steps), dtype=np.float32)
    raise ValueError('unsupported rule weighting')


def prune_rows(matrix, limit=100):
    matrix = matrix.tocsr()
    keep = np.zeros(matrix.nnz, dtype=bool)
    sizes = np.zeros(matrix.shape[0], dtype=np.int64)
    for row in range(matrix.shape[0]):
        a, b = matrix.indptr[row:row+2]
        order = np.lexsort((matrix.indices[a:b], -matrix.data[a:b]))[:int(limit)]
        keep[a + order] = True
        sizes[row] = len(order)
    indptr = np.r_[0, np.cumsum(sizes)]
    out = sparse.csr_matrix((matrix.data[keep], matrix.indices[keep], indptr), shape=matrix.shape, dtype=np.float32)
    out.eliminate_zeros()
    out.sort_indices()
    return out


def models_from_offsets(offsets, frequency):
    if len(offsets) != MAX_OFFSET:
        raise ValueError('exactly ten offset matrices are required')
    K = len(frequency)
    if any(m.shape != (K, K) for m in offsets):
        raise ValueError('offset catalogue shapes disagree')
    frequency = np.asarray(frequency, dtype=np.int64)
    if np.any(frequency < 0) or frequency.sum() <= 0:
        raise ValueError('TRAIN target frequency is empty or negative')
    totals = np.asarray(offsets[0].sum(axis=1)).ravel().astype(np.float32)
    inv = np.zeros_like(totals)
    inv[totals > 0] = np.float32(1) / totals[totals > 0]
    mc = (sparse.diags(inv) @ offsets[0]).tocsr().astype(np.float32)
    seq, pctm = None, None
    seq_w = rule_weights()
    pctm_w = np.float32(1) / np.sqrt(np.arange(1, 11, dtype=np.float32))
    for offset, sw, pw in zip(offsets, seq_w, pctm_w):
        current = offset.multiply(np.float32(sw))
        seq = current if seq is None else seq + current
        current = offset.multiply(np.float32(pw))
        pctm = current if pctm is None else pctm + current
    seq = prune_rows(seq.astype(np.float32), 100)
    pctm = pctm.tocsr().astype(np.float32)
    pctm.data = np.log1p(pctm.data * K / RECIPES['PCTM']['tau']).astype(np.float32)
    log_pop = np.log((frequency.astype(np.float64) + 1) / (frequency.sum() + K)).astype(np.float32)
    background = (np.float32(RECIPES['PCTM']['pop_boost']) * log_pop).astype(np.float32)
    result = {'MC': (mc, np.zeros(K, dtype=np.float32)),
              'SeqRules': (seq, np.zeros(K, dtype=np.float32)),
              'PCTM': (pctm, background)}
    for matrix, bg in result.values():
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.isfinite(matrix.data).all() or not np.isfinite(bg).all():
            raise ValueError('nonfinite fitted evidence')
    return result


def history_matrix(batch, K, method):
    """Only causal input keys are consumed. Unknown ages are never compressed."""
    if method not in METHODS:
        raise ValueError('unknown training-free method')
    classes, lengths, valid = causal_input(batch, K)
    row_ids, cols, weights = [], [], []
    for row, n in enumerate(lengths.tolist()):
        limit = min(n, 1 if method == 'MC' else 10 if method == 'SeqRules' else 50)
        if not limit:
            continue
        positions = n - 1 - np.arange(limit, dtype=np.int64)
        if method == 'MC':
            w = np.ones(limit, dtype=np.float32)
        elif method == 'SeqRules':
            w = np.float32(1) / np.square(np.arange(1, limit+1, dtype=np.float32))
        else:
            w = np.power(np.float32(0.6), np.arange(limit, dtype=np.float32)).astype(np.float32)
            w /= w.sum()
        known = valid[row, positions]
        row_ids.extend([row] * int(known.sum()))
        cols.extend(classes[row, positions[known]].tolist())
        weights.extend(w[known].tolist())
    H = sparse.csr_matrix((np.asarray(weights, dtype=np.float32), (row_ids, cols)),
                          shape=(len(lengths), K), dtype=np.float32)
    H.sum_duplicates()
    H.sort_indices()
    return H


class BackgroundRanks:
    """Full-catalogue background CDF and sorted class IDs of each exact score tie."""
    def __init__(self, background):
        self.values = np.asarray(background, dtype=np.float32)
        if self.values.ndim != 1 or not np.isfinite(self.values).all():
            raise ValueError('background must be finite [K]')
        self.K = len(self.values)
        order = np.argsort(self.values, kind='stable')
        self.sorted = self.values[order]
        levels, starts = np.unique(self.sorted, return_index=True)
        ends = np.r_[starts[1:], self.K]
        self.equal_ids = {float(level): order[a:b] for level, a, b in zip(levels, starts, ends)}

    def rank(self, cols, evidence, target):
        cols = np.asarray(cols, dtype=np.int64)
        evidence = np.asarray(evidence, dtype=np.float32)
        if cols.shape != evidence.shape or np.any(cols < 0) or np.any(cols >= self.K):
            raise ValueError('invalid sparse row')
        if cols.size > 1 and np.any(cols[1:] <= cols[:-1]):
            raise ValueError('sparse columns must be sorted and unique')
        if not np.isfinite(evidence).all():
            raise ValueError('nonfinite evidence')
        target = int(target)
        if target < 0 or target >= self.K:
            raise ValueError('rank target outside catalogue')
        final = (self.values[cols] + evidence).astype(np.float32)
        if not np.isfinite(final).all():
            raise ValueError('nonfinite final scores')
        at = np.searchsorted(cols, target)
        st = final[at] if at < len(cols) and cols[at] == target else self.values[target]
        greater = self.K - int(np.searchsorted(self.sorted, st, side='right'))
        tie_ids = self.equal_ids.get(float(st), np.empty(0, dtype=np.int64))
        equal = len(tie_ids)
        before = int(np.searchsorted(tie_ids, target))
        old = self.values[cols]
        greater += int(np.count_nonzero(final > st)) - int(np.count_nonzero(old > st))
        equal += int(np.count_nonzero(final == st)) - int(np.count_nonzero(old == st))
        before += int(np.count_nonzero((cols < target) & (final == st))) - int(np.count_nonzero((cols < target) & (old == st)))
        return 1 + greater + before, equal - 1, before


class TrainingFreeScorer:
    def __init__(self, method, matrix, background):
        self.method = method
        self.matrix = matrix.tocsr().astype(np.float32)
        self.background = BackgroundRanks(background)
        if self.matrix.shape != (self.background.K, self.background.K):
            raise ValueError('matrix and full catalogue disagree')
        if not np.isfinite(self.matrix.data).all():
            raise ValueError('nonfinite fitted matrix')

    def context(self, batch):
        result = (history_matrix(batch, self.background.K, self.method) @ self.matrix).tocsr()
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()
        return result

    def dense_oracle(self, batch):
        return (self.context(batch).toarray() + self.background.values).astype(np.float32)

    def ranks(self, batch, targets):
        targets = _array(targets, np.int64)
        if np.any(targets < -1):
            raise ValueError('only -1 denotes an unrankable row')
        result = self.context(batch)
        if targets.shape != (result.shape[0],):
            raise ValueError('target vector shape disagrees')
        ranks, ties, before = [np.full(result.shape[0], -1, dtype=np.int64) for _ in range(3)]
        for row, target in enumerate(targets.tolist()):
            if target < 0:
                continue
            a, b = result.indptr[row:row+2]
            ranks[row], ties[row], before[row] = self.background.rank(result.indices[a:b], result.data[a:b], target)
        return ranks, ties, before


def source_binding():
    from v4eval import evaluator_code_sha256
    from v4int import providers, verify
    return {'adapter_sha256': sha_file(__file__), 'protocol_sha256': hashlib.sha256(canonical(PROTOCOL)).hexdigest(),
            'evaluator_code_sha256': evaluator_code_sha256(), 'provider_sha256': sha_file(providers.__file__),
            'manifest_verifier_sha256': sha_file(verify.__file__), 'registry_expected_sha256': REGISTRY_SHA256}


def paths_and_role(phase, data_root):
    refuse_phase(phase)
    root = Path(data_root) if data_root else Path(os.environ['PPSI_TECHNICAL_WORK'])
    registry = root / 'contracts/manifest_registry/REGISTRY_V2.json'
    if sha_file(registry) != REGISTRY_SHA256:
        raise ValueError('registry file identity changed before provider/data access')
    doc = json.loads(registry.read_text())
    expected = ROLE_IDS[phase]
    entry = doc['roles'][phase]
    if (entry['train'], entry['eval']) != expected:
        raise ValueError('registry role IDs differ from exact allowed TRAIN/VALIDATION pair')
    if any('TEST' in mid.upper() for mid in expected):
        raise ValueError('TEST role ID refused before provider/data access')
    from v4int.verify import DataPaths
    return DataPaths.default(root), registry, *expected


def reserve_output(out, operation, phase):
    refuse_phase(phase)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'STARTED.json', {'operation': operation, 'phase': phase, 'utc': utc(), 'protocol': PROTOCOL})
    return out


def fit(phase, data_root, out):
    refuse_phase(phase)
    from v4base.guard import RssGuard
    from v4int.providers import StoreTrainProvider
    paths, registry, train_id, _ = paths_and_role(phase, data_root)
    out = reserve_output(out, 'TRAIN_FIT', phase)
    started = time.monotonic()
    guard = RssGuard()
    provider = StoreTrainProvider(train_id, history_policy='USER_HISTORY_L50', paths=paths, manifest_registry=registry, registry_root=registry.parents[2])
    K = int(provider.verified.header['catalogue_K'])
    guard.check()
    chunks = [[] for _ in range(MAX_OFFSET)]
    frequency = np.zeros(K, dtype=np.int64)
    n_pairs = np.zeros(MAX_OFFSET, dtype=np.int64)
    for start in range(0, provider.n_decisions, 2048):
        pos = np.arange(start, min(start+2048, provider.n_decisions), dtype=np.int64)
        batch = provider.batch(pos)
        keys = causal_pair_keys(batch, K)
        frequency += np.bincount(_array(batch['target_class'], np.int64), minlength=K)
        for index, current in enumerate(keys):
            chunks[index].append(current)
            n_pairs[index] += len(current)
        guard.check()
        if start % (2048 * 100) == 0:
            print(f'TRAIN fit {start}/{provider.n_decisions}', flush=True)
    offsets = []
    for index in range(MAX_OFFSET):
        keys = np.concatenate(chunks[index]) if chunks[index] else np.empty(0, dtype=np.int64)
        chunks[index] = None
        matrix = offset_matrix(keys, K)
        offsets.append(matrix)
        del keys
        gc.collect()
        guard.check()
    models = models_from_offsets(offsets, frequency)
    np.save(out / 'target_frequency_int64.npy', frequency, allow_pickle=False)
    for method, (matrix, background) in models.items():
        sparse.save_npz(out / f'{method}.matrix.npz', matrix, compressed=False)
        np.save(out / f'{method}.background_float32.npy', background, allow_pickle=False)
    files = {p.name: sha_file(p) for p in sorted(out.iterdir()) if p.name != 'STARTED.json'}
    receipt = {'document': 'REES46_TRAINING_FREE_FIT', 'status': 'COMPLETE', 'phase': phase, 'label': LABEL,
               'protocol': PROTOCOL, 'source_binding': source_binding(), 'TRAIN': provider.describe(),
               'registry': {'path': str(registry), 'file_sha256': sha_file(registry)},
               'n_train_decisions': provider.n_decisions, 'catalogue_K': K, 'n_pairs_by_offset': n_pairs.tolist(),
               'nnz_by_method': {m: matrix.nnz for m, (matrix, _) in models.items()}, 'files_sha256': files,
               'resource': guard.report(), 'wall_s': time.monotonic()-started, 'utc': utc()}
    if sum(p.stat().st_size for p in out.iterdir()) > 8 * 1024**3:
        raise ValueError('8 GiB fit disk cap exceeded')
    write_json(out / 'FIT_RECEIPT.json', receipt)
    print(json.dumps({'operation': 'FIT', 'phase': phase, 'status': 'COMPLETE', 'wall_s': receipt['wall_s'], 'nnz': receipt['nnz_by_method']}), flush=True)
    return receipt


def load_fit(fit_dir, phase):
    refuse_phase(phase)
    fit_dir = Path(fit_dir)
    receipt = json.loads((fit_dir / 'FIT_RECEIPT.json').read_text())
    if receipt['status'] != 'COMPLETE' or receipt['phase'] != phase or receipt['protocol'] != PROTOCOL:
        raise ValueError('fit phase/protocol mismatch')
    if receipt['source_binding'] != source_binding():
        raise ValueError('fit source identity changed')
    for name, expected in receipt['files_sha256'].items():
        if sha_file(fit_dir / name) != expected:
            raise ValueError('fit file identity changed')
    models = {m: TrainingFreeScorer(m, sparse.load_npz(fit_dir / f'{m}.matrix.npz'),
                  np.load(fit_dir / f'{m}.background_float32.npy', allow_pickle=False)) for m in METHODS}
    return receipt, models


def predict_sparse_models(models, provider, cfg, model_sha, *, guard=None):
    from v4eval import frozen_config as fc
    from v4eval.predict import PredictionArtifact, _Cursor
    fc.require_frozen(cfg)
    fc.check_runtime_flags(cfg, __import__('torch').device('cpu'))
    manifest = provider.manifest
    K = int(manifest.catalogue_K)
    if set(models) != set(METHODS) or any(m.background.K != K for m in models.values()):
        raise ValueError('three models and the full manifest catalogue are required')
    arts = {}
    for method in METHODS:
        eval_config = cfg.eval_config(PATH, fc.CPU_DEVICE_CLASS)
        eval_config['eval_stream'] = 'E2E_ROWS'
        header = {**dict(manifest.hashes), 'manifest_id': manifest.manifest_id, 'model_sha256': model_sha,
                  'catalogue_K': K, 'candidate_order_hash': manifest.header.get('candidate_order_hash'),
                  'eval_config': eval_config, 'exact_sparse_algorithm': PATH, 'method': method}
        arts[method] = PredictionArtifact(manifest.decision_id.copy(), *[np.full(manifest.n, -1, dtype=np.int64) for _ in range(3)], header=header)
    cursor = _Cursor(manifest, cfg.eval_batch)
    if guard is not None:
        guard.check()
    rows_done = 0
    for batch in provider.batches(cfg.eval_batch):
        rows = cursor.take(batch)
        targets = manifest.target_class[rows].copy()
        targets[~manifest.rankable[rows]] = -1
        for method, scorer in models.items():
            r, t, b = scorer.ranks(batch, targets)
            arts[method].ranks[rows], arts[method].n_tied[rows], arts[method].eq_before[rows] = r, t, b
        rows_done += len(rows)
        if rows_done % (cfg.eval_batch * 20) == 0:
            print(f'VALIDATION full-K {rows_done}/{int(manifest.eval_mask.sum())}', flush=True)
        if guard is not None:
            guard.check()
    if cursor.finish() != 'E2E_ROWS':
        raise ValueError('validation stream is not E2E_ROWS')
    return arts


def evaluate_validation(phase, data_root, fit_dir, out):
    refuse_phase(phase)
    from v4base.guard import RssGuard
    from v4base.scoring import build_rows
    from v4eval import frozen_config as fc
    from v4eval.evaluate import evaluate
    from v4eval.predict import align
    from v4int.providers import StoreEvalProvider, comparison_key_base, manifest_binding_of
    from types import SimpleNamespace
    paths, registry, train_id, eval_id = paths_and_role(phase, data_root)
    receipt, models = load_fit(fit_dir, phase)
    if receipt['registry']['file_sha256'] != sha_file(registry):
        raise ValueError('fit registry identity changed')
    if receipt['TRAIN']['manifest_id'] != train_id:
        raise ValueError('TRAIN manifest mismatch')
    from v4int.verify import verify_manifest_by_registry
    vm = verify_manifest_by_registry(train_id, registry, paths, root=registry.parents[2], expect_kind='TRAIN')
    if receipt['TRAIN']['manifest_binding'] != manifest_binding_of(vm):
        raise ValueError('fit TRAIN identity changed')
    if receipt['n_train_decisions'] != len(vm.canonical_decision_id):
        raise ValueError('fit TRAIN decision count changed')
    out = reserve_output(out, 'VALIDATION', phase)
    started = time.monotonic()
    cfg = fc.load()
    fc.apply_runtime_flags(cfg)
    provider = StoreEvalProvider(eval_id, history_policy='USER_HISTORY_L50', paths=paths, manifest_registry=registry, registry_root=registry.parents[2])
    manifest = provider.manifest
    K = int(manifest.catalogue_K)
    train_identity = SimpleNamespace(verified=vm, history_policy=receipt['TRAIN']['history_policy'],
                                     transform_state=receipt['TRAIN']['transform_state'], manifest_id=train_id)
    key_base = comparison_key_base(paths, train_identity, provider)
    if receipt['catalogue_K'] != K:
        raise ValueError('fit catalogue mismatch')
    model_sha = hashlib.sha256(canonical({'fit_receipt_sha256': sha_file(Path(fit_dir)/'FIT_RECEIPT.json'),
                                          'source_binding': source_binding()})).hexdigest()
    guard = RssGuard()
    arts = predict_sparse_models(models, provider, cfg, model_sha, guard=guard)
    for method, art in arts.items():
        align(manifest, art)
        result = evaluate(manifest, art, config=cfg, evaluator_code_sha256=source_binding()['evaluator_code_sha256'])
        method_dir = out / method
        method_dir.mkdir()
        for name, array in [('ranks_int32', art.ranks), ('n_tied_int32', art.n_tied), ('eq_before_int32', art.eq_before)]:
            np.save(method_dir / f'{name}.npy', array.astype(np.int32), allow_pickle=False)
        write_json(method_dir / 'RESULT.json', result)
        rows = build_rows(result, dict(key_base, family=f'BASELINE_{method}'), method=method)
        for row in rows:
            row['experiment_label'] = LABEL
        write_json(method_dir / 'rows.json', rows)
    result_receipt = {'document': 'REES46_TRAINING_FREE_VALIDATION', 'status': 'COMPLETE', 'label': LABEL,
                      'phase': phase, 'protocol': PROTOCOL, 'source_binding': source_binding(),
                      'registry': {'path': str(registry), 'file_sha256': sha_file(registry)},
                      'fit_receipt_sha256': sha_file(Path(fit_dir)/'FIT_RECEIPT.json'), 'EVAL': provider.describe(),
                      'n_rows': manifest.n, 'catalogue_K': K, 'resource': guard.report(),
                      'wall_s': time.monotonic()-started, 'utc': utc(),
                      'files_sha256': {str(p.relative_to(out)): sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out / 'EVALUATION_RECEIPT.json', result_receipt)
    print(json.dumps({'operation': 'VALIDATION', 'phase': phase, 'status': 'COMPLETE', 'wall_s': result_receipt['wall_s']}), flush=True)
    return result_receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['fit', 'evaluate'])
    parser.add_argument('--phase', choices=['CALIBRATION', 'MAIN'], required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--fit-dir')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    refuse_phase(args.phase)
    import torch
    torch.set_num_threads(4)
    if hasattr(os, 'nice') and os.nice(0) < 19:
        os.nice(19-os.nice(0))
    if args.operation == 'fit':
        fit(args.phase, args.data_root, args.out)
    elif args.fit_dir is None:
        parser.error('evaluate requires --fit-dir')
    else:
        evaluate_validation(args.phase, args.data_root, args.fit_dir, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
