"""Synthetic canary definitions, private fixed-roster injection and scoring.

This library has no training or filesystem entry point. Real runs additionally
require a reviewed CALIB protocol/source/initializer/accountant binding. It is
an item-only audit: no synthetic side-channel features are supplied. Public
definitions and references precede fair private membership coins. Roster keys
and scheduled denominator never depend on membership. Published audit outcomes
are separate from the model-training mechanism's claimed DP release.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import secrets
import numpy as np
import torch

CALIB_TRAIN = 'CALIB_TRAIN_5_100_32768'
PREFIX = 'eid_e2_canary::'


def _int(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return int(value)


def _integer_array(value, name, ndim):
    a = np.asarray(value)
    if a.ndim != ndim or a.dtype.kind not in 'iu':
        raise ValueError(name + ' must be an integer array')
    return np.array(a, dtype=np.int64, copy=True)


def _readonly(a):
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class CanaryDefinitions:
    histories: np.ndarray
    targets: np.ndarray
    catalogue_K: int
    max_len: int = 50

    def __post_init__(self):
        k = _int(self.catalogue_K, 'catalogue_K')
        length = _int(self.max_len, 'max_len')
        h = _integer_array(self.histories, 'histories', 2)
        t = _integer_array(self.targets, 'targets', 1)
        if not len(t) or h.shape[0] != len(t) or not 1 <= h.shape[1] <= length:
            raise ValueError('nonempty aligned histories and targets required')
        if np.any(h < 0) or np.any(h >= k) or np.any(t < 0) or np.any(t >= k):
            raise ValueError('known catalogue classes only')
        object.__setattr__(self, 'histories', _readonly(h))
        object.__setattr__(self, 'targets', _readonly(t))
        object.__setattr__(self, 'catalogue_K', k)
        object.__setattr__(self, 'max_len', length)

    @property
    def count(self):
        return len(self.targets)

    def commitment(self):
        # Only public synthetic definitions: no membership seed or bits.
        h = hashlib.sha256()
        h.update(f'EID_E2_DEFINITIONS_V1:{self.catalogue_K}:{self.max_len}:{self.histories.shape}:'.encode())
        h.update(self.histories.astype('<i8', copy=False).tobytes())
        h.update(self.targets.astype('<i8', copy=False).tobytes())
        return h.hexdigest()

    def batch(self, indices, repeats=1):
        idx = _integer_array(indices, 'indices', 1)
        repeats = _int(repeats, 'repeats')
        if np.any(idx < 0) or np.any(idx >= self.count):
            raise ValueError('canary index outside definitions')
        idx = np.repeat(idx, repeats)
        n, length = len(idx), self.histories.shape[1]
        tokens = np.zeros((n, self.max_len), dtype=np.int64)
        tokens[:, :length] = self.histories[idx] + 3
        mask = np.arange(self.max_len)[None, :] < np.full((n, 1), length)
        positions = np.where(mask, np.arange(self.max_len)[None, :] + 1, 0)
        return dict(item_tokens=torch.from_numpy(tokens), lengths=torch.full((n,), length, dtype=torch.long),
                    attention_mask=torch.from_numpy(mask), position_ids=torch.from_numpy(positions),
                    target_class=torch.from_numpy(self.targets[idx].copy()), loss_mask=torch.ones(n, dtype=torch.bool))


def long_tail_pool(train_target_counts, fraction=.2):
    """Positive-count bottom fraction; class IDs break frequency ties.

    Counts must come from the pinned CALIB TRAIN provider. Zero-frequency
    classes are excluded. This is a declared synthetic distribution, rather
    than a replay of real customers or a full-secret-space exposure estimate.
    """
    counts = _integer_array(train_target_counts, 'train_target_counts', 1)
    if np.any(counts < 0) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('nonnegative counts and fraction in (0,1] required')
    present = np.flatnonzero(counts)
    if not len(present):
        raise ValueError('empty TRAIN-supported catalogue')
    order = np.lexsort((present, counts[present]))
    return _readonly(present[order[:max(1, math.ceil(len(present) * fraction))]])


def public_definitions(pool, catalogue_K, *, n_canaries=1000, n_references=10000,
                       history_length=20, max_len=50, seed=20261014):
    """Independent public draw; this function never accepts membership bits."""
    k = _int(catalogue_K, 'catalogue_K')
    p = _integer_array(pool, 'pool', 1)
    if not len(p) or len(np.unique(p)) != len(p) or np.any(p < 0) or np.any(p >= k):
        raise ValueError('unique known TRAIN-supported classes required')
    nc, nr = _int(n_canaries, 'n_canaries'), _int(n_references, 'n_references')
    length = _int(history_length, 'history_length')
    maximum = _int(max_len, 'max_len')
    seed = _int(seed, 'seed', 0)
    if length > maximum:
        raise ValueError('history longer than model window')
    rng = np.random.default_rng(seed)  # isolated from global NumPy/Torch RNG
    def draw(n):
        return CanaryDefinitions(rng.choice(p, size=(n, length)), rng.choice(p, size=n), k, maximum)
    return draw(nc), draw(nr)


def private_fair_signs(n_canaries):
    """Draw independent fair signs AFTER the public commitment is frozen.

    Production uses system entropy. This private vector/entropy is never part
    of the model's public receipt. Controls inject prescribed signs separately.
    """
    n = _int(n_canaries, 'n_canaries')
    # Each system-entropy byte contributes one exact fair bit.
    b = np.frombuffer(secrets.token_bytes(n), dtype=np.uint8)
    return np.where(b & 1, 1, -1).astype(np.int8)


class FixedRosterCanaries:
    """A private get_client wrapper; all potential canaries stay scheduled."""
    def __init__(self, real_keys, get_real_client, definitions, private_signs, *,
                 train_manifest_id, repeats=10):
        if train_manifest_id != CALIB_TRAIN:
            raise ValueError('canary injection only admits exact CALIB TRAIN')
        keys = tuple(real_keys)
        if any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
            raise ValueError('real keys must be unique strings')
        if not isinstance(definitions, CanaryDefinitions):
            raise ValueError('public canary definitions required')
        signs = np.asarray(private_signs)
        if signs.dtype.kind != 'i' or signs.shape != (definitions.count,) or not np.isin(signs, [-1, 1]).all():
            raise ValueError('one fair private sign per potential client required')
        synthetic = tuple(PREFIX + f'{i:06d}' for i in range(definitions.count))
        if set(keys) & set(synthetic):
            raise ValueError('synthetic IDs collide with real clients')
        self.keys = keys + synthetic
        self._indices = {key: i for i, key in enumerate(synthetic)}
        self._real = frozenset(keys)
        self._get_real_client = get_real_client
        self._definitions = definitions
        self._signs = np.array(signs, dtype=np.int8, copy=True)
        self._repeats = _int(repeats, 'repeats')

    def get_client(self, key):
        if key in self._real:
            return self._get_real_client(key)
        if key not in self._indices:
            raise KeyError('unknown public roster key')
        from fl.client import ClientData
        index = self._indices[key]
        idx = np.array([index] if self._signs[index] == 1 else [], dtype=np.int64)
        return ClientData(key, self._definitions.batch(idx, repeats=self._repeats))

    def public_receipt(self):
        return dict(roster_count=len(self.keys), potential_canaries=self._definitions.count,
                    roster_sha256=hashlib.sha256('\n'.join(self.keys).encode()).hexdigest(),
                    definitions_sha256=self._definitions.commitment(), examples_per_included_client=self._repeats,
                    membership_count_disclosed=False, train_manifest_id=CALIB_TRAIN)


def full_catalogue_logprob(model, batch, catalogue_K):
    """Strict FP32 forward/logsumexp, no labels forwarded to the model.

    Restores each module's training flag; scoring neither consumes dropout RNG
    nor changes model state. Intended for item-only SASRec, with full K logits.
    """
    from fl.numerics import assert_strict_fp32
    assert_strict_fp32()
    if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
        raise ValueError('autocast is incompatible with strict FP32 audit scoring')
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled() or \
            not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
        raise ValueError('STRICT deterministic runtime flags required for audit scoring')
    k = _int(catalogue_K, 'catalogue_K')
    if bool(getattr(getattr(model, 'cfg', None), 'rich', False)):
        raise ValueError('synthetic audit supports item-only model')
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in model.state_dict().values()):
        raise ValueError('audit model must be strict FP32')
    targets = batch['target_class']
    if targets.dtype != torch.long or targets.ndim != 1 or torch.any((targets < 0) | (targets >= k)):
        raise ValueError('known int64 target classes required')
    flags = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            logits = model(item_tokens=batch['item_tokens'], lengths=batch['lengths'],
                           attention_mask=batch['attention_mask'])
            if logits.dtype != torch.float32 or logits.shape != (len(targets), k) or not torch.isfinite(logits).all():
                raise ValueError('finite strict FP32 full catalogue logits required')
            return logits.gather(1, targets[:, None]).squeeze(1) - torch.logsumexp(logits, dim=1)
    finally:
        for module, flag in flags:
            module.training = flag


def drift_scores(initializer, final_model, definitions, *, device='cpu', batch_size=64):
    """Score public definitions without access to membership bits."""
    size = _int(batch_size, 'batch_size')
    scores = np.empty(definitions.count, dtype=np.float64)
    for start in range(0, definitions.count, size):
        end = min(start + size, definitions.count)
        batch = {key: value.to(device) for key, value in definitions.batch(np.arange(start, end)).items()}
        before = full_catalogue_logprob(initializer, batch, definitions.catalogue_K).double()
        after = full_catalogue_logprob(final_model, batch, definitions.catalogue_K).double()
        scores[start:end] = (after - before).cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite canary scores')
    return scores


def finite_reference_exposure(canary_scores, reference_scores):
    """Conservative >= ties, +1 smoothing; bounded by log2(nrefs+1)."""
    c, r = np.asarray(canary_scores, dtype=np.float64), np.asarray(reference_scores, dtype=np.float64)
    if c.ndim != 1 or r.ndim != 1 or not len(c) or not len(r) or not np.isfinite(c).all() or not np.isfinite(r).all():
        raise ValueError('nonempty finite score vectors required')
    counts = len(r) - np.searchsorted(np.sort(r), c, side='left')
    return -np.log2((1.0 + counts) / (len(r) + 1.0))
