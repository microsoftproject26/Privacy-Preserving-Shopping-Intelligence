"""Limited-coordinate fixed-point simulation; attacks use the real raw aggregate.

This independently checks mask cancellation on 4096 predeclared raw coordinates
and 512 linear encoder sketch coordinates. It does not simulate the full model,
does not claim production cryptography and does not replace the training path.
"""
from __future__ import annotations
import hashlib
from collections import OrderedDict
import numpy as np

from .analysis import CountSketch, _np
from fl.secagg import SecAggConfig, fixed_point_scale, encode_fixed, run_round


class SecAggProjector:
    def __init__(self, state, *, raw_coordinates=4096, sketch_dimensions=512, seed=20261013):
        self.keys = tuple(state)
        self.input_key = 'item_embed.weight'
        self.encoder_keys = tuple(k for k in self.keys if k != self.input_key)
        self.shapes = {k: tuple(state[k].shape) for k in self.keys}
        self.sizes = {k: int(np.prod(self.shapes[k])) for k in self.keys}
        self.total = sum(self.sizes.values())
        self.encoder_size = sum(self.sizes[k] for k in self.encoder_keys)
        if self.encoder_size < 1 or self.total < raw_coordinates:
            raise ValueError('insufficient shared coordinates for the pinned audit projection')
        self.sketch = CountSketch(self.encoder_size, dimensions=sketch_dimensions, seed=seed)
        rng = np.random.Generator(np.random.PCG64(seed + 1))
        self.coordinates = np.sort(rng.choice(self.total, raw_coordinates, replace=False))
        self.seed = seed

    def __call__(self, delta, encoder):
        if tuple(delta) != self.keys:
            raise ValueError('raw delta key order differs from predeclared SecAgg coordinates')
        raw = np.empty(self.coordinates.size, dtype=np.float64)
        offset = 0
        for key in self.keys:
            values = np.asarray(_np(delta[key])).reshape(-1)
            if tuple(delta[key].shape) != self.shapes[key]:
                raise ValueError('raw delta shape differs from audit binding')
            selected = (self.coordinates >= offset) & (self.coordinates < offset + values.size)
            raw[selected] = values[self.coordinates[selected] - offset]
            offset += values.size
        return np.concatenate([raw, self.sketch.transform(encoder)])

    def describe(self):
        return dict(raw_coordinates=int(self.coordinates.size), raw_coordinate_space=self.total,
                    shared_key_order=list(self.keys), shapes={k:list(v) for k,v in self.shapes.items()},
                    raw_coordinate_seed=self.seed+1,
                    raw_coordinate_sha256=hashlib.sha256(self.coordinates.astype('<i8').tobytes()).hexdigest(),
                    encoder_sketch=self.sketch.describe(), production_crypto=False,
                    scope='limited_shared_raw_coordinates_and_linear_encoder_sketch; not_full_model')


class SecAggValidator:
    def __init__(self, projector, *, clip=1.0, seed=20261015):
        self.projector, self.clip, self.seed = projector, float(clip), seed

    def __call__(self, cohort, contributions):
        planned = cohort['planned_keys']
        observed = cohort['observed_keys']
        if len(set(planned)) != len(planned) or len(set(observed)) != len(observed):
            raise ValueError('duplicate SecAgg participants')
        values = {key:(np.asarray(_np(vector), dtype=np.float64), int(weight))
                  for key, vector, weight in contributions}
        if len(values) != len(contributions) or set(values) != set(observed):
            raise ValueError('SecAgg contributions differ from the actual survivor cohort')
        if not set(observed) <= set(planned) or any(w <= 0 for _,w in values.values()):
            raise ValueError('SecAgg weights/population are invalid')
        weight = sum(w for _,w in values.values())
        dimension = self.projector.coordinates.size + self.projector.sketch.dimensions
        n = len(planned)
        cfg = SecAggConfig(n=n, d=dimension, k=32 if n > 32 else None,
                           clip=self.clip, seed=self.seed + int(cohort['round']))
        scale = fixed_point_scale(self.clip, n)
        encoded = {}
        largest = 0.0
        nonzero = 0
        for key, (vector, w) in values.items():
            weighted = vector * (w / weight)
            if weighted.shape != (dimension,) or not np.isfinite(weighted).all():
                raise ValueError('invalid limited-coordinate SecAgg contribution')
            maximum = float(np.max(np.abs(weighted)))
            largest = max(largest, maximum)
            if maximum > self.clip:
                raise ValueError('contribution exceeds the pinned SecAgg bound; silent clipping refused')
            encoded[key] = encode_fixed(weighted, scale, self.clip)
            nonzero += int(np.count_nonzero(encoded[key]))
        if not nonzero:
            raise ValueError('all-zero quantized simulation is a trivial identity control')
        dropped = [i for i,key in enumerate(planned) if key not in values]
        result = run_round(cfg, lambda i: encoded[planned[i]], dropped, check_plain=True)
        if not result.success or result.bit_identical is not True:
            raise ValueError('SecAgg unmasking/identity control failed: '+str(result.failure_reason))
        return dict(bit_identical=True, success=True, production_crypto=False,
                    n_planned=n, n_observed=len(observed), n_dropped=len(dropped), dimension=int(dimension),
                    graph_k=cfg.k, global_threshold=cfg.t_global, share_threshold=cfg.t_share,
                    fixed_point_clip=self.clip, fixed_point_scale=scale,
                    largest_weighted_coordinate=largest, nonzero_encoded_words=nonzero,
                    unmask_cpu_s=result.unmask_cpu_s,
                    weighted_contributions='n_consumed/sum_n_consumed_times_projected_raw_delta',
                    scope=self.projector.describe()['scope'],
                    attack_aggregate='ideal_FP64_consumed_weighted_mean_of_observed_FP32_deltas; not_server_FP32_full_weight_release')


def secagg_factory(state):
    from .capture import INPUT_KEY, EXCLUDED_ENCODER, canonical_sha
    server = state['server']
    shared = server.manifest.shared_keys
    keys = tuple(([INPUT_KEY] if INPUT_KEY in shared else []) +
                 [k for k in shared if k not in EXCLUDED_ENCODER])
    theta = server.adapter.broadcast_state()
    projector = SecAggProjector(OrderedDict((k, theta[k]) for k in keys))
    validator = SecAggValidator(projector)
    binding = dict(projector.describe(), fixed_point_clip=validator.clip,
                   graph_k=32, secagg_seed=validator.seed,
                   aggregation='n_consumed_weighted_mean',
                   attack_aggregate='ideal_FP64_mean_of_observed_FP32_deltas; distinct_from_server_FP32_full_weight_mean')
    binding['self_sha256'] = canonical_sha(binding)
    state['secagg_binding'] = binding
    return validator, projector
