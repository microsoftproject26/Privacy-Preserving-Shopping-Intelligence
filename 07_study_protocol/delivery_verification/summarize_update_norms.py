"""P5/T09: summarize existing D1-ext logs without training or reading TEST data.

L2 norms are not per-coordinate magnitudes. RMS is derived as L2/sqrt(numel).
An exact share of coordinates below tau cannot be recovered from these logs;
the reported lower bound follows count(|x| >= tau) <= ||x||^2/tau^2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ('agg_delta_l2', 'agg_delta_rms', 'step_l2', 'step_rms', 'pre_dist_l2', 'rel_pre_dist',
           'fraction_coordinates_below_tau_lower_bound')
FIELDS = ('run', 'scope', 'optimizer', 'tau_used', 'tau_reference', 'dp', 'group', 'numel',
          'metric', 'n', 'missing', 'min', 'median', 'p99', 'max', 'mean')


def sha(p: Path) -> str:
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def quantile(values, p):
    if not values:
        return None
    xs = sorted(values)
    x = (len(xs) - 1) * p
    a = int(math.floor(x))
    b = int(math.ceil(x))
    return xs[a] + (xs[b] - xs[a]) * (x - a)


def coordinate_lower_bound(l2: float, numel: int, tau: float) -> float:
    if not math.isfinite(l2) or l2 < 0 or numel <= 0 or not math.isfinite(tau) or tau <= 0:
        raise ValueError('l2 must be nonnegative, numel positive and tau positive, all finite')
    return max(0.0, 1.0 - (l2 / tau) ** 2 / numel)


def derive(group: dict, tau: float) -> dict:
    n = group['numel']
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('numel must be a positive integer')
    out = {}
    for k in ['agg_delta_l2', 'step_l2', 'pre_dist_l2', 'pre_l2', 'rel_pre_dist']:
        x = group.get(k)
        if x is not None and (not isinstance(x, (float, int)) or isinstance(x, bool) or not math.isfinite(x) or x < 0):
            raise ValueError(f'invalid {k}: {x!r}')
        out[k] = x
    for source, dest in [('agg_delta_l2', 'agg_delta_rms'), ('step_l2', 'step_rms')]:
        out[dest] = out[source] / math.sqrt(n) if out[source] is not None else None
    out['fraction_coordinates_below_tau_lower_bound'] = (
        coordinate_lower_bound(out['agg_delta_l2'], n, tau) if out['agg_delta_l2'] is not None else None)
    return out


def read_log(p: Path, tau: float):
    previous = -1
    dims = None
    with p.open() as f:
        for line, text in enumerate(f, 1):
            if not text.strip():
                raise ValueError(f'{p}:{line}: blank record')
            rec = json.loads(text)
            r = rec.get('round')
            if not isinstance(r, int) or isinstance(r, bool) or r != previous + 1:
                raise ValueError(f'{p}:{line}: rounds must begin at 0 and be contiguous, got {r}')
            previous = r
            if not isinstance(rec.get('server_step'), bool):
                raise ValueError(f'{p}:{line}: missing server_step flag')
            if not rec['server_step']:
                yield rec, {}
                continue
            groups = rec['groups']
            if not groups or 'total_T' in groups:
                raise ValueError(f'{p}:{line}: empty groups or reserved group name')
            cur = {k: v['numel'] for k, v in groups.items()}
            if dims is not None and dims != cur:
                raise ValueError(f'{p}:{line}: tensor groups or dimensions changed')
            dims = cur
            total = rec['total_T']
            if total['numel'] != sum(cur.values()):
                raise ValueError(f'{p}:{line}: group dimensions do not sum to total_T')
            derived = {k: derive(v, tau) for k, v in {**groups, 'total_T': total}.items()}
            for field in ['agg_delta_l2', 'step_l2', 'pre_dist_l2', 'pre_l2']:
                vals = [g.get(field) for g in groups.values()]
                if total.get(field) is not None and all(x is not None for x in vals):
                    want = math.sqrt(sum(x * x for x in vals))
                    if not math.isclose(total[field], want, rel_tol=1e-9, abs_tol=1e-15):
                        raise ValueError(f'{p}:{line}: {field} does not equal the norm over all groups')
            cos = rec.get('cos_prev')
            if cos is not None and (not math.isfinite(cos) or not -1.0000000001 <= cos <= 1.0000000001):
                raise ValueError(f'{p}:{line}: invalid cosine')
            yield rec, derived
    if previous < 0:
        raise ValueError(f'{p}: empty log')


def summarize(v: Path, out: Path, tau: float) -> dict:
    paths = sorted([*v.glob('runs_p2/MAIN/*/UPDATE_NORM_LOG.jsonl'), *v.glob('runs_p2/WA/*/UPDATE_NORM_LOG.jsonl')])
    if not paths:
        raise ValueError('no Phase-2 MAIN/WA logs found')
    if out.resolve().is_relative_to(v.resolve()):
        raise ValueError('outputs must be outside the experiment workspace')
    out.mkdir(parents=True, exist_ok=False)
    sources, summaries, runs = [], [], []
    round_fields = ('run', 'scope', 'dp', 'round', 'server_step', 'finite', 'n_clients', 'cos_prev', 'numel', *METRICS)
    with (out / 'TOTAL_BY_ROUND.csv').open('w', newline='') as rf:
        round_writer = csv.DictWriter(rf, fieldnames=round_fields)
        round_writer.writeheader()
        for p in paths:
            h0 = sha(p)
            summary_path = p.parent / 'RUN_SUMMARY.json'
            summary_sha = sha(summary_path) if summary_path.exists() else None
            meta = json.loads(summary_path.read_text()) if summary_sha else {}
            cfg = meta.get('server_opt', {}).get('cfg', {})
            run = p.parent.name
            dp = str(meta.get('arm', '')).startswith('P2_WSTAR_DP')
            scope = p.parent.parent.name
            values, missing = defaultdict(list), defaultdict(int)
            rounds = steps = nonfinite = 0
            dims = {}
            last_total = None
            cosines = []
            rms_below = defaultdict(int)
            for rec, derived in read_log(p, tau):
                rounds += 1
                base = dict(run=run, scope=scope, dp=dp, round=rec['round'], server_step=rec['server_step'],
                            finite=rec.get('finite'), n_clients=rec.get('n_clients'), cos_prev=rec.get('cos_prev'))
                if dp and 'n_clients' in rec:
                    raise ValueError(f'{p}: DP log contains a client count')
                if not derived:
                    round_writer.writerow(base)
                    continue
                steps += 1
                nonfinite += rec.get('finite') is False
                if rec.get('cos_prev') is not None:
                    cosines.append(rec['cos_prev'])
                for group, data in derived.items():
                    g = rec['total_T'] if group == 'total_T' else rec['groups'][group]
                    dims[group] = g['numel']
                    for metric in METRICS:
                        x = data[metric]
                        if x is None:
                            missing[group, metric] += 1
                        else:
                            values[group, metric].append(x)
                    if data['agg_delta_rms'] is not None and data['agg_delta_rms'] < tau:
                        rms_below[group] += 1
                last_total = derived['total_T']
                round_writer.writerow(dict(base, numel=rec['total_T']['numel'],
                                           **{k: last_total[k] for k in METRICS}))
            if meta.get('done') is True and isinstance(meta.get('rounds'), int) and rounds != meta['rounds']:
                raise ValueError(f'{p}: logged {rounds} rounds but completed summary says {meta["rounds"]}')
            for group in sorted(dims):
                for metric in METRICS:
                    xs = values[group, metric]
                    summaries.append(dict(run=run, scope=scope, optimizer=cfg.get('name'), tau_used=cfg.get('tau'),
                                          tau_reference=tau, dp=dp, group=group, numel=dims[group], metric=metric,
                                          n=len(xs), missing=missing[group, metric], min=min(xs) if xs else None,
                                          median=quantile(xs, .5), p99=quantile(xs, .99), max=max(xs) if xs else None,
                                          mean=statistics.mean(xs) if xs else None))
            runs.append(dict(run=run, scope=scope, dp=dp, optimizer=cfg.get('name'), tau_used=cfg.get('tau'),
                             rounds=rounds, server_steps=steps, nonfinite_rounds=nonfinite, final_total=last_total,
                             cos_prev_median=quantile(cosines, .5), cos_prev_missing=steps-len(cosines),
                             per_group_rms_below_tau_reference_round_share={
                                 g: rms_below[g] / len(values[g, 'agg_delta_rms']) if values[g, 'agg_delta_rms'] else None
                                 for g in dims},
                             exact_fraction_coordinates_below_tau=None))
            if h0 != sha(p) or (summary_sha is not None and summary_sha != sha(summary_path)):
                raise ValueError(f'{p}: input changed during analysis')
            sources.append(dict(path=str(p.relative_to(v)), sha256=h0, bytes=p.stat().st_size,
                                summary_sha256=summary_sha))
            print(f'{scope}/{run}: {rounds} rounds, {steps} steps', flush=True)
    with (out / 'GROUP_SUMMARY.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(summaries)
    report = dict(label='DESCRIPTIVE', task='P5 / T09 D1-ext', tau_reference=tau, sources=sources, runs=runs,
                  measurements={'norms': 'Existing per-group L2 norms; RMS = L2 / sqrt(numel).',
                                'coordinate_fraction': 'Exact fraction NOT_AVAILABLE: norms do not identify coordinate magnitudes.',
                                'bound': 'fraction(|delta| < tau) >= max(0, 1 - L2^2 / (numel * tau^2)); a lower bound, not an estimate.',
                                'round_share': 'Fraction of logged finite server-step RMS values below reference tau, not a coordinate fraction.',
                                'dp': 'DP inputs contain only the post-noise released aggregate; never per-client data.',
                                'cosine': 'Existing cos_prev values only; first step and resume boundaries may be missing.'},
                  no_training=True, no_test_evaluation=True, input_hashes_unchanged=True)
    (out / 'UPDATE_NORM_SUMMARY.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tau-reference', type=float, default=1e-3)
    args = ap.parse_args()
    if not math.isfinite(args.tau_reference) or args.tau_reference <= 0:
        ap.error('--tau-reference must be finite and positive')
    report = summarize(args.v, args.out, args.tau_reference)
    print(json.dumps({'runs': len(report['runs']), 'out': str(args.out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
