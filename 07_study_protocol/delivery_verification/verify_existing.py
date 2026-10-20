"""Read-only master-plan V1-V4/V8/V10/V11 checks of previously registered results.

This reads existing result JSONs; it never scores a model or creates a TEST look.
Outputs must be outside the experiment workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

EXPECTED = {
    'runs_ds3/ANA/out_test_v3/RETENTION_RESULT.json': '28b7649231eed2b1f1ae6fdeccb9d9f523fd643fe165d8f023d5fcdaa2258ca4',
    'runs_p2/TEST/p2_out_test_v1/RETENTION_RESULT.json': 'c44382c5c4c0e9ca200f42a6b8005eedb690a4be1e65fa45ab9d0a5cb05b8d38',
    'company/gates/ALL_METHODS_LOCK.json': '04a3cd196149e2acf2ab5bb004f58c997199aa52e32f3c11c9ae647255beed23',
    'company/gates/P2_METHODS_LOCK.json': 'b3a2f62a7d98d48e504e1bd9231de317d65c65d9224aa45cfd4e8e706e6162ca',
    'company/gates/ANA_SCRIPT_FREEZE_v3.json': '59c83c861ff104d65292a15658a38a927626a286f9e628a1d0814c2de01d8134',
    'company/gates/P2_SCRIPT_FREEZE.json': '3077963f',
    'company/gates/EXP_BENCH_X_TEST_LOCK.json': '025a59dc2b7eeb6f087d5ca8800c65121068250f3d10f5b280b24889f14680ec',
    'company/gates/EXP_XREES46_TEST_LOCK.json': 'a67b19ca50527faf67d3eab9007b477e6f4688c539d3b2fc36f507b61d74332f',
    'company/gates/EXP_PB100_TEST_LOCK.json': '6ee1c65b',
    'company/gates/EXP_PB100_TEST_LOCK_v2.json': 'c3a67aafdb3bddb33c814037ddc60373017ec362e75a19ce645d7c728f71e36d',
    'tools/retention_report.py': '9a3b2c2a918ef2baacc5299426941f1eb94c366f5bb6aa7c29b4bfdbca0afe58',
    'tools/test_exposure_register.py': '7d59e828',
    'company/registers/TEST_EXPOSURE_REGISTER.json': 'dfdcbaea90557e0d6eca71d8aad85479b0048bcb5951f95e6e3a2fceacd4131f',
    'runs_bench/s3_beauty/TEST_RESULTS.json': '93603eb96a6880d9',
    'runs_bench/s3_sports/TEST_RESULTS.json': 'df7f330b34468846',
    'runs_bench/s3_toys/TEST_RESULTS.json': '6c4cbe7b100d1352',
    'runs_bench/amazon2023_video_games/TEST_RESULTS.json': 'a210da2ed58d4826',
    'runs_bench/amazon2023_baby_products/TEST_RESULTS.json': 'fa5e15d03b42ed78',
    'runs_bench_gpu/ml_1m/TEST_RESULTS.json': '76286a8df0e847fe',
    'runs_bench_x/TEST/ANALYSIS_EXP_BENCH_X/X_RESULT.json': 'e1b6368d1329347c',
    'runs_x_rees46/TEST/ANALYSIS_EXP_XREES46_TEST_LOCK/X_RESULT.json': 'df9ad2c67880ac9c600959f6eff02c6d2f277b068b0f525647b9edb633ea5221',
    'runs_x_rees46/TEST/ANALYSIS_EXP_PB100_TEST_LOCK_v2/X_RESULT.json': '200532ecd191cc682f5fc7e64710ee9117248d5fb1af3ac75837437206ce31f9',
}


def sha(p: Path) -> str:
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(p: Path):
    return json.loads(p.read_text())


def verify(v: Path, h: Path) -> dict:
    checks, details = [], {}

    def check(name, ok, **data):
        checks.append(dict(name=name, passed=bool(ok), **data))

    for rel, expected in EXPECTED.items():
        p = v / rel
        actual = sha(p) if p.is_file() else None
        check('sha:' + rel, actual is not None and actual.startswith(expected),
              expected=expected, actual=actual, comparison='full' if len(expected) == 64 else 'prefix')
    # Stop before inspecting values if any frozen input differs from its registered version.
    if not all(c['passed'] for c in checks):
        return dict(success=False, checks=checks, details=details)
    result = subprocess.run([sys.executable, '-B', str(v / 'tools/test_exposure_register.py'), 'verify'],
                            cwd=v, text=True, capture_output=True, check=False)
    register = json.loads(result.stdout) if result.returncode == 0 else {'output': result.stdout, 'stderr': result.stderr}
    check('V4:server_register', result.returncode == 0 and register == {'entries': 19, 'errors': []}, result=register)
    for rel in ['company/registers/TEST_EXPOSURE_REGISTER.json', 'company/gates/ALL_METHODS_LOCK.json']:
        check('V4:handover_server_equal:' + rel, sha(v / rel) == sha(h / 'ppsi_v4' / rel))

    p1 = load(v / 'runs_ds3/ANA/out_test_v3/RETENTION_RESULT.json')
    p2 = load(v / 'runs_p2/TEST/p2_out_test_v1/RETENTION_RESULT.json')
    headlines = {}
    for doc, ids, key in [(p1, ['K20b', 'K21', 'K2', 'K3', 'K9c', 'K12'], 'confirmatory'), (p2, ['Q1'], 'contrasts')]:
        indexed = {x['id']: x for x in doc[key]}
        for name in ids:
            rule = 'CONTROLLED_BUDGET' if name == 'K12' else 'PRACTICAL_BEST'
            row = indexed[name]['by_rule'][rule]
            headlines[name] = dict(rule=rule, **{k: row.get(k) for k in ['point', 'ci95', 'n_users', 'n_decisions', 'label']})
    details['headlines_from_existing_JSON'] = headlines
    for name, digits, expected in [('K20b', 2, [88.49, 87.73, 89.18]), ('K21', 6, [-0.000508, -0.000933, -0.000037]),
                                    ('Q1', 6, [-0.000081, -0.000179, 0.000024])]:
        row = headlines[name]
        got = [round(row['point'], digits), *[round(x, digits) for x in row['ci95']]]
        check('V2:' + name, got == expected and row['n_users'] == 26254 and row['n_decisions'] == 229742,
              printed=got, expected=expected)
    check('V2:Q1_equivalent', p2['q1_reading']['conditions_true'] == ['EQUIVALENT'])
    check('V2:deployment_grade_false', p2['deployment_grade']['deployment_grade'] is False)
    check('V2:Holm_zero_rejections', p2['holm']['n_rejected'] == 0)

    benchmarks = []
    for ds, expected in [('s3_beauty', .0437), ('s3_sports', .0224), ('s3_toys', .0410),
                          ('amazon2023_video_games', .0487), ('amazon2023_baby_products', .0164), ('ml_1m', .1936)]:
        doc = load(v / ('runs_bench_gpu' if ds == 'ml_1m' else 'runs_bench') / ds / 'TEST_RESULTS.json')
        means = {}
        for arm in (['C_FULL', 'FA', 'PRE'] if ds == 'ml_1m' else ['C_FULL']):
            seeds = [2026, 2027, 2028] if ds == 'ml_1m' and arm == 'C_FULL' else [2026, 2027]
            vals = {s: doc['runs'][f'{arm}_s{s}']['CONTROLLED_BUDGET']['ALL']['ndcg@10'] for s in seeds}
            mean = statistics.mean(vals.values())
            means[arm] = dict(per_seed=vals, mean=mean)
            want = expected if arm == 'C_FULL' else {'FA': .1572, 'PRE': .1092}[arm]
            check('V8:' + ds + ':' + arm, round(mean, 4) == want, mean=mean, printed=round(mean, 4), expected=want)
        benchmarks.append(dict(dataset=ds, rule='CONTROLLED_BUDGET', population='ALL', means=means))
    details['benchmark_spot_checks'] = benchmarks
    exploratory = []
    for rel, want in [('runs_x_rees46/TEST/ANALYSIS_EXP_XREES46_TEST_LOCK/X_RESULT.json', [.00022, -.00011, .00057]),
                      ('runs_x_rees46/TEST/ANALYSIS_EXP_PB100_TEST_LOCK_v2/X_RESULT.json', [.00911, .00810, .01009])]:
        doc = load(v / rel)
        row = doc['contrasts'][0]
        mrr = row['mrr@20']
        got = [round(mrr['point'], 5), *[round(x, 5) for x in mrr['ci95']]]
        check('V10:' + rel, got == want, printed=got, expected=want)
        exploratory.append(dict(path=rel, id=row['id'], label=doc['label'], point=mrr['point'], ci95=mrr['ci95']))
    details['registered_exploratory_results'] = exploratory

    pairs = [('company/gates/INVENTORY_BEFORE_EXP_XREES46.json', 'runs_x_rees46/locks/INVENTORY_AFTER_T2.json', 'combined_sha256', 'trees'),
             ('runs_x_rees46/locks/INVENTORY_BEFORE_T3.json', 'runs_x_rees46/locks/INVENTORY_AFTER_T3.json', 'combined_sha256', 'trees'),
             ('runs_bench_x/x2_state/INVENTORY_CONF_before.json', 'runs_bench_x/x2_state/INVENTORY_CONF_after.json', 'digest', 'per_dir')]
    for before, after, digest, contents in pairs:
        a, b = load(v / before), load(v / after)
        check('V11:' + before, a[digest] == b[digest] and a[contents] == b[contents], digest=a[digest],
              scope='historical before/after inventory receipts, not a new full-tree rehash')
    return dict(success=all(c['passed'] for c in checks), checks=checks, details=details,
                scope='existing registered artifacts only; no GitHub, no training, no new TEST evaluation')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v', type=Path, required=True)
    ap.add_argument('--handover', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.resolve().is_relative_to(args.v.resolve()):
        ap.error('outputs must be outside the experiment workspace')
    report = verify(args.v.resolve(), args.handover.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'success': report['success'], 'checks': len(report['checks']),
                      'failed': [c['name'] for c in report['checks'] if not c['passed']], 'report': str(args.out)}))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
