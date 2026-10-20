"""Run legacy suites in separate processes, each with its own scratch directory.

Their test helper module names overlap and their conftests bind global code roots;
collecting the suites together causes import collisions. No frozen tree is edited.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--suites', nargs='+', default=['tests_fl', 'tests_ana', 'tests_int'],
                    choices=['tests_fl', 'tests_ana', 'tests_int'])
    args = ap.parse_args()
    code = args.workspace.resolve() / 'code'
    review = args.workspace.resolve().parent
    evidence = args.evidence.resolve()
    scratch = args.scratch.resolve()
    if not evidence.is_relative_to(review) or not scratch.is_relative_to(review):
        ap.error('evidence and scratch must stay within the work review directory')
    evidence.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', TMPDIR='/tmp')
    report = dict(scope='existing local handover code, one process per suite', suites=[])
    for suite in args.suites:
        xml = evidence / (suite + '.xml')
        command = [sys.executable, '-B', str(Path(__file__).with_name('protect_work.py')),
                   '--allow', str(review), '--allow', str(review.parent / 'runtime'), '--',
                   sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                   '--basetemp', str(scratch / suite), '--junitxml', str(xml), suite]
        print('Running ' + suite, flush=True)
        with (evidence / (suite + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        item = dict(suite=suite, returncode=result.returncode, command=command, cwd=str(code), failures=[])
        if xml.exists():
            root = ET.parse(xml).getroot()
            item['totals'] = {k: sum(int(s.get(k, '0')) for s in root.iter('testsuite'))
                              for k in ['tests', 'failures', 'errors', 'skipped']}
            for case in root.iter('testcase'):
                for kind in ['failure', 'error']:
                    problem = case.find(kind)
                    if problem is not None:
                        item['failures'].append(dict(test=case.get('classname') + '::' + case.get('name'),
                                                     type=kind, message=problem.get('message'),
                                                     traceback=problem.text))
        report['suites'].append(item)
        report['success'] = all(s['returncode'] == 0 for s in report['suites'])
        (evidence / 'LOCAL_TEST_STATUS.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'suite': suite, 'returncode': result.returncode, 'totals': item.get('totals')}), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
