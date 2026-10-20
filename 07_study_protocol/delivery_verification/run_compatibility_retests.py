"""Recheck the corrected Linux fixtures without collecting unrelated suites."""
import json
import os
import subprocess
import sys
import ppsi_paths

review = ppsi_paths.DELIVERY_REVIEW_ROOT.resolve()   # the DELIVERY_REVIEW folder
evidence = review / 'evidence' / 'COMPATIBILITY_RETESTS'
scratch = review / 'scratch' / 'compatibility_retests'
evidence.mkdir(parents=True, exist_ok=True)
scratch.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', TMPDIR='/tmp')
targets = ['tests_int/test_p2_run1_lane.py', 'tests_int/test_p2_reeval.py']
command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
           '--basetemp', str(scratch / 'pytest'), '--junitxml', str(evidence / 'RETESTS.xml'), *targets]
with (evidence / 'RETESTS.log').open('w') as log:
    result = subprocess.run(command, cwd=review / 'ppsi_v4' / 'code', env=env,
                            stdout=log, stderr=subprocess.STDOUT)
(evidence / 'COMMAND.json').write_text(json.dumps(dict(command=command, returncode=result.returncode), indent=2)+'\n')
print((evidence / 'RETESTS.log').read_text()[-7000:])
raise SystemExit(result.returncode)
