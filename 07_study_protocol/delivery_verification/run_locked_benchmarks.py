"""Run the four reviewed, BIND-gated benchmark units with status-only output.

Each child enters the existing isolated write guard. No rank or metric value is
read here. The frozen package itself verifies review, BIND and all pinned bytes.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import ppsi_paths

REVIEW = ppsi_paths.DELIVERY_REVIEW_ROOT
RUN = REVIEW / 'ppsi_v4/optional_test_audit/run.sh'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not a.out.resolve().is_relative_to(REVIEW):
        p.error('Operator output must stay in review workspace')
    a.out.mkdir(parents=True, exist_ok=False)
    units = ['ml_1m', 's3_beauty', 's3_sports', 's3_toys']
    env = dict(os.environ, PPSI_OPTIONAL_THREADS='16', PYTHONDONTWRITEBYTECODE='1')
    def run(ds):
        command = ['bash', str(RUN), 'bench', 'score', '--lock', str(a.lock),
                   '--review', str(a.review), '--dataset', ds, '--threads', '16']
        start = time.monotonic()
        print('START ' + ds + ': 16 CPU threads, gated scoring', flush=True)
        with (a.out / (ds + '.operator.log')).open('x') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        return dict(unit=ds, returncode=result.returncode, command=command,
                    elapsed_s=time.monotonic()-start, log=str(a.out / (ds+'.operator.log')))
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(run, ds) for ds in units]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(json.dumps({k:result[k] for k in ['unit','returncode','elapsed_s']}), flush=True)
            report = dict(utc=datetime.now(timezone.utc).isoformat(), label='POST_HOC_EXPLORATORY',
                          reads_new_metric_values=False, results=results,
                          units_completed=len(results), all_units_complete=len(results)==4,
                          success=len(results)==4 and all(r['returncode']==0 for r in results))
            (a.out/'OPERATOR_STATUS.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if all(r['returncode']==0 for r in results) else 1


if __name__ == '__main__':raise SystemExit(main())
