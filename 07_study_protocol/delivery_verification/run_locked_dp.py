"""Score four BIND-gated DP8 endpoints, two sequential queues on idle GPUs.

Never reads model metrics or sealed scorer logs. Each scorer enforces reviewed
lock, register, source, accountant, checkpoint and holdout bindings itself.
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
RUN = REVIEW/'ppsi_v4/optional_test_audit/run.sh'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',choices=['0','1','both'],default='both')
    a=p.parse_args()
    if not a.out.resolve().is_relative_to(REVIEW):p.error('Output escapes review')
    a.out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PPSI_OPTIONAL_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
    results=[]
    lanes=[(0,2026),(1,2027)] if a.gpu=='both' else [(int(a.gpu),2026+int(a.gpu))]
    expected=2*len(lanes)
    def lane(gpu,seed):
        done=[]
        for line in ['cold','pb100']:
            unit=f'DP8_{line}_s{seed}_CB'
            command=['bash',str(RUN),'dp','score','--lock',str(a.lock),'--review',str(a.review),
                     '--entry',unit,'--gpu',str(gpu)]
            print(f'START {unit}: GPU{gpu}; gated scoring',flush=True)
            start=time.monotonic()
            with (a.out/(unit+'.operator.log')).open('x') as log:
                r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            item=dict(unit=unit,gpu=gpu,returncode=r.returncode,command=command,
                      elapsed_s=time.monotonic()-start,log=str(a.out/(unit+'.operator.log')))
            done.append(item)
            print(json.dumps({k:item[k] for k in ['unit','gpu','returncode','elapsed_s']}),flush=True)
            (a.out/f'GPU{gpu}_STATUS.json').write_text(json.dumps(dict(results=done),indent=2)+'\n')
            if r.returncode:break
        return done
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(lane,gpu,seed) for gpu,seed in lanes]
        for f in as_completed(pending):
            results.extend(f.result())
            report=dict(utc=datetime.now(timezone.utc).isoformat(),label='POST_HOC_EXPLORATORY',
                        results=results,reads_new_metric_values=False,units_completed=len(results),
                        expected_units=expected,
                        success=len(results)==expected and all(r['returncode']==0 for r in results))
            (a.out/'OPERATOR_STATUS.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if len(results)==expected and all(r['returncode']==0 for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
