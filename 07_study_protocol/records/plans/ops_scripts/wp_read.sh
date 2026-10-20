/home/jovyan/venvs/ppsi-gpu/bin/python - <<'PY'
import json, os, glob
B='/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/WARM_PREVIEW/'
for d in sorted(glob.glob(B+'*')):
    if not os.path.isdir(d): continue
    out=[]
    for x in open(d+'/metrics.jsonl'):
        m=json.loads(x); p=m['rows_file']; p=p if os.path.isabs(p) else d+'/'+p
        rows=json.load(open(p))['rows']
        v=[r['value'] for r in rows if r['metric']=='mrr' and r.get('k')==20 and r['aggregation']=='macro' and r.get('eval_stream')=='E2E_ROWS']
        out.append((m['mark'],v[0]))
    best=max(out,key=lambda t:t[1])
    print(f"{os.path.basename(d):28s} best {best[1]:.4f}@{best[0]}  " + ' '.join(f"{mk}:{v:.3f}" for mk,v in out))
PY
