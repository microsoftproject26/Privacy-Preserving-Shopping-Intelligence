/home/jovyan/venvs/ppsi-gpu/bin/python - <<'PY'
import json, os
B='/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/'
for d in ['LRSCREEN_S3/A2_FEDADAM_eta0.01_FLCONST_SGD0.1','LRSCREEN_S3/A2_FEDADAM_eta0.01_FLCONST_SGD1.0','LRSCREEN_S3/W_FEDADAM_eta0.01_FLCONST_SGD0.01','LRSCREEN_S3/W_FEDADAM_eta0.01_FLCONST_ADAMW','LRSCREEN_S2/FEDADAM_eta0.01_tau1e-5_x1']:
    out=[]
    for x in open(B+d+'/metrics.jsonl'):
        m=json.loads(x); p=m['rows_file']; p=p if os.path.isabs(p) else B+d+'/'+p
        rows=json.load(open(p))['rows']
        v=[r['value'] for r in rows if r['metric']=='mrr' and r.get('k')==20 and r['aggregation']=='macro' and r.get('eval_stream')=='E2E_ROWS']
        out.append(f"{m['mark']}:{v[0]:.4f}" if v else f"{m['mark']}:?{len(rows)}")
    print(d.ljust(32),' '.join(out))
PY
