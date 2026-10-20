set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
export V
/home/jovyan/venvs/ppsi-gpu/bin/python -B - <<'PY'
import json, os, torch
from pathlib import Path
v=Path(os.environ['V'])
for run in [v/'runs_p3/SASREC/02B_SASREC_s2026',v/'runs_ds/C_SMALL_s2026']:
    start=json.loads((run/'RUN_START.json').read_text())
    p=run/'sealed/checkpoints/best.pt'
    obj=torch.load(p,map_location='cpu',weights_only=False)
    print('RUN',run.name,'schema_keys',sorted(start['schema']),'config_dropout',start['config'].get('dropout'))
    print('CHECKPOINT',obj.get('kind'), 'meta_keys',sorted(obj.get('meta',{})), 'model_tensors',len(obj.get('model',{})))
    print('META_FIELDS',{k:obj.get('meta',{}).get(k) for k in ('model_variant','seed','config_sha256','efe','exposures','mark')})
PY
