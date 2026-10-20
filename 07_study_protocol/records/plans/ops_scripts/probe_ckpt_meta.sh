set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
export V
/home/jovyan/venvs/ppsi-gpu/bin/python -B - <<'PY'
import json, os
from pathlib import Path
v=Path(os.environ['V'])
for run in [v/'runs_p3/SASREC/02B_SASREC_s2026', v/'runs_ds/C_SMALL_s2026']:
    p=run/'RUN_START.json'
    print('RUN', p, p.exists())
    if p.exists():
        o=json.loads(p.read_text())
        print('TOP_KEYS', sorted(o))
        print('CONFIG_FIELDS', {k:o.get('config',{}).get(k) for k in ('run_id','family','seed','recipe_id','recipe_registry','model_variant','schema_registry','manifest_registry','data_version','theta0_init_sha256')})
        print('KEY_BASE_FIELDS', sorted(o.get('key_base',{})))
    for kind in ('best.pt','endpoint_6.0.pt'):
        c=run/'sealed/checkpoints'/kind
        print('CKPT', c.exists(), c.stat().st_size if c.exists() else None)
PY
