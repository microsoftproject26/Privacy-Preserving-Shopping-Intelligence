set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
export V
/home/jovyan/venvs/ppsi-gpu/bin/python -B - <<'PY'
import json, os
from pathlib import Path
v=Path(os.environ['V'])
p=v/'runs_ds3/T3_A40_r2/out/T3_GPU_RECEIPT.json'
o=json.loads(p.read_text())
print('T3',o['status'],o.get('all_checks_pass'))
for k,f in o['families'].items():
    print('T3_FAMILY',k,f['checks_pass'],f.get('a16_reference'))
p=v/'data_cfull/receipts/DATA08_SERVER_RECEIPT.json'
o=json.loads(p.read_text())
print('DATA08',o.get('overall_pass'),o.get('status'))
PY
tail -10 "$V/runs_ds3/pre/gate.log"
