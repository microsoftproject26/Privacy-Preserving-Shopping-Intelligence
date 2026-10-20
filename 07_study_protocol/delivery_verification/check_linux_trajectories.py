"""Emit toy trajectories from a specified code tree; never touch its receipts.

Run separately for the historical frozen tree and the current working tree.
This keeps imports isolated and compares real results on the same Linux host.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--code', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--scratch', type=Path, required=True)
ap.add_argument('--warm', action='store_true')
args = ap.parse_args()
code = args.code.resolve()
sys.path.insert(0, str(code))
sys.path.insert(0, str(code / 'tests_int'))
import ppsi_env
import torch
from fl.numerics import enforce_strict_fp32
from test_fl10_fedadam_launch import F, PRE_FL10_TRAJ, _tiny, run_tiny

torch.set_num_threads(1)
enforce_strict_fp32()
args.scratch.mkdir(parents=True, exist_ok=False)
digests = {}
if args.warm:
    from test_fl12_launch import make_pre, warm_world
    from test_fl16 import S4_POINT, tiny_spec, PRE_FL16_WARM_TRAJ
    checkpoint, adapter = make_pre(args.scratch / 'pre')
    state, ident, launch = F.load_pretrain_init(checkpoint, model_variant='SASREC_D64_B2')
    pre = dict(path=checkpoint, adapter=adapter, state=state, ident=ident, launch=launch)
    for arm in F.WARM_ARMS:
        for name, kw in [('default', {}), ('s4', S4_POINT)]:
            tag = f'{arm}_{name}'
            output = run_tiny(args.scratch, tiny_spec(arm, rounds=12, init_from=ident, **kw),
                              tag, world=warm_world(pre))
            loss_sha = hashlib.sha256((args.scratch / tag / F.TRAIN_LOSS_FILE).read_bytes()).hexdigest()
            digests[f'{arm}|{name}'] = [output['summary']['server_digest'], loss_sha]
    registered_pins = PRE_FL16_WARM_TRAJ
else:
    for arm in F.ARMS:
        digests[arm] = run_tiny(args.scratch, _tiny(arm), arm)['summary']['server_digest']
    registered_pins = PRE_FL10_TRAJ
args.out.parent.mkdir(parents=True, exist_ok=True)
record = dict(code=str(code), torch=torch.__version__, threads=torch.get_num_threads(),
              server_digests=digests, registered_pins=registered_pins,
              code_file_sha256={rel: hashlib.sha256((code / rel).read_bytes()).hexdigest()
                                for rel in ['v4int/launch_fl.py', 'fl/runtime.py', 'fl/server.py', 'fl/client.py']})
args.out.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(dict(output=str(args.out), server_digests=digests)))
