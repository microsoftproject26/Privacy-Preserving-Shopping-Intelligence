"""Explicit reviewed recovery of the interrupted FULL_PRE seed2027 run.

Adds an operational exception to its original no-resume declaration. Numerical
source/config/budget stay unchanged. The original engine restores optimizer,
RNG, cursor and completed marks; uncheckpointed work is replayed, old logs stay.
No automatic retry and no arbitrary role/run/seed/checkpoint admission.
"""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
from experiment_extensions.pretraining import pretrain_seed as P

PROTOCOL = P.EXT / 'protocols/FULL_PRE_SEED2027_PROTOCOL_v2.json'
PROTOCOL_SHA = '318b74df29c177e9ba6d79e7270d2ba0b978677ab60c52bdd22fb6314b8cf4e6'
REVIEW = P.EXT / 'reviews/FULL_PRE_SEED2027_INDEPENDENT_PRE_TRAIN_REVIEW_v2.json'
REVIEW_SHA = '3615b0b5d733e477e1f519cf855ec89764993e280164fc9cc814e73e4031af41'
PREP = P.EXT / 'evidence/PRETRAIN_FULL_PROVIDER_PREPARATION_v2.json'
PREP_SHA = 'db941bd382e395075d9877d470a1d5d4eaa81c080cf36e5da58161b33f7dadbe'
FORMAT = 'EID_FULLPRE_MANUAL_RESUME_AMENDMENT_1'
AUTHOR = '/root'


def source_paths():
    return [Path(__file__), P.CODE / 'experiment_extensions/test_pretrain_resume.py',
            P.CODE / 'experiment_extensions/__init__.py']


def original_gate():
    P.load_protocol(PROTOCOL, PROTOCOL_SHA)
    P.load_review(REVIEW, REVIEW_SHA, PROTOCOL_SHA)
    if P.sha_file(PREP) != PREP_SHA:
        raise P.Refused('original provider preparation changed')
    prep = P.strict(PREP)
    if prep['status'] != 'PASS' or prep['fixed_declaration'] != P.fixed_declaration():
        raise P.Refused('original provider preparation identity')
    if P.binding(prep['raw_preparation']['path']) != prep['raw_preparation']:
        raise P.Refused('raw provider preparation changed')
    started = P.strict(P.RUN_DIR / 'EXTENSION_TRAIN_STARTED.json')
    if started['protocol_sha256'] != PROTOCOL_SHA or started['review_sha256'] != REVIEW_SHA or \
            started['preparation_sha256'] != PREP_SHA or started['fixed_declaration'] != P.fixed_declaration() or \
            started['launch_argv'] != P.launch_argv(0):
        raise P.Refused('original start receipt differs from approved fixedGPU0 run')


def verify_meta(start, entry, status):
    meta = entry['meta']
    if start['config']['run_id'] != P.RUN_ID or start['config']['seed'] != P.SEED or start['N'] != P.N or \
            start['theta0_init_sha256'] != P.THETA0 or status.get('status') != 'RUNNING':
        raise P.Refused('partial run role/seed/initializer/status mismatch')
    for key, expected in [('run_id', P.RUN_ID), ('seed', P.SEED), ('init_sha256', P.THETA0),
                          ('run_role', 'MAIN'), ('model_variant', 'SASREC_D64_B2'),
                          ('config_sha256', start['config_sha256']), ('device_pin_sha256', start['device_pin_sha256'])]:
        if meta.get(key) != expected:
            raise P.Refused('latest checkpoint differs: ' + key)
    if meta.get('updates') != 94101 or meta.get('exposures') != P.N or meta.get('efe') != 1.0 or \
            meta.get('attempt') != 0 or entry['file'] != 'checkpoints/latest.pt':
        raise P.Refused('this amendment admits only the verified first1EFE interruption')
    schema_identity = {key: start['schema'][key] for key in ('schema_id', 'schema_self_sha256')}
    if meta['determinism'] != start['determinism'] or meta['schema'] != schema_identity:
        raise P.Refused('checkpoint schema/determinism differs')


def partial_gate():
    for name in ['EXTENSION_COMPLETE.json', 'PRETRAIN_FOR_FL_INIT.json', 'RUN_SUMMARY.json',
                 'sealed/RUN_SUMMARY.json', 'sealed/checkpoints/endpoint_6.0.pt']:
        if (P.RUN_DIR / name).exists():
            raise P.Refused('completed/endpoint artifact prevents this recovery: ' + name)
    start = P.strict(P.RUN_DIR / 'RUN_START.json')
    index = P.strict(P.RUN_DIR / 'sealed/CHECKPOINTS.json')
    status = P.strict(P.RUN_DIR / 'STATUS.json')
    verify_meta(start, index['latest'], status)
    for key in ('latest', 'best'):
        entry = index[key]
        path = P.RUN_DIR / 'sealed' / entry['file']
        if P.sha_file(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise P.Refused('checkpoint byte integrity failed: ' + key)
    expected = {f'eval_{mark}.{suffix}' for mark in ('0.5', '1.0')
                for suffix in ('ranks_int32.npy', 'result.json', 'rows.json')}
    actual = {p.name for p in (P.RUN_DIR / 'sealed/evals').iterdir() if p.is_file()}
    if actual != expected:
        raise P.Refused('unexpected/orphan validation artifacts; do not silently rescore')
    return index


def build_amendment(out):
    P.fresh_output(out)
    original_gate()
    index = partial_gate()
    snapshot = P.EXT / 'evidence/PRETRAIN_INTERRUPTION_SNAPSHOT_01'
    snapshot.mkdir(parents=True, exist_ok=False)
    retained = []
    for path in sorted(P.RUN_DIR.rglob('*')):
        if not path.is_file():
            continue
        binding = P.binding(path)
        dest = snapshot / path.relative_to(P.RUN_DIR)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with path.open('rb') as src, dest.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        if P.sha_file(dest) != binding['sha256']:
            raise P.Refused('interruption snapshot copy changed')
        retained.append(dict(original=binding, snapshot=P.binding(dest)))
    obj = dict(format=FORMAT, utc=P.utc(), original_protocol_sha256=PROTOCOL_SHA,
               author=AUTHOR, fixed_run_id=P.RUN_ID, physical_gpu=0, latest=index['latest'],
               source_files=[P.binding(p) for p in source_paths()], retained_before=retained,
               amendment='Explicit manual resume from1EFE only; original no-resume records unchanged; '
                         'restore unchanged trainer weights/AdamW/RNG/cursor/marks; fixed6EFE endpoint; '
                         'preserve old104000update monitor evidence; replay from94101 under next attempt; '
                         'no adaptive continuation or TEST; no automatic retry',
               numerics_changed=False, original_review_sha256=REVIEW_SHA)
    obj['self_sha256'] = P.canonical_sha(obj)
    P.write_once(out, obj)
    return dict(status='PASS_AMENDMENT_BOUND_NO_TRAINING', amendment_path=str(P.physical(out)),
                amendment_sha256=P.sha_file(out), latest_sha256=index['latest']['sha256'],
                retained_files=len(retained))


def load_amendment(path, digest):
    path = P.inside(path)
    if P.sha_file(path) != digest:
        raise P.Refused('resume amendment SHA mismatch')
    obj = P.strict(path)
    if obj.get('format') != FORMAT or obj.get('self_sha256') != P.canonical_sha(obj) or \
            obj.get('fixed_run_id') != P.RUN_ID or obj.get('physical_gpu') != 0 or \
            obj.get('original_protocol_sha256') != PROTOCOL_SHA:
        raise P.Refused('resume amendment declaration/self hash mismatch')
    if obj['source_files'] != [P.binding(p) for p in source_paths()]:
        raise P.Refused('resume adapter/test/import bytes changed')
    original_gate()
    index = partial_gate()
    if obj['latest'] != index['latest']:
        raise P.Refused('latest checkpoint changed since recovery commitment')
    for item in obj['retained_before']:
        if P.binding(item['original']['path']) != item['original'] or P.binding(item['snapshot']['path']) != item['snapshot']:
            raise P.Refused('interruption evidence changed before launch')
    return obj


def resume(path, digest, review_path, review_digest):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise P.Refused('physicalGPU0 must be exported before first numerical import')
    obj = load_amendment(path, digest)
    review_path = P.inside(review_path)
    if P.sha_file(review_path) != review_digest:
        raise P.Refused('resume review SHA mismatch')
    review = P.strict(review_path)
    if review.get('status') != 'PASS' or review.get('amendment_sha256') != digest or \
            review.get('module_sha256') != P.sha_file(__file__) or \
            review.get('tests_sha256') != P.sha_file(P.CODE / 'experiment_extensions/test_pretrain_resume.py') or \
            review.get('reviewer') in (None, '', AUTHOR):
        raise P.Refused('independent exact adapter/amendment review required')
    receipt = P.RUN_DIR / 'EXTENSION_MANUAL_RESUME_01_STARTED.json'
    P.fresh_output(receipt)
    S = P.install(0)
    P.write_once(receipt, dict(status='EXPLICIT_RESUME_STARTED', utc=P.utc(), amendment_sha256=digest,
        independent_review_sha256=review_digest, from_update=94101, from_exposures=P.N,
        original_start_unchanged=True, automatic_retry=False))
    rc = S.main(['--ext-registry', str(P.WORK / P.REG_REL), '--'] + P.launch_argv(0) + ['--resume'])
    if rc:
        raise P.Refused('unchanged trainer resume refused; evidence retained: exit' + str(rc))
    endpoint = P.verify_endpoint(P.RUN_DIR)
    P.load_protocol(PROTOCOL, PROTOCOL_SHA)
    for item in obj['retained_before']:
        p = Path(item['original']['path'])
        if 'evals' in p.parts or p.name in ('RUN_START.json', 'EXTENSION_TRAIN_STARTED.json'):
            if P.binding(p) != item['original']:
                raise P.Refused('old checkpointed evaluation or start changed during recovery')
    result = dict(status='COMPLETE_FIXED6_VALIDATION_DESCRIPTIVE', utc=P.utc(), label=P.LABEL,
                  protocol_sha256=PROTOCOL_SHA, review_sha256=REVIEW_SHA, endpoint=endpoint,
                  operational_resume_amendment_sha256=digest, resume_review_sha256=review_digest,
                  test_scored=False, run_directory=str(P.RUN_DIR))
    P.write_once(P.RUN_DIR / 'EXTENSION_COMPLETE.json', result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build-amendment'); build.add_argument('--out', required=True, type=Path)
    run = sub.add_parser('resume')
    for name in ('amendment', 'amendment-sha256', 'review', 'review-sha256'):
        run.add_argument('--'+name, required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == 'build-amendment':
            result = build_amendment(args.out)
        else:
            with (P.EXT / 'evidence/PRETRAIN_GPU0_MANUAL_RECOVERY.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = resume(args.amendment, args.amendment_sha256, args.review, args.review_sha256)
        import json
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except (P.Refused, OSError, ValueError, KeyError) as exc:
        print('FULL_PRE_MANUAL_RESUME_REFUSED: ' + str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
