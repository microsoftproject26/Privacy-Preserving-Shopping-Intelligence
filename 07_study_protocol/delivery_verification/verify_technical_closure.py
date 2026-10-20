"""Verify technical completion from reviewed artifacts; never train or score TEST.

DP LOOK 23 is checked before opening any new DP result, manifest or scorer log.
The final byte inventories are verified, then current archived metadata is
compared; this helper does not repeat the already recorded full archive rehash.
Run with the copied Python through START_TECHNICAL_WORK.sh.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET
import ppsi_paths

sys.dont_write_bytecode = True
OPENED_JSON = []
HASH_CACHE = {}


class Incomplete(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def physical(path):
    p = str(Path(path))
    for prefix in [alias + '/' for alias in ppsi_paths.LEGACY_ALIASES]:
        if p.startswith(prefix):
            p = ppsi_paths.SERVER_BASE.as_posix() + '/' + p[len(prefix):]
            break
    return Path(p)


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def sha(path):
    p = physical(path)
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns, st.st_ino)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        end = p.stat()
        need((end.st_size, end.st_mtime_ns, end.st_ino) == key[1:], 'File changed while hashing: ' + str(p))
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def read(path):
    p = physical(path)
    OPENED_JSON.append(str(p))
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(p.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(RuntimeError('Nonfinite JSON')))


def pins(files):
    for filename, expected in files.items():
        need(sha(filename) == expected, 'Pinned bytes changed: ' + filename)


def verify_chain(reg):
    previous = None
    for number, entry in enumerate(reg['entries'], 1):
        body = {k: v for k, v in entry.items() if k != 'entry_sha256'}
        need(entry['entry_id'] == number and entry['prev_entry_sha256'] == previous,
             'Register chain order differs')
        need(hashlib.sha256(canon(body)).hexdigest() == entry['entry_sha256'], 'Register hash mismatch')
        previous = entry['entry_sha256']


def mirrors(root):
    source = root.parent / 'Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4'
    paths = [source / 'company/registers/TEST_EXPOSURE_REGISTER.json',
             root / 'ppsi_v4/optional_test_audit/state/local/company/registers/TEST_EXPOSURE_REGISTER.json',
             root / 'evidence/OPTIONAL_TEST_AUDIT/server/company/registers/TEST_EXPOSURE_REGISTER.json']
    old, local, server = [read(p) for p in paths]
    for reg in [old, local, server]:
        verify_chain(reg)
    need(len(old['entries']) == 19, 'Historical register must remain 19 entries')
    need(local == server and sha(paths[1]) == sha(paths[2]), 'New register mirrors differ')
    need(local['entries'][:19] == old['entries'], 'Historical prefix changed')
    identities = {(p.stat().st_dev, p.stat().st_ino) for p in paths}
    need(len(identities) == 3, 'Historical/local/server register inodes are not distinct')
    return source, paths, old, local


def look_identity(root, source, reg, kind):
    """Absence refuses before touching any new DP value/output path."""
    number = 23 if kind == 'DP' else 22
    lock_path = root / 'ppsi_v4/optional_test_audit/state/locks' / (
        'DP8_TEST_LOCK_20261013_v1.json' if kind == 'DP' else 'BENCH_TEST_LOCK_20261013_v1.json')
    lock = read(lock_path)
    need(lock['draft'] is False and lock['label'] == 'POST_HOC_EXPLORATORY', 'Final lock invalid')
    need(lock['self_sha256'] == hashlib.sha256(canon({k: v for k, v in lock.items()
                                                   if k != 'self_sha256'})).hexdigest(), 'Lock self hash mismatch')
    lock_sha = sha(lock_path)
    need(sha(source / 'company/registers/TEST_EXPOSURE_REGISTER.json') ==
         lock['register_fork']['historical_file_sha256'], 'Historical register pin changed')
    looks = [e for e in reg['entries'] if e.get('entry_id') == number
             and e.get('observations') == 'METRICS' and e.get('all_methods_lock_sha256') == lock_sha
             and physical(e.get('source', '')) == lock_path]
    if not looks:
        raise Incomplete(f'Exact mirrored {kind} LOOK{number} absent; no {kind} metric/output opened')
    need(len(looks) == 1, 'Ambiguous LOOK')
    entry = looks[0]
    manifest_path = root / 'evidence/OPTIONAL_TEST_AUDIT' / (kind + '_OUTPUT_MANIFEST_v4.json')
    need(entry['access_purpose'] == 'LOCKED_EVALUATION' and entry['split'] == 'RETRO_TEST', 'LOOK purpose invalid')
    need(physical(entry['receipt_path']) == manifest_path, 'LOOK receipt differs')
    need(f'output_manifest_sha256={sha(manifest_path)}' in entry['observations_description'], 'LOOK manifest pin differs')
    return lock_path, lock, entry, manifest_path


def test_provenance(root, source, reg, kind):
    lock_path, lock, entry, manifest_path = look_identity(root, source, reg, kind)
    evidence = root / 'evidence/OPTIONAL_TEST_AUDIT'
    proof_path = evidence / (kind + '_INVENTORY_RECEIPT_v4.json')
    proof = read(proof_path)
    need(proof['status'] == 'PASS' and not any(proof[k] for k in ['changed', 'added', 'removed']), 'TEST archive diff nonempty')
    need(proof['lock_sha256'] == sha(lock_path) and proof['output_manifest_sha256'] == sha(manifest_path), 'TEST provenance pins differ')
    for key in ['before', 'after']:
        need(sha(proof[key]) == proof[key + '_sha256'], 'Inventory artifact changed')
    dl_pin = proof['shas_only_decision_log_receipt']
    need(sha(dl_pin['path']) == dl_pin['sha256'], 'SHA-only log receipt changed')
    dl = read(dl_pin['path'])
    need(dl['scope'] == 'SHA_ONLY_NO_NEW_METRIC_VALUES_READ' and dl['output_manifest_sha256'] == sha(manifest_path), 'SHA-only log binding invalid')
    need(dl['exact_appended_row'] in (source / 'company/DECISION_LOG.md').read_text(), 'SHA-only row absent from source decision log')
    manifest = read(manifest_path)
    need(manifest['status'] == 'COMPLETE' and manifest['lock_sha256'] == sha(lock_path), 'Output manifest incomplete')
    need(manifest['unit_count'] == 4, 'Locked score unit count differs')
    pins(lock['package_files'])
    package = root / 'ppsi_v4/optional_test_audit'
    package_files = {str(p): sha(p) for p in sorted(list(package.glob('*.py')) + list(package.glob('*.sh')))}
    need(package_files == lock['package_files'], 'Locked package file set changed')
    pins(manifest['files'])
    pre_review = read(root / 'reviews' / ('DP8_TEST_FINAL_REVIEW_v4.json' if kind == 'DP' else 'BENCH_TEST_FINAL_REVIEW_v4.json'))
    need(pre_review['status'] == 'PASS' and pre_review['lock_sha256'] == sha(lock_path)
         and pre_review['reviewed_files'] == lock['package_files'], 'Pre-score exact review changed')
    bind_number = 21 if kind == 'DP' else 20
    bind = reg['entries'][bind_number - 1]
    need(bind['observations'] == 'NONE' and bind['all_methods_lock_sha256'] == sha(lock_path), 'BIND missing')
    need(bind['utc'] < entry['utc'], 'LOOK precedes BIND')
    for unit in lock['units']:
        receipt = read(Path(lock['output_root']) / unit / 'SCORE_RECEIPT.json')
        need(receipt['status'] == 'COMPLETE' and receipt['lock_sha256'] == sha(lock_path), 'Score receipt incomplete')
        need(receipt['bind']['entry_id'] == bind_number, 'Score receipt BIND mismatch')
    post_path = root / 'reviews' / (kind + '_POST_LOOK_FINAL_REVIEW.json')
    if not post_path.exists():
        raise Incomplete('Final independent post-LOOK review missing: ' + kind)
    post = read(post_path)
    need(post['status'] == 'PASS' and post['LOOK_entry_id'] == entry['entry_id'], 'Post-LOOK review incomplete')
    analysis_path = evidence / (kind + '_ANALYSIS_v4.json')
    for key, path in [('lock_sha256', lock_path), ('output_manifest_sha256', manifest_path),
                      ('inventory_receipt_sha256', proof_path), ('analysis_sha256', analysis_path)]:
        need(post[key] == sha(path), 'Post-review pin changed: ' + key)
    analysis = read(analysis_path)
    need(analysis['label'] == 'POST_HOC_EXPLORATORY' and analysis['lock_sha256'] == sha(lock_path), 'Analysis label/binding invalid')
    if kind == 'BENCH':
        pairs = sum(len(c['checkpoints']) for c in lock['cells'])
        need(len(lock['cells']) == 80 and pairs == 147, 'Benchmark locked scope differs')
        need(sum(len(d['means']) for d in analysis['datasets'].values()) == 147, '147 benchmark pairs not complete')
        need(sum(len(d['contrasts']) for d in analysis['datasets'].values()) == 90, '90 valid contrasts not complete')
        need(sum(len(d['unavailable_contrasts']) for d in analysis['datasets'].values()) == 22, 'Early-stop omissions disappeared')
        return dict(status='PASS', LOOK=22, cells=80, checkpoint_pairs=147, contrasts=90, unavailable_contrasts=22)
    need(len(lock['cells']) == 4 and len(analysis['points']) == 8 and len(analysis['contrasts']) == 2, 'DP8 scope differs')
    return dict(status='PASS', LOOK=23, own_checkpoint_completions=4, reference_and_DP_points=8, contrasts=2)


def e3(root):
    review = read(root / 'reviews/E3_AMENDED_BOTH_FINAL_POST_REVIEW.json')
    need(review['status'] == 'PASS' and review['exact_reviewed_code_files'] == 612, 'E3 final review incomplete')
    amendment = read(root / 'reviews/E3_AMENDMENT_FINAL_PRE_REPLAY_REVIEW.json')
    need(sha(root / 'reviews/E3_AMENDMENT_FINAL_PRE_REPLAY_REVIEW.json') == review['amendment_review_sha256'], 'E3 amendment review changed')
    pins(amendment['reviewed_files'])
    code = root / 'ppsi_v4/code'
    current = {str(p.relative_to(code)): sha(p) for p in sorted(code.rglob('*.py')) if '__pycache__' not in p.parts}
    need(len(current) == 612 and hashlib.sha256(canon(current)).hexdigest() == review['code_sha256'], 'E3 source inventory changed')
    baseline = read(root / 'reviews/E3_INITIAL_RUNS_POST_REVIEW.json')['variants']
    results = {}
    for variant in ['unfrozen', 'frozen']:
        pins(baseline[variant]['file_sha256'])
        pins(review['final_result_artifacts_sha256'][variant])
        p = root / 'evidence' / ('E3_REAL_' + variant.upper() + '_02')
        rows = [json.loads(l) for l in (p / 'ROUND_LOG.jsonl').read_text().splitlines()]
        fields = ['round', 'state_digest', 'lr', 'efe', 'weight']
        need([{k: r[k] for k in fields} for r in rows] == baseline[variant]['all50_trajectory_bindings'], 'E3 numerical trajectories changed')
        result = read(p / 'ANALYSIS_RESULT.json')
        need(result['status'] == 'COMPLETED_VALID_DESCRIPTIVE' and result['population_privacy_certified'] is False, 'E3 invalid interpretation')
        need(result['coverage']['n_individual'] == 512 and result['coverage']['n_cohorts'] == 50, 'E3 coverage incomplete')
        need(result['item_set_attack'] == baseline[variant]['item_set_attack'], 'Original item metrics/CIs changed')
        need(result['secagg_identity']['valid'] and len(result['secagg_identity']['receipts']) == 50, 'E3 crypto coverage incomplete')
        need(all(r['bit_identical'] and r['success'] and r['nonzero_encoded_words'] > 0
                 for r in result['secagg_identity']['receipts']), 'E3 crypto identity failed')
        need(all(d['status'] == 'CONVERGED' and d['convergence_warning_count'] == 0
                 for d in result['category_probe']['optimizer_diagnostics'].values()), 'E3 optimizer incomplete')
        for attempt in ['01', '02']:
            folder = root / 'evidence' / ('E3_REAL_' + variant.upper() + '_' + attempt)
            cleanup = read(folder / 'CAPTURE_DELETION_RECEIPT.json')
            need(cleanup['private_directory_removed'] and cleanup['remaining_raw_capture_paths'] == 0, 'E3 raw cleanup incomplete')
            need(not list(folder.glob('E3_PRIVATE_UPLOADS_*')) and not list(folder.rglob('*.pt')), 'E3 raw files remain')
        item = result['item_set_attack']
        if variant == 'unfrozen':
            need(item['verdict'] == 'AGGREGATE_ATTACK_REMAINS' and item['aggregate_advantage']['mean'] > .02, 'Negative aggregate finding lost')
            need(item['metrics']['single_f1']['mean'] == item['metrics']['aggregate_recall']['mean'] == 1.0, 'Support recovery finding changed')
            need(result['additional_defenses_required_by_measured_attack'] is True, 'Required defenses flag lost')
        else:
            need(item['status'] == 'NOT_APPLICABLE' and item['zero_frozen_deltas_are_not_privacy_evidence'], 'Frozen absence misinterpreted')
        need(result['verdict'] == 'SECAGG_NEEDED_NOT_DEMONSTRATED', 'E3 joint point rule changed')
        results[variant] = dict(status='PASS', exact_original_trajectory_rounds=50, targets=512,
                                item_status=item['status'], point_verdict=result['verdict'])
    return dict(status='PASS', variants=results, privacy_certified=False,
                technical_note='Aggregate input support attack remains above2pp; joint NOT_DEMONSTRATED never means privacyPASS')


def protection(root, source):
    before_path = root / 'evidence/PROTECTED_SHA256_BEFORE_EXECUTION.json'
    after_path = root / 'evidence/PROTECTED_SHA256_AFTER_ALL.json'
    proof_path = root / 'evidence/PROTECTED_SHA256_ALL_COMPARISON.json'
    if not after_path.exists() or not proof_path.exists():
        raise Incomplete('Final AFTER_ALL full byte inventory not ready')
    before, after, proof = [read(p) for p in [before_path, after_path, proof_path]]
    need(proof['status'] == 'PASS' and not any(proof[k] for k in ['added', 'removed', 'changed']), 'Final protected diff nonempty')
    need(sha(before_path) == proof['before_sha256'] and sha(after_path) == proof['after_sha256'], 'Protected proof pins changed')
    need(before['files'] == after['files'] and before['roots'] == after['roots'], 'Protected inventory differs')
    need(before['entries'] == after['entries'] == len(after['files']) == 78413, 'Protected scope differs')
    paths = {p for name in after['roots'] for p in (source / name).rglob('*') if p.is_file() or p.is_symlink()}
    need({str(p.relative_to(source)) for p in paths} == set(after['files']), 'Current protected file set differs')
    for relative, info in after['files'].items():
        p = source / relative
        st = p.lstat()
        for key, value in [('size', st.st_size), ('mtime_ns', st.st_mtime_ns), ('inode', st.st_ino), ('mode', st.st_mode)]:
            need(info[key] == value, 'Protected metadata changed since AFTER_ALL: ' + relative)
        if 'symlink' in info:
            need(p.is_symlink() and str(p.readlink()) == info['symlink'], 'Protected symlink changed')
    return dict(status='PASS', full_byte_inventory_entries=78413, empty_diff=True,
                current_metadata_matches_after_all=True, full_rehash_by_this_helper=False)


def data_bindings(root):
    final = read(root / 'evidence/WORK_DATA_BINDINGS.json')
    need(final['status'] == 'PASS' and not final['content_conflicts'] and not final['protected_data_mutations'], 'Canonical data binding conflict')
    need(final['no_Python_or_package_code_changed'] and final['no_TEST_table_or_CSV_parsed'], 'Data binding scope invalid')
    for pin in final['evidence'].values():
        need(sha(pin['path']) == pin['sha256'], 'Canonical data evidence changed')
    views = read(root / 'evidence/WORK_DATA_BINDINGS_FILE_VIEWS.json')
    need(views['status'] == 'PASS' and not views['changed'] and views['file_view_count'] == 996, 'Data views incomplete')
    for relative, info in views['file_views'].items():
        p = root / 'ppsi_v4' / relative
        need(sha(p) == info['sha256'], 'Canonical file view bytes changed: ' + relative)
    for link in final['links']:
        p = physical(link['link'])
        need(p.is_symlink() and physical(p.readlink()) == physical(link['target']), 'Canonical link changed')
    need(len(final['links']) == 40 and final['existing_handover_files_sha256_preserved'] == 151, 'Data binding preservation count differs')
    need(len(final['default_benchmark_TRAIN_verified']) == 4 and len(final['default_core_TRAIN_verified']) == 2, 'Default TRAIN verification incomplete')
    guard = read(root / 'evidence/WORK_DATA_BINDINGS_GUARD.json')
    need(guard['status'] == 'PASS' and guard['negative_controls_count'] == 7
         and all(v['nontruncating_write_open_denied'] and v['no_write_called'] for v in guard['negative_controls']), 'Canonical data write guard failed')
    return dict(status='PASS', file_views=996, missing_only_links=40, preserved_handover_files=151,
                actual_default_TRAIN_loaders=6, denied_nontruncating_write_controls=7)


def local_checks(root):
    local = read(root / 'evidence/CLOSURE_LOCAL_TEST_VERIFICATION.json')
    need(local['success'] and local['passed'] == 1087 and local['expected_negative_controls'] == 328
         and local['total_unique_tests'] == 1415, 'Scoped local verification incomplete')
    independent_path = root / 'reviews/CANONICAL_DATA_AND_LOCAL_CHECKS_REVIEW.json'
    need(sha(independent_path) == '603acb9668a492822c29a390d4498a576128c8ace7a4c8a20428a9734c0f8c45',
         'Independent canonical/local review changed')
    independent = read(independent_path)
    need(independent['status'] == 'PASS' and not independent['claims_exceeding_evidence'],
         'Independent canonical/local review incomplete')
    need(independent['closure_receipt_sha256'] == sha(root / 'evidence/CLOSURE_LOCAL_TEST_VERIFICATION.json')
         and independent['readiness_receipt_sha256'] == sha(root / 'evidence/WORK_DATA_BINDINGS.json'),
         'Independent review no longer binds current readiness/local receipts')
    for filename, expected in independent['independent_checks'].items():
        need(sha(root / 'reviews' / filename) == expected, 'Independent verification artifact changed')
    recount = read(root / 'reviews/CLOSURE_LOCAL_CHECKS_INDEPENDENT_RECOUNT.json')
    need(recount['status'] == 'PASS' and recount['closure_receipt_sha256'] == independent['closure_receipt_sha256'],
         'Independent corrected-case recount binding changed')
    need(recount['independently_merged_unique_tests'] == 1415 and recount['passed'] == 1087
         and recount['expected_negative_controls'] == 328 and recount['arbitrary_skips'] == 0
         and recount['unresolved_failure_or_error'] == 0, 'Independent scoped test counts differ')
    need(len(recount['XML_evidence_sha256']) == 8, 'Independent XML provenance coverage incomplete')
    # Original failed INT XML stays visible; corrected properties are merged by
    # the pinned independent recount rather than relabelling the original run.
    for relative, expected in recount['XML_evidence_sha256'].items():
        need(sha(root / relative) == expected, 'Historical/current XML evidence changed: ' + relative)
    for group in local['suites'].values():
        if 'sha256' in group:
            p = root / group['evidence']
            need(sha(p) == group['sha256'], 'Local XML changed')
            cases = list(ET.parse(p).getroot().iter('testcase'))
            need(len(cases) == group['unique_tests'], 'Local XML coverage differs')
            need(not any(c.find('failure') is not None or c.find('error') is not None for c in cases), 'Local XML failure/error')
    syntax_path = root / 'evidence/FINAL_PYTHON_SYNTAX.json'
    syntax = read(syntax_path)
    files = [p for folder in [root / 'ppsi_v4', root / 'code'] for p in folder.rglob('*.py') if '__pycache__' not in p.parts]
    need(syntax['success'] and not syntax['errors'] and syntax['files'] == len(files), 'Final syntax evidence incomplete/stale')
    need(syntax['interpreter'].startswith('3.11.14') and sys.version.startswith('3.11.14'), 'Use copied actual Python3.11.14')
    need(syntax_path.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in files), 'Syntax report predates final source')
    for p in files:
        ast.parse(p.read_bytes(), filename=str(p))
    historic = read(root / 'evidence/EXISTING_RESULTS_AUDIT_EXECUTION_FINAL.json')
    need(historic['success'] and len(historic['checks']) == 44 and all(c['passed'] for c in historic['checks']), 'Historical44-check audit incomplete')
    return dict(status='PASS', passed=1087, expected_negative_controls=328, scoped_cases=1415,
                current_parsed_python_files=len(files), historical_registered_artifact_checks=44,
                independently_pinned_XML_files=8,
                independent_data_and_local_review_sha256=sha(independent_path),
                scope=local['scope'], this_helper_gate_control_excluded_from_scoped_counts=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, default=ppsi_paths.DELIVERY_REVIEW_ROOT.resolve())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate-check-only', action='store_true')
    args = parser.parse_args()
    root = physical(args.review_root)
    output = physical(args.output)
    need(output.is_relative_to(root) and not output.exists(), 'Output must be a new review artifact')
    result = dict(format='PPSI_TECHNICAL_CLOSURE_VERIFICATION_1', utc=datetime.now(timezone.utc).isoformat(),
                  mode='GATE_ONLY' if args.gate_check_only else 'FINAL', sections={}, pending_DP_values_read=False)
    code = 0
    try:
        source, paths, old, reg = mirrors(root)
        result['sections']['mirrors'] = dict(status='PASS', entries=len(reg['entries']), preserved_historical_prefix=19, distinct_inodes=3)
        # The first dynamic operation concerning DP outcomes is this absence guard.
        look_identity(root, source, reg, 'DP')
        if args.gate_check_only:
            result['status'] = 'PASS_GATE_ONLY_NOT_TECHNICAL_CLOSURE'
        else:
            need(len(reg['entries']) == 23, 'Unexpected additional TEST exposures beyond reviewed closure')
            result['sections']['E3'] = e3(root)
            result['sections']['BENCH'] = test_provenance(root, source, reg, 'BENCH')
            result['sections']['DP8'] = test_provenance(root, source, reg, 'DP')
            result['sections']['protected_archives'] = protection(root, source)
            result['sections']['canonical_data'] = data_bindings(root)
            result['sections']['local_verification'] = local_checks(root)
            result['status'] = 'PASS_TECHNICAL_CLOSURE_WITH_RECORDED_SCIENTIFIC_LIMITATIONS'
    except Incomplete as exc:
        result['status'], result['reason'], code = 'INCOMPLETE', str(exc), 2
    except Exception as exc:
        result['status'], result['reason'], code = 'FAIL', str(exc), 1
    result['opened_JSON_paths'] = OPENED_JSON
    result['DP_value_JSON_opened'] = any('/DP_ANALYSIS_' in p or '/outputs/dp/' in p for p in OPENED_JSON)
    result['scientific_limitations'] = ['Technical closure is separate from deployment/privacy approval.',
                                      'Unfrozen E3 aggregate input attack remains above fixed2pp ceiling.',
                                      'Post-hoc benchmark/DP results never replace confirmatory evidence.',
                                      'Independent reviews were performed by AI agents.']
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({k: result.get(k) for k in ['status', 'reason', 'DP_value_JSON_opened']}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
