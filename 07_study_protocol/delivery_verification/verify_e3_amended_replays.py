"""Read-only E3 replay/provenance verification; does not train or score TEST."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import ppsi_paths


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(review_root, variant):
    root = Path(review_root)
    review = read(root / 'reviews/E3_AMENDMENT_FINAL_PRE_REPLAY_REVIEW.json')
    baseline = read(root / 'reviews/E3_INITIAL_RUNS_POST_REVIEW.json')['variants'][variant]
    for filename, expected in baseline['file_sha256'].items():
        assert sha(filename) == expected, ('initial artifact changed', filename)
    old = root / 'evidence' / ('E3_REAL_' + variant.upper() + '_01')
    new = root / 'evidence' / ('E3_REAL_' + variant.upper() + '_02')
    log = [json.loads(line) for line in (new / 'ROUND_LOG.jsonl').read_text().splitlines()]
    fields = ['round', 'state_digest', 'lr', 'efe', 'weight']
    trajectory = [{key: row[key] for key in fields} for row in log]
    assert len(trajectory) == 50
    assert trajectory == baseline['all50_trajectory_bindings'], 'numerical trajectory changed'
    summary = read(new / 'REPLAY_SUMMARY.json')
    result = read(new / 'ANALYSIS_RESULT.json')
    cleanup = read(new / 'CAPTURE_DELETION_RECEIPT.json')
    binding = read(new / 'REPLAY_BINDING.json')
    code_root = root / 'ppsi_v4/code'
    reviewed_inventory = {str(Path(p).relative_to(code_root)): h
                          for p, h in review['reviewed_files'].items()}
    assert binding['numerical_code'] == reviewed_inventory
    assert binding['attack_protocol_sha256'] == review['amended_protocol_sha256']
    for filename in ['POPULATION.json', 'LR_TABLE_BINDING.json', 'SECAGG_BINDING.json']:
        assert read(old / filename) == read(new / filename), ('binding changed', filename)
    assert summary['status'] == 'CALIB_REPLAY_COMPLETE'
    assert summary['capture']['n_individual'] == 512
    assert summary['capture']['n_cohorts'] == 50
    assert summary['final_state_digest'] == trajectory[-1]['state_digest']
    assert result['item_set_attack'] == baseline['item_set_attack'], 'item metrics/CIs changed'
    assert result['secagg_identity']['valid']
    receipts = result['secagg_identity']['receipts']
    assert len(receipts) == 50 and all(r['success'] and r['bit_identical'] for r in receipts)
    assert cleanup['outcome'] == 'ANALYSIS_COMPLETED'
    assert cleanup['private_directory_removed'] and cleanup['remaining_raw_capture_paths'] == 0
    assert not list(new.glob('E3_PRIVATE_UPLOADS_*')) and not list(new.rglob('*.pt'))
    category = result['category_probe']
    if category['status'] == 'VALID_DESCRIPTIVE':
        assert all(d['status'] == 'CONVERGED' and d['convergence_warning_count'] == 0
                   for d in category['optimizer_diagnostics'].values())
        assert result['status'] == 'COMPLETED_VALID_DESCRIPTIVE'
    else:
        assert category['status'] == result['status'] == 'INCOMPLETE'
    return {
        'variant': variant, 'numerical_trajectory_identical_rounds': 50,
        'initial_artifacts_hashes_unchanged': True, 'item_metrics_and_CIs_identical': True,
        'targets': 512, 'cohorts': 50, 'secagg_bit_identical_rounds': 50,
        'raw_capture_deleted': True, 'status': result['status'], 'verdict': result['verdict'],
        'category_status': category['status'],
        'optimizer_diagnostics': category.get('optimizer_diagnostics'),
        'elapsed_s': summary['replay_elapsed_s'],
        'binding_sha256': binding['self_sha256'],
        'files_sha256': {p.name: sha(p) for p in sorted(new.iterdir()) if p.is_file()},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, default=ppsi_paths.DELIVERY_REVIEW_ROOT.resolve())
    parser.add_argument('--variant', choices=['unfrozen', 'frozen'], action='append')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {
        'format': 'E3_AMENDED_REPLAY_VERIFICATION_1',
        'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'label': 'POST_HOC_EXPLORATORY_DESCRIPTIVE', 'TEST_opened': False,
        'variants': [verify(args.review_root, v) for v in (args.variant or ['unfrozen', 'frozen'])],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({v['variant']: {'status': v['status'], 'verdict': v['verdict'],
                                   'optimizer': v['optimizer_diagnostics']}
                      for v in result['variants']}, indent=2))
