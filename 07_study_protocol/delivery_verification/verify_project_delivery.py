"""Read-only delivery integrity check. Never trains, scores, or initializes CUDA."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import ppsi_paths

BASE = ppsi_paths.SERVER_BASE.resolve()   # the folder that holds DELIVERY_REVIEW
DEFAULT = BASE / 'DELIVERY_REVIEW/DELIVERY_MANIFEST_20261017.json'

def validate(manifest, base=BASE):
    errors = []
    if manifest.get('format') != 'PPSI_TECHNICAL_DELIVERY_MANIFEST_1':
        return ['unsupported delivery manifest']
    if manifest.get('status') != 'CLOSED_SELECTED_CODE_AND_COMPLETED_RESULTS_WITH_LIMITATIONS':
        errors.append('delivery scope is not closed')
    for relative in manifest.get('required_stop_files', []):
        path = (base / relative).resolve()
        if not path.is_relative_to(base.resolve()) or not path.is_file():
            errors.append('missing automation stop: ' + relative)
    rows = manifest.get('files', [])
    if not rows:
        errors.append('empty delivery inventory')
    seen = set()
    for row in rows:
        relative = row['relative_path']
        if relative in seen:
            errors.append('duplicate inventory path: ' + relative)
            continue
        seen.add(relative)
        path = (base / relative).resolve()
        if not path.is_relative_to(base.resolve()) or not path.is_file():
            errors.append('missing or external artifact: ' + relative)
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != row['bytes'] or digest != row['sha256']:
            errors.append('artifact bytes changed: ' + relative)
    return errors

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        errors = validate(manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors = [type(error).__name__ + ': ' + str(error)]
        manifest = {}
    print(json.dumps({'status': 'PASS' if not errors else 'FAIL',
                      'scope': 'Declared delivery inventory and automation stop files only',
                      'files_checked': len(manifest.get('files', [])), 'errors': errors,
                      'training_scoring_or_CUDA': False}, indent=2))
    return 1 if errors else 0

if __name__ == '__main__':
    sys.exit(main())
