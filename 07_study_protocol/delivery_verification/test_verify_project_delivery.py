import copy
import hashlib
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('delivery_check', Path(__file__).with_name('verify_project_delivery.py'))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

def fixture(tmp_path):
    (tmp_path / 'STOP').write_text('owner stopped experiments')
    p = tmp_path / 'model_metadata.json'
    p.write_text('{"status":"COMPLETE"}')
    return {'format':'PPSI_TECHNICAL_DELIVERY_MANIFEST_1',
            'status':'CLOSED_SELECTED_CODE_AND_COMPLETED_RESULTS_WITH_LIMITATIONS',
            'required_stop_files':['STOP'],
            'files':[{'relative_path':p.name,'bytes':p.stat().st_size,
                      'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}]}

def test_valid_saved_delivery_does_not_modify_files(tmp_path):
    m = fixture(tmp_path)
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert V.validate(m, tmp_path) == []
    assert before == {p.name:p.read_bytes() for p in tmp_path.iterdir()}

def test_changed_artifact_fails(tmp_path):
    m = fixture(tmp_path)
    (tmp_path / 'model_metadata.json').write_text('{"status":"FAILED__"}')
    assert any('bytes changed' in e for e in V.validate(m, tmp_path))

def test_missing_stop_or_artifact_fails(tmp_path):
    m = fixture(tmp_path)
    (tmp_path / 'STOP').unlink()
    (tmp_path / 'model_metadata.json').unlink()
    errors = V.validate(m, tmp_path)
    assert len(errors) == 2

def test_duplicate_or_external_artifact_fails(tmp_path):
    m = fixture(tmp_path)
    m['files'].append(copy.deepcopy(m['files'][0]))
    assert any('duplicate' in e for e in V.validate(m, tmp_path))
    m['files'][0]['relative_path'] = '../outside'
    assert any('external' in e for e in V.validate(m, tmp_path))
