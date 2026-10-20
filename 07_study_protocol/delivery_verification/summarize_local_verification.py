"""Combine full-suite coverage and explicit corrective reruns by unique test ID.

Initial failures remain in their original XML/logs. This report records the
latest result for each test without counting reruns as additional coverage.
"""
import json
import xml.etree.ElementTree as ET
from collections import Counter
import ppsi_paths

review = ppsi_paths.DELIVERY_REVIEW_ROOT.resolve()   # the DELIVERY_REVIEW folder
evidence = review / 'evidence'


def cases(relative):
    result = {}
    for case in ET.parse(evidence / relative).getroot().iter('testcase'):
        key = case.get('classname') + '::' + case.get('name')
        status = 'passed'
        reason = None
        for tag in ['failure', 'error', 'skipped']:
            issue = case.find(tag)
            if issue is not None:
                status = tag
                reason = issue.get('message')
                break
        result[key] = dict(status=status, reason=reason, evidence=relative)
    return result


groups = {}
fl = cases('LOCAL_SUITES/tests_fl.xml')
fl.update(cases('LOCAL_SUITES/gpu_receipt_fix.xml'))
groups['tests_fl'] = fl
groups['tests_ana'] = cases('LOCAL_SUITES/tests_ana.xml')
integration = cases('INT_CANONICAL_WORKSPACE/tests_int.xml')
# Same refusal property, renamed when its obsolete real-host-state assumption
# was replaced by an explicit absent-lock fixture. Do not double-count it.
old = 'test_p2_reeval::test_test_role_refused_on_this_host_now'
replacement = 'test_p2_reeval::test_test_role_refused_when_p2_lock_missing'
compatibility = cases('COMPATIBILITY_RETESTS/RETESTS.xml')
assert replacement in compatibility and old in integration
integration.pop(old)
integration.update(compatibility)
integration.update(cases('LINUX_PIN_RETESTS/RETESTS.xml'))
groups['tests_int'] = integration
groups['new_P5_checks'] = cases('P5_GUARDED_TESTS.xml')

expected = dict(tests_fl=362, tests_ana=340, tests_int=631, new_P5_checks=12)
report = dict(scope='Three existing FL/ANA/INT suites plus new P5 scientific checks; not every repository test suite',
              method='Latest observed case result; corrective reruns replace cases rather than add coverage',
              suites={}, unresolved=[], skipped_other_than_negative_controls=[])
for group, data in groups.items():
    assert len(data) == expected[group], (group, len(data), expected[group])
    counts = dict(Counter(item['status'] for item in data.values()))
    report['suites'][group] = dict(unique_tests=len(data), counts=counts,
                                   evidence=sorted({item['evidence'] for item in data.values()}))
    for name, item in data.items():
        if item['status'] in ['failure', 'error']:
            report['unresolved'].append(dict(suite=group, test=name, **item))
        if item['status'] == 'skipped' and 'NEGATIVE CONTROL:' not in (item['reason'] or ''):
            report['skipped_other_than_negative_controls'].append(dict(suite=group, test=name, **item))
report['total_unique_tests'] = sum(len(c) for c in groups.values())
report['passed'] = sum(item['status'] == 'passed' for data in groups.values() for item in data.values())
report['expected_negative_control_failures'] = sum(item['status'] == 'skipped' for data in groups.values()
                                                  for item in data.values())
report['success'] = not report['unresolved'] and not report['skipped_other_than_negative_controls']
report['platform_baseline_provenance'] = ['LINUX_TRAJECTORIES_COMPARISON.json', 'LINUX_WARM_TRAJECTORIES_COMPARISON.json']
report['canonical_root_coverage'] = 'INT and all corrective reruns use ppsi_v4; FL/ANA full runs used the earlier same-content TECHNICAL_WORK copy. Production numeric files used by those suites were not subsequently edited.'
(evidence / 'FINAL_LOCAL_TEST_VERIFICATION.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(0 if report['success'] else 1)
