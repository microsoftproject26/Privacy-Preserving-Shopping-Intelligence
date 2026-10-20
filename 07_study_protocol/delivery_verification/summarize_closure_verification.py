"""Combine current scoped verification without counting independent reruns twice."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from xml.etree import ElementTree as ET
import ppsi_paths

REVIEW = ppsi_paths.DELIVERY_REVIEW_ROOT.resolve()   # the DELIVERY_REVIEW folder


def xml_summary(relative):
    path=REVIEW/relative
    cases={}
    for c in ET.parse(path).getroot().iter('testcase'):
        key=c.get('classname')+'::'+c.get('name')
        if key in cases:raise RuntimeError('Duplicate test ID: '+key)
        status='passed'
        for tag in ['failure','error','skipped']:
            element=c.find(tag)
            if element is not None:
                if tag=='skipped' and 'NEGATIVE CONTROL:' in element.get('message',''):
                    status='expected_negative_control'
                else:status=tag
                break
        cases[key]=status
    return dict(unique_tests=len(cases),counts=dict(Counter(cases.values())),
                evidence=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    base=json.loads((REVIEW/'evidence/FINAL_LOCAL_TEST_VERIFICATION.json').read_text())
    if not base['success']:raise RuntimeError('Earlier scoped verification was incomplete')
    groups={k:v for k,v in base['suites'].items() if k!='tests_fl'}
    # The entire FL suite was run again after the optional observation hook.
    groups['tests_fl']=xml_summary('evidence/FL_AFTER_E3_OBSERVER.xml')
    groups['new_E3_checks']=xml_summary('evidence/E3_AMENDMENT_TESTS.xml')
    groups['new_optional_TEST_gate_checks']=xml_summary('reviews/INDEPENDENT_OPTIONAL_GATE_TESTS_v4.xml')
    if groups['tests_fl']['unique_tests']!=362:raise RuntimeError('Incomplete FL coverage')
    if groups['new_E3_checks']['unique_tests']!=48:raise RuntimeError('Incomplete E3 controls')
    if groups['new_optional_TEST_gate_checks']['unique_tests']!=22:raise RuntimeError('Incomplete TEST gate controls')
    totals=Counter()
    for group in groups.values():
        for status,count in group['counts'].items():
            totals['expected_negative_control' if status=='skipped' else status]+=count
    success=not any(totals[k] for k in ['failure','error','skipped'])
    result=dict(utc=datetime.now(timezone.utc).isoformat(),success=success,suites=groups,
                total_unique_tests=sum(g['unique_tests'] for g in groups.values()),
                passed=totals['passed'],expected_negative_controls=totals['expected_negative_control'],
                scope='Existing FL/ANA/INT plus P5, E3 and optional TEST gate controls; not every repository suite',
                provenance='FL rerun after server/runtime hooks. ANA/INT/P5 retain earlier corrected-case coverage. E3 current amended source and gate current locked source verified. Independent duplicate reruns do not increase counts.',
                historical_verification='evidence/FINAL_LOCAL_TEST_VERIFICATION.json')
    out=REVIEW/'evidence/CLOSURE_LOCAL_TEST_VERIFICATION.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['success','total_unique_tests','passed','expected_negative_controls','scope']}))
    if not success:raise SystemExit(1)


if __name__=='__main__':main()
