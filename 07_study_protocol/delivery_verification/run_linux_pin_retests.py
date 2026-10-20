"""Recheck all five affected modules using independently validated Linux references."""
import json, os, subprocess, sys
import ppsi_paths
review = ppsi_paths.DELIVERY_REVIEW_ROOT.resolve()   # the DELIVERY_REVIEW folder
evidence=review/'evidence/LINUX_PIN_RETESTS';evidence.mkdir(parents=True,exist_ok=True)
scratch=review/'scratch/linux_pin_retests';scratch.mkdir(parents=True,exist_ok=True)
targets=['tests_int/test_fl10_fedadam_launch.py', 'tests_int/test_fl12_launch.py', 'tests_int/test_fl14_06l.py', 'tests_int/test_fl15.py', 'tests_int/test_fl16.py']
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR='/tmp')
command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--basetemp',str(scratch/'pytest'),'--junitxml',str(evidence/'RETESTS.xml'),*targets]
with (evidence/'RETESTS.log').open('w') as log:
 result=subprocess.run(command,cwd=review/'ppsi_v4/code',env=env,stdout=log,stderr=subprocess.STDOUT)
(evidence/'COMMAND.json').write_text(json.dumps(dict(command=command,returncode=result.returncode),indent=2)+'\n')
print((evidence/'RETESTS.log').read_text()[-10000:])
raise SystemExit(result.returncode)
