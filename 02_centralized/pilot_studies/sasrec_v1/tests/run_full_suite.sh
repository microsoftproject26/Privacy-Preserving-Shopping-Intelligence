#!/usr/bin/env bash
S=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1
cd $S/code
CUDA_VISIBLE_DEVICES=0 ~/venvs/ppsi-gpu/bin/python -m pytest tests/test_attention.py tests/test_head_objective.py tests/test_schedule_stopping.py tests/test_eval_and_precision.py tests/test_rich.py tests/test_trainer_exposure_resume.py tests/test_data_adapter.py tests/test_runtime.py -q -p no:cacheprovider --junitxml=$S/tests/evidence/FULL_SUITE_junit.xml > $S/tests/evidence/FULL_SUITE.log 2>&1
echo "exit=$?" >> $S/tests/evidence/FULL_SUITE.log
