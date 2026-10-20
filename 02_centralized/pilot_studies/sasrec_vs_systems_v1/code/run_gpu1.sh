#!/usr/bin/env bash
# run_gpu1.sh (owner: Opus): launch vs_systems.py on GPU1 only, per review/AUTO_DECISION_GPU1.md.
# Refuses to start without the Fable code-review pass file. Detached in tmux vss-gpu1; no LLM polling.
set -u
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_vs_systems_v1
[ -f $V/review/CODE_REVIEW_PASS.json ] || { echo "no CODE_REVIEW_PASS.json"; exit 2; }
mkdir -p "$V/logs" "$V/state" "$V/results"   # blocker 2: log dir must exist before the redirect
cd $V/code
export CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
       OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec nice -n 10 ~/venvs/ppsi-gpu/bin/python vs_systems.py >> $V/logs/vs_systems.log 2>&1
