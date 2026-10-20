#!/usr/bin/env bash
# ML-1M GPU jobs: "python -B -m v4bench.run ARGS" -> the same module via runpy under -c (alter_sys=False), so that
# multiprocessing "spawn" pool workers do NOT re-import v4bench.run as __mp_main__ (its import-time line sets
# CUDA_VISIBLE_DEVICES="" and the workers then see 0 GPUs). No code-tree file changes; every other call passes through.
P=/home/jovyan/venvs/ppsi-gpu/bin/python
if [ "$1" = "-B" ] && [ "$2" = "-m" ] && [ "$3" = "v4bench.run" ]; then
  shift 3; exec "$P" -B -c 'import runpy; runpy.run_module("v4bench.run", run_name="__main__", alter_sys=False)' "$@"
fi
exec "$P" "$@"
