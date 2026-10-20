#!/usr/bin/env python
"""run.py - 04_on_device: on-device arms, export and latency measurements.

The same command line as the root run_pipeline.py, pinned to this stage: ``--mode auto``
decides skip / resume / start per training unit, ``--dry-run`` prints the plan, ``--force``
moves old outputs aside, ``--from-results`` rebuilds this stage's tables and figures from
``results/``. See ``00_common/code/ppsi_pipeline/`` for the rules.
"""
from __future__ import annotations

from ppsi_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main(only_stage="04_on_device"))
