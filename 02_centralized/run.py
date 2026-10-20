#!/usr/bin/env python
"""run.py - 02_centralized: central training (contest, cloud references, server pretraining).

The same command line as the root run_pipeline.py, pinned to this stage: ``--mode auto``
decides skip / resume / start per training unit, ``--dry-run`` prints the plan, ``--force``
moves old outputs aside, ``--from-results`` rebuilds this stage's tables and figures from
``results/``. See ``00_common/code/ppsi_pipeline/`` for the rules.
"""
from __future__ import annotations

from ppsi_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main(only_stage="02_centralized"))
