#!/usr/bin/env python
"""run_pipeline.py - run every stage's training units and from-results builders.

Stages run in order 01_data .. 06_evaluation_and_benchmarks; ``--stages 03 04`` selects a
subset. See ``uv run python run_pipeline.py --help`` and the ppsi_pipeline package
(``00_common/code/ppsi_pipeline/``) for the skip / resume / start rules, the ``--force``
move-aside and the ``--from-results`` table and figure rebuilders.
"""
from __future__ import annotations

from ppsi_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
