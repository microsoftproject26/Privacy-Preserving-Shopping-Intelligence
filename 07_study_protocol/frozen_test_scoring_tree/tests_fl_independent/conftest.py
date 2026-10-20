"""QA-03 — independent FL/PFL/LO test suite (Sonnet QA engineer, a different author from FL-01).

Expectations in this suite are derived from the package specification
(06_FEDERATED_AND_PERSONALIZED.md, 07_LOCAL_ONLY_AND_DEVICE.md sections 1-2,
12_ACCEPTANCE_TESTS.md T23-T34) and hand-computed toy arithmetic worked out BEFORE `code/tests_fl/`
was read. `code/tests_fl/` was read only afterwards, to avoid duplication and to build the
coverage-gap list in `company/handoffs/QA-03.md`.

This file mirrors the project convention (see `code/fl/families.py`, `code/tests_fl/conftest.py` per
FL-01's handoff): put `code/` on sys.path, enforce strict FP32 and 1 thread for every test in this
session (Fable G0 decision 5; `fl.numerics`).
"""
from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]          # .../ppsi_v4/code
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import torch  # noqa: E402
from fl.numerics import enforce_strict_fp32  # noqa: E402

torch.set_num_threads(1)
enforce_strict_fp32(deterministic=True)
