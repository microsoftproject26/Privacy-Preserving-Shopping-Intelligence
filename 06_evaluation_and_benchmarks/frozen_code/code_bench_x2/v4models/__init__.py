"""v4models — Clean V2 model variants of the two executed candidate families (PPSI V4 task MODEL-01; closes OI-05).

Width-parameterized copies of the snapshot model modules (the snapshot `code/snap_v1` is never edited):
  gru_v2      copy of snap_v1/reuse/model.py   (ContextGRU; user-context / mask widths from the spec)
  sasrec_v2   copy of snap_v1/sasrec_model.py  (SASRecCE v3; numeric / flags / user widths from a `widths` object)
  DIFF_VS_SNAPSHOT.md   the exact line diff against the snapshot with a justification per hunk

Support modules:
  schema      FEATURE_SCHEMA_CLEAN_V2.json -> InputWidths (15 / 15 / 12 / 4, 53 channels), self-hash verified; batch
              layout check (widths, name arrays, feature_view)
  build       build(family, schema, seed, device) -> BuiltModel (module, FL adapter, theta0 sha256; SASRec theta0 is
              the recentered init, Fable G0(b) C7); ADAPTERS registration for fl.adapter
  monitors    gradient and common-mode drift monitors (T19)
  synthetic   synthetic Clean V2 decisions (tests only)
  inventory   CLI: parameter-role manifest + parameter inventory per family at the Clean V2 widths (full K)

Import order: `ppsi_env` is imported first (by `build`), so every snapshot module comes from code/snap_v1.
Run Python with -B (or PYTHONDONTWRITEBYTECODE=1) so no bytecode cache is written into the snapshot.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_CODE = _Path(__file__).resolve().parents[1]
if str(_CODE) not in _sys.path:
    _sys.path.insert(0, str(_CODE))
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

from .schema import (CleanV2Schema, InputWidths, SchemaError, WidthError, check_batch_layout,  # noqa: E402
                     load_schema, v1_reference_widths)
from .build import ADAPTERS, FAMILIES, TOPOLOGY, BuiltModel, build, make_adapter  # noqa: E402

__version__ = "MODEL-01.0"
__all__ = ["CleanV2Schema", "InputWidths", "SchemaError", "WidthError", "check_batch_layout", "load_schema",
           "v1_reference_widths", "ADAPTERS", "FAMILIES", "TOPOLOGY", "BuiltModel", "build", "make_adapter"]
