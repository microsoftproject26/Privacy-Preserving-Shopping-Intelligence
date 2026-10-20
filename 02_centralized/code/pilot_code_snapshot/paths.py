"""Workspace constants and import path setup for the SASRec campaign (owner: Opus).

Importing this module puts the read-only reused GRU v1 modules (`code/reuse`, a
byte-identical hashed copy) and their vendored `rees46_eda` on `sys.path`, and sets
the per-process thread defaults. It never modifies anything on disk.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# owner directive D2: 3 intra-op threads per trainer (8-CPU pod, two trainers + supervisor)
for _v, _n in (("OMP_NUM_THREADS", "3"), ("MKL_NUM_THREADS", "3"), ("OPENBLAS_NUM_THREADS", "1"),
               ("NUMEXPR_NUM_THREADS", "1")):
    os.environ.setdefault(_v, _n)

S = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1")
CODE = Path(__file__).resolve().parent
REUSE = CODE / "reuse"
GRU_WORKSPACE = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1")
DATASET_ROOT = Path("/home/jovyan/shared_data/rees46_nextitem_v1")
VERSION_HASH = "860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd"

K = 158_486
ITEM_VOCAB = 158_489
PAD, MISSING, OOV = 0, 1, 2
L = 50
N_TRAIN = 40_180_608
N_VS = 8_826_488
N_VS_USERS = 790_459
N_COHORT = 112_365
EFFECTIVE_BATCH = 256
U_PROD = 156_956  # ceil(N_TRAIN / 256)

REGISTRY_LOCKED = S / "config" / "run_registry.locked.json"
COHORT_PARQUET = GRU_WORKSPACE / "artifacts" / "validation_select_cohort.parquet"

for _p in (str(REUSE / "vendor"), str(REUSE), str(CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
