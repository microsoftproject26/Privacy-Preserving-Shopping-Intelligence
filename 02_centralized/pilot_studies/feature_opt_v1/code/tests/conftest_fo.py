"""conftest_fo.py (owner: Sonnet B, independent test worker) -- shared fixtures for test_fo_*.py.

Local only: no server, no network, no git, no pip. Puts `code/` on sys.path, points
FO_NORMALIZERS at the real fitted normalizers (read-only), and gives every test that writes
a fresh FO_ROOT_OVERRIDE tmp dir (never the real study folders).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
STUDY_ROOT = CODE_DIR.parent
NORMALIZERS_DEFAULT = STUDY_ROOT / "evidence" / "snap" / "dataset" / "normalizers.json"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("FO_NORMALIZERS", str(NORMALIZERS_DEFAULT))

import fo_env  # noqa: E402
import feature_view as fv  # noqa: E402
import fo_models as fm  # noqa: E402
from synthetic_store import SyntheticStore  # noqa: E402
from sasrec_model import SASRecConfig, SASRecCE  # noqa: E402

TINY_REGISTRY = {"id": "A1_RICH256", "group": "rich_frozen_channels", "d_model": 16, "blocks": 1,
                  "heads": 2, "ff_multiplier": 4, "dropout": 0.0, "peak_lr": 1e-3, "label_smoothing": 0.0}


def reload_env_modules(*names: str) -> None:
    """Re-import already-imported modules whose module-level constants read an env var at
    import time (fo_env.FO_ROOT, pin_baseline.CAMPAIGN), so a changed env var takes effect."""
    for name in names:
        mod = sys.modules.get(name)
        if mod is not None:
            importlib.reload(mod)
        else:
            importlib.import_module(name)


@pytest.fixture
def set_fo_root(monkeypatch):
    """Callable: point FO_ROOT_OVERRIDE at a directory and reload fo_env so
    fo_env.study_path()/FO_ROOT picks it up. Restores the original env var after the test."""
    created = []

    def _set(path: Path, mkdirs=("audit", "config", "state", "runs")):
        monkeypatch.setenv("FO_ROOT_OVERRIDE", str(path))
        reload_env_modules("fo_env")
        for d in mkdirs:
            (path / d).mkdir(parents=True, exist_ok=True)
        created.append(path)
        return path

    yield _set
    monkeypatch.delenv("FO_ROOT_OVERRIDE", raising=False)
    reload_env_modules("fo_env")


@pytest.fixture
def set_fo_campaign_root(monkeypatch):
    """Callable: point FO_CAMPAIGN_ROOT at a directory and reload pin_baseline so its
    module-level CAMPAIGN/RUNS constants pick it up."""
    def _set(path: Path):
        monkeypatch.setenv("FO_CAMPAIGN_ROOT", str(path))
        reload_env_modules("pin_baseline")
        return path

    yield _set
    monkeypatch.delenv("FO_CAMPAIGN_ROOT", raising=False)
    reload_env_modules("pin_baseline")


def tiny_store(n_train=64, n_vs=64, K=97, seed=0, task="mapping", channels="F4"):
    return SyntheticStore.build(channels=channels, n_train=n_train, n_vs=n_vs, K=K, seed=seed, task=task)


def tiny_inner(store, seed=1, perturb=True, gate=0.5):
    """The packet's 'tiny rich model': A1_RICH256 shrunk to d_model=16, perturbed weights,
    side_gate/user_gate=0.5 so the side paths actually matter."""
    m = SASRecCE(SASRecConfig.from_registry(TINY_REGISTRY), vocab_sizes=store.vocab_sizes,
                 product_idx_of_class=store.product_idx_of_class, seed=seed)
    with torch.no_grad():
        if perturb:
            g = torch.Generator().manual_seed(seed)
            for p in m.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
        m.side_gate.fill_(gate)
        m.user_gate.fill_(gate)
    return m


def full_kwargs(store, ds, positions):
    import sasrec_data
    b = store.batch(ds, np.asarray(positions))
    base = fm.FOModel(tiny_inner(store), view="full")
    return sasrec_data.predictor_kwargs(b, base), b
