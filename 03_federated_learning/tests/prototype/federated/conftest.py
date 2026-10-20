"""Run test_fl_synthetic_smoke.py from a view of the prototype repository layout.

That module reads every path relative to the working directory, the way the prototype
repository had them (`docs/evidence/s1-pr-02/...`, `scripts/federated/fl_synthetic_smoke.py`,
`config/...`). It cannot be edited: the S1-PR-02 artifact manifest it verifies records the
SHA-256 of the module itself (`lifecycle_test`). So its tests run with the working directory set
to a byte copy of the current files laid out the old way (`ppsi_paths.prototype_view`). The
manifest is checked against the current bytes all the same.
"""

from __future__ import annotations

import pytest

from ppsi_paths import prototype_view

PROTOTYPE_CWD_MODULES = {"test_fl_synthetic_smoke.py"}

# Every path the module reads relative to the working directory, and every path the S1-PR-02
# manifest (docs/evidence/s1-pr-02/artifact_manifest.v1.json) records.
_VIEW_PATHS = (
    "config/fl_synthetic_smoke.v1.json",
    "scripts/federated",
    "tests/federated/test_fl_synthetic_smoke.py",
    "docs/federated-synthetic-smoke.md",
    "docs/evidence/s1-pr-02",
    "uv.lock",
)


@pytest.fixture(scope="session")
def _s1_pr_02_view(tmp_path_factory):
    return prototype_view(tmp_path_factory.mktemp("prototype_layout"), _VIEW_PATHS)


@pytest.fixture(autouse=True)
def _prototype_layout_cwd(request, monkeypatch):
    if request.path.name in PROTOTYPE_CWD_MODULES:
        monkeypatch.chdir(request.getfixturevalue("_s1_pr_02_view"))
