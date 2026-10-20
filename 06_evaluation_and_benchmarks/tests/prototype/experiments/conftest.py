"""Make the shared record builders of these tests importable by name.

Several test modules reuse the builders in ``test_schemas`` and ``test_primitives``
(``from test_schemas import _make_experiment_config``). Before the stage layout they were
imported as ``tests.experiments.*`` from the repository root; ``scripts.experiments`` now comes
from the installed package. Putting this folder on sys.path keeps the builders importable in any
pytest import mode.
"""

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
