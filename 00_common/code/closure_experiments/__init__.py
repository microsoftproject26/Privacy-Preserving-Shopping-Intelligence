"""Owner-authorized fresh exploratory experiment modules.

The closure-phase experiments are split by stage. Each subpackage lives in its stage folder
(the folders carry the descriptive name `closure_experiments`; the import name stays
`experiment_extensions`, as used in the recorded runs):

    experiment_extensions.pretraining  02_centralized/code/closure_experiments/pretraining
    experiment_extensions.federated    03_federated_learning/code/closure_experiments/federated
    experiment_extensions.on_device    04_on_device/code/closure_experiments/on_device
    experiment_extensions.privacy      05_privacy_and_security/code/closure_experiments/privacy
    experiment_extensions.baselines    06_evaluation_and_benchmarks/code/closure_experiments/baselines

`pip install -e .` maps each one through `[tool.setuptools.package-dir]`. The stage folders are
also added to `__path__` below, so the subpackages import from a source checkout as well.
"""
import ppsi_paths as _ppsi_paths

for _stage in ("02_centralized", "03_federated_learning", "04_on_device", "05_privacy_and_security",
               "06_evaluation_and_benchmarks"):
    _dir = _ppsi_paths.REPO_ROOT / _stage / "code" / "closure_experiments"
    if _dir.is_dir() and str(_dir) not in __path__:
        __path__.append(str(_dir))
del _ppsi_paths, _stage, _dir
