"""Owner-authorized fresh exploratory experiment modules.

The closure-phase experiments are split by stage. Each subpackage lives in its stage folder:

    experiment_extensions.pretraining  02_centralized/code/experiment_extensions/pretraining
    experiment_extensions.federated    03_federated_learning/code/experiment_extensions/federated
    experiment_extensions.on_device    04_on_device/code/experiment_extensions/on_device
    experiment_extensions.privacy      05_privacy_and_security/code/experiment_extensions/privacy
    experiment_extensions.baselines    06_evaluation_and_benchmarks/code/experiment_extensions/baselines

`pip install -e .` maps each one through `[tool.setuptools.package-dir]`. The stage folders are
also added to `__path__` below, so the subpackages import from a source checkout as well.
"""
import ppsi_paths as _ppsi_paths

for _stage in ("02_centralized", "03_federated_learning", "04_on_device", "05_privacy_and_security",
               "06_evaluation_and_benchmarks"):
    _dir = _ppsi_paths.REPO_ROOT / _stage / "code" / "experiment_extensions"
    if _dir.is_dir() and str(_dir) not in __path__:
        __path__.append(str(_dir))
del _ppsi_paths, _stage, _dir
