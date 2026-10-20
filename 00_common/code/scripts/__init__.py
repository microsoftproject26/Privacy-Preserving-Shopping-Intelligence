"""Compatibility package for the prototype's `scripts.*` import names.

The prototype kept its scripts in one root `scripts/` folder and imported them as a package
(`from scripts.experiments.schemas import ...`, `from scripts import training_smoke`). Installed
code does too: `ppsi.training.initialization` and `ppsi.training.result` import
`scripts.experiments`. The restructure moved each script folder to its stage:

    scripts.data                                        01_data/scripts/prototype/data
    scripts.baselines, .model, .t2, .t3, .t5, .t8,      02_centralized/scripts/prototype
      scripts.training_smoke
    scripts.federated, scripts.mvp                      03_federated_learning/scripts/prototype
    scripts.deployment                                  04_on_device/scripts/prototype
    scripts.evaluation, scripts.experiments             06_evaluation_and_benchmarks/scripts/prototype
    scripts.generate_training_artifact_manifest,        tools/
      scripts.checkpoint_manifest, scripts.env_smoke

`pip install -e .` maps the script subpackages through `[tool.setuptools.package-dir]`. The
folders are also added to `__path__` below, which is what makes the loose modules
(`scripts.training_smoke`, the tools) and the folders without `__init__.py` importable, and keeps
every old import name working without edits.
"""
import ppsi_paths as _ppsi_paths

for _dir in [*sorted(_ppsi_paths.REPO_ROOT.glob("0*/scripts/prototype")), _ppsi_paths.REPO_ROOT / "tools"]:
    if _dir.is_dir() and str(_dir) not in __path__:
        __path__.append(str(_dir))
del _ppsi_paths, _dir
