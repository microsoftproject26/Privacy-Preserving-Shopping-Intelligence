"""``--from-results``: rebuild each stage's summary tables and figures from saved results.

Every builder reads only the curated files under ``<stage>/results/`` (no GPU, no data, no
training) and writes its table copy plus a Markdown rendering into ``<stage>/results/summary/``
and its figure as PNG and PDF into ``<stage>/figures/``. A builder whose source file is missing
is skipped with a warning; a stage without a ``results/`` tree prints one clear "no results
found" line and counts as nothing to do.

Two builder kinds exist:

* ``runs_inventory`` walks ``results/runs/**/RUN_SUMMARY.json`` (the sealed record of every
  run) and tabulates run / status / arm, with a status-overview figure.
* ``csv_table`` re-renders one curated CSV. The thesis-facing figures of the study are plotted
  from columns recognised by name (``mark``/``efe``/``round`` on x, ``value``/``metric``/
  ``p50``/... on y, a low-cardinality text column as the series); when the columns of a file
  are not recognisable the table is still rebuilt and the figure is skipped with a warning that
  lists the columns found. Nothing is ever recomputed: numbers are copied, never derived.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .decide import results_root
from .units import STAGE_ORDER

# Column-name fragments used to recognise plot axes in a curated CSV (matched case-insensitively
# as substrings). They are name hints only: an unrecognised file keeps its table, loses its
# figure.
X_HINTS = ("mark", "efe", "round", "eta", "server_lr", "lr", "seed", "step", "epoch", "k",
           "cells", "band", "users")
Y_HINTS = ("value", "metric", "mrr", "ndcg", "retention", "ratio", "estimate", "delta", "auc",
           "p50", "p95", "latency", "ms", "overhead", "norm", "server_step", "acc", "recall")
GROUP_HINTS = ("arm", "run", "dataset", "device", "model", "variant", "family", "method",
               "series", "cell", "scope", "browser", "precision")


@dataclass(frozen=True)
class Builder:
    """One table/figure builder: a name, its source files and how to build it."""

    name: str                     # base name of the outputs
    sources: tuple[str, ...]      # relative to <stage>/results/ (posix)
    build: Callable[[Builder, Path, Path, Path], list[str]]
    label: str = ""


def _stage_dirs(stage: str) -> tuple[Path, Path, Path]:
    """(results, figures, summary) directories of a stage under the configured results root."""
    root = results_root()
    return (root / stage / "results", root / stage / "figures", root / stage / "results" / "summary")


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _write_outputs(summary_dir: Path, figures_dir: Path, name: str, header: list[str],
                   rows: list[list[object]], md_prefix: str, figure: Callable | None) -> list[str]:
    """Write <name>.csv + <name>.md into summary_dir and, when `figure` builds one, the pair
    <name>.png + <name>.pdf into figures_dir. Returns the messages to print."""
    import pandas as pd

    summary_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=header)
    frame.to_csv(summary_dir / f"{name}.csv", index=False)
    (summary_dir / f"{name}.md").write_text(md_prefix + _markdown_table(header, rows),
                                            encoding="utf-8")
    messages = [f"table {name}: {len(rows)} rows -> {summary_dir / f'{name}.csv'}"]
    if figure is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        figure(frame, figures_dir, name)
        messages += [f"figure {name}: {figures_dir / f'{name}.png'} (+ .pdf)"]
    return messages


def _pick(frame, hints: tuple[str, ...], used: set[str]) -> str | None:
    """The first column whose name contains a hint (case-insensitive), skipping used columns."""
    for hint in hints:
        for column in frame.columns:
            if column not in used and hint in column.lower():
                return column
    return None


def _csv_figure(x_hints: tuple[str, ...], y_hints: tuple[str, ...],
                group_hints: tuple[str, ...]) -> Callable:
    """Build the plotting callable for a curated CSV (see module docstring for the rules)."""

    def draw(frame, figures_dir: Path, name: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        used: set[str] = set()
        y = _pick(frame, y_hints, used)
        if y is None:
            print(f"warning: figure {name}: no plottable value column "
                  f"(columns: {', '.join(map(str, frame.columns))}); table only")
            return
        used.add(y)
        x = _pick(frame, x_hints, used)
        if x is not None:
            used.add(x)
        group = _pick(frame, group_hints, used)
        x_numeric = x is not None and frame[x].dtype.kind in "if"
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        if group is not None and frame[group].nunique() <= 16:
            for key, part in frame.groupby(group, sort=True):
                xs = part[x] if x is not None else part.index
                if x_numeric:
                    ax.plot(xs, part[y], marker="o", linewidth=1.2, label=str(key))
                else:
                    ax.bar(range(len(part)), part[y], label=str(key))
            ax.legend(fontsize=7, title=group)
        else:
            xs = frame[x] if x is not None else frame.index
            if x_numeric:
                ax.plot(xs, frame[y], marker="o", linewidth=1.2)
            else:
                ax.bar(range(len(frame)), frame[y])
        if x is not None:
            ax.set_xlabel(str(x))
            ax.set_ylabel(y)
            if not x_numeric:
                ax.set_xticks(range(len(frame)))
                ax.set_xticklabels([str(v) for v in frame[x]], rotation=45, ha="right",
                                   fontsize=7)
        else:
            ax.set_ylabel(y)
        if y.lower().startswith(("server_step", "norm")) or "norm" in y.lower():
            ax.set_yscale("log")
        ax.set_title(name.replace("_", " "), fontsize=9)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{name}.png", dpi=150)
        fig.savefig(figures_dir / f"{name}.pdf")
        plt.close(fig)

    return draw


def _csv_table(name: str, source: str, x_hints: tuple[str, ...] = X_HINTS,
               y_hints: tuple[str, ...] = Y_HINTS,
               group_hints: tuple[str, ...] = GROUP_HINTS) -> Builder:
    def build(builder: Builder, results_dir: Path, figures_dir: Path, summary_dir: Path
              ) -> list[str]:
        import pandas as pd

        frame = pd.read_csv(results_dir / source)
        header = [str(c) for c in frame.columns]
        rows = frame.astype(object).where(frame.notna(), None).values.tolist()
        prefix = (f"Rebuilt from `{source}` by `run_pipeline.py --from-results` "
                  f"(numbers copied, never recomputed).\n\n")
        return _write_outputs(summary_dir, figures_dir, builder.name, header, rows, prefix,
                              _csv_figure(x_hints, y_hints, group_hints))

    return Builder(name=name, sources=(source,), build=build, label=f"curated table {source}")


def _runs_inventory() -> Builder:
    def build(builder: Builder, results_dir: Path, figures_dir: Path, summary_dir: Path
              ) -> list[str]:
        runs_dir = results_dir / "runs"
        rows = []
        for summary_path in sorted(runs_dir.rglob("RUN_SUMMARY.json")):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                status = summary.get("status", "?") if isinstance(summary, dict) else "?"
            except (OSError, ValueError):
                status = "UNREADABLE"
            rel = summary_path.relative_to(runs_dir)
            parts = rel.parts
            phase = parts[0] if parts and parts[0] in ("main_study", "closure_phase") else ""
            arm = ""
            if isinstance(summary, dict) and isinstance(summary.get("arm"), str):
                arm = summary["arm"]
            rows.append([phase, rel.parent.as_posix(), arm, status])
        header = ["phase", "run", "arm", "status"]

        def figure(frame, figures_dir_: Path, name: str) -> None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            counts = (frame.groupby(["phase", "status"]).size().rename("runs")
                      .reset_index().sort_values(["phase", "status"]))
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            labels = [f"{phase} {status}" for phase, status in zip(counts["phase"],
                                                                   counts["status"])]
            ax.bar(labels, counts["runs"])
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("runs")
            ax.set_title("run status overview", fontsize=9)
            fig.tight_layout()
            fig.savefig(figures_dir_ / f"{name}.png", dpi=150)
            fig.savefig(figures_dir_ / f"{name}.pdf")
            plt.close(fig)

        prefix = ("Every run directory under `results/runs/` with its RUN_SUMMARY status, "
                  "rebuilt by `run_pipeline.py --from-results`.\n\n")
        return _write_outputs(summary_dir, figures_dir, builder.name, header, rows, prefix,
                              figure)

    return Builder(name="runs_inventory", sources=("runs",), build=build,
                   label="run inventory from RUN_SUMMARY.json files")


def _stage_builders(stage: str) -> list[Builder]:
    """The explicit per-stage builder list (see the module docstring for the two kinds)."""
    builders: dict[str, list[Builder]] = {
        "01_data": [_runs_inventory()],
        "02_centralized": [
            _runs_inventory(),
            _csv_table("closure_points_frozen_preseed",
                       "closure_phase/WARM_FROZEN_PRESEED_PARTIAL_01/POINTS.csv"),
            _csv_table("closure_points_full_preseed",
                       "closure_phase/WARM_FULL_PRESEED_INDEPENDENT_03/POINTS.csv"),
        ],
        "03_federated_learning": [
            _runs_inventory(),
            _csv_table("warm_tau_grid_calibration", "exploratory_warm_tau_grid_calibration.csv",
                       y_hints=("value", "metric", "mrr", "calib")),
            _csv_table("update_norm_total_by_round",
                       "closure_phase/update_norm_analysis/TOTAL_BY_ROUND.csv",
                       x_hints=("round",), y_hints=("server_step", "norm")),
        ],
        "04_on_device": [
            _runs_inventory(),
            _csv_table("device_latency", "device_latency.csv",
                       x_hints=("variant", "precision", "threads", "model"),
                       y_hints=("p50", "p95", "ms", "latency")),
        ],
        "05_privacy_and_security": [
            _runs_inventory(),
            _csv_table("privacy_dp_cost", "privacy_dp_cost.csv"),
            _csv_table("privacy_membership_inference", "privacy_membership_inference.csv",
                       y_hints=("auc", "value", "attack", "leakage")),
            _csv_table("secure_aggregation", "secure_aggregation.csv",
                       y_hints=("overhead", "ratio", "value")),
        ],
        "06_evaluation_and_benchmarks": [
            _runs_inventory(),
            _csv_table("confirmatory_main_analysis_arms",
                       "rees46_confirmatory/confirmatory_phase1_arms.csv"),
            _csv_table("confirmatory_main_analysis_contrasts",
                       "rees46_confirmatory/confirmatory_phase1_contrasts.csv"),
            _csv_table("confirmatory_followup_analysis_contrasts",
                       "rees46_confirmatory/confirmatory_phase2_contrasts.csv"),
            _csv_table("benchmarks_track_a", "benchmarks/confirmatory_track_a.csv",
                       x_hints=("dataset",), y_hints=("ndcg", "value", "metric")),
            _csv_table("benchmarks_track_b", "benchmarks/confirmatory_track_b.csv",
                       x_hints=("dataset",), y_hints=("ndcg", "value", "metric")),
        ],
    }
    return builders.get(stage, [])


def run_from_results(stages: Iterable[str], dry_run: bool = False) -> int:
    """Rebuild the tables and figures of `stages` from their saved results.

    Returns 0 even when a stage has no results (that is "nothing to do", not a failure);
    missing source files of a single builder are warnings.
    """
    for stage in stages:
        builders = _stage_builders(stage)
        results_dir, figures_dir, summary_dir = _stage_dirs(stage)
        if not results_dir.is_dir():
            print(f"{stage}: no results found (looked for {results_dir}); nothing to do")
            continue
        print(f"{stage}: rebuilding {len(builders)} table/figure builder(s) from {results_dir}")
        for builder in builders:
            missing = [s for s in builder.sources if not (results_dir / s).exists()]
            if missing:
                print(f"warning: {stage}: skipped {builder.name}: missing "
                      f"{', '.join(missing)}")
                continue
            if dry_run:
                print(f"  would build {builder.name} from {', '.join(builder.sources)}")
                continue
            for message in builder.build(builder, results_dir, figures_dir, summary_dir):
                print(f"  {message}")
    return 0


__all__ = ["Builder", "run_from_results", "STAGE_ORDER"]
