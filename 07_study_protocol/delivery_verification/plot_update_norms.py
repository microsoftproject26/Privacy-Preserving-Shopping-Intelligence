"""Generate standalone scientific P5 figures from the descriptive CSV outputs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    main_runs, wa = defaultdict(list), defaultdict(list)
    with (args.directory / 'TOTAL_BY_ROUND.csv').open() as f:
        for row in csv.DictReader(f):
            if row['server_step'] != 'True' or row['finite'] != 'True':
                continue
            if row['run'] in ['P2_WSTAR_SMALL_s2026', 'P2_WSTAR_SMALL_s2027']:
                main_runs[row['run']].append(row)
            elif row['scope'] == 'WA' and 'FEDADAM' in row['run']:
                wa[row['run']].append(row)
    if len(main_runs) != 2:
        raise ValueError('both W* MAIN seeds are required for the main figures')
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for name, rows in sorted(main_runs.items()):
        rounds = [int(r['round']) for r in rows]
        for ax, metric in zip(axes.flat, ['agg_delta_rms', 'step_rms', 'pre_dist_l2', 'rel_pre_dist']):
            ys = [float(r[metric]) for r in rows]
            ax.plot(rounds, ys, linewidth=.7, label=name.rsplit('_', 1)[-1])
            ax.set_xlabel('Round (0-based)')
            ax.set_ylabel({'agg_delta_rms': 'Aggregate delta RMS', 'step_rms': 'Server step RMS',
                           'pre_dist_l2': 'Distance from initial model (L2)', 'rel_pre_dist': 'Relative drift on trained parameters'}[metric])
            ax.legend(loc='best')
            ax.grid(alpha=.2)
    axes[0, 0].axhline(1e-3, color='black', linestyle='--', linewidth=1, label='Reference tau = 1e-3')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 1].set_yscale('log')
    fig.suptitle('DESCRIPTIVE: existing W* MAIN trajectories (trained subset only)')
    for suffix in ['svg', 'png', 'pdf']:
        fig.savefig(args.directory / ('WSTAR_TRAJECTORIES.' + suffix), dpi=180)
    plt.close(fig)
    if wa:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for name, rows in sorted(wa.items()):
            label = name.replace('P2_W_FEDADAM_', '').replace('_CALIB_s2026', '')
            xs = [int(r['round']) for r in rows]
            for ax, metric in zip(axes, ['step_rms', 'rel_pre_dist']):
                ax.plot(xs, [float(r[metric]) for r in rows], linewidth=.7, label=label)
        axes[0].set_ylabel('Server step RMS')
        axes[0].set_yscale('log')
        axes[1].set_ylabel('Relative drift on trained parameters')
        for ax in axes:
            ax.set_xlabel('Round (0-based)')
            ax.grid(alpha=.2)
        axes[1].legend(fontsize=7)
        fig.suptitle('DESCRIPTIVE: existing FedAdam CALIB WA trajectories')
        for suffix in ['svg', 'png', 'pdf']:
            fig.savefig(args.directory / ('WA_FEDADAM_TRAJECTORIES.' + suffix), dpi=180)
        plt.close(fig)
    print('Created W* and CALIB figures from existing norm CSVs.')


if __name__ == '__main__':
    main()
