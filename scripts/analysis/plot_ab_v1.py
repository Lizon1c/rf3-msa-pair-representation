#!/usr/bin/env python3
"""Plot AB-V1 epoch trajectories. Panels = cells, thin lines = individual runs,
thick = mean. The full-set figure merges results_full.json (sp_no/sp_msa/
x_msa_faesm) with results_full_esm.json (sp_faesm/x_no_faesm) into 5 panels.
Run: python plot_ab_v1.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/k/output_heads/rbd/ab_bestepoch')
FIG = Path(__file__).parent.parent.parent / 'figures'
FIG.mkdir(exist_ok=True)
EPOCHS = 300


def plot_runs(runs, cells, out_name, with_train=False, suptitle=''):
    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    x = np.arange(EPOCHS)
    for ax, c in zip(axes, cells):
        rs = [r for r in runs if r['cond'] == c]
        if not rs:
            ax.set_title(f'{c}\n(no runs)')
            continue
        v1 = np.array([r['traj1'] for r in rs])
        v2 = np.array([r['traj2'] for r in rs])
        for row in v1:
            ax.plot(x, row, color='#4c72b0', lw=0.5, alpha=0.25)
        for row in v2:
            ax.plot(x, row, color='#dd8452', lw=0.5, alpha=0.25)
        ax.plot(x, v1.mean(0), color='#4c72b0', lw=2.2, label='val1 (selection set)')
        ax.plot(x, v2.mean(0), color='#dd8452', lw=2.2, label='val2 (held-out)')
        if with_train and 'traj_tr' in rs[0]:
            tr = np.array([r['traj_tr'] for r in rs])
            for row in tr:
                ax.plot(x, row, color='#55a868', lw=0.4, alpha=0.15)
            ax.plot(x, tr.mean(0), color='#55a868', lw=2.2, label='train')
        b1 = int(np.argmax(v1.mean(0)))
        ax.scatter([b1], [v1.mean(0)[b1]], color='#4c72b0', zorder=5, s=30)
        sel = np.mean([r['best_v1'] - r['v2_at_best_v1'] for r in rs])
        late = float(np.mean(v2.mean(0)[250:] - v1.mean(0)[250:]))
        tr_r = np.mean([r['traj_corr'] for r in rs])
        ax.set_title(f'{c}\nsel_opt={sel:+.3f}  late_gap(v2−v1)={late:+.3f}  traj_r={tr_r:.2f}',
                     fontsize=10)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('Spearman ρ')
    axes[0].set_ylim(-0.1, 1.05)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / out_name, dpi=150, bbox_inches='tight')
    print(f'saved {out_name}')


if __name__ == '__main__':
    p1 = ROOT / 'results_main.json'
    if p1.exists():
        plot_runs(json.load(open(p1))['runs'], ['sp_no', 'sp_msa', 'x_msa_faesm'],
                  'ab_v1_val12.png', with_train=False,
                  suptitle='AB-V1: per-epoch ρ trajectories, val1 vs val2 (RBD 500-subset, '
                           '60/20/20 cross-position; thin=run, thick=mean; dot=mean-curve best epoch)')
    p3 = ROOT / 'results_full.json'
    p4 = ROOT / 'results_full_esm.json'
    if p3.exists():
        runs = json.load(open(p3))['runs']
        cells = ['sp_no', 'sp_msa', 'x_msa_faesm']
        if p4.exists():
            runs = runs + json.load(open(p4))['runs']
            cells = ['sp_no', 'sp_msa', 'sp_faesm', 'x_no_faesm', 'x_msa_faesm']
        plot_runs(runs, cells, 'ab_v1_full_train_val12.png', with_train=True,
                  suptitle='AB-V1 RBD full set (N=3998, 60/20/20 cross-position, 15 runs/cell): '
                           'train green, val1 blue, val2 orange; thin=run, thick=mean. '
                           'sp_msa is the only cell with a persistent val1>val2 offset (−0.06, locked by ep5); '
                           'fusion late gaps ≈0 and track FAESM\'s difficulty ordering (r=0.90), not Z_msa\'s (r=0.37).')
    # CASP3 replication
    pc = Path('/mnt/k/output_heads/casp3/ab_bestepoch/results_main.json')
    if pc.exists():
        plot_runs(json.load(open(pc))['runs'],
                  ['sp_no', 'sp_msa', 'sp_faesm', 'x_no_faesm', 'x_msa_faesm'],
                  'ab_v1_casp3_train_val12.png', with_train=True,
                  suptitle='AB-V1 CASP3 replication (N=1567, L=488, 69-row MSA, 60/20/20 cross-position, '
                           '15 runs/cell): train green, val1 blue, val2 orange; thin=run, thick=mean.')
