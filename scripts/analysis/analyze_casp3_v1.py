#!/usr/bin/env python3
"""Analyze the CASP3 V1 replication (ab_bestepoch_casp3.py) against the
pre-registered RBD findings.

Pre-registration (from RBD V1):
  (1) sp_msa shows a persistent val1>val2 late gap, locked by ~ep5, 5/5 splits;
      if pinning-driven, CASP3's should be LARGER than RBD's −0.061 (CASP3
      pinning −42% vs RBD −26%).
  (2) sp_faesm late gap ≈ 0 with high trajectory agreement (RBD: −0.010, r=0.69).
  (3) fusion (x_msa_faesm/x_no_faesm) late gap ≈ 0, and per-split late-gap
      ordering tracks sp_faesm (RBD r=0.90/0.76), not sp_msa (RBD r=0.37).

Run: python analyze_casp3_v1.py   (works on partial data)
"""
import json
import numpy as np
from pathlib import Path

P = Path('/mnt/k/output_heads/casp3/ab_bestepoch/results_main.json')
runs = json.load(open(P))['runs']
CELLS = ['sp_no', 'sp_msa', 'sp_faesm', 'x_no_faesm', 'x_msa_faesm']
RBD_REF = {  # RBD full-set reference values
    'late_gap': {'sp_no': +0.021, 'sp_msa': -0.061, 'sp_faesm': -0.010,
                 'x_no_faesm': -0.006, 'x_msa_faesm': -0.006},
    'traj_r': {'sp_no': 0.117, 'sp_msa': 0.209, 'sp_faesm': 0.687,
               'x_no_faesm': 0.336, 'x_msa_faesm': 0.290},
}

print(f'CASP3 V1: {len(runs)} runs;',
      {c: sum(r['cond'] == c for r in runs) for c in CELLS})

print('\n=== per-cell summary (CASP3 vs RBD reference) ===')
late = {}
for c in CELLS:
    rs = [r for r in runs if r['cond'] == c]
    if not rs:
        print(f'{c:<12} (no runs yet)')
        continue
    m = lambda k: np.mean([r[k] for r in rs])
    v1 = np.mean([r['traj1'] for r in rs], 0)
    v2 = np.mean([r['traj2'] for r in rs], 0)
    late[c] = float(np.mean(v2[250:] - v1[250:]))
    ep5 = float(v2[5] - v1[5])
    sel = [r['best_v1'] - r['v2_at_best_v1'] for r in rs]
    bysplit = {}
    for r in rs:
        l1 = np.mean(r['traj1'][250:]); l2 = np.mean(r['traj2'][250:])
        bysplit.setdefault(r['split'], []).append(l2 - l1)
    ps = {s: np.mean(v) for s, v in sorted(bysplit.items())}
    n_neg = sum(1 for v in ps.values() if v < 0)
    print(f'{c:<12} n={len(rs):>2} best_v1={m("best_v1"):.3f} v2@b1={m("v2_at_best_v1"):.3f} '
          f'best_v2={m("best_v2"):.3f} final={m("final_v1"):.3f}/{m("final_v2"):.3f}')
    print(f'{"":12} late_gap={late[c]:+.3f} (ep5 {ep5:+.3f}) [RBD {RBD_REF["late_gap"][c]:+.3f}] '
          f'traj_r={m("traj_corr"):.2f} [RBD {RBD_REF["traj_r"][c]:.2f}] '
          f'sel_opt={np.mean(sel):+.3f} late-gap splits {n_neg}/{len(ps)} neg '
          f'{ {s: round(v, 3) for s, v in ps.items()} }')

if len(late) >= 3:
    print('\n=== per-split late-gap correlation matrix (CASP3) ===')
    per_split = {}
    for c in late:
        rs = [r for r in runs if r['cond'] == c]
        g = {}
        for r in rs:
            g.setdefault(r['split'], []).append(np.mean(r['traj2'][250:]) - np.mean(r['traj1'][250:]))
        per_split[c] = {s: np.mean(v) for s, v in g.items()}
    cs = sorted(late)
    print('             ' + ' '.join(f'{c:>12}' for c in cs))
    for a in cs:
        row = []
        for b in cs:
            sa = sorted(set(per_split[a]) & set(per_split[b]))
            if len(sa) < 3:
                row.append(float('nan'))
            else:
                row.append(np.corrcoef([per_split[a][s] for s in sa],
                                       [per_split[b][s] for s in sa])[0, 1])
        print(f'{a:<12} ' + ' '.join(f'{x:>+12.2f}' if not np.isnan(x) else f'{"(n<3)":>12}' for x in row))
    if 'x_msa_faesm' in late and 'sp_faesm' in late and 'sp_msa' in late:
        print('\nPrediction (3): fusion should track sp_faesm (RBD 0.90/0.76), not sp_msa (RBD 0.37).')

print('\n=== verdict vs pre-registration ===')
if 'sp_msa' in late:
    v = late['sp_msa']
    print(f'(1) sp_msa late gap {v:+.3f} vs RBD −0.061: '
          + ('LARGER/more negative — pinning-driven hypothesis supported'
             if v < -0.061 else
             'smaller — RBD offset not fully replicated; pinning-driven hypothesis weakened'))
if 'sp_faesm' in late:
    rs = [r for r in runs if r['cond'] == 'sp_faesm']
    print(f'(2) sp_faesm late gap {late["sp_faesm"]:+.3f}, traj_r {np.mean([r["traj_corr"] for r in rs]):.2f} '
          f'vs RBD (−0.010, 0.69)')
