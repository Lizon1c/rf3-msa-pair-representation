#!/usr/bin/env python3
"""Analyze AB-V2 full-set fusion-cell strict runs against the archived v3
(overall) run. Pairing: identical (split, init) grid (seeds 100-104, inits
7/107/207), so strict-vs-overall differences are within-run paired.

Answers: do the dead-stream-corrected triad marginals move under the correct
preprocessing boundary? (Subset evidence: fusion cells ±0.005 — unaffected.)

Run: python analyze_strict_fusion.py
"""
import json
import numpy as np
from pathlib import Path

RBD = Path('/mnt/k/output_heads/rbd')
strict = json.load(open(RBD / 'ab_std_split/results_full_fusion.json'))['runs']
v3m = json.load(open(RBD / 'e2_matrix_complete/results_v3.json'))['runs']
v3f = json.load(open(RBD / 'e2_four_condition/results_v3.json'))['runs']
overall = v3m + v3f

CELLS = ['x_msa_faesm', 'x_msa_zeros', 'x_no_faesm', 'x_no_zeros', 'tri', 'tri_z']


def grid(runs, cond):
    return {(r['split'], r['init']): r['val'] for r in runs if r['cond'] == cond}


print('=== per-cell: strict vs overall (paired by split×init) ===')
means = {}
for c in CELLS:
    o, s = grid(overall, c), grid(strict, c)
    ks = sorted(set(o) & set(s))
    if not ks:
        print(f'{c:<13} (missing: overall={len(o)} strict={len(s)})')
        continue
    ds = [s[k] - o[k] for k in ks]
    bysplit = {}
    for k in ks:
        bysplit.setdefault(k[0], []).append(s[k] - o[k])
    ps = [np.mean(v) for _, v in sorted(bysplit.items())]
    means[c] = {'overall': np.mean([o[k] for k in ks]), 'strict': np.mean([s[k] for k in ks]),
                'delta': np.mean(ds)}
    print(f'{c:<13} overall={means[c]["overall"]:.4f} strict={means[c]["strict"]:.4f} '
          f'Δ={np.mean(ds):+.4f} ({sum(d < 0 for d in ds)}/{len(ds)} neg) '
          f'per-split={[round(x, 3) for x in ps]}')

print('\n=== dead-stream-corrected marginals under each convention ===')


def marginal(a, b, table):
    if a not in table or b not in table:
        return None
    return table[a] - table[b]


def fmt(x):
    return f'{x:+.4f}' if x is not None else '(pending)'


for conv in ['overall', 'strict']:
    t = {c: means[c][conv] for c in means}
    print(f'[{conv}]')
    print(f'  FAESM over Z_msa        (x_msa_faesm − x_msa_zeros) = '
          f'{fmt(marginal("x_msa_faesm", "x_msa_zeros", t))}')
    print(f'  FAESM over Z_no         (x_no_faesm − x_no_zeros)   = '
          f'{fmt(marginal("x_no_faesm", "x_no_zeros", t))}')
    print(f'  Z_msa over Z_no         (x_msa_zeros − x_no_zeros)  = '
          f'{fmt(marginal("x_msa_zeros", "x_no_zeros", t))}')
    print(f'  Z_no over (Z_msa+FAESM) (tri − tri_z)               = '
          f'{fmt(marginal("tri", "tri_z", t))}')

print('\n=== marginal deltas (strict − overall), paired at the run level ===')
for name, a, b in [('FAESM over Z_msa', 'x_msa_faesm', 'x_msa_zeros'),
                   ('FAESM over Z_no', 'x_no_faesm', 'x_no_zeros'),
                   ('Z_msa over Z_no', 'x_msa_zeros', 'x_no_zeros'),
                   ('Z_no over (Z_msa+FAESM)', 'tri', 'tri_z')]:
    oa, ob, sa, sb = grid(overall, a), grid(overall, b), grid(strict, a), grid(strict, b)
    ks = sorted(set(oa) & set(ob) & set(sa) & set(sb))
    if not ks:
        print(f'{name:<26} (missing)')
        continue
    m_over = np.mean([oa[k] - ob[k] for k in ks])
    m_strict = np.mean([sa[k] - sb[k] for k in ks])
    dd = [(sa[k] - sb[k]) - (oa[k] - ob[k]) for k in ks]
    print(f'{name:<26} overall={m_over:+.4f} strict={m_strict:+.4f} '
          f'strict−overall={np.mean(dd):+.4f} ({sum(d < 0 for d in dd)}/{len(dd)} neg)')

print('\nDecision: if all strict−overall marginal deltas are within ±0.02,')
print('the triad conclusions (H1/H2) are convention-robust and the archived')
print('overall numbers stand; only single-modality absolutes need strict values.')
