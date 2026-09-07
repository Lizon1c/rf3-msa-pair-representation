#!/usr/bin/env python3
"""Analyze AB-V3 (ESM noise ladder) and AB-V4 (f_enc LR pace) on x_msa_faesm
(RBD full set). Reads the partial-or-complete results.json written by
ab_esm_noise.py / ab_pace_fenc.py and reports, per level:

  - mean/std val Spearman (n per level; warns if < 15)
  - paired Delta vs the baseline arm (nl=0 / lam=1.0) by (split, init)
  - late gradient economy: mean z, f and the f/z ratio (rebalancing readout)

Discriminating tests (pre-registered in the overnight report §4/§5):
  (A) does the best non-baseline level beat its own baseline (fusion alone)?
  (B) does it break the sp_msa ceiling (0.644 overall / 0.589 strict)? If yes,
      FAESM was actively interfering via gradient suppression, not redundant.
  (C) inverted-U: is the argmax level an INTERIOR level (not the endpoint)?

Robust to incomplete runs — reports n and flags partial arms.
"""
import json
import numpy as np
from pathlib import Path

K = Path('/mnt/k/output_heads/rbd')
SP_MSA_OVERALL = 0.644
SP_MSA_STRICT = 0.589
FULL_N = 15  # 5 splits x 3 inits


def load(tag):
    p = K / tag / 'results.json'
    if not p.exists():
        return None
    return json.load(open(p))


def analyze(tag, key, baseline_val, levels):
    d = load(tag)
    print(f"\n===== {tag} (key={key}, baseline {key}={baseline_val}) =====")
    if d is None or not d.get('runs'):
        print("  no results yet")
        return
    runs = d['runs']
    by = {lv: [r for r in runs if abs(r[key] - lv) < 1e-9] for lv in levels}
    base_pairs = {(r['split'], r['init']): r['val']
                  for r in by.get(baseline_val, [])}

    print(f"  {'level':>7} {'n':>3} {'mean':>8} {'std':>8} "
          f"{'paired_d':>9} {'z':>7} {'f':>7} {'f/z':>6}")
    best_lv, best_mean = None, -1
    for lv in levels:
        rs = by[lv]
        n = len(rs)
        if n == 0:
            print(f"  {lv:>7.3f} {0:>3}   (none)")
            continue
        vals = [r['val'] for r in rs]
        m, s = float(np.mean(vals)), float(np.std(vals))
        zs = [r['grad_late']['z'] for r in rs if r.get('grad_late')]
        fs = [r['grad_late']['f'] for r in rs if r.get('grad_late')]
        zm = float(np.mean(zs)) if zs else float('nan')
        fm = float(np.mean(fs)) if fs else float('nan')
        # paired delta vs baseline (only over shared split x init)
        arm_pairs = {(r['split'], r['init']): r['val'] for r in rs}
        ks = sorted(set(arm_pairs) & set(base_pairs))
        if ks and lv != baseline_val:
            deltas = [arm_pairs[k] - base_pairs[k] for k in ks]
            ps = {}
            for k in ks:
                ps.setdefault(k[0], []).append(arm_pairs[k] - base_pairs[k])
            per_split = [float(np.mean(v)) for _, v in sorted(ps.items())]
            pd = f"{np.mean(deltas):+.4f}"
            extra = f"  paired_n={len(ks)} per_split={['%+.3f' % p for p in per_split]}"
        else:
            pd = "  --   "
            extra = ""
        flag = "" if n == FULL_N else f"  [PARTIAL n<{FULL_N}]"
        print(f"  {lv:>7.3f} {n:>3} {m:>8.4f} {s:>8.4f} {pd:>9} "
              f"{zm:>7.3f} {fm:>7.3f} {fm/zm:>6.2f}{flag}{extra}")
        if m > best_mean:
            best_lv, best_mean = lv, m

    # verdicts
    base_mean = float(np.mean([r['val'] for r in by.get(baseline_val, [])])) \
        if by.get(baseline_val) else float('nan')
    interior = levels[1:-1]
    interior_best = best_lv in interior
    print(f"  argmax level = {best_lv} (mean {best_mean:.4f}); "
          f"baseline mean = {base_mean:.4f}")
    print(f"  (A) beats baseline: "
          f"{'YES' if best_mean > base_mean and best_lv != baseline_val else 'no'} "
          f"({best_mean - base_mean:+.4f})")
    print(f"  (B) breaks sp_msa ceiling: "
          f"overall(0.644)={'YES' if best_mean > SP_MSA_OVERALL else 'no'} / "
          f"strict(0.589)={'YES' if best_mean > SP_MSA_STRICT else 'no'}")
    print(f"  (C) inverted-U (interior argmax): {'YES' if interior_best else 'no'}")
    if any(len(by[lv]) < FULL_N for lv in levels):
        print("  NOTE: partial data — verdicts provisional until n=15/level.")


if __name__ == '__main__':
    analyze('ab_esm_noise', 'nl', 0.0, [0.0, 0.25, 0.5, 0.75, 1.0])
    analyze('ab_pace_fenc', 'lam', 1.0, [1.0, 0.25, 0.125, 0.06])
