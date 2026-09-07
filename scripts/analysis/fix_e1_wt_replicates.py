#!/usr/bin/env python3
"""Fix E1 arm outputs: WT-identical manifest rows (201 of them, one per site) have no
mutation site, so the extractor saved full [201,201,128] blocks. Convert those to the
standard [2,201,128] row+col format sliced at position 0 — matching the old pipeline's
_find_mut_pos default. Idempotent; skips WT_zii.pt (intentionally full block) and any
file already in [2,201,128]. Safe to re-run after each arm completes.

Run: python -B fix_e1_wt_replicates.py [arm ...]   (default: all 8 arms)
"""
import sys
import torch
from pathlib import Path

base = Path('/mnt/k/output_heads/rbd')
arms = sys.argv[1:] or ['full15', 'colshuffle', 'consensus', 'depth01',
                        'depth04', 'depth08', 'rowshuffle', 'random']
for arm in arms:
    d = base / f'zii_e1_{arm}'
    if not d.exists():
        continue
    fixed, weird = 0, 0
    for p in sorted(d.glob('*_zii.pt')):
        if p.stem == 'WT_zii':
            continue
        try:
            z = torch.load(p, map_location='cpu', weights_only=True)
            if isinstance(z, dict):
                z = list(z.values())[0]
            if tuple(z.shape) == (201, 201, 128):
                rowcol = torch.stack([z[0, :, :], z[:, 0, :]]).half()
                torch.save(rowcol, p)
                fixed += 1
            elif tuple(z.shape) != (2, 201, 128):
                print(f'  UNEXPECTED {tuple(z.shape)}: {p}')
                weird += 1
        except Exception as e:
            print(f'  LOAD FAIL {p}: {e}')
            weird += 1
    print(f'{arm}: fixed={fixed} problems={weird}')
print('done')
