#!/usr/bin/env python3
"""Build E1 MSA content-control arms from the GOLD_MSA (rbd_SARS_CoV_2_WH1.a3m).

Arms (row 0 = WT RBD query kept verbatim in every arm; the extraction pipeline
overrides row 0 with each mutant's sequence at inference time — msa.py:643):
  depth01    query only (pipeline-level noMSA-equivalent control)
  depth04    query + 3 homologs (seed 0)
  depth08    query + 7 homologs
  full15     query + all 14 homologs (= original GOLD_MSA content)
  colshuffle full15, each column independently permuted among homolog rows
             (keeps per-column composition/conservation, destroys covariation & row coherence)
  consensus  homolog rows all replaced by per-column majority consensus of the 14 homologs
             (maximal conservation, zero diversity, zero covariation)
  rowshuffle each homolog row internally shuffled
             (destroys per-position conservation too; keeps AA composition & depth)
  random     homolog rows = uniform random 20-AA strings (destroys everything, keeps depth)

Discriminating logic (E1):
  - colshuffle collapses Z_II like full15 -> collapse carried by profile/conservation
    (retrieval-key / template channel), NOT by pairwise covariation.
  - only full15 collapses -> covariation is the carrier (coevolution account).
  - consensus/rowshuffle/random calibrate the "any extra rows" baseline.
Output: inputs/e1_msa_arms/*.a3m + arms_manifest.json
"""
import json
import numpy as np
from pathlib import Path

SRC = Path('/mnt/j/conda_envs/foundry/DMS_Project/inputs/GOLD_MSA/rbd_SARS_CoV_2_WH1.a3m')
OUT = Path('/mnt/j/conda_envs/foundry/DMS_Project/inputs/e1_msa_arms')
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(0)
AA = list('ACDEFGHIKLMNPQRSTVWY')

hdrs, seqs = [], []
h = None
for line in open(SRC):
    line = line.strip()
    if not line:
        continue
    if line.startswith('>'):
        hdrs.append(line)
        seqs.append('')
    else:
        seqs[-1] += line
# keep all rows matching the query length (gaps allowed — they are alignment content);
# rows with insert characters (len != query len) cannot be column-aligned char-wise: drop them
L = len(seqs[0])
assert '-' not in seqs[0], 'query row must be gap-free'
full = [(h, s) for h, s in zip(hdrs, seqs) if len(s) == L]
query_h, query_s = full[0]
hom_h = [h for h, _ in full[1:]]
hom_s = [s for _, s in full[1:]]
M = len(hom_s)
n_gap = sum('-' in s for s in hom_s)
print(f'query: {query_h} L={L}; homologs kept: {M} (of {len(seqs)-1}, {n_gap} contain gaps)')

arms = {}


def write(name, rows):
    """rows: list of (header, seq); query always first."""
    with open(OUT / f'{name}.a3m', 'w') as f:
        f.write(query_h + '\n' + query_s + '\n')
        for hh, ss in rows:
            f.write(hh + '\n' + ss + '\n')
    arms[name] = {'n_homolog_rows': len(rows), 'L': L}


write('depth01', [])
idx = rng.permutation(M)
write('depth04', [(hom_h[i], hom_s[i]) for i in sorted(idx[:3])])
write('depth08', [(hom_h[i], hom_s[i]) for i in sorted(idx[:7])])
write('full15', list(zip(hom_h, hom_s)))

# colshuffle: independent permutation per column among homolog rows
arr = np.array([list(s) for s in hom_s])
cs = arr.copy()
for j in range(L):
    cs[:, j] = arr[rng.permutation(M), j]
write('colshuffle', [(hom_h[i], ''.join(cs[i])) for i in range(M)])

# consensus: all homolog rows = per-column majority (ties -> rng pick among maxima)
cons = []
for j in range(L):
    vals, cnts = np.unique(arr[:, j], return_counts=True)
    top = vals[cnts == cnts.max()]
    cons.append(rng.choice(top))
cons = ''.join(cons)
write('consensus', [(hom_h[i], cons) for i in range(M)])

# rowshuffle: shuffle within each homolog row
rs = [''.join(rng.permutation(list(s))) for s in hom_s]
write('rowshuffle', [(hom_h[i], rs[i]) for i in range(M)])

# random: uniform 20-AA
rd = [''.join(rng.choice(AA, size=L)) for _ in range(M)]
write('random', [(hom_h[i], rd[i]) for i in range(M)])

with open(OUT / 'arms_manifest.json', 'w') as f:
    json.dump({'source': str(SRC), 'arms': arms,
               'note': 'row0=WT query verbatim; pipeline replaces row0 per mutant (msa.py:643)'},
              f, indent=2)
print('wrote', len(arms), 'arms to', OUT)
for k, v in arms.items():
    print(f"  {k}: {v['n_homolog_rows']} homolog rows")
