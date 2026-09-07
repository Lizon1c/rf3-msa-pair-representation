#!/usr/bin/env python3
"""E0: MSA pinning geometry — additive-constant vs multiplicative-suppression test.

Uses EXISTING RBD Z_II embeddings (no-MSA vs WH1-MSA, N=3998, row part [201,128]).
No new GPU extraction. CPU only.

Analyses:
  A. Replication of report table (centered PCA on flattened [N, 201*128]):
     PC1 ratio, eff-rank 90/95, per-position cosine, per-position CKA, global CKA.
  B. Mean-offset decomposition: raw-mean energy fraction, cos(mean_no, mean_msa),
     cos(PC1_centered, raw_mean) — how much of the collapse is a shared constant
     vs. structure in the deviations. (Centered PCA already removes any additive
     constant; B quantifies the additive part separately.)
  C. Delta (=PCA-centered feature) geometry: per-mutant norm ratio ||Δmsa||/||Δno||
     (suppression factor) and cos(Δmsa, Δno) (direction preservation);
     per-position suppression profile s_p and direction profile; top-PC alignment
     between conditions; Spearman of top-PC scores with bind_avg.
  D. Per-position suppression vs MSA column conservation (GOLD_MSA a3m).

Output: /mnt/k/output_heads/rbd/e0_msa_geometry/results.json + stdout summary.
"""
import json, math, time
import numpy as np
import torch
from pathlib import Path
from scipy import stats

torch.set_num_threads(32)
t0 = time.time()

RBD = Path('/mnt/k/output_heads/rbd')
TMP = Path('/tmp/e0')
OUT = RBD / 'e0_msa_geometry'
OUT.mkdir(exist_ok=True)
A3M = Path('/mnt/j/conda_envs/foundry/DMS_Project/inputs/GOLD_MSA/rbd_SARS_CoV_2_WH1.a3m')
L, D = 201, 128
K_TOP = 8


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


# ═══════ Load ═══════
meta = [json.loads(l) for l in open(RBD / 'metadata.jsonl') if l.strip()]
meta_sorted = sorted(meta, key=lambda m: m['mutant_id'])
mmap = {m['mutant_id']: i for i, m in enumerate(meta_sorted)}
mids = [m['mutant_id'] for m in meta]
bind = np.array([m['bind_avg'] for m in meta], dtype=np.float64)
site = np.array([m['site_rbd'] for m in meta], dtype=np.int64)
N = len(mids)
log(f'N={N} mutants')


def load(which):
    X = np.empty((N, L, D), np.float32)
    for i, mid in enumerate(mids):
        if which == 'no':
            z = torch.load(TMP / 'zii' / f'{mid}_zii.pt', map_location='cpu')
            if isinstance(z, dict):
                z = list(z.values())[0]
        else:
            z = torch.load(TMP / 'zii_msa' / f'mutant_{mmap[mid]:04d}_zii.pt', map_location='cpu')
            if isinstance(z, dict):
                z = list(z.values())[0]
        X[i] = z[0].float().numpy()
        if i % 1000 == 999:
            log(f'  {which}: {i+1}/{N}')
    return torch.from_numpy(X)


Xno = load('no')
log(f'Xno {tuple(Xno.shape)}')
Xmsa = load('msa')
log(f'Xmsa {tuple(Xmsa.shape)}')

res = {'N': N, 'L': L, 'D': D}


# ═══════ Helpers ═══════
def flat(X):
    return X.reshape(N, -1)  # [N, L*D]


def spectrum(Xc):
    """Xc: centered [N, F]. Returns evals (desc), ratio, top-k feature dirs [k, F]."""
    n = Xc.shape[0]
    G = (Xc @ Xc.T) / n
    evals, evecs = torch.linalg.eigh(G)
    evals = evals.flip(0).clamp_min(0)
    evecs = evecs.flip(1)
    ratio = evals / evals.sum().clamp_min(1e-12)
    Vt = evecs[:, :K_TOP].T @ Xc  # [k, F]
    Vt = Vt / Vt.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return evals, ratio, Vt


def effrank(ratio, thr):
    c = torch.cumsum(ratio, 0)
    return int((c < thr).sum().item()) + 1


def lin_cka(Xc, Yc):
    """Linear CKA via sample grams (feature-centered inputs)."""
    Kx = Xc @ Xc.T
    Ky = Yc @ Yc.T
    num = (Kx * Ky).sum()
    den = Kx.norm() * Ky.norm()
    return (num / den.clamp_min(1e-20)).item()


def cos(a, b, dim=-1):
    return (torch.nn.functional.cosine_similarity(a, b, dim=dim))


# ═══════ A. Raw means & replication ═══════
fno, fmsa = flat(Xno), flat(Xmsa)
m_no, m_msa = fno.mean(0), fmsa.mean(0)
e_no = (fno ** 2).sum(1).mean()
e_msa = (fmsa ** 2).sum(1).mean()
res['mean'] = {
    'energy_frac_no': float((m_no ** 2).sum() / e_no),
    'energy_frac_msa': float((m_msa ** 2).sum() / e_msa),
    'cos_mean_no_msa': float(cos(m_no, m_msa)),
    'rms_no': float(e_no.sqrt()),
    'rms_msa': float(e_msa.sqrt()),
}
log(f"mean energy frac: no={res['mean']['energy_frac_no']:.4f} msa={res['mean']['energy_frac_msa']:.4f}; "
    f"cos(mean_no,mean_msa)={res['mean']['cos_mean_no_msa']:.4f}")

# Centered (== delta from cross-mutant mean)
Dno = fno - m_no
Dmsa = fmsa - m_msa
log('computing spectra (Gram + eigh)...')
ev_no, ra_no, V_no = spectrum(Dno)
log('  no done')
ev_msa, ra_msa, V_msa = spectrum(Dmsa)
log('  msa done')

res['pca'] = {
    'pc1_no': float(ra_no[0]), 'pc1_msa': float(ra_msa[0]),
    'eff90_no': effrank(ra_no, 0.90), 'eff90_msa': effrank(ra_msa, 0.90),
    'eff95_no': effrank(ra_no, 0.95), 'eff95_msa': effrank(ra_msa, 0.95),
    'top64_ratio_no': [float(x) for x in ra_no[:64]],
    'top64_ratio_msa': [float(x) for x in ra_msa[:64]],
}
log(f"PC1: no={res['pca']['pc1_no']:.4f} msa={res['pca']['pc1_msa']:.4f}; "
    f"eff90: {res['pca']['eff90_no']} vs {res['pca']['eff90_msa']}")

# cos(PC1, raw mean): is the collapsed axis the pinning/mean axis?
res['pc1_vs_mean'] = {
    'cos_pc1no_meanno': float(abs(cos(V_no[0], m_no))),
    'cos_pc1msa_meamsa': float(abs(cos(V_msa[0], m_msa))),
    'cos_pc1msa_meanno': float(abs(cos(V_msa[0], m_no))),
    'cos_pc1_no_msa': float(abs(cos(V_no[0], V_msa[0]))),
    'cos_topk_no_msa': [float(abs(cos(V_no[j], V_msa[j]))) for j in range(K_TOP)],
}
log(f"cos(pc1_msa, mean_msa)={res['pc1_vs_mean']['cos_pc1msa_meamsa']:.4f}; "
    f"cos(pc1_no, pc1_msa)={res['pc1_vs_mean']['cos_pc1_no_msa']:.4f}")

# ═══════ CKA (global + per-position) ═══════
res['cka'] = {'global_centered': lin_cka(Dno, Dmsa)}
log(f"global CKA(centered)={res['cka']['global_centered']:.4f}")

pp_cka, pp_cos_raw, pp_cos_delta = [], [], []
Dno3 = Dno.view(N, L, D)
Dmsa3 = Dmsa.view(N, L, D)
Xno3c = Xno - Xno.mean(0, keepdim=True)
Xmsa3c = Xmsa - Xmsa.mean(0, keepdim=True)
for p in range(L):
    a = Dno3[:, p, :]
    b = Dmsa3[:, p, :]
    pp_cka.append(lin_cka(a, b))
    pp_cos_raw.append(float(cos(Xno[:, p, :], Xmsa[:, p, :]).mean()))
    pp_cos_delta.append(float(cos(a, b).mean()))
res['per_position'] = {
    'cka_mean': float(np.mean(pp_cka)),
    'cos_raw_mean': float(np.mean(pp_cos_raw)),
    'cos_delta_mean': float(np.mean(pp_cos_delta)),
    'cka': pp_cka, 'cos_raw': pp_cos_raw, 'cos_delta': pp_cos_delta,
}
log(f"per-pos: cos_raw={res['per_position']['cos_raw_mean']:.4f} "
    f"cos_delta={res['per_position']['cos_delta_mean']:.4f} cka={res['per_position']['cka_mean']:.4f}")

# ═══════ C. Delta suppression & direction ═══════
n_no = Dno.norm(dim=1)
n_msa = Dmsa.norm(dim=1)
r = (n_msa / n_no.clamp_min(1e-12)).numpy()
c = cos(Dmsa, Dno).numpy()
res['delta'] = {
    'norm_ratio_mean': float(r.mean()), 'norm_ratio_median': float(np.median(r)),
    'norm_ratio_deciles': [float(x) for x in np.percentile(r, np.arange(0, 101, 10))],
    'cos_mean': float(c.mean()), 'cos_median': float(np.median(c)),
    'cos_deciles': [float(x) for x in np.percentile(c, np.arange(0, 101, 10))],
    'cos_frac_above_0.5': float((c > 0.5).mean()),
    'cos_frac_above_0.7': float((c > 0.7).mean()),
    'norm_ratio_per_mutant': r.tolist(), 'cos_per_mutant': c.tolist(),
}
log(f"delta: ||Δmsa||/||Δno|| median={res['delta']['norm_ratio_median']:.4f}; "
    f"cos(Δmsa,Δno) median={res['delta']['cos_median']:.4f}")

# per-position suppression profile
sp = (Dmsa3.norm(dim=2).mean(0) / Dno3.norm(dim=2).mean(0).clamp_min(1e-12)).numpy()
res['per_position']['suppression'] = sp.tolist()
res['per_position']['suppression_deciles'] = [float(x) for x in np.percentile(sp, np.arange(0, 101, 10))]

# top-PC fitness link
for tag, Dc, V in [('no', Dno, V_no), ('msa', Dmsa, V_msa)]:
    S = (Dc @ V.T).numpy()  # [N, k]
    res[f'pc_fitness_{tag}'] = [float(stats.spearmanr(S[:, j], bind)[0]) for j in range(K_TOP)]
log(f"Spearman(PCj, bind) no: {['%.3f' % x for x in res['pc_fitness_no'][:5]]}")
log(f"Spearman(PCj, bind) msa: {['%.3f' % x for x in res['pc_fitness_msa'][:5]]}")

# ═══════ D. Suppression vs MSA column conservation ═══════
try:
    seqs = []
    cur = []
    for line in open(A3M):
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if cur:
                seqs.append(''.join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        seqs.append(''.join(cur))
    # match alignment columns to the 201 RBD sites via the query (first) row:
    # keep columns where the query has a residue (not '-')
    q = seqs[0]
    qcols = [i for i, ch in enumerate(q) if ch != '-']
    assert len(qcols) == L, f'query nongap cols {len(qcols)} != L={L}'
    arr = np.array([[s[col] for col in qcols] for s in seqs if len(s) == len(q)])
    depth = arr.shape[0]
    cons = np.zeros(L)
    for j in range(L):
        vals, cnts = np.unique(arr[:, j], return_counts=True)
        cons[j] = cnts.max() / cnts.sum()
    res['msa'] = {'n_rows_used': int(depth), 'aln_cols': len(q),
                  'conservation_mean': float(cons.mean())}
    valid = ~np.isnan(sp)
    res['msa']['spearman_suppression_vs_conservation'] = float(
        stats.spearmanr(sp[valid], cons[valid])[0])
    res['msa']['conservation_per_position'] = cons.tolist()
    log(f"MSA rows={depth}; Spearman(suppression, conservation)="
        f"{res['msa']['spearman_suppression_vs_conservation']:.4f}")
except Exception as e:
    res['msa'] = {'error': repr(e)}
    log(f'MSA conservation step failed: {e!r}')

# ═══════ Save ═══════
with open(OUT / 'results.json', 'w') as f:
    json.dump(res, f)
log(f'saved {OUT}/results.json')

# ═══════ Summary ═══════
print('\n════════ E0 SUMMARY ════════')
print(f"A. replication: PC1 {res['pca']['pc1_no']:.3f}→{res['pca']['pc1_msa']:.3f} | "
      f"eff90 {res['pca']['eff90_no']}→{res['pca']['eff90_msa']} | "
      f"CKA {res['cka']['global_centered']:.3f} | "
      f"perpos cos_raw {res['per_position']['cos_raw_mean']:.3f} cka {res['per_position']['cka_mean']:.3f}")
print(f"B. mean: energy frac no {res['mean']['energy_frac_no']:.3f} / msa {res['mean']['energy_frac_msa']:.3f} | "
      f"cos(means) {res['mean']['cos_mean_no_msa']:.3f} | cos(pc1_msa, mean_msa) {res['pc1_vs_mean']['cos_pc1msa_meamsa']:.3f}")
print(f"C. delta: norm ratio med {res['delta']['norm_ratio_median']:.3f} (deciles {['%.2f' % x for x in res['delta']['norm_ratio_deciles'][::2]]}) | "
      f"cos med {res['delta']['cos_median']:.3f} (>0.5: {res['delta']['cos_frac_above_0.5']:.2f}, >0.7: {res['delta']['cos_frac_above_0.7']:.2f})")
print(f"D. suppression-vs-conservation Spearman: {res['msa'].get('spearman_suppression_vs_conservation')}")
