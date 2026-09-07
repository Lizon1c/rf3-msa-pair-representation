#!/usr/bin/env python3
"""E1 analysis: MSA dose x content matrix — geometry per arm vs no-MSA reference.

For each arm in zii_e1_<arm>/ (subset mutants, [2,201,128] row+col):
  - PC1 ratio, eff-rank 90/95 (centered PCA, row part, flattened [N, 201*128])
  - global CKA(arm, noMSA) and CKA(arm, original wh1_msa)
  - per-position raw cosine vs noMSA (mean)
  - delta geometry vs noMSA: ||Δ_arm||/||Δ_no|| per mutant (suppression),
    cos(Δ_arm, Δ_no) per mutant (direction preservation)
  - top-64 spectrum per arm (for plotting)

Discriminating readout:
  colshuffle collapses like full15  -> profile/conservation channel (retrieval-key/template)
  colshuffle stays high-rank like depth01 -> pairwise covariation is the carrier (coevolution)

Output: /mnt/k/output_heads/rbd/e1_msa_arms_analysis/results.json + stdout table.
Run: python -B e1_analyze.py   (CPU only; copies arm data from K: to /tmp first)
"""
import json, os, time
import numpy as np
import torch
from pathlib import Path

torch.set_num_threads(32)
t0 = time.time()

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'e1_msa_arms_analysis'
OUT.mkdir(exist_ok=True)
TMP_NO = Path('/tmp/e0/zii')          # all 3998 noMSA files (already local)
TMP_MSA = Path('/tmp/e0/zii_msa')     # all 3998 original-MSA files (already local)
ARMS = os.environ.get('E1_ANALYZE_ARMS', '').split(',') if os.environ.get('E1_ANALYZE_ARMS') else \
    [f'zii_e1_{a}' for a in ["full15", "colshuffle", "consensus", "depth01",
                             "depth04", "depth08", "rowshuffle", "random"]]
TAG = os.environ.get('E1_ANALYZE_TAG', 'results')
L, D = 201, 128


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


# metadata FILE order == manifest CSV order == extraction order (verified 2026-07-25).
# Do NOT sort by mutant_id: alphabetical sort puts site-100 first and scrambles the
# pairing (mutant_XXXX.pt files are indexed by file order). rbd_repro.py's sorted
# mutant_map was exactly this bug.
meta = [json.loads(l) for l in open(RBD / 'metadata.jsonl') if l.strip()]
mids = [m['mutant_id'] for m in meta]
STRIDE = int(os.environ.get('E1_ANALYZE_STRIDE', '8'))
sub_idx = list(range(0, len(mids), STRIDE))
log(f'subset: {len(sub_idx)} mutants (stride {STRIDE})')


def load_set(files):
    X = np.empty((len(files), L, D), np.float32)
    for i, p in enumerate(files):
        z = torch.load(p, map_location='cpu', weights_only=True)
        if isinstance(z, dict):
            z = list(z.values())[0]
        X[i] = z[0].float().numpy()
    return torch.from_numpy(X)


def spectrum(Xc):
    n = Xc.shape[0]
    G = (Xc @ Xc.T) / n
    evals, _ = torch.linalg.eigh(G)
    evals = evals.flip(0).clamp_min(0)
    return evals / evals.sum().clamp_min(1e-12)


def effrank(ratio, thr):
    return int((torch.cumsum(ratio, 0) < thr).sum().item()) + 1


def lin_cka(Xc, Yc):
    Kx = Xc @ Xc.T
    Ky = Yc @ Yc.T
    return ((Kx * Ky).sum() / (Kx.norm() * Ky.norm()).clamp_min(1e-20)).item()


cos = torch.nn.functional.cosine_similarity

# reference sets
files_no = [TMP_NO / f'{mids[i]}_zii.pt' for i in sub_idx]
files_orig = [TMP_MSA / f'mutant_{i:04d}_zii.pt' for i in sub_idx]
Xno = load_set(files_no)
log('noMSA loaded')
Xorig = load_set(files_orig)
log('original-MSA loaded')


def flat(X):
    return X.reshape(X.shape[0], -1)


def arm_metrics(Xarm, Xno, Xorig):
    fa, fn, fo = flat(Xarm), flat(Xno), flat(Xorig)
    Da = fa - fa.mean(0)
    Dn = fn - fn.mean(0)
    Do = fo - fo.mean(0)
    ra = spectrum(Da)
    out = {
        'pc1': float(ra[0]),
        'eff90': effrank(ra, 0.90),
        'eff95': effrank(ra, 0.95),
        'cka_vs_noMSA': lin_cka(Da, Dn),
        'cka_vs_origMSA': lin_cka(Da, Do),
        'pp_cos_raw_vs_noMSA': float(cos(Xarm, Xno).mean()),
        'delta_norm_ratio_median': float((Da.norm(dim=1) / Dn.norm(dim=1).clamp_min(1e-12)).median()),
        'delta_cos_vs_noMSA_median': float(cos(Da, Dn).median()),
        'delta_cos_vs_noMSA_mean': float(cos(Da, Dn).mean()),
        'top64_ratio': [float(x) for x in ra[:64]],
    }
    return out


results = {'subset_n': len(sub_idx), 'arms': {}}
# baseline rows for reference
results['arms']['noMSA_ref'] = arm_metrics(Xno, Xno, Xorig)
results['arms']['origMSA_ref'] = arm_metrics(Xorig, Xno, Xorig)
log('references done')

for arm in ARMS:
    d = RBD / arm
    files = [d / f'mutant_{i:04d}_zii.pt' for i in sub_idx]
    missing = [p for p in files if not p.exists()]
    if missing:
        log(f'{arm}: {len(missing)} files missing, skipping')
        results['arms'][arm] = {'error': f'{len(missing)} missing'}
        continue
    Xa = load_set(files)
    results['arms'][arm] = arm_metrics(Xa, Xno, Xorig)
    r = results['arms'][arm]
    log(f"{arm}: PC1={r['pc1']:.3f} eff90={r['eff90']} cka_no={r['cka_vs_noMSA']:.3f} "
        f"sup={r['delta_norm_ratio_median']:.3f} dcos={r['delta_cos_vs_noMSA_median']:.3f}")
    del Xa

with open(OUT / f'results_{TAG}.json', 'w') as f:
    json.dump(results, f, indent=1)
log(f'saved {OUT}/results_{TAG}.json')

print('\n════════ E1 ARMS TABLE ════════')
hdr = f"{'arm':<12}{'PC1':>7}{'eff90':>7}{'eff95':>7}{'CKA_no':>8}{'CKA_orig':>9}{'ppCos':>7}{'supp':>7}{'dCos':>7}"
print(hdr)
for arm, r in results['arms'].items():
    if 'error' in r:
        print(f'{arm:<12} {r["error"]}')
        continue
    print(f"{arm:<12}{r['pc1']:>7.3f}{r['eff90']:>7}{r['eff95']:>7}{r['cka_vs_noMSA']:>8.3f}"
          f"{r['cka_vs_origMSA']:>9.3f}{r['pp_cos_raw_vs_noMSA']:>7.3f}"
          f"{r['delta_norm_ratio_median']:>7.3f}{r['delta_cos_vs_noMSA_median']:>7.3f}")
