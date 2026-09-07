#!/usr/bin/env python3
"""CI cross-protein replication: MSA vs no-MSA Z_II geometry + supervised.
Clean zii_clean/ (noMSA, true-site row) vs zii_msa/ (lambda-repressor MSA).
Same protocol as e4_casp3_msa_analysis.py.
Output: /mnt/k/output_heads/ci/e4_msa_analysis/results.json
"""
import csv, json, os, re, time
import numpy as np
import torch
from pathlib import Path
from scipy import stats

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
torch.set_float32_matmul_precision('high')
import sys
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import DEVICE, BS, EPOCHS, LR, WD
from fusion_v3 import SinglePredictorV2
import torch.nn as nn

torch.set_num_threads(24)
BASE = Path('/mnt/k/output_heads/ci')
PROJ = Path('/mnt/j/conda_envs/foundry/DMS_Project')
OUT = BASE / 'e4_msa_analysis'
OUT.mkdir(exist_ok=True)
L = 237
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


rows = list(csv.DictReader(open(PROJ / 'data' / 'DMS_ProteinGym_substitutions' /
                                  'RPC1_LAMBD_Li_2019_high-expression.csv')))
N = len(rows)
y = np.array([float(r['DMS_score']) for r in rows], np.float32)
pos = np.array([int(re.match(r'[A-Z](\d+)[A-Z]', r['mutant']).group(1)) - 1 for r in rows])


def load_rows(which):
    d = BASE / which
    X = np.empty((N, L, 128), np.float32)
    for i in range(N):
        z = torch.load(d / f'ci_{i:03d}_zii.pt', map_location='cpu', weights_only=True)
        if isinstance(z, dict):
            z = list(z.values())[0]
        X[i] = z[0].float().numpy()
    return torch.from_numpy(X)


def spectrum(Xc):
    G = (Xc @ Xc.T) / Xc.shape[0]
    ev = torch.linalg.eigh(G)[0].flip(0).clamp_min(0)
    return ev / ev.sum().clamp_min(1e-12)


def effrank(r, thr):
    return int((torch.cumsum(r, 0) < thr).sum().item()) + 1


def lin_cka(Xc, Yc):
    Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
    return float((Kx * Ky).sum() / (Kx.norm() * Ky.norm()).clamp_min(1e-20))


def cross_split(seed):
    rng = np.random.RandomState(seed)
    ap = sorted(np.unique(pos))
    rng.shuffle(ap)
    vp = set(ap[:max(1, int(len(ap) * 0.3))])
    return (np.array([i for i in range(N) if pos[i] not in vp]),
            np.array([i for i in range(N) if pos[i] in vp]))


def std_rows(X):
    for i in range(X.shape[1]):
        mu = X[:, i].mean(0)
        sd = X[:, i].std(0).clamp_min(1e-8)
        X[:, i] = (X[:, i] - mu) / sd
    return X


def train_one(X, tr, va, init):
    torch.manual_seed(init)
    model = SinglePredictorV2(128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    yt = torch.from_numpy(y[tr]).to(DEVICE)
    Xt, Xv = X[tr], X[va]
    best = -1
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(model(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xv[i:i + BS]) for i in range(0, len(va), BS)]).cpu().numpy()
        r, _ = stats.spearmanr(pv, y[va])
        best = max(best, r if not np.isnan(r) else -1)
    del model
    torch.cuda.empty_cache()
    return best


log(f'N={N} CI mutants')
Xno = load_rows('zii_clean')
log('clean noMSA loaded')
Xms = load_rows('zii_msa')
log('MSA loaded')

fno, fms = Xno.reshape(N, -1), Xms.reshape(N, -1)
Dno, Dms = fno - fno.mean(0), fms - fms.mean(0)
cos = torch.nn.functional.cosine_similarity
ra, rb = spectrum(Dno), spectrum(Dms)
geo = {
    'pc1_no': float(ra[0]), 'pc1_msa': float(rb[0]),
    'eff90_no': effrank(ra, 0.9), 'eff90_msa': effrank(rb, 0.9),
    'eff95_no': effrank(ra, 0.95), 'eff95_msa': effrank(rb, 0.95),
    'cka': lin_cka(Dno, Dms),
    'supp_median': float((Dms.norm(dim=1) / Dno.norm(dim=1).clamp_min(1e-12)).median()),
    'dcos_median': float(cos(Dms, Dno).median()),
}
log(f"geometry: PC1 {geo['pc1_no']:.3f}→{geo['pc1_msa']:.3f} eff90 {geo['eff90_no']}→{geo['eff90_msa']} "
    f"CKA {geo['cka']:.3f} supp {geo['supp_median']:.3f} dCos {geo['dcos_median']:.3f}")

Xno_g = std_rows(Xno.clone()).to(DEVICE)
Xms_g = std_rows(Xms.clone()).to(DEVICE)
runs = []
for cond, X in [('sp_no', Xno_g), ('sp_msa', Xms_g)]:
    for sd in [100, 101, 102, 103, 104]:
        tr, va = cross_split(sd)
        for init in [7, 107, 207]:
            r = train_one(X, tr, va, init)
            runs.append({'cond': cond, 'split': sd, 'init': init, 'val': r})
            log(f'{cond} s{sd} i{init}: {r:.4f}')
            with open(OUT / 'results.json', 'w') as f:
                json.dump({'geo': geo, 'runs': runs}, f)

summary = {}
for cond in ['sp_no', 'sp_msa']:
    vals = [r['val'] for r in runs if r['cond'] == cond]
    summary[cond] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
per_split = [float(np.mean([r['val'] for r in runs if r['cond'] == 'sp_msa' and r['split'] == sd]) -
                  np.mean([r['val'] for r in runs if r['cond'] == 'sp_no' and r['split'] == sd]))
             for sd in [100, 101, 102, 103, 104]]
summary['paired_delta'] = {'per_split': per_split, 'mean': float(np.mean(per_split))}
with open(OUT / 'results.json', 'w') as f:
    json.dump({'geo': geo, 'runs': runs, 'summary': summary}, f, indent=1)
log(f"sp_no {summary['sp_no']['mean']:.4f} | sp_msa {summary['sp_msa']['mean']:.4f} | "
    f"paired {['%+.3f' % d for d in per_split]} μ={summary['paired_delta']['mean']:+.4f}")
log('saved')
