#!/usr/bin/env python3
"""E5: complete the fusion matrix on CASP3 / CI (cross-protein V-series replication).

Cells (SP noMSA/MSA done in e4_*): sp_faesm, x_no_msa, x_msa_faesm,
x_no_zeros, x_msa_zeros, tri, tri_z — with per-encoder gradient tracking.
Protocol: 5 splits x 3 inits, 300 ep, BS=64, AdamW 1e-4/0.05, cosine, MSE,
best-epoch val Spearman, raw labels. CASP3: row slices z[pos,:488,:] + tiled
ESM-244->488 (documented tile caveat); CI: z[0] of [2,237,128] row+col.
Run: python -B e5_matrix_protein.py casp3|ci [cond ...]
Output: {base}/e5_matrix/results_<protein>.json
"""
import csv, json, os, re, sys, time
import numpy as np

PROTEIN = sys.argv[1] if len(sys.argv) > 1 else 'casp3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1" if PROTEIN == 'casp3' else "0"

import torch, torch.nn as nn
from pathlib import Path
from scipy import stats

torch.set_float32_matmul_precision('high')
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import XAttn, DEVICE, BS, EPOCHS, LR, WD
from e2_matrix_complete import TriAttn, grad_norms
from fusion_v3 import SinglePredictorV2

CONDS = sys.argv[2:] or ['sp_faesm', 'x_no_msa', 'x_msa_faesm',
                         'x_no_zeros', 'x_msa_zeros', 'tri', 'tri_z']
assert PROTEIN in ('casp3', 'ci')
BASE = Path(f'/mnt/k/output_heads/{PROTEIN}')
OUT = BASE / 'e5_matrix'
OUT.mkdir(exist_ok=True)
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


if PROTEIN == 'casp3':
    meta = [json.loads(l) for l in open(BASE / 'manifest.jsonl') if l.strip()]
    N = len(meta)
    y = np.array([m['label'] for m in meta], np.float32)
    pos = np.array([m['position'] - 1 for m in meta])
    L = 488

    def load_z(which):
        d = BASE / ('zii' if which == 'no' else which)
        X = np.empty((N, L, 128), np.float32)
        for i, m in enumerate(meta):
            z = torch.load(d / f"{m['mutant_id']}_zii.pt", map_location='cpu', weights_only=True)
            X[i] = z[m['position'] - 1].float().numpy()
            if i % 500 == 499:
                log(f'  {which} {i+1}/{N}')
        return torch.from_numpy(X)

    def load_f():
        X = np.empty((N, L, 1280), np.float32)
        for i, m in enumerate(meta):
            f = torch.load(BASE / 'faesm' / f"{m['mutant_id']}.pt", map_location='cpu', weights_only=True)
            if isinstance(f, dict):
                f = list(f.values())[0]
            f = f.float()
            if f.shape[0] == 244:  # tile 244 -> 488 (documented tile caveat)
                f = f.repeat(2, 1)
            X[i] = f[:L].numpy()
            if i % 500 == 499:
                log(f'  faesm {i+1}/{N}')
        return torch.from_numpy(X)
else:
    PROJ = Path('/mnt/j/conda_envs/foundry/DMS_Project')
    rows = list(csv.DictReader(open(PROJ / 'data' / 'DMS_ProteinGym_substitutions' /
                                      'RPC1_LAMBD_Li_2019_high-expression.csv')))
    N = len(rows)
    y = np.array([float(r['DMS_score']) for r in rows], np.float32)
    pos = np.array([int(re.match(r'[A-Z](\d+)[A-Z]', r['mutant']).group(1)) - 1 for r in rows])
    L = 237

    def load_z(which):
        d = BASE / ('zii_clean' if which == 'no' else which)
        X = np.empty((N, L, 128), np.float32)
        for i in range(N):
            z = torch.load(d / f'ci_{i:03d}_zii.pt', map_location='cpu', weights_only=True)
            if isinstance(z, dict):
                z = list(z.values())[0]
            X[i] = z[0].float().numpy()
        return torch.from_numpy(X)

    def load_f():
        X = np.empty((N, L, 1280), np.float32)
        for i in range(N):
            X[i] = torch.load(BASE / 'faesm' / f'ci_{i}.pt', map_location='cpu',
                              weights_only=True).float().numpy()
        return torch.from_numpy(X)


def std_rows(X):
    for i in range(X.shape[1]):
        mu = X[:, i].mean(0)
        sd = X[:, i].std(0).clamp_min(1e-8)
        X[:, i] = (X[:, i] - mu) / sd
    return X


def cross_split(seed):
    rng = np.random.RandomState(seed)
    ap = sorted(np.unique(pos))
    rng.shuffle(ap)
    vp = set(ap[:max(1, int(len(ap) * 0.3))])
    return (np.array([i for i in range(N) if pos[i] not in vp]),
            np.array([i for i in range(N) if pos[i] in vp]))


def train_one(model, inputs, tr, va, init, groups):
    torch.manual_seed(init)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    yt = torch.from_numpy(y[tr]).to(DEVICE)
    tr_t, va_t = torch.from_numpy(tr).to(DEVICE), torch.from_numpy(va).to(DEVICE)
    ins_tr = [x[tr_t] for x in inputs]
    ins_va = [x[va_t] for x in inputs]

    def fwd(*xs):
        out = model(*xs) if len(xs) > 1 else model(xs[0])
        return out[0] if isinstance(out, tuple) else out

    best, g_late = -1, None
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        ep_g = {k: 0.0 for k in groups}
        nb = 0
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(fwd(*[x[b] for x in ins_tr]), yt[b])
            opt.zero_grad(); loss.backward()
            g = grad_norms(model, groups)
            for k in groups:
                ep_g[k] += g[k]
            nb += 1
            opt.step()
        sched.step()
        if ep >= EPOCHS - 50:
            g_late = {k: v / max(nb, 1) for k, v in ep_g.items()} if g_late is None else \
                     {k: g_late[k] + ep_g[k] / max(nb, 1) / 50 for k in groups}
        model.eval()
        with torch.no_grad():
            pv = torch.cat([fwd(*[x[i:i + BS] for x in ins_va])
                            for i in range(0, len(va), BS)]).cpu().numpy()
        r, _ = stats.spearmanr(pv, y[va])
        best = max(best, r if not np.isnan(r) else -1)
    del model
    torch.cuda.empty_cache()
    return best, g_late


Z_no = std_rows(load_z('no')).to(DEVICE)
log('z_no ready')
Z_msa = std_rows(load_z('zii_msa_sp69' if PROTEIN == 'casp3' else 'zii_msa')).to(DEVICE)
log('z_msa ready')
F = std_rows(load_f()).to(DEVICE)
log('faesm ready')
Zeros = torch.zeros_like(Z_no)


def inputs_for(c):
    if c == 'sp_faesm':
        return SinglePredictorV2(1280), [F], {'f': 'enc'}
    if c == 'x_no_msa':
        return XAttn(128, 128), [Z_no, Z_msa], {'z': 'z_enc', 'm': 'f_enc'}
    if c == 'x_msa_faesm':
        return XAttn(128, 1280), [Z_msa, F], {'m': 'z_enc', 'f': 'f_enc'}
    if c == 'x_no_zeros':
        return XAttn(128, 128), [Z_no, Zeros], {'z': 'z_enc', 'x': 'f_enc'}
    if c == 'x_msa_zeros':
        return XAttn(128, 128), [Z_msa, Zeros], {'m': 'z_enc', 'x': 'f_enc'}
    if c == 'tri':
        return TriAttn(128, 128, 1280), [Z_no, Z_msa, F], {'z': 'e1', 'm': 'e2', 'f': 'e3'}
    if c == 'tri_z':
        return TriAttn(128, 128, 1280), [Zeros, Z_msa, F], {'x': 'e1', 'm': 'e2', 'f': 'e3'}
    raise ValueError(c)


res_path = OUT / f"results_{PROTEIN}{os.environ.get('E5_TAG', '')}.json"
done, results = set(), []
if res_path.exists():
    prev = json.load(open(res_path))
    done = {(r['cond'], r['split'], r['init']) for r in prev['runs']}
    results = prev['runs']
for c in CONDS:
    for sd in [100, 101, 102, 103, 104]:
        tr, va = cross_split(sd)
        for init in [7, 107, 207]:
            if (c, sd, init) in done:
                continue
            model, inputs, names = inputs_for(c)
            groups = {k: getattr(model, v) for k, v in names.items()}
            r, g = train_one(model, inputs, tr, va, init, groups)
            results.append({'cond': c, 'split': sd, 'init': init, 'val': r, 'grad_late': g})
            log(f'{c} s{sd} i{init}: {r:.4f} grads={ {k: round(v, 3) for k, v in g.items()} }')
            with open(res_path, 'w') as f:
                json.dump({'runs': results}, f)

summary = {}
for c in CONDS:
    vals = [r['val'] for r in results if r['cond'] == c]
    summary[c] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    log(f"{c}: μ={summary[c]['mean']:.4f} σ={summary[c]['std']:.4f}")
with open(res_path, 'w') as f:
    json.dump({'runs': results, 'summary': summary}, f, indent=1)
log('saved')
