#!/usr/bin/env python3
"""Corrected supervised comparison: noMSA vs e1_full15 Z_II, 500-mutant subset.

The published "MSA −40%" (0.494 vs 0.295) is INVALID — the June wh1_msa extraction
sliced every mutant's Z_II at position 0 (wrong WT reference in _find_mut_pos),
so those files carry no mutation-site information. This script reruns the comparison
with correctly-sliced, correctly-paired embeddings (zii/ vs zii_e1_full15/, subset
every-8th mutant, metadata FILE order == CSV order == extraction order).

Protocol: rbd_repro.py V2 (ZII_Model: Encoder 3L DM=256 NH=8 + AttnPool + MLP),
cross-position 70/30 split, 5 splits × 3 inits, seeds 100-104, 300 ep, BS=64,
AdamW lr1e-4 wd0.05, cosine, MSE, best-epoch val Spearman. bind_avg labels.
GPU shared OK (~1GB). Output: /mnt/k/output_heads/rbd/e1_supervised/results.json
"""
import json, time
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from scipy import stats
import math, os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.set_float32_matmul_precision('high')
RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'e1_supervised'
OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda')
DM, NH, BS = 256, 8, 64
EPOCHS, LR, WD = 300, 1e-4, 0.05
FINAL_DIM, DROPOUT = 128, 0.1
L = 201
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


class PosEnc(nn.Module):
    def __init__(self, d, m=500):
        super().__init__()
        pe = torch.zeros(m, d)
        pos = torch.arange(m).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.proj = nn.Linear(d_in, DM)
        self.pe = PosEnc(DM)
        el = nn.TransformerEncoderLayer(DM, NH, DM * 4, DROPOUT, batch_first=True)
        self.enc = nn.TransformerEncoder(el, 3)
        self.out = nn.Sequential(nn.Linear(DM, DM * 2), nn.GELU(), nn.Linear(DM * 2, FINAL_DIM))

    def forward(self, x):
        return self.out(self.enc(self.pe(self.proj(x))))


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scorer = nn.Linear(d, 1)

    def forward(self, x):
        w = torch.softmax(self.scorer(x).squeeze(-1), dim=1)
        return (x * w.unsqueeze(-1)).sum(dim=1)


class ZII_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(128)
        self.pool = AttnPool(128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.pool(self.enc(x))).squeeze(-1)


# ── data: metadata FILE order (== CSV/extraction order) ──
meta = [json.loads(l) for l in open(RBD / 'metadata.jsonl') if l.strip()]
STRIDE = int(os.environ.get('E1_SUP_STRIDE', '8'))
sub = list(range(0, len(meta), STRIDE))
TAG = os.environ.get('E1_SUP_TAG', 'results')
y = np.array([meta[i]['bind_avg'] for i in sub], np.float32)
pos = np.array([meta[i]['site_rbd'] - 1 for i in sub])


def load_zii(which):
    X = np.empty((len(sub), L, 128), np.float32)
    for k, i in enumerate(sub):
        if which == 'no':
            p = RBD / 'zii' / f"{meta[i]['mutant_id']}_zii.pt"
        else:  # full dir name, e.g. zii_e1_full15 or zii_e7_k1
            p = RBD / which / f'mutant_{i:04d}_zii.pt'
        z = torch.load(p, map_location='cpu', weights_only=True)
        if isinstance(z, dict):
            z = list(z.values())[0]
        X[k] = z[0].float().numpy()
    for i in range(L):
        s = X[:, i].std()
        X[:, i] = (X[:, i] - X[:, i].mean()) / max(s, 1e-8)
    return torch.from_numpy(X)


def cross_split(seed):
    rng = np.random.RandomState(seed)
    ap = sorted(np.unique(pos))
    rng.shuffle(ap)
    vp = set(ap[:max(1, int(len(ap) * 0.3))])
    tr = [i for i in range(len(pos)) if pos[i] not in vp]
    va = [i for i in range(len(pos)) if pos[i] in vp]
    return np.array(tr), np.array(va)


def train_one(Xt, yt, Xv, yv, init):
    torch.manual_seed(init)
    m = ZII_Model().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    yt, yv = yt.to(DEVICE), yv.to(DEVICE)
    best = -1
    for ep in range(EPOCHS):
        m.train()
        idx = torch.randperm(len(Xt))
        for bi in range(0, len(Xt), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(m(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            pv = torch.cat([m(Xv[i:i + BS]) for i in range(0, len(Xv), BS)]).cpu().numpy()
        va, _ = stats.spearmanr(pv, yv.cpu().numpy())
        va = va if not np.isnan(va) else -1
        best = max(best, va)
    return best


cond = os.environ.get('E1_SUP_COND', 'no,full15').split(',')
Xcache = {}
for c in cond:
    Xcache[c] = load_zii(c).to(DEVICE)
    log(f'loaded {c}')

results = []
for c in cond:
    X = Xcache[c]
    for si, sd in enumerate([100, 101, 102, 103, 104]):
        tr, va = cross_split(sd)
        for init in [7, 107, 207]:
            r = train_one(X[tr], torch.from_numpy(y[tr]), X[va], torch.from_numpy(y[va]), init)
            results.append({'cond': c, 'split': sd, 'init': init, 'val': r})
            log(f'{c} split{sd} init{init}: {r:.4f}')

summary = {}
for c in cond:
    vals = [r['val'] for r in results if r['cond'] == c]
    summary[c] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                  'min': float(np.min(vals)), 'max': float(np.max(vals))}
    log(f"{c}: μ={summary[c]['mean']:.4f} σ={summary[c]['std']:.4f} "
        f"min={summary[c]['min']:.4f} max={summary[c]['max']:.4f}")

with open(OUT / f'results_{TAG}.json', 'w') as f:
    json.dump({'runs': results, 'summary': summary}, f, indent=1)
log('saved')
