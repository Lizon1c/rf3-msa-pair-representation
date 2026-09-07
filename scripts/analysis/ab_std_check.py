#!/usr/bin/env python3
"""A/B: scalar-per-position vs per-channel-per-position standardization,
for sp_faesm / x_no_faesm / sp_no / sp_msa (2 splits x 2 inits, quick check).

NOTE: the 2026-07-27 run of this script was print-only — its values survive
only hard-coded in make_figures.py fig8. This version writes
/mnt/k/output_heads/rbd/ab_std_check/results.json so reruns are archived."""
import json, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import sys
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import (XAttn, load_z_no, load_z_msa, load_faesm,
                               DEVICE, BS, EPOCHS, LR, WD, y, pos, cross_split)
from fusion_v3 import SinglePredictorV2
from scipy import stats


def std_scalar(X):
    for i in range(X.shape[1]):
        s = X[:, i].std()
        X[:, i] = (X[:, i] - X[:, i].mean()) / max(float(s), 1e-8)
    return X


def std_channel(X):
    for i in range(X.shape[1]):
        mu = X[:, i].mean(0)
        sd = X[:, i].std(0).clamp_min(1e-8)
        X[:, i] = (X[:, i] - mu) / sd
    return X


def train_one(model, inputs, tr, va, init):
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

    best = -1
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(fwd(*[x[b] for x in ins_tr]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([fwd(*[x[i:i + BS] for x in ins_va])
                            for i in range(0, len(va), BS)]).cpu().numpy()
        r, _ = stats.spearmanr(pv, y[va])
        best = max(best, r if not np.isnan(r) else -1)
    del model
    torch.cuda.empty_cache()
    return best


def main():
    Zn, Zm, F = load_z_no(), load_z_msa(), load_faesm()
    runs = []
    for tag, stdf in [('scalar', std_scalar), ('channel', std_channel)]:
        Zs, Zms, Fs = stdf(Zn.clone()).to(DEVICE), stdf(Zm.clone()).to(DEVICE), stdf(F.clone()).to(DEVICE)
        for cond, mk in [('sp_faesm', lambda: (SinglePredictorV2(1280), [Fs])),
                         ('sp_no', lambda: (SinglePredictorV2(128), [Zs])),
                         ('sp_msa', lambda: (SinglePredictorV2(128), [Zms])),
                         ('x_no_faesm', lambda: (XAttn(128, 1280), [Zs, Fs]))]:
            vals = []
            for sd in [100, 101]:
                tr, va = cross_split(sd)
                for init in [7, 107]:
                    m, ins = mk()
                    v = train_one(m, ins, tr, va, init)
                    vals.append(v)
                    runs.append({'std': tag, 'cond': cond, 'split': sd,
                                 'init': init, 'val': float(v)})
            print(f'{tag} {cond}: {["%.4f" % v for v in vals]} mean={np.mean(vals):.4f}', flush=True)
    out = Path('/mnt/k/output_heads/rbd/ab_std_check')
    out.mkdir(exist_ok=True)
    with open(out / 'results.json', 'w') as f:
        json.dump({'runs': runs,
                   'note': '2 splits x 2 inits quick check on full-set RBD'}, f, indent=1)
    print('saved', out / 'results.json', flush=True)


if __name__ == '__main__':
    main()
