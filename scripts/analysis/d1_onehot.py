#!/usr/bin/env python3
"""D1-onehot: map PURE one-hot sequence (each AA a fixed, mutually-orthogonal
128-dim vector) -> Z_msa. Cleanest memorization control: the input carries ONLY
per-residue amino-acid identity (no evolutionary / structural context). If a deep
net reconstructs Z_msa from this well, Z_msa is largely sequence-identity-determined
(pure memorization of the sequence->structure map); if it plateaus (like zonly),
sequence identity alone is insufficient and context matters.

Orthogonal codebook: 20 AA -> 20 mutually-orthogonal 128-dim vectors (fixed/frozen,
from QR of a random matrix). TF ladder {2,4,8} matches d1_deep for comparison.
3 splits x 2 inits. Output: d1_distill/onehot.json
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import (load_z_msa, std_per_residue, cross_split,
                               DEVICE, BS, EPOCHS, LR, WD, y, pos)
from d1_deep import TF, lin_cka, metrics
from pathlib import Path
import json, csv

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'd1_distill'
PROJ = Path('/mnt/j/conda_envs/foundry/DMS_Project')
AA = 'ACDEFGHIKLMNPQRSTVWY'
aa2idx = {a: i for i, a in enumerate(AA)}
DEPTHS = [2, 4, 8]


def orthogonal_codebook(k=20, d=128, seed=0):
    """k mutually-orthogonal unit vectors in R^d (d >= k), fixed seed -> frozen code."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k].T.contiguous()  # [k, d], orthonormal rows


def main():
    # mutant sequences in metadata order (manifest CSV order == metadata file order)
    manifest = list(csv.DictReader(open(PROJ / 'data' / 'sarbecovirus' /
                                        'manifest_SARS_CoV_2_WH1.csv')))
    seqs = [r['protein'] for r in manifest]
    L = len(seqs[0])
    N = len(seqs)
    code = orthogonal_codebook(20, 128).to(DEVICE)  # [20,128] frozen
    # verify orthogonality
    gram = code @ code.T
    off = (gram - torch.eye(20, device=DEVICE)).abs().max()
    print(f'codebook orthogonality max off-diag deviation: {off:.2e}', flush=True)

    # build one-hot inputs [N, L, 128]
    X = torch.zeros(N, L, 128, device=DEVICE)
    for i, s in enumerate(seqs):
        idx = torch.tensor([aa2idx.get(c, 0) for c in s], device=DEVICE)
        X[i] = code[idx]
    print(f'one-hot input {tuple(X.shape)}', flush=True)

    Z_msa = std_per_residue(load_z_msa()).to(DEVICE)
    results = []
    for nl in DEPTHS:
        for sd in [100, 101, 102]:
            tr, va = cross_split(sd)
            for init in [7, 107]:
                torch.manual_seed(init)
                model = TF(128, nl, 8).to(DEVICE)
                nparams = sum(p.numel() for p in model.parameters())
                opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
                tr_t, va_t = torch.from_numpy(tr).to(DEVICE), torch.from_numpy(va).to(DEVICE)
                Yt = Z_msa[tr_t]
                for ep in range(EPOCHS):
                    model.train()
                    idx = torch.randperm(len(tr), device=DEVICE)
                    for bi in range(0, len(tr), BS):
                        b = idx[bi:bi + BS]
                        loss = nn.MSELoss()(model(X[tr_t][b]), Yt[b])
                        opt.zero_grad(); loss.backward(); opt.step()
                    sched.step()
                model.eval()
                with torch.no_grad():
                    pred_tr = torch.cat([model(X[tr_t][i:i + BS]) for i in range(0, len(tr), BS)])
                    pred_va = torch.cat([model(X[va_t][i:i + BS]) for i in range(0, len(va), BS)])
                m_val = metrics(pred_va, Z_msa[va_t])
                m_train = metrics(pred_tr, Yt)
                del model; torch.cuda.empty_cache()
                r = {**{f'val_{k}': v for k, v in m_val.items()},
                     **{f'train_{k}': v for k, v in m_train.items()},
                     'depth': nl, 'split': sd, 'init': init, 'nparams': nparams}
                results.append(r)
                print(f"onehot TF{nl} s{sd} i{init}: val_CKA={r['val_cka']:.3f} "
                      f"train_CKA={r['train_cka']:.3f} gap={r['train_cka']-r['val_cka']:+.3f} "
                      f"val_R2={r['val_r2']:.3f}", flush=True)
                with open(OUT / 'onehot.json', 'w') as f:
                    json.dump({'runs': results}, f)

    print(f"\n{'depth':>6}{'valCKA':>8}{'trCKA':>8}{'gap':>7}{'valR2':>7}")
    for nl in DEPTHS:
        rs = [r for r in results if r['depth'] == nl]
        print(f"{nl:>6}{np.mean([r['val_cka'] for r in rs]):>8.3f}"
              f"{np.mean([r['train_cka'] for r in rs]):>8.3f}"
              f"{np.mean([r['train_cka']-r['val_cka'] for r in rs]):>+7.3f}"
              f"{np.mean([r['val_r2'] for r in rs]):>7.3f}")
    print('saved', OUT / 'onehot.json')


if __name__ == '__main__':
    main()
