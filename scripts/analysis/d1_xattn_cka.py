#!/usr/bin/env python3
"""D1-XAttn-CKA: reconstruct Z_msa with SEPARATE per-modality encoders (XAttn-style,
so the weak Z_no signal is not swamped by F in a shared projection), the per-residue
interaction + joint encoder, but NO attention pooling — keep the per-residue [L,128]
representation and optimize CKA directly as the loss (the metric we actually care about).

Tests whether the naive-concat both<fonly was an architecture artifact: with separate
encoders + CKA loss, does Z_no contribute (both > fonly)? Variants both/fonly/zonly,
joint depths {2,4}. 3 splits x 2 inits. Output: d1_distill/xattn_cka.json
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import (load_z_no, load_z_msa, load_faesm, std_per_residue,
                               cross_split, DEVICE, EPOCHS, LR, WD, y, pos)
from d1_deep import lin_cka
from fusion_v3 import Encoder, PosEnc
from pathlib import Path
import json

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'd1_distill'
BS = 32
import os as _os
DEPTHS = [int(x) for x in _os.environ.get("D1X_DEPTHS", "2,4").split(",")]


class XAttnRecon(nn.Module):
    """Separate encoders per modality -> per-residue interaction -> joint encoder ->
    per-residue [L,128] output (no attention pool, no scalar head)."""
    def __init__(self, d_ins, nl):
        super().__init__()
        self.encs = nn.ModuleList([Encoder(d) for d in d_ins])
        n = len(d_ins)
        self.n = n
        if n > 1:
            el = nn.TransformerEncoderLayer(128, 4, 512, 0.1, batch_first=True)
            self.interaction = nn.TransformerEncoder(el, 1)
        else:
            self.interaction = None
        self.pe = PosEnc(128 * n)
        el2 = nn.TransformerEncoderLayer(128 * n, 8, (128 * n) * 2, 0.1, batch_first=True)
        self.joint = nn.TransformerEncoder(el2, nl)
        self.out = nn.Linear(128 * n, 128)

    def forward(self, xs):
        encs = [enc(x) for enc, x in zip(self.encs, xs)]
        if self.interaction is not None:
            B, L, D = encs[0].shape
            tokens = torch.stack(encs, dim=2).view(B * L, self.n, D)
            inter = self.interaction(tokens).view(B, L, self.n, D)
            encs = [inter[:, :, i, :] for i in range(self.n)]
        j = torch.cat(encs, dim=-1)
        j = self.pe(j)
        j = self.joint(j)
        return self.out(j)  # [B, L, 128]


def _cka_feature(p, t):
    # feature-space linear CKA: only [128,128] matrices (memory-safe, == Gram-space CKA)
    XtX = p.T @ p
    YtY = t.T @ t
    XtY = p.T @ t
    return (XtY.norm() ** 2) / (XtX.norm() * YtY.norm()).clamp_min(1e-12)


def cka_loss(pred, target):
    p = pred.reshape(-1, 128)
    t = target.reshape(-1, 128)
    p = p - p.mean(0, keepdim=True)
    t = t - t.mean(0, keepdim=True)
    return 1 - _cka_feature(p, t)


def val_cka(pred, target):
    p = pred.reshape(-1, 128).cpu()
    t = target.reshape(-1, 128).cpu()
    p = p - p.mean(0, keepdim=True)
    t = t - t.mean(0, keepdim=True)
    return float(_cka_feature(p, t))


def run_one(d_ins, make_xs, Y, tr, va, nl, init):
    torch.manual_seed(init)
    model = XAttnRecon(d_ins, nl).to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    tr_t, va_t = torch.from_numpy(tr).to(DEVICE), torch.from_numpy(va).to(DEVICE)
    Yt = Y[tr_t]
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            xs = [x[tr_t][b] for x in make_xs()]
            loss = cka_loss(model(xs), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        xs_tr = [x[tr_t] for x in make_xs()]
        xs_va = [x[va_t] for x in make_xs()]
        pred_tr = torch.cat([model([x[i:i + BS] for x in xs_tr]) for i in range(0, len(tr), BS)])
        pred_va = torch.cat([model([x[i:i + BS] for x in xs_va]) for i in range(0, len(va), BS)])
    vc = val_cka(pred_va, Y[va_t])
    tc = val_cka(pred_tr, Yt)
    del model; torch.cuda.empty_cache()
    return vc, tc, nparams


def main():
    Z_no = std_per_residue(load_z_no()).to(DEVICE)
    Z_msa = std_per_residue(load_z_msa()).to(DEVICE)
    F = std_per_residue(load_faesm()).to(DEVICE)
    configs = {
        'both': ([128, 1280], lambda: [Z_no, F]),
        'fonly': ([1280], lambda: [F]),
        'zonly': ([128], lambda: [Z_no]),
    }
    results = []
    for name, (d_ins, mk) in configs.items():
        for nl in DEPTHS:
            for sd in [100, 101, 102]:
                tr, va = cross_split(sd)
                for init in [7, 107]:
                    vc, tc, np_ = run_one(d_ins, mk, Z_msa, tr, va, nl, init)
                    results.append({'src': name, 'depth': nl, 'split': sd, 'init': init,
                                    'val_cka': vc, 'train_cka': tc, 'nparams': np_})
                    print(f"{name} nl{nl} s{sd} i{init}: val_CKA={vc:.3f} train_CKA={tc:.3f} "
                          f"gap={tc-vc:+.3f} params={np_//1000}k", flush=True)
                    with open(OUT / (_os.environ.get('D1X_TAG', 'xattn_cka') + '.json'), 'w') as f:
                        json.dump({'runs': results}, f)
    print(f"\n{'src':<7}{'depth':>6}{'valCKA':>8}{'trCKA':>8}{'gap':>7}")
    for name in configs:
        for nl in DEPTHS:
            rs = [r for r in results if r['src'] == name and r['depth'] == nl]
            print(f"{name:<7}{nl:>6}{np.mean([r['val_cka'] for r in rs]):>8.3f}"
                  f"{np.mean([r['train_cka'] for r in rs]):>8.3f}"
                  f"{np.mean([r['train_cka']-r['val_cka'] for r in rs]):>+7.3f}")
    print('saved', OUT / (_os.environ.get('D1X_TAG', 'xattn_cka') + '.json'))


if __name__ == '__main__':
    main()
