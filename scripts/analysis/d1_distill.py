#!/usr/bin/env python3
"""D1: distillation — can (Z_no + FAESM) reconstruct Z_msa, and at what capacity?

Question: is the MSA's content already latent in the no-MSA structural features +
ESM evolutionary features? Train mappings (Z_no, FAESM) -> Z_msa on the full RBD
set (3998 mutants), cross-position splits, and measure reconstruction fidelity
vs architecture capacity. If a LINEAR map suffices, the MSA signal is linearly
present in the other two modalities; if deep nets are required, the MSA adds
nonlinearly synthesized content.

Ladder (shared per-residue MLP on concat[z_i, f_i] -> z_msa_i):
  lin      : Linear(1408 -> 128)
  mlp256   : 1408 -> 256 -> GELU -> 128
  mlp512   : 1408 -> 512 -> GELU -> 256 -> GELU -> 128
  tf1      : proj 256 + PE + 1L Transformer(4h) -> 128
  tf2      : proj 256 + PE + 2L Transformer(8h) -> 128
Ablations (mlp512): z_no-only, faesm-only.
Splits: cross-position 3 x 2 inits. Target standardized per-residue.
Metrics: per-position cos(recon, real) mean; global R^2 (variance explained);
CKA(recon, real); per-channel R^2 summary.
Output: /mnt/k/output_heads/rbd/d1_distill/results.json + stdout table.
"""
import json, math, os, time
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from scipy import stats

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.set_float32_matmul_precision('high')
import sys
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import (load_z_no, load_z_msa, load_faesm,
                               std_per_residue, cross_split, DEVICE, L)
from fusion_v3 import PosEnc

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'd1_distill'
OUT.mkdir(exist_ok=True)
EPOCHS, BS, LR, WD = 300, 64, 1e-4, 0.05
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


class MLP(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        layers = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, 128)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TF(nn.Module):
    def __init__(self, d_in, nl, nh):
        super().__init__()
        self.proj = nn.Linear(d_in, 256)
        self.pe = PosEnc(256)
        el = nn.TransformerEncoderLayer(256, nh, 512, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(el, nl)
        self.out = nn.Linear(256, 128)

    def forward(self, x):
        return self.out(self.enc(self.pe(self.proj(x))))


ARCHS = {
    'lin': lambda d: MLP(d, []),
    'mlp256': lambda d: MLP(d, [256]),
    'mlp512': lambda d: MLP(d, [512, 256]),
    'tf1': lambda d: TF(d, 1, 4),
    'tf2': lambda d: TF(d, 2, 8),
}


def lin_cka(Xc, Yc):
    Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
    return float((Kx * Ky).sum() / (Kx.norm() * Ky.norm()).clamp_min(1e-20))


def run_split(Xin, Y, tr, va, arch, init):
    torch.manual_seed(init)
    d_in = Xin.shape[-1]
    model = ARCHS[arch](d_in).to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    yt_mean = Y[tr].mean(0, keepdim=True)
    best_mse, best_pred = float('inf'), None
    tr_t, va_t = torch.from_numpy(tr).to(DEVICE), torch.from_numpy(va).to(DEVICE)
    Xin_t, Xin_v = Xin[tr_t], Xin[va_t]
    Yt, Yv = Y[tr_t], Y[va_t]
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(model(Xin_t[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xin_v[i:i + BS]) for i in range(0, len(va), BS)])
            mse = float(nn.MSELoss()(pv, Yv))
        if mse < best_mse:
            best_mse, best_pred = mse, pv
    # metrics on val
    pv = best_pred
    Yv_c = Yv - Yv.mean(0, keepdim=True)
    pv_c = pv - pv.mean(0, keepdim=True)
    r2 = float(1 - ((pv - Yv) ** 2).sum() / ((Yv - Yv.mean(0, keepdim=True)) ** 2).sum())
    cos = torch.nn.functional.cosine_similarity
    pcos = float(cos(pv_c.reshape(len(va), -1), Yv_c.reshape(len(va), -1)).mean())
    cka = lin_cka(pv_c.reshape(len(va), -1), Yv_c.reshape(len(va), -1))
    r2_ch = 1 - ((pv - Yv) ** 2).mean(0) / ((Yv - Yv.mean(0, keepdim=True)) ** 2).mean(0).clamp_min(1e-12)
    del model
    torch.cuda.empty_cache()
    return {'r2': r2, 'pcos': pcos, 'cka': cka, 'mse': best_mse,
            'r2_ch_median': float(r2_ch.median()), 'nparams': nparams}


def main():
    Z_no = std_per_residue(load_z_no())
    log('z_no ready')
    Z_msa = std_per_residue(load_z_msa())
    log('z_msa ready')
    F = std_per_residue(load_faesm())
    log('faesm ready')
    Xin_both = torch.cat([Z_no, F], dim=-1).to(DEVICE)
    Xin_z = Z_no.to(DEVICE)
    Xin_f = F.to(DEVICE)
    Y = Z_msa.to(DEVICE)

    jobs = [('both', a) for a in ['lin', 'mlp256', 'mlp512', 'tf1', 'tf2']] + \
           [('zonly', 'mlp512'), ('fonly', 'mlp512')]
    results = []
    meta = [json.loads(l) for l in open(RBD / 'metadata.jsonl') if l.strip()]
    pos = np.array([m['site_rbd'] - 1 for m in meta])
    import e2_four_condition as _e2
    _e2.pos = pos  # cross_split reads module-level pos
    for src, arch in jobs:
        Xin = {'both': Xin_both, 'zonly': Xin_z, 'fonly': Xin_f}[src]
        for sd in [100, 101, 102]:
            tr, va = cross_split(sd)
            for init in [7, 107]:
                r = run_split(Xin, Y, tr, va, arch, init)
                r.update({'src': src, 'arch': arch, 'split': sd, 'init': init})
                results.append(r)
                log(f"{src}/{arch} s{sd} i{init}: R2={r['r2']:.3f} cos={r['pcos']:.3f} "
                    f"CKA={r['cka']:.3f} params={r['nparams'] // 1000}k")
                with open(OUT / 'results.json', 'w') as f:
                    json.dump({'runs': results}, f)

    print(f"\n{'src':<7}{'arch':<8}{'params':>8}{'R2':>7}{'cos':>7}{'CKA':>7}")
    for src, arch in jobs:
        rs = [r for r in results if r['src'] == src and r['arch'] == arch]
        print(f"{src:<7}{arch:<8}{rs[0]['nparams'] // 1000:>6}k"
              f"{np.mean([r['r2'] for r in rs]):>7.3f}{np.mean([r['pcos'] for r in rs]):>7.3f}"
              f"{np.mean([r['cka'] for r in rs]):>7.3f}")
    log('saved')


if __name__ == '__main__':
    main()
