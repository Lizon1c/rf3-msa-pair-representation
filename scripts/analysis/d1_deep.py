#!/usr/bin/env python3
"""D1-deep: extend the distillation capacity ladder and test for implicit memorization.

(1) Depth ladder TF{2,3,4,6,8} (both-input) -> where does val-CKA saturate?
(2) Matched-depth single-modality controls (z-only, f-only) -> does the both-vs-single
    gap persist or close at depth? (if it closes, the net reconstructs from memorized
    Z_msa structure, not from the input modality)
(3) train-CKA vs val-CKA gap at each depth -> direct memorization detector
    (growing train-val gap = the net memorizes Z_msa patterns rather than learning
    a generalizable input->Z_msa map).

All CKA/R2/cos computed on the cross-position VAL split (unseen positions); train-CKA
on the train split. 3 splits x 2 inits. Output: d1_distill/deep.json + table.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import (load_z_no, load_z_msa, load_faesm,
                               std_per_residue, cross_split, DEVICE, BS, EPOCHS, LR, WD, y, pos)
from d1_distill import TF
from scipy import stats
from pathlib import Path
import json

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'd1_distill'
DEPTHS = [2, 3, 4, 6, 8]


def lin_cka(Xc, Yc):
    Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
    return float((Kx * Ky).sum() / (Kx.norm() * Ky.norm()).clamp_min(1e-20))


def metrics(pred, real):
    pred_c = pred - pred.mean(0, keepdim=True)
    real_c = real - real.mean(0, keepdim=True)
    cos = torch.nn.functional.cosine_similarity
    r2 = float(1 - ((pred - real) ** 2).sum() / ((real - real.mean(0, keepdim=True)) ** 2).sum())
    pcos = float(cos(pred_c.reshape(len(pred), -1), real_c.reshape(len(pred), -1)).mean())
    cka = lin_cka(pred_c.reshape(len(pred), -1), real_c.reshape(len(pred), -1))
    return {'r2': r2, 'cos': pcos, 'cka': cka}


def run_one(Xin, Y, tr, va, nl, init):
    torch.manual_seed(init)
    model = TF(Xin.shape[-1], nl, 8).to(DEVICE)
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
            loss = nn.MSELoss()(model(Xin[tr_t][b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred_tr = torch.cat([model(Xin[tr_t][i:i + BS]) for i in range(0, len(tr), BS)])
        pred_va = torch.cat([model(Xin[va_t][i:i + BS]) for i in range(0, len(va), BS)])
    m_val = metrics(pred_va, Y[va_t])
    m_train = metrics(pred_tr, Yt)
    del model
    torch.cuda.empty_cache()
    return {**{f'val_{k}': v for k, v in m_val.items()},
            **{f'train_{k}': v for k, v in m_train.items()},
            'nparams': nparams}


def main():
    Z_no = std_per_residue(load_z_no()).to(DEVICE)
    Z_msa = std_per_residue(load_z_msa()).to(DEVICE)
    F = std_per_residue(load_faesm()).to(DEVICE)
    Xin_both = torch.cat([Z_no, F], dim=-1)
    all_inputs = {'both': Xin_both, 'zonly': Z_no, 'fonly': F}
    src_filter = os.environ.get('D1_SRC', '').split(',') if os.environ.get('D1_SRC') else None
    inputs = {k: v for k, v in all_inputs.items() if not src_filter or k in src_filter}
    tag = os.environ.get('D1_TAG', 'deep')
    out_path = OUT / f'{tag}.json'
    results = []
    for src, Xin in inputs.items():
        for nl in DEPTHS:
            for sd in [100, 101, 102]:
                tr, va = cross_split(sd)
                for init in [7, 107]:
                    r = run_one(Xin, Z_msa, tr, va, nl, init)
                    r.update({'src': src, 'depth': nl, 'split': sd, 'init': init})
                    results.append(r)
                    print(f"{src} TF{nl} s{sd} i{init}: val_CKA={r['val_cka']:.3f} "
                          f"train_CKA={r['train_cka']:.3f} gap={r['train_cka']-r['val_cka']:+.3f} "
                          f"val_R2={r['val_r2']:.3f} params={r['nparams']//1000}k", flush=True)
                    with open(out_path, 'w') as f:
                        json.dump({'runs': results}, f)

    print(f"\n{'src':<7}{'depth':>6}{'params':>8}{'valCKA':>8}{'trCKA':>8}{'gap':>7}{'valR2':>7}")
    for src in inputs:
        for nl in DEPTHS:
            rs = [r for r in results if r['src'] == src and r['depth'] == nl]
            if not rs:
                continue
            print(f"{src:<7}{nl:>6}{rs[0]['nparams']//1000:>6}k"
                  f"{np.mean([r['val_cka'] for r in rs]):>8.3f}"
                  f"{np.mean([r['train_cka'] for r in rs]):>8.3f}"
                  f"{np.mean([r['train_cka']-r['val_cka'] for r in rs]):>+7.3f}"
                  f"{np.mean([r['val_r2'] for r in rs]):>7.3f}")
    print('saved', out_path)


if __name__ == '__main__':
    main()
