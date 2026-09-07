#!/usr/bin/env python3
"""D1-deep trajectory probe: epoch-resolved train/val CKA + R² for the naive-concat
distillation ladder (TF{2,4,8} × {both, fonly, zonly}), to test whether the
TF2→TF8 collapse of the train-val CKA gap (fig6: +0.233 → +0.093) is grokking-like
TEMPORAL dynamics (train saturates early, val CKA jumps LATE in training) or just
a depth effect visible at the last epoch.

Records per epoch: val R² (cheap); every 10 epochs: val CKA and train CKA
(sample-Gram linear CKA on flattened [N, L*128], identical formula to d1_deep.py
so last-epoch values are comparable to the archived deep_{both,f,z}.json).
3 splits (100-102) × 2 inits (7, 107) × 3 depths × 3 sources = 54 runs,
matching d1_deep's grid. Labels: reconstruction target = standardized Z_msa
(same as d1_deep).

Run: CUDA_VISIBLE_DEVICES=0 python -B d1_deep_traj.py
Output: /mnt/k/output_heads/rbd/d1_distill/deep_traj.json
"""
import json, os, time
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch, torch.nn as nn
from pathlib import Path

torch.set_float32_matmul_precision('high')
import sys
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import (load_z_no, load_z_msa, load_faesm, std_per_residue,
                               cross_split, DEVICE, BS, EPOCHS, LR, WD)
from d1_distill import TF

OUT = Path('/mnt/k/output_heads/rbd/d1_distill')
DEPTHS = [2, 4, 8]
SRCS = ['both', 'fonly', 'zonly']
CKA_EVERY = 10
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


def lin_cka(Xc, Yc):
    Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
    return float((Kx * Ky).sum() / (Kx.norm() * Ky.norm()).clamp_min(1e-20))


def metrics(pred, real):
    """CKA + R² on [N, L*128] flattened, centered (identical to d1_deep.metrics)."""
    p = pred.reshape(len(pred), -1)
    r = real.reshape(len(real), -1)
    pc = p - p.mean(0, keepdim=True)
    rc = r - r.mean(0, keepdim=True)
    r2 = float(1 - ((p - r) ** 2).sum() / ((r - r.mean(0, keepdim=True)) ** 2).sum())
    return lin_cka(pc, rc), r2


def run_one(Xin, Y, tr, va, nl, init):
    torch.manual_seed(init)
    model = TF(Xin.shape[-1], nl, 8).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    tr_t = torch.from_numpy(tr).to(DEVICE)
    va_t = torch.from_numpy(va).to(DEVICE)
    Yt, Yv = Y[tr_t], Y[va_t]
    # val_r2 every epoch (cheap); val/train CKA every CKA_EVERY epochs + final
    # (full sample-Gram CKA is ~1-2 s per eval, so not per-epoch)
    traj = {'r2_ep': [], 'val_r2': [], 'cka_ep': [], 'val_cka': [], 'tr_cka': []}
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(model(Xin[tr_t][b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        cka_now = (ep % CKA_EVERY == 0) or (ep == EPOCHS - 1)
        with torch.no_grad():
            pv = torch.cat([model(Xin[va_t][i:i + BS])
                            for i in range(0, len(va), BS)])
            if cka_now:
                val_cka, val_r2 = metrics(pv, Yv)
                pt = torch.cat([model(Xin[tr_t][i:i + BS])
                                for i in range(0, len(tr), BS)])
                tr_cka, _ = metrics(pt, Yt)
                traj['cka_ep'].append(ep)
                traj['val_cka'].append(round(val_cka, 4))
                traj['tr_cka'].append(round(tr_cka, 4))
            else:
                val_r2 = float(1 - ((pv - Yv) ** 2).sum() /
                               ((Yv - Yv.mean(0, keepdim=True)) ** 2).sum())
            traj['r2_ep'].append(ep)
            traj['val_r2'].append(round(val_r2, 4))
    # final-epoch values (comparable to archived deep_*.json last-epoch metrics)
    final = {'val_cka': traj['val_cka'][-1], 'val_r2': traj['val_r2'][-1],
             'train_cka': traj['tr_cka'][-1]}
    del model
    torch.cuda.empty_cache()
    return traj, final


def main():
    log('loading + standardizing')
    Z_no = std_per_residue(load_z_no()).to(DEVICE)
    Z_msa = std_per_residue(load_z_msa()).to(DEVICE)
    F = std_per_residue(load_faesm()).to(DEVICE)
    inputs = {'both': torch.cat([Z_no, F], dim=-1), 'fonly': F, 'zonly': Z_no}
    res_path = OUT / 'deep_traj.json'
    done, results = set(), []
    if res_path.exists():
        prev = json.load(open(res_path))
        done = {(r['src'], r['depth'], r['split'], r['init']) for r in prev['runs']}
        results = prev['runs']
    for src in SRCS:
        Xin = inputs[src]
        for nl in DEPTHS:
            for sd in [100, 101, 102]:
                tr, va = cross_split(sd)
                for init in [7, 107]:
                    if (src, nl, sd, init) in done:
                        continue
                    traj, final = run_one(Xin, Z_msa, tr, va, nl, init)
                    results.append({'src': src, 'depth': nl, 'split': sd, 'init': init,
                                    'traj': traj, 'final': final})
                    log(f'{src} TF{nl} s{sd} i{init}: final val_cka={final["val_cka"]:.3f} '
                        f'train_cka={final["train_cka"]:.3f} gap={final["train_cka"]-final["val_cka"]:+.3f}')
                    with open(res_path, 'w') as f:
                        json.dump({'runs': results}, f)
    # summary: per (src, depth) mean final gap + grokking diagnostic
    # (max late-epoch val_cka jump: val_cka[ep299] - val_cka[ep150])
    summary = {}
    for src in SRCS:
        for nl in DEPTHS:
            rs = [r for r in results if r['src'] == src and r['depth'] == nl]
            if not rs:
                continue
            gaps = [r['final']['train_cka'] - r['final']['val_cka'] for r in rs]

            def cka_at(r, ep_want):
                d = dict(zip(r['traj']['cka_ep'], r['traj']['val_cka']))
                return d.get(ep_want, d[max(k for k in d if k <= ep_want)])

            late_jump = [r['traj']['val_cka'][-1] - cka_at(r, 150) for r in rs]
            half_jump = [cka_at(r, 150) - cka_at(r, 50) for r in rs]
            summary[f'{src}_TF{nl}'] = {
                'val_cka': float(np.mean([r['final']['val_cka'] for r in rs])),
                'train_cka': float(np.mean([r['final']['train_cka'] for r in rs])),
                'gap': float(np.mean(gaps)),
                'val_cka_ep50': float(np.mean([cka_at(r, 50) for r in rs])),
                'val_cka_ep150': float(np.mean([cka_at(r, 150) for r in rs])),
                'late_jump_150_299': float(np.mean(late_jump)),
                'mid_jump_50_150': float(np.mean(half_jump)),
            }
            s = summary[f'{src}_TF{nl}']
            log(f"{src} TF{nl}: val_cka={s['val_cka']:.3f} gap={s['gap']:+.3f} "
                f"ep50={s['val_cka_ep50']:.3f} ep150={s['val_cka_ep150']:.3f} ep299={s['val_cka']:.3f} "
                f"(mid {s['mid_jump_50_150']:+.3f}, late {s['late_jump_150_299']:+.3f})")
    with open(res_path, 'w') as f:
        json.dump({'runs': results, 'summary': summary}, f, indent=1)
    log('saved')


if __name__ == '__main__':
    main()
