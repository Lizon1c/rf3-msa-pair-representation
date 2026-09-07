#!/usr/bin/env python3
"""AB-V3: ESM-noise ladder on x_msa_faesm (RBD full set) — does noising FAESM
improve Z_msa+FAESM fusion, as it did for Z_no+FAESM fusion on CASP3?

fusion_as_validator paradigm (e26/noise_casp3): interpolation noise
x' = (1-nl)*x + nl*N(0,1)*sigma_token, sigma_token = per-token per-channel std
of the standardized FAESM tensor (computed once; ~1 post-standardization),
fresh resample per batch, train AND val noised at the same nl, 300 ep,
best-epoch val Spearman. Levels {0, 0.25, 0.5, 0.75, 1.0}.

Context and pre-registration:
- fusion_as_validator found the ESM-noise inverted-U on CASP3 (peak nl=0.75,
  +0.04) but NOT on RBD (monotone cost) — for Z_no+FAESM fusion.
- Here the fusion is Z_msa+FAESM, where FAESM's information marginal over
  Z_msa is ~0 on RBD while its encoder dominates late gradients (f/m~2.7).
  Mechanism hypothesis: FAESM's gradient dominance suppresses Z_msa learning
  (OGM-GE/pace family); noise may rebalance gradients and unlock Z_msa's
  contribution. Discriminating readout: does noised fusion exceed the sp_msa
  ceiling (archived 0.644 overall / 0.589 strict)? If yes, FAESM was actively
  interfering, not merely redundant. Gradient tracking (z_enc/f_enc late
  norms) tests the rebalancing directly.
- nl=0 arm reproduces the archived v3 grid (seeds 100-104, inits 7/107/207,
  70/30 cross-position) -> baseline 0.651 expected.
- Labels raw (RBD convention, unlike CASP3 noise scripts which standardized).

Run: CUDA_VISIBLE_DEVICES=1 python -B ab_esm_noise.py [nl ...]
Output: /mnt/k/output_heads/rbd/ab_esm_noise/results.json
"""
import json, os, time
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch, torch.nn as nn
from pathlib import Path
from scipy import stats

torch.set_float32_matmul_precision('high')
import sys
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import (load_z_msa, load_faesm, std_per_residue,
                               cross_split, DEVICE, BS, EPOCHS, LR, WD)
from fusion_v3 import CrossAttnOrthoConcatFusion

OUT = Path('/mnt/k/output_heads/rbd/ab_esm_noise')
OUT.mkdir(exist_ok=True)
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


log('loading + standardizing (overall, per-token per-channel)')
Z_msa = std_per_residue(load_z_msa()).to(DEVICE)
F = std_per_residue(load_faesm()).to(DEVICE)
esm_std = F.std(dim=0, keepdim=True)  # [1, L, 1280], per-token per-channel (~1)
log(f'Z_msa {tuple(Z_msa.shape)}  F {tuple(F.shape)}  esm_std mean={esm_std.mean():.3f}')
import e2_four_condition as _e2
y = _e2.y  # raw bind_avg labels, full metadata order


def grad_norms(model):
    out = {}
    for name, mod in [('z', model.z_enc), ('f', model.f_enc)]:
        s = 0.0
        for p in mod.parameters():
            if p.grad is not None:
                s += float(p.grad.norm() ** 2)
        out[name] = s ** 0.5
    return out


def run_one(nl, tr, va, init):
    torch.manual_seed(init)
    model = CrossAttnOrthoConcatFusion().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    tr_t = torch.from_numpy(tr).to(DEVICE)
    va_t = torch.from_numpy(va).to(DEVICE)
    yt = torch.from_numpy(y[tr]).to(DEVICE)
    yv = y[va]
    Ztr, Zva = Z_msa[tr_t], Z_msa[va_t]

    def f_noisy(ids):
        if nl == 0.0:
            return F[ids]
        return (1.0 - nl) * F[ids] + nl * torch.randn(len(ids), F.shape[1],
                                                      F.shape[2], device=DEVICE) * esm_std

    def fwd(z, f):
        out = model(z, f)
        return out[0] if isinstance(out, tuple) else out

    best, traj = -1, []
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        ep_g = {'z': 0.0, 'f': 0.0}
        nb = 0
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            ftr = f_noisy(tr_t[b])  # fresh noise per batch
            loss = nn.MSELoss()(fwd(Ztr[b], ftr), yt[b])
            opt.zero_grad(); loss.backward()
            g = grad_norms(model)
            ep_g['z'] += g['z']; ep_g['f'] += g['f']; nb += 1
            opt.step()
        sched.step()
        if ep >= EPOCHS - 50:
            g_late = {k: ep_g[k] / max(nb, 1) for k in ('z', 'f')}
            traj.append(g_late)
        model.eval()
        with torch.no_grad():
            fva = f_noisy(va_t)  # val noised at same nl (canonical protocol)
            pv = torch.cat([fwd(Zva[i:i + BS], fva[i:i + BS])
                            for i in range(0, len(va), BS)]).cpu().numpy()
        r, _ = stats.spearmanr(pv, yv)
        best = max(best, r if not np.isnan(r) else -1)
    gsum = {k: float(np.mean([t[k] for t in traj])) for k in ('z', 'f')} if traj else {}
    del model
    torch.cuda.empty_cache()
    return best, gsum


def main():
    levels = [float(a) for a in (sys.argv[1:] or ['0', '0.25', '0.5', '0.75', '1.0'])]
    log(f'noise levels: {levels}')
    res_path = OUT / 'results.json'
    done, results = set(), []
    if res_path.exists():
        prev = json.load(open(res_path))
        done = {(r['nl'], r['split'], r['init']) for r in prev['runs']}
        results = prev['runs']
    for nl in levels:
        for sd in [100, 101, 102, 103, 104]:
            tr, va = cross_split(sd)
            for init in [7, 107, 207]:
                if (nl, sd, init) in done:
                    continue
                r, g = run_one(nl, tr, va, init)
                results.append({'nl': nl, 'split': sd, 'init': init, 'val': float(r),
                                'grad_late': g})
                log(f'nl={nl:<4} s{sd} i{init}: {r:.4f} grads={ {k: round(v, 3) for k, v in g.items()} }')
                with open(res_path, 'w') as f:
                    json.dump({'runs': results}, f)

    summary = {}
    for nl in levels:
        rs = [r for r in results if r['nl'] == nl]
        if not rs:
            continue
        zs = [r['grad_late']['z'] for r in rs if r.get('grad_late')]
        fs = [r['grad_late']['f'] for r in rs if r.get('grad_late')]
        summary[f'nl{nl}'] = {'mean': float(np.mean([r['val'] for r in rs])),
                              'std': float(np.std([r['val'] for r in rs])),
                              'grad_z': float(np.mean(zs)) if zs else None,
                              'grad_f': float(np.mean(fs)) if fs else None}
        s = summary[f'nl{nl}']
        log(f"nl={nl:<4} μ={s['mean']:.4f} σ={s['std']:.4f} "
            f"grad z/f={s['grad_z']:.3f}/{s['grad_f']:.3f} f/m={s['grad_f']/max(s['grad_z'],1e-9):.2f}")
    # paired deltas vs nl=0
    base = {(r['split'], r['init']): r['val'] for r in results if r['nl'] == 0.0}
    for nl in levels:
        if nl == 0.0:
            continue
        arm = {(r['split'], r['init']): r['val'] for r in results if r['nl'] == nl}
        ks = sorted(set(base) & set(arm))
        if not ks:
            continue
        bysplit = {}
        for k in ks:
            bysplit.setdefault(k[0], []).append(arm[k] - base[k])
        ps = [float(np.mean(v)) for _, v in sorted(bysplit.items())]
        summary[f'nl{nl}']['vs_nl0'] = {'per_split': ps, 'mean': float(np.mean(ps))}
        log(f"nl{nl}-nl0 paired: μ={np.mean(ps):+.4f} per_split={['%+.3f' % p for p in ps]}")
    with open(res_path, 'w') as f:
        json.dump({'runs': results, 'summary': summary}, f, indent=1)
    log('saved')


if __name__ == '__main__':
    main()
