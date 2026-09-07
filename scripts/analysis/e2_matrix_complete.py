#!/usr/bin/env python3
"""E2-matrix: complete the fusion matrix (V-series cells) on clean full-set RBD data.

New cells beyond the four-condition table:
  sp_faesm      SP(FAESM)
  x_no_faesm    XAttn(Z_no, FAESM)
  x_faesm_zeros XAttn(FAESM, zeros)      [ESM-side dead-stream control]
  tri           TriAttn(Z_no, Z_msa, FAESM)
  tri_z         TriAttn(zeros, Z_msa, FAESM)  [same-stream-count dead control; E27-style]

Same-stream-count information estimates:
  FAESM info over Z_msa  = x_msa_faesm − x_msa_zeros  (four-condition)
  FAESM info over Z_no   = x_no_faesm  − x_no_zeros   (four-condition)
  Z_no  info over (Z_msa+FAESM) = tri − tri_z

TriAttn: per-residue 3-token interaction (e12 style) → concat 384 → PE → 2L joint →
AttnPool → MLP. Protocol: 5 splits × 3 inits, 300 ep, BS=64, AdamW 1e-4/0.05,
cosine, MSE, best-epoch val Spearman, raw bind_avg. Output: results_<TAG>.json
"""
import json, os, time
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from scipy import stats

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
torch.set_float32_matmul_precision('high')
import sys
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import (XAttn, load_z_no, load_z_msa, load_faesm,
                               std_per_residue, cross_split, train_one,
                               DEVICE, L, y, BS)
from fusion_v3 import SinglePredictorV2, Encoder, AttnPool, PosEnc

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'e2_matrix_complete'
OUT.mkdir(exist_ok=True)
FINAL_DIM, DM3, NH, DROPOUT = 128, 384, 8, 0.1
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


class TriAttn(nn.Module):
    def __init__(self, d1=128, d2=128, d3=1280):
        super().__init__()
        self.e1, self.e2, self.e3 = Encoder(d1), Encoder(d2), Encoder(d3)
        el = nn.TransformerEncoderLayer(d_model=FINAL_DIM, nhead=NH // 2,
                                        dim_feedforward=FINAL_DIM * 4,
                                        dropout=DROPOUT, batch_first=True)
        self.interaction = nn.TransformerEncoder(el, num_layers=1)
        self.pe = PosEnc(DM3)
        el2 = nn.TransformerEncoderLayer(d_model=DM3, nhead=NH,
                                         dim_feedforward=DM3 * 4,
                                         dropout=DROPOUT, batch_first=True)
        self.joint_enc = nn.TransformerEncoder(el2, num_layers=2)
        self.pool = AttnPool(DM3)
        self.head = nn.Sequential(nn.Linear(DM3, DM3 // 2), nn.GELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(DM3 // 2, 1))

    def forward(self, x1, x2, x3):
        a, b, c = self.e1(x1), self.e2(x2), self.e3(x3)
        B, L_, D = a.shape
        tokens = torch.stack([a, b, c], dim=2).view(B * L_, 3, D)
        inter = self.interaction(tokens).view(B, L_, 3, D)
        j = torch.cat([inter[:, :, 0, :], inter[:, :, 1, :], inter[:, :, 2, :]], dim=-1)
        j = self.pe(j)
        j = self.joint_enc(j)
        j = self.pool(j)
        return self.head(j).squeeze(-1)


def grad_norms(model, groups):
    out = {}
    for name, mod in groups.items():
        s = 0.0
        for p in mod.parameters():
            if p.grad is not None:
                s += float(p.grad.norm() ** 2)
        out[name] = s ** 0.5
    return out


def train_one_grad(model, inputs, tr, va, init, groups):
    """train_one + per-encoder gradient-norm tracking (mean over batches per epoch)."""
    torch.manual_seed(init)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300)
    yt = torch.from_numpy(y[tr]).to(DEVICE)
    tr_t, va_t = torch.from_numpy(tr).to(DEVICE), torch.from_numpy(va).to(DEVICE)
    ins_tr = [x[tr_t] for x in inputs]
    ins_va = [x[va_t] for x in inputs]

    def fwd(*xs):
        out = model(*xs) if len(xs) > 1 else model(xs[0])
        return out[0] if isinstance(out, tuple) else out

    best, traj = -1, []
    for ep in range(300):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        ep_g = {k: 0.0 for k in groups}
        nb = 0
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(fwd(*[x[b] for x in ins_tr]), yt[b])
            opt.zero_grad()
            loss.backward()
            g = grad_norms(model, groups)
            for k in groups:
                ep_g[k] += g[k]
            nb += 1
            opt.step()
        sched.step()
        traj.append({k: v / max(nb, 1) for k, v in ep_g.items()})
        model.eval()
        with torch.no_grad():
            pv = torch.cat([fwd(*[x[i:i + BS] for x in ins_va])
                            for i in range(0, len(va), BS)]).cpu().numpy()
        r, _ = stats.spearmanr(pv, y[va])
        best = max(best, r if not np.isnan(r) else -1)
    del model
    torch.cuda.empty_cache()
    return best, traj


def main():
    conds = sys.argv[1:] or ['sp_faesm', 'x_no_faesm', 'x_faesm_zeros', 'tri', 'tri_z']
    log(f'conds: {conds}')
    Z_no = std_per_residue(load_z_no()).to(DEVICE)
    log('z_no ready')
    Z_msa = std_per_residue(load_z_msa()).to(DEVICE)
    log('z_msa ready')
    F_aesm = std_per_residue(load_faesm()).to(DEVICE)
    log('faesm ready')
    Zeros = torch.zeros_like(Z_no)

    def inputs_for(c):
        if c == 'sp_faesm':
            return SinglePredictorV2(1280), [F_aesm]
        if c == 'x_no_faesm':
            return XAttn(128, 1280), [Z_no, F_aesm]
        if c == 'x_faesm_zeros':
            return XAttn(1280, 128), [F_aesm, Zeros]
        if c == 'tri':
            return TriAttn(128, 128, 1280), [Z_no, Z_msa, F_aesm]
        if c == 'tri_z':
            return TriAttn(128, 128, 1280), [Zeros, Z_msa, F_aesm]
        raise ValueError(c)

    res_path = OUT / f"results_{os.environ.get('E2M_TAG', 'main')}.json"
    done, results = set(), []
    if res_path.exists():
        prev = json.load(open(res_path))
        done = {(r['cond'], r['split'], r['init']) for r in prev['runs']}
        results = prev['runs']
    for c in conds:
        for sd in [100, 101, 102, 103, 104]:
            tr, va = cross_split(sd)
            for init in [7, 107, 207]:
                if (c, sd, init) in done:
                    continue
                model, inputs = inputs_for(c)
                if isinstance(model, TriAttn):
                    groups = {'z': model.e1, 'm': model.e2, 'f': model.e3}
                elif isinstance(model, XAttn):
                    groups = {'z': model.z_enc, 'f': model.f_enc}
                else:
                    groups = {'f': model.enc}
                r, traj = train_one_grad(model, inputs, tr, va, init, groups)
                late = traj[-50:] if len(traj) >= 50 else traj
                gsum = {k: float(np.mean([t[k] for t in late])) for k in groups}
                results.append({'cond': c, 'split': sd, 'init': init, 'val': r,
                                'grad_late': gsum, 'grad_traj': traj})
                log(f'{c} s{sd} i{init}: {r:.4f} grads={ {k: round(v, 3) for k, v in gsum.items()} }')
                with open(res_path, 'w') as f:
                    json.dump({'runs': results}, f)

    summary = {}
    for c in conds:
        vals = [r['val'] for r in results if r['cond'] == c]
        summary[c] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        log(f"{c}: μ={summary[c]['mean']:.4f} σ={summary[c]['std']:.4f}")
    with open(res_path, 'w') as f:
        json.dump({'runs': results, 'summary': summary}, f, indent=1)
    log('saved')


if __name__ == '__main__':
    main()
