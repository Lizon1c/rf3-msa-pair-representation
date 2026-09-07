#!/usr/bin/env python3
"""D1 functional equivalence: does reconstructed Z_msa (from Z_no+FAESM) predict
bind_avg as well as real Z_msa? For each split: train tf2 recon on train, apply to
all samples, then SP(recon) vs SP(real Z_msa) vs SP(Z_no) on the same val set.
3 splits x 2 inits. Output: /mnt/k/output_heads/rbd/d1_distill/functional.json
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import (load_z_no, load_z_msa, load_faesm,
                               std_per_residue, cross_split, DEVICE, BS, EPOCHS, LR, WD, y, pos)
from d1_distill import TF
from fusion_v3 import SinglePredictorV2
from scipy import stats
from pathlib import Path
import json

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'd1_distill'


def main():
    Z_no = std_per_residue(load_z_no()).to(DEVICE)
    Z_msa = std_per_residue(load_z_msa()).to(DEVICE)
    F = std_per_residue(load_faesm()).to(DEVICE)
    Xin = torch.cat([Z_no, F], dim=-1)
    results = []
    for sd in [100, 101, 102]:
        tr, va = cross_split(sd)
        for init in [7, 107]:
            # train tf2 recon on train split, apply to train+val samples
            torch.manual_seed(init)
            model = TF(Xin.shape[-1], 2, 8).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            tr_t, va_t = torch.from_numpy(tr).to(DEVICE), torch.from_numpy(va).to(DEVICE)
            for ep in range(EPOCHS):
                model.train()
                idx = torch.randperm(len(tr), device=DEVICE)
                for bi in range(0, len(tr), BS):
                    b = idx[bi:bi + BS]
                    loss = nn.MSELoss()(model(Xin[tr_t][b]), Z_msa[tr_t][b])
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                rec_tr = torch.cat([model(Xin[tr_t][i:i + BS]) for i in range(0, len(tr), BS)])
                rec_va = torch.cat([model(Xin[va_t][i:i + BS]) for i in range(0, len(va), BS)])
            del model
            torch.cuda.empty_cache()
            def sp2(Xtr, Xva, ini):
                torch.manual_seed(ini)
                m = SinglePredictorV2(128).to(DEVICE)
                o = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
                sc = torch.optim.lr_scheduler.CosineAnnealingLR(o, EPOCHS)
                yt = torch.from_numpy(y[tr]).to(DEVICE)
                best = -1
                for ep in range(EPOCHS):
                    m.train(); idx = torch.randperm(len(tr), device=DEVICE)
                    for bi in range(0, len(tr), BS):
                        b = idx[bi:bi + BS]
                        loss = nn.MSELoss()(m(Xtr[b]), yt[b])
                        o.zero_grad(); loss.backward(); o.step()
                    sc.step(); m.eval()
                    with torch.no_grad():
                        pv = torch.cat([m(Xva[i:i + BS]) for i in range(0, len(va), BS)]).cpu().numpy()
                    r, _ = stats.spearmanr(pv, y[va]); best = max(best, r if not np.isnan(r) else -1)
                del m; torch.cuda.empty_cache()
                return best
            r_rec = sp2(rec_tr, rec_va, init)
            r_real = sp2(Z_msa[tr_t], Z_msa[va_t], init)
            r_no = sp2(Z_no[tr_t], Z_no[va_t], init)
            results.append({'split': sd, 'init': init, 'recon': r_rec, 'real': r_real, 'no': r_no})
            print(f"s{sd} i{init}: recon={r_rec:.4f} real={r_real:.4f} no={r_no:.4f}", flush=True)
            with open(OUT / 'functional.json', 'w') as f:
                json.dump({'runs': results}, f)
    print(f"\nrecon μ={np.mean([r['recon'] for r in results]):.4f} | "
          f"real μ={np.mean([r['real'] for r in results]):.4f} | "
          f"no μ={np.mean([r['no'] for r in results]):.4f}")
    print('saved', OUT / 'functional.json')


if __name__ == '__main__':
    main()
