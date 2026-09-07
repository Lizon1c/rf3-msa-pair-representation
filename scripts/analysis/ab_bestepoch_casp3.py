#!/usr/bin/env python3
"""AB-V1 CASP3 replication: best-epoch selection optimism + dual-val trajectory
agreement + val1/val2 late-gap mechanism check, on CASP3 (N=1567, L=488,
69-row MSA, eff90 pinning −42% vs RBD's −26%).

Sister script of ab_bestepoch_testset.py (RBD), identical protocol: 60/20/20
cross-position splits (seeds 100-104), inits 7/107/207, 300 ep, BS=64,
AdamW 1e-4/0.05, cosine, MSE, raw labels, per-token per-channel std computed
on all samples (current convention — matches the RBD V1 runs for comparability).
Per run: full 300-epoch train/val1/val2 Spearman trajectories; best-val1 epoch
and val2 at that epoch; best-val2 (oracle); final-epoch both; trajectory Pearson.
Cells: sp_no, sp_msa, sp_faesm, x_no_faesm, x_msa_faesm.

CASP3 loaders follow e5_matrix_protein.py: Z_II row slices z[position-1] of the
full [488,488,128] pairs (zii/ noMSA, zii_msa_sp69/ with MSA); FAESM 244-token
tiled to 488 (documented tile confound, E41: ~harmless +0.005).

Pre-registered predictions (from RBD V1): (1) sp_msa shows a persistent
val1>val2 late gap, locked by ~ep5, 5/5 splits; if the gap is pinning-driven,
it should be LARGER here than RBD's −0.061 (stronger pinning). (2) sp_faesm
late gap ≈ 0 with high trajectory agreement (RBD: r=0.69). (3) x_msa_faesm
late gap ≈ 0 and its per-split gap ordering tracks sp_faesm's (RBD r=0.90),
not sp_msa's (RBD r=0.37).

Run: CUDA_VISIBLE_DEVICES=0 python -B ab_bestepoch_casp3.py [cond ...]
Output: /mnt/k/output_heads/casp3/ab_bestepoch/results_<AB1C_TAG>.json
"""
import json, os, time
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch, torch.nn as nn
from pathlib import Path
from scipy import stats

torch.set_float32_matmul_precision('high')
import sys
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import XAttn
from fusion_v3 import SinglePredictorV2, CrossAttnOrthoConcatFusion

BASE = Path('/mnt/k/output_heads/casp3')
OUT = BASE / 'ab_bestepoch'
OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda')
BS, EPOCHS, LR, WD, L = 64, 300, 1e-4, 0.05, 488
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


# ── data: CASP3 manifest, metadata FILE order ──
meta = [json.loads(l) for l in open(BASE / 'manifest.jsonl') if l.strip()]
STRIDE = int(os.environ.get('AB1C_STRIDE', '1'))
sub = list(range(0, len(meta), STRIDE))
y = np.array([meta[i]['label'] for i in sub], np.float32)
pos = np.array([meta[i]['position'] - 1 for i in sub])
log(f'N={len(sub)} mutants, {len(np.unique(pos))} unique sites')


def load_z(which):
    d = BASE / ('zii' if which == 'no' else 'zii_msa_sp69')
    X = np.empty((len(sub), L, 128), np.float32)
    for k, i in enumerate(sub):
        z = torch.load(d / f"{meta[i]['mutant_id']}_zii.pt",
                       map_location='cpu', weights_only=True)
        X[k] = z[meta[i]['position'] - 1].float().numpy()
        if k % 500 == 499:
            log(f'  z_{which} {k+1}/{len(sub)}')
    return torch.from_numpy(X)


def load_faesm():
    X = np.empty((len(sub), L, 1280), np.float32)
    for k, i in enumerate(sub):
        f = torch.load(BASE / 'faesm' / f"{meta[i]['mutant_id']}.pt",
                       map_location='cpu', weights_only=True)
        if isinstance(f, dict):
            f = list(f.values())[0]
        f = f.float()
        if f.shape[0] == 244:  # tile 244 -> 488 (documented tile caveat)
            f = f.repeat(2, 1)
        X[k] = f[:L].numpy()
        if k % 500 == 499:
            log(f'  faesm {k+1}/{len(sub)}')
    return torch.from_numpy(X)


def std_overall(X):
    """Per-token per-channel z-score over all samples (current convention,
    matches RBD V1 runs)."""
    X = X.clone()
    for i in range(X.shape[1]):
        mu = X[:, i].mean(0)
        sd = X[:, i].std(0).clamp_min(1e-8)
        X[:, i] = (X[:, i] - mu) / sd
    return X


def split3(seed):
    """Cross-position 60/20/20: train / val1 / val2 over unique mutation sites."""
    rng = np.random.RandomState(seed)
    ap = sorted(np.unique(pos))
    rng.shuffle(ap)
    n1 = max(1, int(len(ap) * 0.2))
    n2 = max(1, int(len(ap) * 0.2))
    vp1, vp2 = set(ap[:n1]), set(ap[n1:n1 + n2])
    tr = [i for i in range(len(pos)) if pos[i] not in vp1 and pos[i] not in vp2]
    va1 = [i for i in range(len(pos)) if pos[i] in vp1]
    va2 = [i for i in range(len(pos)) if pos[i] in vp2]
    return np.array(tr), np.array(va1), np.array(va2)


def train_2val(model, inputs, tr, va1, va2, init):
    """Train once; record per-epoch Spearman on train, val1, val2."""
    torch.manual_seed(init)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    tr_t = torch.from_numpy(tr).to(DEVICE)
    v1_t = torch.from_numpy(va1).to(DEVICE)
    v2_t = torch.from_numpy(va2).to(DEVICE)
    yt = torch.from_numpy(y[tr]).to(DEVICE)
    yv1, yv2 = y[va1], y[va2]
    ins_tr = [x[tr_t] for x in inputs]
    ins_v1 = [x[v1_t] for x in inputs]
    ins_v2 = [x[v2_t] for x in inputs]

    def fwd(*xs):
        out = model(*xs) if len(xs) > 1 else model(xs[0])
        return out[0] if isinstance(out, tuple) else out

    def eval_rho(ins_v, yv):
        with torch.no_grad():
            pv = torch.cat([fwd(*[x[i:i + BS] for x in ins_v])
                            for i in range(0, len(yv), BS)]).cpu().numpy()
        r, _ = stats.spearmanr(pv, yv)
        return float(r) if not np.isnan(r) else -1.0

    traj_tr, traj1, traj2 = [], [], []
    ytr = y[tr]
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(fwd(*[x[b] for x in ins_tr]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        traj_tr.append(eval_rho(ins_tr, ytr))
        traj1.append(eval_rho(ins_v1, yv1))
        traj2.append(eval_rho(ins_v2, yv2))

    t1, t2 = np.array(traj1), np.array(traj2)
    b1 = int(np.argmax(t1)); b2 = int(np.argmax(t2))
    corr = float(np.corrcoef(t1, t2)[0, 1])
    del model
    torch.cuda.empty_cache()
    return {'best_v1': float(t1[b1]), 'ep_best_v1': b1, 'v2_at_best_v1': float(t2[b1]),
            'best_v2': float(t2[b2]), 'ep_best_v2': b2, 'v1_at_best_v2': float(t1[b2]),
            'final_v1': float(t1[-1]), 'final_v2': float(t2[-1]),
            'best_tr': float(max(traj_tr)), 'final_tr': float(traj_tr[-1]),
            'traj_corr': corr, 'traj_tr': [round(v, 4) for v in traj_tr],
            'traj1': [round(v, 4) for v in traj1],
            'traj2': [round(v, 4) for v in traj2]}


def main():
    conds = sys.argv[1:] or ['sp_no', 'sp_msa', 'sp_faesm', 'x_no_faesm', 'x_msa_faesm']
    log(f'conds: {conds}')
    Z_no = std_overall(load_z('no')).to(DEVICE)
    log('z_no ready')
    Z_msa = std_overall(load_z('msa')).to(DEVICE)
    log('z_msa ready')
    F_aesm = None

    def inputs_for(c):
        nonlocal F_aesm
        if c == 'sp_no':
            return SinglePredictorV2(128), [Z_no]
        if c == 'sp_msa':
            return SinglePredictorV2(128), [Z_msa]
        if c == 'sp_faesm':
            if F_aesm is None:
                F_aesm = std_overall(load_faesm()).to(DEVICE)
                log('faesm ready')
            return SinglePredictorV2(1280), [F_aesm]
        if c == 'x_no_faesm':
            if F_aesm is None:
                F_aesm = std_overall(load_faesm()).to(DEVICE)
                log('faesm ready')
            return XAttn(128, 1280), [Z_no, F_aesm]
        if c == 'x_msa_faesm':
            if F_aesm is None:
                F_aesm = std_overall(load_faesm()).to(DEVICE)
                log('faesm ready')
            return CrossAttnOrthoConcatFusion(), [Z_msa, F_aesm]
        raise ValueError(c)

    res_path = OUT / f"results_{os.environ.get('AB1C_TAG', 'main')}.json"
    done, results = set(), []
    if res_path.exists():
        prev = json.load(open(res_path))
        done = {(r['cond'], r['split'], r['init']) for r in prev['runs']}
        results = prev['runs']
    for c in conds:
        for sd in [100, 101, 102, 103, 104]:
            tr, va1, va2 = split3(sd)
            for init in [7, 107, 207]:
                if (c, sd, init) in done:
                    continue
                model, inputs = inputs_for(c)
                r = train_2val(model, inputs, tr, va1, va2, init)
                r.update({'cond': c, 'split': sd, 'init': init,
                          'n_tr': len(tr), 'n_v1': len(va1), 'n_v2': len(va2)})
                results.append(r)
                log(f'{c} s{sd} i{init}: best_v1={r["best_v1"]:.3f}(ep{r["ep_best_v1"]}) '
                    f'v2@same={r["v2_at_best_v1"]:.3f} best_v2={r["best_v2"]:.3f}(ep{r["ep_best_v2"]}) '
                    f'final={r["final_v1"]:.3f}/{r["final_v2"]:.3f} traj_r={r["traj_corr"]:.3f}')
                with open(res_path, 'w') as f:
                    json.dump({'runs': results}, f)

    summary = {}
    for c in conds:
        rs = [r for r in results if r['cond'] == c]
        if not rs:
            continue
        def m(k):
            return float(np.mean([r[k] for r in rs]))
        sel = [r['best_v1'] - r['v2_at_best_v1'] for r in rs]
        summary[c] = {
            'n': len(rs),
            'best_v1': m('best_v1'), 'v2_at_best_v1': m('v2_at_best_v1'),
            'best_v2': m('best_v2'), 'final_v1': m('final_v1'), 'final_v2': m('final_v2'),
            'traj_corr': m('traj_corr'),
            'sel_optimism_mean': float(np.mean(sel)),
            'sel_optimism_per_split': [float(np.mean([r['best_v1'] - r['v2_at_best_v1']
                                          for r in rs if r['split'] == sd]))
                                       for sd in [100, 101, 102, 103, 104]],
        }
        s = summary[c]
        log(f"{c}: best_v1={s['best_v1']:.4f} v2@best1={s['v2_at_best_v1']:.4f} "
            f"best_v2={s['best_v2']:.4f} final={s['final_v1']:.4f}/{s['final_v2']:.4f} "
            f"sel_optim={s['sel_optimism_mean']:+.4f} traj_r={s['traj_corr']:.3f}")
    with open(res_path, 'w') as f:
        json.dump({'runs': results, 'summary': summary}, f, indent=1)
    log('saved')


if __name__ == '__main__':
    main()
