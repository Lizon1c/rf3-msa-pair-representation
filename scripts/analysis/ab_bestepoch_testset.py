#!/usr/bin/env python3
"""AB-V1: best-epoch selection optimism + dual-validation trajectory agreement.

Collaborator concern 1: selecting the best epoch on a single val set may overfit
that val set. Fix/quantification: hold out TWO independent val sets (val1, val2),
select best epoch on val1, report the same epoch's rho on val2, and show the
val1/val2 epoch trajectories agree.

Design: RBD 500-subset (stride 8, E1 convention), cross-position 3-way split
(60% train / 20% val1 / 20% val2 of unique mutation sites), split seeds 100-104,
inits 7/107/207; otherwise July protocol (300 ep, BS=64, AdamW 1e-4/0.05, cosine,
MSE, raw bind_avg labels, per-position per-channel input std = current convention).
Per run: full 300-epoch val1/val2 Spearman trajectories; best-val1 epoch and
val2 at that epoch; best-val2 (oracle upper bound); final-epoch both; trajectory
Pearson r and argmax-epoch discrepancy. Cells: sp_no, sp_msa, sp_faesm, x_no_faesm, x_msa_faesm (conds via argv).

Run: CUDA_VISIBLE_DEVICES=0 python -B ab_bestepoch_testset.py [cond ...]
Output: /mnt/k/output_heads/rbd/ab_bestepoch/results.json
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
from e2_four_condition import XAttn, std_per_residue
from fusion_v3 import SinglePredictorV2, CrossAttnOrthoConcatFusion

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'ab_bestepoch'
OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda')
BS, EPOCHS, LR, WD, L = 64, 300, 1e-4, 0.05, 201
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


# ── data: 500-subset, metadata FILE order ──
meta = [json.loads(l) for l in open(RBD / 'metadata.jsonl') if l.strip()]
STRIDE = int(os.environ.get('AB1_STRIDE', '8'))
sub = list(range(0, len(meta), STRIDE))
y = np.array([meta[i]['bind_avg'] for i in sub], np.float32)
pos = np.array([meta[i]['site_rbd'] - 1 for i in sub])
log(f'subset N={len(sub)} mutants, {len(np.unique(pos))} unique sites')


def load_z_no():
    X = np.empty((len(sub), L, 128), np.float32)
    for k, i in enumerate(sub):
        z = torch.load(RBD / 'zii' / f"{meta[i]['mutant_id']}_zii.pt",
                       map_location='cpu', weights_only=True)
        if isinstance(z, dict):
            z = list(z.values())[0]
        X[k] = z[0].float().numpy()
    return torch.from_numpy(X)


def load_z_msa():
    X = np.empty((len(sub), L, 128), np.float32)
    for k, i in enumerate(sub):
        z = torch.load(RBD / 'zii_e1_full15' / f'mutant_{i:04d}_zii.pt',
                       map_location='cpu', weights_only=True)
        if isinstance(z, dict):
            z = list(z.values())[0]
        X[k] = z[0].float().numpy()
    return torch.from_numpy(X)


def load_faesm():
    X = np.empty((len(sub), L, 1280), np.float32)
    for k, i in enumerate(sub):
        X[k] = torch.load(RBD / 'faesm' / f"{meta[i]['mutant_id']}.pt",
                          map_location='cpu', weights_only=True)[:L].float().numpy()
    return torch.from_numpy(X)


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
    """Train once; record per-epoch Spearman on BOTH val sets."""
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
    conds = sys.argv[1:] or ['sp_no', 'sp_msa', 'x_msa_faesm']
    log(f'conds: {conds}')
    Z_no = std_per_residue(load_z_no()).to(DEVICE)
    log('z_no ready')
    Z_msa = std_per_residue(load_z_msa()).to(DEVICE)
    log('z_msa ready')
    F_aesm = None

    def inputs_for(c):
        nonlocal F_aesm
        if c == 'sp_no':
            return SinglePredictorV2(128), [Z_no]
        if c == 'sp_msa':
            return SinglePredictorV2(128), [Z_msa]
        if c == 'x_msa_faesm':
            if F_aesm is None:
                F_aesm = std_per_residue(load_faesm()).to(DEVICE)
                log('faesm ready')
            return CrossAttnOrthoConcatFusion(), [Z_msa, F_aesm]
        if c == 'sp_faesm':
            if F_aesm is None:
                F_aesm = std_per_residue(load_faesm()).to(DEVICE)
                log('faesm ready')
            return SinglePredictorV2(1280), [F_aesm]
        if c == 'x_no_faesm':
            if F_aesm is None:
                F_aesm = std_per_residue(load_faesm()).to(DEVICE)
                log('faesm ready')
            return XAttn(128, 1280), [Z_no, F_aesm]
        raise ValueError(c)

    res_path = OUT / f"results_{os.environ.get('AB1_TAG', 'main')}.json"
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

    # summary per cond
    summary = {}
    for c in conds:
        rs = [r for r in results if r['cond'] == c]
        if not rs:
            continue
        def m(k):
            return float(np.mean([r[k] for r in rs]))
        sel_opt = [r['best_v1'] - r['v2_at_best_v1'] for r in rs]   # selection optimism
        oracle = [r['best_v1'] - r['best_v2'] for r in rs]          # vs oracle best_v2
        final_gap = [r['final_v1'] - r['final_v2'] for r in rs]     # no-selection symmetry
        ep_gap = [abs(r['ep_best_v1'] - r['ep_best_v2']) for r in rs]
        summary[c] = {
            'n': len(rs),
            'best_v1': m('best_v1'), 'v2_at_best_v1': m('v2_at_best_v1'),
            'best_v2': m('best_v2'), 'final_v1': m('final_v1'), 'final_v2': m('final_v2'),
            'traj_corr': m('traj_corr'),
            'sel_optimism_mean': float(np.mean(sel_opt)),
            'sel_optimism_per_split': [float(np.mean([r['best_v1'] - r['v2_at_best_v1']
                                          for r in rs if r['split'] == sd]))
                                       for sd in [100, 101, 102, 103, 104]],
            'oracle_gap_mean': float(np.mean(oracle)),
            'final_gap_mean': float(np.mean(final_gap)),
            'argmax_ep_gap_mean': float(np.mean(ep_gap)),
        }
        s = summary[c]
        log(f"{c}: best_v1={s['best_v1']:.4f} v2@best1={s['v2_at_best_v1']:.4f} "
            f"best_v2={s['best_v2']:.4f} | sel_optim={s['sel_optimism_mean']:+.4f} "
            f"oracle_gap={s['oracle_gap_mean']:+.4f} final_gap={s['final_gap_mean']:+.4f} "
            f"traj_r={s['traj_corr']:.3f} argmax_Δep={s['argmax_ep_gap_mean']:.0f}")
    with open(res_path, 'w') as f:
        json.dump({'runs': results, 'summary': summary}, f, indent=1)
    log('saved')


if __name__ == '__main__':
    main()
