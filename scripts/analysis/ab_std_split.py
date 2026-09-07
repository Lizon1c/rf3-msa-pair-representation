#!/usr/bin/env python3
"""AB-V2: preprocessing-leakage A/B — overall vs split-first standardization.

Collaborator concern 2: standardization fit on ALL samples before the split is
preprocessing leakage; the correct boundary is to fit per-position per-channel
mean/std on TRAIN only and apply the frozen stats to val/test.

NOTE: "per-position" here = per TOKEN position (residue index), while splits are
by MUTATION site — the two are orthogonal, so overall (pre-split) stats DO pool
val mutants into the stats applied to train. This experiment quantifies whether
that boundary violation matters.

Arms (labels raw in ALL arms; only input standardization varies):
  none     no input standardization
  overall  current convention: per-token per-channel stats over ALL mutants,
           computed before the split (std_per_residue)
  strict   collaborator boundary: split first; per-token per-channel stats from
           TRAIN mutants only; frozen stats applied to val
  tglob    per-channel GLOBAL stats pooled over train (all tokens x train
           mutants), applied to both sides — the non-degenerate "train-fit stats
           transferred across positions" variant (AGENTS.md convention (b))

Design: RBD 500-subset (stride 8), cross-position 70/30 splits seeds 100-104
(identical to the audited runs), inits 7/107/207, 300 ep, BS=64, AdamW 1e-4/0.05,
cosine, MSE, best-epoch val Spearman (full set when AB2_STRIDE=1). Cells:
sp_no, sp_msa, x_no_faesm, x_msa_faesm, x_no_zeros, x_msa_zeros, tri, tri_z
(single-modality + fusion matrix cells incl. dead-stream controls; conds
selectable via argv). Paired per (cell, split, init) across arms.

Run: CUDA_VISIBLE_DEVICES=1 python -B ab_std_split.py [cond ...]
Output: /mnt/k/output_heads/rbd/ab_std_split/results_<AB2_TAG>.json
(default tag 'main'). Archived: results_main.json (subset, 4 arms × 4 cells),
results_full.json (full set, overall/strict × sp_no/sp_msa),
results_full_fusion.json (full set, strict × fusion cells — pair against the
archived v3 overall run, identical split/init grid).
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
from e2_four_condition import XAttn
from e2_matrix_complete import TriAttn
from fusion_v3 import SinglePredictorV2, CrossAttnOrthoConcatFusion

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'ab_std_split'
OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda')
BS, EPOCHS, LR, WD, L = 64, 300, 1e-4, 0.05, 201
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


# ── data: 500-subset, metadata FILE order ──
meta = [json.loads(l) for l in open(RBD / 'metadata.jsonl') if l.strip()]
STRIDE = int(os.environ.get('AB2_STRIDE', '8'))
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


# ── standardization arms ──
def std_none(X):
    return X.clone()


def std_overall(X):
    """Current convention: per-token per-channel over ALL mutants, pre-split."""
    X = X.clone()
    for i in range(X.shape[1]):
        mu = X[:, i].mean(0)
        sd = X[:, i].std(0).clamp_min(1e-8)
        X[:, i] = (X[:, i] - mu) / sd
    return X


def std_strict(X, tr, va):
    """Collaborator boundary: per-token per-channel stats from TRAIN mutants
    only; the frozen stats are applied to val."""
    X = X.clone()
    tr_t = torch.from_numpy(tr).long()
    va_t = torch.from_numpy(va).long()
    for i in range(X.shape[1]):
        mu = X[tr_t, i].mean(0)
        sd = X[tr_t, i].std(0).clamp_min(1e-8)
        X[tr_t, i] = (X[tr_t, i] - mu) / sd
        X[va_t, i] = (X[va_t, i] - mu) / sd
    return X


def std_tglob(X, tr, va):
    """Per-channel GLOBAL: one mean/std vector per channel pooled over all
    tokens of train mutants; applied to both sides."""
    X = X.clone()
    tr_t = torch.from_numpy(tr).long()
    va_t = torch.from_numpy(va).long()
    C = X.shape[2]
    tr_flat = X[tr_t].reshape(-1, C)
    mu = tr_flat.mean(0)
    sd = tr_flat.std(0).clamp_min(1e-8)
    X[tr_t] = (X[tr_t] - mu) / sd
    X[va_t] = (X[va_t] - mu) / sd
    return X


ARMS = {'none': std_none, 'overall': std_overall, 'strict': std_strict, 'tglob': std_tglob}


def cross_split(seed):
    rng = np.random.RandomState(seed)
    ap = sorted(np.unique(pos))
    rng.shuffle(ap)
    vp = set(ap[:max(1, int(len(ap) * 0.3))])
    tr = [i for i in range(len(pos)) if pos[i] not in vp]
    va = [i for i in range(len(pos)) if pos[i] in vp]
    return np.array(tr), np.array(va)


def train_one(model, inputs, tr, va, init):
    torch.manual_seed(init)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    tr_t = torch.from_numpy(tr).to(DEVICE)
    va_t = torch.from_numpy(va).to(DEVICE)
    yt = torch.from_numpy(y[tr]).to(DEVICE)
    yv = y[va]
    ins_tr = [x[tr_t] for x in inputs]
    ins_va = [x[va_t] for x in inputs]

    def fwd(*xs):
        out = model(*xs) if len(xs) > 1 else model(xs[0])
        return out[0] if isinstance(out, tuple) else out

    best = -1
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(fwd(*[x[b] for x in ins_tr]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([fwd(*[x[i:i + BS] for x in ins_va])
                            for i in range(0, len(va), BS)]).cpu().numpy()
        r, _ = stats.spearmanr(pv, yv)
        best = max(best, r if not np.isnan(r) else -1)
    del model
    torch.cuda.empty_cache()
    return best


def main():
    conds = sys.argv[1:] or ['sp_no', 'sp_msa', 'x_no_faesm', 'x_msa_faesm']
    log(f'conds: {conds}')
    raw = {'z_no': load_z_no(), 'z_msa': load_z_msa(), 'f': load_faesm()}
    log('raw tensors loaded')

    def tensors_for(arm, tr, va):
        """Standardize a fresh copy of the raw tensors under the given arm."""
        f = ARMS[arm]
        if arm in ('none', 'overall'):
            Z_no = f(raw['z_no']).to(DEVICE)
            Z_msa = f(raw['z_msa']).to(DEVICE)
            F = f(raw['f']).to(DEVICE)
        else:
            Z_no = f(raw['z_no'], tr, va).to(DEVICE)
            Z_msa = f(raw['z_msa'], tr, va).to(DEVICE)
            F = f(raw['f'], tr, va).to(DEVICE)
        return Z_no, Z_msa, F

    def inputs_for(c, Z_no, Z_msa, F, Zeros):
        if c == 'sp_no':
            return SinglePredictorV2(128), [Z_no]
        if c == 'sp_msa':
            return SinglePredictorV2(128), [Z_msa]
        if c == 'x_no_faesm':
            return XAttn(128, 1280), [Z_no, F]
        if c == 'x_msa_faesm':
            return CrossAttnOrthoConcatFusion(), [Z_msa, F]
        if c == 'x_no_zeros':
            return XAttn(128, 128), [Z_no, Zeros]
        if c == 'x_msa_zeros':
            return XAttn(128, 128), [Z_msa, Zeros]
        if c == 'tri':
            return TriAttn(128, 128, 1280), [Z_no, Z_msa, F]
        if c == 'tri_z':
            return TriAttn(128, 128, 1280), [Zeros, Z_msa, F]
        raise ValueError(c)

    arm_list = os.environ.get('AB2_ARMS', 'none,overall,strict,tglob').split(',')
    res_path = OUT / f"results_{os.environ.get('AB2_TAG', 'main')}.json"
    done, results = set(), []
    if res_path.exists():
        prev = json.load(open(res_path))
        done = {(r['arm'], r['cond'], r['split'], r['init']) for r in prev['runs']}
        results = prev['runs']
    for arm in arm_list:
        for c in conds:
            for sd in [100, 101, 102, 103, 104]:
                tr, va = cross_split(sd)
                todo = [i for i in [7, 107, 207] if (arm, c, sd, i) not in done]
                if not todo:
                    continue
                Z_no, Z_msa, F = tensors_for(arm, tr, va)
                Zeros = torch.zeros_like(Z_no)
                for init in todo:
                    model, inputs = inputs_for(c, Z_no, Z_msa, F, Zeros)
                    r = train_one(model, inputs, tr, va, init)
                    results.append({'arm': arm, 'cond': c, 'split': sd,
                                    'init': init, 'val': float(r)})
                    log(f'{arm:<8} {c:<12} s{sd} i{init}: {r:.4f}')
                    with open(res_path, 'w') as f:
                        json.dump({'runs': results}, f)
                del Z_no, Z_msa, F
                torch.cuda.empty_cache()

    # summary + paired deltas vs overall
    summary = {}
    for c in conds:
        summary[c] = {}
        for arm in arm_list:
            vals = [r['val'] for r in results if r['cond'] == c and r['arm'] == arm]
            if vals:
                summary[c][arm] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        # paired per (split, init) vs overall, per-split means
        def paired(arm):
            da = {(r['split'], r['init']): r['val'] for r in results
                  if r['cond'] == c and r['arm'] == arm}
            db = {(r['split'], r['init']): r['val'] for r in results
                  if r['cond'] == c and r['arm'] == 'overall'}
            ks = sorted(set(da) & set(db))
            if not ks:
                return None
            bysplit = {}
            for k in ks:
                bysplit.setdefault(k[0], []).append(da[k] - db[k])
            return {'per_split': [float(np.mean(v)) for _, v in sorted(bysplit.items())],
                    'mean': float(np.mean([da[k] - db[k] for k in ks])),
                    'per_run_abs': float(np.mean([abs(da[k] - db[k]) for k in ks]))}
        for arm in [a for a in arm_list if a != 'overall']:
            p = paired(arm)
            if p:
                summary[c][f'{arm}-overall'] = p
                log(f"{c} {arm}-overall: μ={p['mean']:+.4f} per_split="
                    f"{['%+.3f' % d for d in p['per_split']]} mean|Δ|={p['per_run_abs']:.4f}")
        for arm in arm_list:
            if arm in summary[c]:
                s = summary[c][arm]
                log(f"{c} {arm}: μ={s['mean']:.4f} σ={s['std']:.4f}")
    with open(res_path, 'w') as f:
        json.dump({'runs': results, 'summary': summary}, f, indent=1)
    log('saved')


if __name__ == '__main__':
    main()
